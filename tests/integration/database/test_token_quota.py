from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

from api.quota import (
    QuotaExceededError,
    QuotaNotFoundError,
    reserve_tokens,
    settle_tokens,
)
from database.models import User


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


@pytest.fixture
def quota():
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(bind=engine)
    user_id = uuid.uuid4()
    with factory() as session, session.begin():
        session.add(User(id=user_id, username=f"quota-{user_id.hex[:8]}", password_hash="x"))

    def set_usage(*, limit, used, period_end):
        with factory() as session, session.begin():
            session.execute(
                text(
                    """
                    INSERT INTO user_usage
                        (id, user_id, token_limit, tokens_used, period_start, period_end)
                    VALUES (gen_random_uuid(), :u, :limit, :used, :start, :end)
                    ON CONFLICT (user_id) DO UPDATE SET
                        token_limit = :limit, tokens_used = :used,
                        period_start = :start, period_end = :end
                    """
                ),
                {"u": user_id, "limit": limit, "used": used,
                 "start": period_end - timedelta(days=30), "end": period_end},
            )

    def usage():
        with factory() as session:
            return session.execute(
                text("SELECT tokens_used, period_end FROM user_usage WHERE user_id = :u"),
                {"u": user_id},
            ).one()

    try:
        yield factory, user_id, set_usage, usage
    finally:
        with factory() as session, session.begin():
            session.execute(delete(User).where(User.id == user_id))
        engine.dispose()


FUTURE = datetime.now(timezone.utc) + timedelta(days=10)


def test_reserve_then_settle_to_actual_usage(quota):
    factory, user_id, set_usage, usage = quota
    set_usage(limit=100_000, used=1_000, period_end=FUTURE)

    with factory() as db:
        assert reserve_tokens(db, user_id, 8_000) == 8_000
    assert usage().tokens_used == 9_000

    with factory() as db:
        settle_tokens(db, user_id, reserved=8_000, actual=1_500)
    assert usage().tokens_used == 2_500


def test_reservation_is_capped_by_remaining_quota(quota):
    factory, user_id, set_usage, usage = quota
    set_usage(limit=10_000, used=7_000, period_end=FUTURE)

    with factory() as db:
        assert reserve_tokens(db, user_id, 8_000) == 3_000
    with factory() as db, pytest.raises(QuotaExceededError):
        reserve_tokens(db, user_id, 8_000)
    assert usage().tokens_used == 10_000


def test_expired_period_is_reset(quota):
    factory, user_id, set_usage, usage = quota
    set_usage(limit=10_000, used=10_000, period_end=datetime.now(timezone.utc) - timedelta(days=1))

    with factory() as db:
        assert reserve_tokens(db, user_id, 8_000) == 8_000

    row = usage()
    assert row.tokens_used == 8_000
    assert row.period_end > datetime.now(timezone.utc)
    assert row.period_end.day == 1


def test_concurrent_reservations_cannot_overrun_quota(quota):
    factory, user_id, set_usage, usage = quota
    set_usage(limit=10_000, used=9_000, period_end=FUTURE)
    barrier = threading.Barrier(8)
    outcomes = []

    def attempt():
        barrier.wait()
        with factory() as db:
            try:
                outcomes.append(reserve_tokens(db, user_id, 8_000))
            except QuotaExceededError:
                outcomes.append("denied")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes, key=str) == [1_000] + ["denied"] * 7
    assert usage().tokens_used == 10_000


def test_missing_quota_row(quota):
    factory, user_id, _set_usage, _usage = quota

    with factory() as db, pytest.raises(QuotaNotFoundError):
        reserve_tokens(db, user_id, 8_000)
