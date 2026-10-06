from __future__ import annotations

import os
import uuid

import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

import api.main as main
from auth.rate_limit import LoginRateLimiter
from auth.security import generate_totp_secret, hash_password
from auth.service import create_session
from database.models import User


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)

PASSWORD = "correct horse battery"


@pytest.fixture
def env(monkeypatch):
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(main.app.state, "session_factory", factory, raising=False)
    monkeypatch.setattr(main, "login_rate_limiter", LoginRateLimiter(max_attempts=5))
    created: list[uuid.UUID] = []

    def make_user(*, role="USER", totp=False):
        user = User(
            id=uuid.uuid4(),
            username=f"{role.lower()}-{uuid.uuid4().hex[:8]}",
            password_hash=hash_password(PASSWORD),
            role=role,
            totp_secret=generate_totp_secret() if totp else None,
            totp_enabled=totp,
        )
        with factory() as session, session.begin():
            session.add(user)
        created.append(user.id)
        return user

    try:
        yield TestClient(main.app), factory, make_user
    finally:
        with factory() as session, session.begin():
            session.execute(delete(User).where(User.id.in_(created)))
        engine.dispose()


def _login(client, user, password=PASSWORD):
    return client.post(
        "/api/v1/auth/login", json={"username": user.username, "password": password}
    )


def test_new_user_defaults_to_user_role(env):
    _client, factory, _make_user = env
    user = User(id=uuid.uuid4(), username=f"plain-{uuid.uuid4().hex[:8]}", password_hash="x")
    with factory() as session, session.begin():
        session.add(user)
    try:
        with factory() as session:
            assert session.get(User, user.id).role == "USER"
    finally:
        with factory() as session, session.begin():
            session.execute(delete(User).where(User.id == user.id))


def test_admin_without_totp_cannot_log_in(env):
    client, _factory, make_user = env
    admin = make_user(role="ADMIN", totp=False)

    response = _login(client, admin)

    assert response.status_code == 403
    assert "lawchat_session" not in response.headers.get("set-cookie", "")


def test_admin_password_alone_gets_no_session(env):
    client, _factory, make_user = env
    admin = make_user(role="ADMIN", totp=True)

    response = _login(client, admin)

    assert response.status_code == 200
    assert response.json()["requires_totp"] is True
    assert "lawchat_session" not in response.headers.get("set-cookie", "")


def test_admin_totp_code_cannot_be_replayed(env):
    client, _factory, make_user = env
    admin = make_user(role="ADMIN", totp=True)
    body = {
        "username": admin.username,
        "password": PASSWORD,
        "code": pyotp.TOTP(admin.totp_secret).now(),
    }

    first = client.post("/api/v1/auth/totp", json=body)
    second = client.post("/api/v1/auth/totp", json=body)

    assert first.status_code == 200
    assert "lawchat_session" in first.headers.get("set-cookie", "")
    assert second.status_code == 401


def test_existing_admin_session_without_totp_is_refused(env):
    client, factory, make_user = env
    admin = make_user(role="ADMIN", totp=False)
    with factory() as session:
        token = create_session(session, session.get(User, admin.id))

    client.cookies.set("lawchat_session", token)
    response = client.get("/api/v1/admin/ping")

    assert response.status_code == 403


def test_sixth_failed_login_is_rate_limited(env):
    client, _factory, make_user = env
    user = make_user()

    statuses = [_login(client, user, password="wrong").status_code for _ in range(5)]
    locked = _login(client, user)

    assert statuses == [401] * 5
    assert locked.status_code == 429
    assert int(locked.headers["retry-after"]) > 0
