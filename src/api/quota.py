"""Per-user monthly token quota: reserve before an LLM run, settle after it.

Reserving up front in one atomic UPDATE keeps concurrent requests from all
passing a read-only check and overrunning the limit together.
"""
from __future__ import annotations

import os
from uuid import UUID

from sqlalchemy import text


DEFAULT_RESERVE_TOKENS = 8000


class QuotaNotFoundError(LookupError):
    pass


class QuotaExceededError(RuntimeError):
    pass


def reserve_amount() -> int:
    return max(int(os.getenv("LAWCHAT_QUOTA_RESERVE_TOKENS", DEFAULT_RESERVE_TOKENS)), 1)


def reserve_tokens(db, user_id: UUID, amount: int) -> int:
    """Reserve up to ``amount`` tokens and return how many were reserved.

    A period that has ended is rolled over to the current calendar month (UTC)
    with usage reset to zero. Raises QuotaExceededError when nothing is left.
    """
    db.execute(
        text(
            """
            UPDATE user_usage
            SET
                tokens_used = 0,
                period_start = date_trunc('month', now() AT TIME ZONE 'UTC')
                    AT TIME ZONE 'UTC',
                period_end = (
                    date_trunc('month', now() AT TIME ZONE 'UTC') + interval '1 month'
                ) AT TIME ZONE 'UTC',
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id
              AND period_end <= now()
            """
        ),
        {"user_id": user_id},
    )
    reserved = db.execute(
        text(
            """
            WITH current_usage AS (
                SELECT tokens_used, token_limit
                FROM user_usage
                WHERE user_id = :user_id
                FOR UPDATE
            )
            UPDATE user_usage AS usage
            SET
                tokens_used = current_usage.tokens_used + LEAST(
                    :amount, current_usage.token_limit - current_usage.tokens_used
                ),
                updated_at = CURRENT_TIMESTAMP
            FROM current_usage
            WHERE usage.user_id = :user_id
              AND current_usage.tokens_used < current_usage.token_limit
            RETURNING LEAST(
                :amount, current_usage.token_limit - current_usage.tokens_used
            ) AS reserved
            """
        ),
        {"user_id": user_id, "amount": amount},
    ).scalar()
    if reserved is None:
        exists = db.execute(
            text("SELECT 1 FROM user_usage WHERE user_id = :user_id"),
            {"user_id": user_id},
        ).scalar()
        db.rollback()
        if exists is None:
            raise QuotaNotFoundError("Usage quota not found")
        raise QuotaExceededError("Token quota exceeded")
    db.commit()
    return int(reserved)


def settle_tokens(db, user_id: UUID, *, reserved: int, actual: int) -> None:
    """Replace a reservation with the tokens actually used."""
    delta = actual - reserved
    if delta == 0:
        return
    db.execute(
        text(
            """
            UPDATE user_usage
            SET
                tokens_used = GREATEST(0, LEAST(token_limit, tokens_used + :delta)),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id
            """
        ),
        {"user_id": user_id, "delta": delta},
    )
    db.commit()
