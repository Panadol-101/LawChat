from __future__ import annotations

from datetime import datetime, timedelta, timezone
from sqlalchemy import select, text
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from auth.security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    match_totp_step,
    verify_password,
)
from database.models import AuditLog, Session, User


SESSION_LIFETIME = timedelta(hours=12)

def create_user_usage(
    db: DBSession,
    user_id,
    token_limit: int = 100_000,
) -> None:
    """Create the initial token quota for a user."""

    now = datetime.now(timezone.utc)

    period_start = now.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    if period_start.month == 12:
        period_end = period_start.replace(
            year=period_start.year + 1,
            month=1,
        )
    else:
        period_end = period_start.replace(
            month=period_start.month + 1,
        )

    db.execute(
        text(
            """
            INSERT INTO user_usage (
                id,
                user_id,
                token_limit,
                tokens_used,
                period_start,
                period_end
            )
            VALUES (
                :id,
                :user_id,
                :token_limit,
                0,
                :period_start,
                :period_end
            )
            ON CONFLICT (user_id) DO NOTHING
            """
        ),
        {
            "id": uuid4(),
            "user_id": user_id,
            "token_limit": token_limit,
            "period_start": period_start,
            "period_end": period_end,
        },
    )



def register_user(
    db: DBSession,
    username: str,
    password: str,
) -> User:
    """Create a normal USER account."""

    username = username.strip()

    if not username:
        raise ValueError("Username is required")

    if len(username) < 3 or len(username) > 50:
        raise ValueError("Username must be between 3 and 50 characters")

    existing_user = db.scalar(
        select(User).where(User.username == username)
    )

    if existing_user is not None:
        raise ValueError("Username already exists")

    user = User(
        username=username,
        password_hash=hash_password(password),
        role="USER",
        totp_enabled=False,
        is_active=True,
    )

    db.add(user)
    db.flush()

    create_user_usage(
        db,
        user.id,
        token_limit=100_000,
    )

    db.commit()
    db.refresh(user)

    return user

def authenticate_user(
    db: DBSession,
    username: str,
    password: str,
) -> User | None:
    """Authenticate a user with username and password."""

    username = username.strip()

    if not username or not password:
        return None

    user = db.scalar(
        select(User).where(
            User.username == username,
            User.is_active.is_(True),
        )
    )

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


def create_session(
    db: DBSession,
    user: User,
) -> str:
    """Create a new session and return the raw token."""

    raw_token = generate_session_token()

    session = Session(
        user_id=user.id,
        token_hash=hash_session_token(raw_token),
        expires_at=datetime.now(timezone.utc) + SESSION_LIFETIME,
    )

    db.add(session)
    db.commit()

    return raw_token


def get_user_by_session(
    db: DBSession,
    raw_token: str,
) -> User | None:
    """Resolve a valid session token to its user."""

    if not raw_token:
        return None

    token_hash = hash_session_token(raw_token)
    now = datetime.now(timezone.utc)

    session = db.scalar(
        select(Session).where(
            Session.token_hash == token_hash,
            Session.revoked_at.is_(None),
            Session.expires_at > now,
        )
    )

    if session is None:
        return None

    user = db.scalar(
        select(User).where(
            User.id == session.user_id,
            User.is_active.is_(True),
        )
    )

    return user


def revoke_session(
    db: DBSession,
    raw_token: str,
) -> bool:
    """Revoke a session token."""

    if not raw_token:
        return False

    token_hash = hash_session_token(raw_token)

    session = db.scalar(
        select(Session).where(
            Session.token_hash == token_hash,
            Session.revoked_at.is_(None),
        )
    )

    if session is None:
        return False

    session.revoked_at = datetime.now(timezone.utc)
    db.commit()

    return True


def write_audit_log(
    db: DBSession,
    *,
    event: str,
    user_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Write an authentication audit event."""

    audit_log = AuditLog(
        user_id=user_id,
        event=event,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    db.add(audit_log)
    db.commit()
def verify_user_totp(
    db: DBSession,
    user: User,
    code: str,
) -> bool:
    """Verify the user's TOTP code and consume it so it cannot be replayed."""

    if not user.totp_enabled:
        return False

    if not user.totp_secret:
        return False

    step = match_totp_step(user.totp_secret, code)
    if step is None:
        return False

    # Conditional update: of two requests racing with the same code, only one
    # advances the step, so each code is accepted at most once.
    consumed = db.execute(
        text(
            """
            UPDATE users
            SET totp_last_step = :step
            WHERE id = :user_id
              AND (totp_last_step IS NULL OR totp_last_step < :step)
            """
        ),
        {"step": step, "user_id": user.id},
    ).rowcount
    db.commit()
    return consumed == 1
