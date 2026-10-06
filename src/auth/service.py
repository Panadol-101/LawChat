from __future__ import annotations

from datetime import datetime, timedelta, timezone
from sqlalchemy import select, text
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
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


class AccountNotActiveError(Exception):
    """Correct password, but the account is not approved (PENDING/REJECTED)."""

    def __init__(self, user_id: UUID, status: str) -> None:
        super().__init__(f"Account is {status}")
        self.user_id = user_id
        self.status = status


class UserNotFoundError(Exception):
    """The target account does not exist."""


class AdminAccountProtectedError(Exception):
    """ADMIN accounts are never approved, rejected or deleted."""


class InvalidStatusTransitionError(Exception):
    """The account is already in a state the action cannot start from."""


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
        # New accounts wait for administrator approval.
        status="PENDING",
        is_active=False,
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
    """Authenticate a user with username and password.

    Raises AccountNotActiveError for a correct password on a PENDING or
    REJECTED account; the status is only revealed after the password check.
    """

    username = username.strip()

    if not username or not password:
        return None

    user = db.scalar(
        select(User).where(
            User.username == username,
        )
    )

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    if user.status != "ACTIVE":
        raise AccountNotActiveError(user.id, user.status)

    if not user.is_active:
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
            User.status == "ACTIVE",
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
    actor_user_id: UUID | None = None,
    details: dict[str, Any] | None = None,
    commit: bool = True,
) -> None:
    """Write an authentication audit event.

    user_id is the account the event is about; actor_user_id is the
    administrator who performed it, if any.
    """

    audit_log = AuditLog(
        user_id=user_id,
        actor_user_id=actor_user_id,
        event=event,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details,
    )

    db.add(audit_log)
    if commit:
        db.commit()
    else:
        db.flush()


def _lock_target_user(db: DBSession, target_id: UUID) -> User:
    target = db.scalar(
        select(User).where(User.id == target_id).with_for_update()
    )
    if target is None:
        raise UserNotFoundError("User not found")
    if target.role != "USER":
        raise AdminAccountProtectedError(
            "Administrator accounts cannot be approved, rejected or deleted"
        )
    return target


def _target_details(target: User, **extra: Any) -> dict[str, Any]:
    return {
        "target_user_id": str(target.id),
        "target_username": target.username,
        "previous_status": target.status,
        **extra,
    }


def approve_user(
    db: DBSession,
    target_id: UUID,
    *,
    actor_id: UUID,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> User:
    """Approve a PENDING or REJECTED USER account."""

    target = _lock_target_user(db, target_id)
    if target.status not in ("PENDING", "REJECTED"):
        raise InvalidStatusTransitionError(f"User is already {target.status}")

    details = _target_details(target)
    target.status = "ACTIVE"
    target.is_active = True
    db.flush()
    # Accounts registered before quotas existed get one now; a no-op otherwise.
    create_user_usage(db, target.id)
    write_audit_log(
        db,
        event="user_approved",
        user_id=target.id,
        actor_user_id=actor_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details,
        commit=False,
    )
    db.commit()
    return target


def reject_user(
    db: DBSession,
    target_id: UUID,
    *,
    actor_id: UUID,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> User:
    """Reject a PENDING or ACTIVE USER account and revoke all its sessions."""

    target = _lock_target_user(db, target_id)
    if target.status == "REJECTED":
        raise InvalidStatusTransitionError("User is already REJECTED")

    previous_status = target.status
    target.status = "REJECTED"
    target.is_active = False
    revoked = db.execute(
        update(Session)
        .where(
            Session.user_id == target.id,
            Session.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(timezone.utc))
    ).rowcount
    write_audit_log(
        db,
        event="user_rejected",
        user_id=target.id,
        actor_user_id=actor_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details={
            "target_user_id": str(target.id),
            "target_username": target.username,
            "previous_status": previous_status,
            "revoked_sessions": revoked,
        },
        commit=False,
    )
    db.commit()
    return target


def delete_user(
    db: DBSession,
    target_id: UUID,
    *,
    actor_id: UUID,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """Delete a USER account; sessions, chats and quota go with it (CASCADE).

    The audit row is written first in the same transaction: its user_id is set
    to NULL by the cascade, and details keep the target's id and username.
    """

    target = _lock_target_user(db, target_id)
    details = _target_details(target)
    write_audit_log(
        db,
        event="user_deleted",
        user_id=target.id,
        actor_user_id=actor_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details,
        commit=False,
    )
    db.execute(
        delete(User)
        .where(User.id == target.id, User.role == "USER")
        .execution_options(synchronize_session=False)
    )
    db.commit()
    return details


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
