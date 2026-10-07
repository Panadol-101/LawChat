"""Account approval rules in auth.service, checked without a database."""

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

import auth.service as service
from auth.security import hash_password
from auth.service import (
    AccountNotActiveError,
    AdminAccountProtectedError,
    InvalidStatusTransitionError,
    UserNotFoundError,
    approve_user,
    authenticate_user,
    delete_user,
    register_user,
    reject_user,
)
from database.models import AuditLog

PASSWORD = "correct horse battery"
PASSWORD_HASH = hash_password(PASSWORD)


class FakeDB:
    """Records what the service does; scalar() returns queued results."""

    def __init__(self, *scalars, rowcount=0, flush_error=None):
        self.scalars = list(scalars)
        self.rowcount = rowcount
        self.flush_error = flush_error
        self.added = []
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def scalar(self, statement):
        self.statements.append(statement)
        return self.scalars.pop(0) if self.scalars else None

    def execute(self, statement, params=None):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=self.rowcount)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        if self.flush_error is not None:
            raise self.flush_error

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass

    @property
    def audits(self):
        return [obj for obj in self.added if isinstance(obj, AuditLog)]


def _user(*, role="USER", status="ACTIVE", is_active=True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        username=f"u-{uuid.uuid4().hex[:6]}",
        password_hash=PASSWORD_HASH,
        role=role,
        status=status,
        is_active=is_active,
    )


ADMIN_ID = uuid.uuid4()


def test_registration_is_pending_and_inactive_with_quota(monkeypatch):
    quota_calls = []
    monkeypatch.setattr(
        service, "create_user_usage", lambda db, user_id, **kw: quota_calls.append(kw)
    )
    db = FakeDB(None)

    user = register_user(db, "newuser", PASSWORD)

    assert user.status == "PENDING"
    assert user.is_active is False
    assert user.role == "USER"
    assert user.password_hash.startswith("$argon2id$")
    assert quota_calls == [{"token_limit": 100_000}]


def test_correct_password_on_pending_account_raises():
    user = _user(status="PENDING", is_active=False)

    with pytest.raises(AccountNotActiveError) as exc_info:
        authenticate_user(FakeDB(user), user.username, PASSWORD)

    assert exc_info.value.status == "PENDING"
    assert exc_info.value.user_id == user.id


def test_correct_password_on_rejected_only_account_raises_rejected():
    rejected = _user(status="REJECTED", is_active=False)
    # First query (current account) finds nothing; second finds the old row.
    db = FakeDB(None, rejected)

    with pytest.raises(AccountNotActiveError) as exc_info:
        authenticate_user(db, rejected.username, PASSWORD)

    assert exc_info.value.status == "REJECTED"


def test_wrong_password_never_reveals_status():
    pending = _user(status="PENDING", is_active=False)
    rejected = _user(status="REJECTED", is_active=False)

    assert authenticate_user(FakeDB(pending), pending.username, "wrong") is None
    assert authenticate_user(FakeDB(None, rejected), rejected.username, "wrong") is None


def test_disabled_active_account_still_fails_like_before():
    user = _user(status="ACTIVE", is_active=False)

    assert authenticate_user(FakeDB(user), user.username, PASSWORD) is None


def test_active_account_authenticates():
    user = _user()

    assert authenticate_user(FakeDB(user), user.username, PASSWORD) is user


def test_approve_activates_and_audits_actor_and_target(monkeypatch):
    status = "PENDING"
    monkeypatch.setattr(service, "create_user_usage", lambda db, user_id, **kw: None)
    target = _user(status=status, is_active=False)
    db = FakeDB(target)

    approve_user(db, target.id, actor_id=ADMIN_ID)

    assert (target.status, target.is_active) == ("ACTIVE", True)
    [audit] = db.audits
    assert audit.event == "user_approved"
    assert audit.user_id == target.id
    assert audit.actor_user_id == ADMIN_ID
    assert audit.details["target_username"] == target.username
    assert audit.details["previous_status"] == status
    assert db.commits == 1


@pytest.mark.parametrize(
    ("status", "is_active"), [("ACTIVE", True), ("REJECTED", False)]
)
def test_only_pending_users_can_be_approved(status, is_active):
    target = _user(status=status, is_active=is_active)
    db = FakeDB(target)

    with pytest.raises(InvalidStatusTransitionError, match="Only PENDING"):
        approve_user(db, target.id, actor_id=ADMIN_ID)

    assert (target.status, target.is_active) == (status, is_active)
    assert db.audits == []
    assert db.commits == 0


@pytest.mark.parametrize("status", ["PENDING", "ACTIVE"])
def test_reject_deactivates_revokes_sessions_and_audits(status):
    target = _user(status=status, is_active=status == "ACTIVE")
    db = FakeDB(target, rowcount=2)

    reject_user(db, target.id, actor_id=ADMIN_ID)

    assert (target.status, target.is_active) == ("REJECTED", False)
    assert any("UPDATE sessions" in str(s) for s in db.statements)
    [audit] = db.audits
    assert audit.event == "user_rejected"
    assert audit.user_id == target.id
    assert audit.actor_user_id == ADMIN_ID
    assert audit.details["revoked_sessions"] == 2
    assert audit.details["previous_status"] == status
    assert db.commits == 1


def test_reject_already_rejected_is_rejected():
    target = _user(status="REJECTED", is_active=False)

    with pytest.raises(InvalidStatusTransitionError):
        reject_user(FakeDB(target), target.id, actor_id=ADMIN_ID)


def test_delete_audits_before_deleting_and_keeps_username():
    target = _user()
    db = FakeDB(target)

    details = delete_user(db, target.id, actor_id=ADMIN_ID)

    [audit] = db.audits
    assert audit.event == "user_deleted"
    assert audit.user_id == target.id
    assert audit.actor_user_id == ADMIN_ID
    assert audit.details == details
    assert details["target_username"] == target.username
    delete_sql = str(db.statements[-1])
    assert delete_sql.startswith("DELETE FROM users")
    assert "users.role" in delete_sql
    assert db.commits == 1


@pytest.mark.parametrize("action", [approve_user, reject_user, delete_user])
def test_admin_accounts_are_protected(action):
    admin = _user(role="ADMIN")
    db = FakeDB(admin)

    with pytest.raises(AdminAccountProtectedError):
        action(db, admin.id, actor_id=ADMIN_ID)

    assert admin.status == "ACTIVE" and admin.is_active is True
    assert db.audits == []
    assert db.commits == 0


@pytest.mark.parametrize("action", [approve_user, reject_user, delete_user])
def test_missing_user_is_not_found(action):
    with pytest.raises(UserNotFoundError):
        action(FakeDB(None), uuid.uuid4(), actor_id=ADMIN_ID)


def test_target_row_is_locked():
    target = _user(status="PENDING", is_active=False)
    db = FakeDB(target)

    reject_user(db, target.id, actor_id=ADMIN_ID)

    assert "FOR UPDATE" in str(
        db.statements[0].compile(dialect=_postgres_dialect())
    )


def _postgres_dialect():
    from sqlalchemy.dialects import postgresql

    return postgresql.dialect()


def _sql(statement):
    from sqlalchemy.dialects import postgresql

    return str(statement.compile(dialect=postgresql.dialect()))


def test_authentication_query_excludes_rejected_rows():
    user = _user()
    db = FakeDB(user)

    assert authenticate_user(db, user.username, PASSWORD) is user

    [query] = db.statements
    assert "users.status != %(status_1)s" in _sql(query)
    assert query.compile().params["status_1"] == "REJECTED"


def test_current_account_is_used_without_looking_at_rejected_rows():
    pending = _user(status="PENDING", is_active=False)
    db = FakeDB(pending)

    with pytest.raises(AccountNotActiveError) as exc_info:
        authenticate_user(db, pending.username, PASSWORD)

    assert exc_info.value.user_id == pending.id
    assert len(db.statements) == 1


def test_registration_conflict_check_ignores_rejected_rows(monkeypatch):
    monkeypatch.setattr(service, "create_user_usage", lambda db, user_id, **kw: None)
    db = FakeDB(None)

    register_user(db, "sonnytest", PASSWORD)

    [query] = db.statements
    assert "users.status != %(status_1)s" in _sql(query)
    assert query.compile().params["status_1"] == "REJECTED"


def test_registration_of_taken_username_fails():
    with pytest.raises(ValueError, match="Username already exists"):
        register_user(FakeDB(_user(status="PENDING", is_active=False)), "x_user", PASSWORD)


def test_registration_race_on_unique_index_is_username_exists(monkeypatch):
    monkeypatch.setattr(service, "create_user_usage", lambda db, user_id, **kw: None)
    db = FakeDB(None, flush_error=IntegrityError("INSERT", {}, Exception("dup")))

    with pytest.raises(ValueError, match="Username already exists"):
        register_user(db, "sonnytest", PASSWORD)

    assert db.rollbacks == 1
    assert db.commits == 0


def test_approve_refuses_when_username_is_used_by_another_account():
    # Old inconsistent data: another current account already holds the name.
    target = _user(status="PENDING", is_active=False)
    db = FakeDB(target, uuid.uuid4())

    with pytest.raises(InvalidStatusTransitionError, match="already used"):
        approve_user(db, target.id, actor_id=ADMIN_ID)

    assert target.status == "PENDING"
    assert db.audits == []
    assert db.commits == 0


def test_approve_race_on_unique_index_is_conflict(monkeypatch):
    monkeypatch.setattr(service, "create_user_usage", lambda db, user_id, **kw: None)
    target = _user(status="PENDING", is_active=False)
    db = FakeDB(target, None, flush_error=IntegrityError("UPDATE", {}, Exception("dup")))

    with pytest.raises(InvalidStatusTransitionError):
        approve_user(db, target.id, actor_id=ADMIN_ID)

    assert db.rollbacks == 1
    assert db.commits == 0
