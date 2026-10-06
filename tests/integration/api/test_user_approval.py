from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

import api.main as main
from api.quota import settle_tokens
from auth.rate_limit import LoginRateLimiter
from auth.security import generate_totp_secret, hash_password
from auth.service import create_session
from chat import ChatNotFoundError, ChatRepository
from database.models import AuditLog, ChatConversation, Session, User


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

    def make_user(*, role="USER", status="ACTIVE", totp=False):
        user = User(
            id=uuid.uuid4(),
            username=f"{role.lower()}-{uuid.uuid4().hex[:8]}",
            password_hash=hash_password(PASSWORD),
            role=role,
            status=status,
            is_active=status == "ACTIVE",
            totp_secret=generate_totp_secret() if totp else None,
            totp_enabled=totp,
        )
        with factory() as session, session.begin():
            session.add(user)
        created.append(user.id)
        return user

    def admin_client():
        admin = make_user(role="ADMIN", totp=True)
        with factory() as session:
            token = create_session(session, session.get(User, admin.id))
        client = TestClient(main.app)
        client.cookies.set("lawchat_session", token)
        return admin, client

    try:
        yield factory, make_user, admin_client, created
    finally:
        with factory() as session, session.begin():
            session.execute(delete(AuditLog).where(
                AuditLog.user_id.in_(created) | AuditLog.actor_user_id.in_(created)
            ))
            session.execute(delete(User).where(User.id.in_(created)))
        engine.dispose()


def _login(client, username, password=PASSWORD):
    return client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )


def _audits(factory, event, user_id):
    with factory() as session:
        return session.scalars(
            select(AuditLog).where(AuditLog.event == event, AuditLog.user_id == user_id)
        ).all()


def test_registration_waits_for_approval_then_logs_in(env):
    factory, _make_user, admin_client, created = env
    client = TestClient(main.app)
    username = f"reg_{uuid.uuid4().hex[:8]}"

    registered = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": PASSWORD, "confirm_password": PASSWORD},
    )
    assert registered.status_code == 200
    user_id = uuid.UUID(registered.json()["id"])
    created.append(user_id)
    assert registered.json()["status"] == "PENDING"

    with factory() as session:
        user = session.get(User, user_id)
        assert (user.status, user.is_active) == ("PENDING", False)
        assert user.password_hash.startswith("$argon2id$")
        assert session.execute(
            text("SELECT count(*) FROM user_usage WHERE user_id = :id"), {"id": user_id}
        ).scalar_one() == 1

    blocked = _login(client, username)
    assert blocked.status_code == 403
    assert "lawchat_session" not in blocked.headers.get("set-cookie", "")
    assert len(_audits(factory, "login_blocked_pending", user_id)) == 1

    admin, admin_http = admin_client()
    approved = admin_http.post(f"/api/v1/admin/users/{user_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "ACTIVE"
    [audit] = _audits(factory, "user_approved", user_id)
    assert audit.actor_user_id == admin.id
    assert audit.details["target_username"] == username

    assert _login(client, username).status_code == 200


def test_reject_revokes_every_session(env):
    factory, make_user, admin_client, _created = env
    user = make_user()
    with factory() as session:
        tokens = [create_session(session, session.get(User, user.id)) for _ in range(2)]
    user_http = TestClient(main.app)
    user_http.cookies.set("lawchat_session", tokens[0])
    assert user_http.get("/api/v1/auth/me").status_code == 200

    admin, admin_http = admin_client()
    response = admin_http.post(f"/api/v1/admin/users/{user.id}/reject")

    assert response.status_code == 200
    assert response.json() == {
        "id": str(user.id), "username": user.username, "role": "USER",
        "status": "REJECTED", "is_active": False,
    }
    with factory() as session:
        assert session.scalar(
            select(Session).where(Session.user_id == user.id, Session.revoked_at.is_(None))
        ) is None
    assert user_http.get("/api/v1/auth/me").status_code == 401
    assert _login(TestClient(main.app), user.username).status_code == 403
    [audit] = _audits(factory, "user_rejected", user.id)
    assert audit.actor_user_id == admin.id
    assert audit.details["revoked_sessions"] == 2


def test_rejected_user_can_be_approved_again(env):
    _factory, make_user, admin_client, _created = env
    user = make_user(status="REJECTED")
    _admin, admin_http = admin_client()

    assert admin_http.post(f"/api/v1/admin/users/{user.id}/approve").json()["status"] == "ACTIVE"
    assert admin_http.post(f"/api/v1/admin/users/{user.id}/approve").status_code == 409


def test_delete_removes_account_data_but_keeps_audit(env):
    factory, make_user, admin_client, _created = env
    user = make_user()
    repository = ChatRepository(factory)
    conversation = repository.create_conversation(user.id, "local", title="t")
    with factory() as session:
        create_session(session, session.get(User, user.id))
        session.execute(
            text(
                "INSERT INTO user_usage (id, user_id, period_start, period_end) "
                "VALUES (gen_random_uuid(), :id, now(), now() + interval '1 month')"
            ),
            {"id": user.id},
        )
        session.commit()

    admin, admin_http = admin_client()
    response = admin_http.delete(f"/api/v1/admin/users/{user.id}")

    assert response.status_code == 204
    with factory() as session:
        assert session.get(User, user.id) is None
        assert session.get(ChatConversation, conversation.id) is None
        assert session.scalar(select(Session).where(Session.user_id == user.id)) is None
        assert session.execute(
            text("SELECT count(*) FROM user_usage WHERE user_id = :id"), {"id": user.id}
        ).scalar_one() == 0
        [audit] = session.scalars(
            select(AuditLog).where(
                AuditLog.event == "user_deleted",
                AuditLog.actor_user_id == admin.id,
            )
        ).all()
    assert audit.user_id is None  # ON DELETE SET NULL
    assert audit.details["target_user_id"] == str(user.id)
    assert audit.details["target_username"] == user.username


def test_in_flight_chat_of_deleted_user_fails_softly(env):
    """What a running stream does after its user is deleted."""
    factory, make_user, admin_client, _created = env
    user = make_user()
    repository = ChatRepository(factory)
    conversation = repository.create_conversation(user.id, "local", title="t")

    _admin, admin_http = admin_client()
    assert admin_http.delete(f"/api/v1/admin/users/{user.id}").status_code == 204

    # The stream handler turns this into a chat.failed event.
    with pytest.raises(ChatNotFoundError):
        repository.save_assistant_reply(
            user.id, "local", conversation.id,
            client_message_id="m1", payload={"status": "VERIFIED", "answer": "x"},
        )
    # Quota settlement updates no row and does not raise.
    with factory() as session:
        settle_tokens(session, user.id, reserved=8000, actual=10)


@pytest.mark.parametrize(
    ("method", "path"), [("post", "/approve"), ("post", "/reject"), ("delete", "")]
)
def test_admin_accounts_cannot_be_changed(env, method, path):
    factory, make_user, admin_client, _created = env
    other_admin = make_user(role="ADMIN", totp=True)
    admin, admin_http = admin_client()

    for target in (other_admin.id, admin.id):
        response = getattr(admin_http, method)(f"/api/v1/admin/users/{target}{path}")
        assert response.status_code == 403
        with factory() as session:
            row = session.get(User, target)
            assert (row.status, row.is_active, row.role) == ("ACTIVE", True, "ADMIN")
        [audit] = _audits(factory, "admin_action_denied", target)
        assert audit.actor_user_id == admin.id


def test_unknown_user_is_404(env):
    _factory, _make_user, admin_client, _created = env
    _admin, admin_http = admin_client()

    assert admin_http.post(f"/api/v1/admin/users/{uuid.uuid4()}/approve").status_code == 404


def test_plain_user_cannot_approve(env):
    factory, make_user, _admin_client, _created = env
    user = make_user()
    pending = make_user(status="PENDING")
    with factory() as session:
        token = create_session(session, session.get(User, user.id))
    client = TestClient(main.app)
    client.cookies.set("lawchat_session", token)

    assert client.post(f"/api/v1/admin/users/{pending.id}/approve").status_code == 403


def test_admin_user_list_filters_by_status(env):
    _factory, make_user, admin_client, _created = env
    pending = make_user(status="PENDING")
    _admin, admin_http = admin_client()

    rows = admin_http.get("/api/v1/admin/users", params={"status": "PENDING"}).json()

    assert pending.username in {row["username"] for row in rows}
    assert {row["status"] for row in rows} == {"PENDING"}


@pytest.mark.parametrize("status", ["PENDING", "REJECTED"])
def test_unapproved_status_cannot_be_active(env, status):
    factory, _make_user, _admin_client, _created = env
    user = User(
        id=uuid.uuid4(), username=f"bad-{uuid.uuid4().hex[:8]}",
        password_hash="x", status=status, is_active=True,
    )
    with pytest.raises(IntegrityError):
        with factory() as session, session.begin():
            session.add(user)
