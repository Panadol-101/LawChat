from __future__ import annotations

import os
import uuid

import pyotp
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
from frontend.admin.metrics import user_metrics
from frontend.api_client import APIError, LawChatAPI


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

    def make_user(*, role="USER", status="ACTIVE", totp=False, username=None,
                  password=PASSWORD):
        user = User(
            id=uuid.uuid4(),
            username=username or f"{role.lower()}-{uuid.uuid4().hex[:8]}",
            password_hash=hash_password(password),
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


@pytest.mark.parametrize("status", ["REJECTED", "ACTIVE"])
def test_only_pending_users_can_be_approved(env, status):
    factory, make_user, admin_client, _created = env
    user = make_user(status=status)
    _admin, admin_http = admin_client()

    response = admin_http.post(f"/api/v1/admin/users/{user.id}/approve")

    assert response.status_code == 409
    with factory() as session:
        row = session.get(User, user.id)
        assert (row.status, row.is_active) == (status, status == "ACTIVE")
    assert _audits(factory, "user_approved", user.id) == []


def test_rejected_user_can_still_be_deleted(env):
    factory, make_user, admin_client, _created = env
    user = make_user(status="REJECTED")
    _admin, admin_http = admin_client()

    assert admin_http.delete(f"/api/v1/admin/users/{user.id}").status_code == 204
    with factory() as session:
        assert session.get(User, user.id) is None


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


# ---------------------------------------------------------------------------
# REJECTED accounts do not reserve their username
# ---------------------------------------------------------------------------

def _name():
    return f"reuse_{uuid.uuid4().hex[:8]}"


def _register(client, username, password=PASSWORD):
    return client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "confirm_password": password},
    )


def test_rejected_username_can_be_registered_again(env):
    factory, make_user, _admin_client, created = env
    name = _name()
    old = make_user(username=name, status="REJECTED")

    response = _register(TestClient(main.app), name)

    assert response.status_code == 200
    new_id = uuid.UUID(response.json()["id"])
    created.append(new_id)
    with factory() as session:
        rows = session.scalars(
            select(User).where(User.username == name).order_by(User.created_at)
        ).all()
    assert [(r.id, r.status, r.is_active, r.role) for r in rows] == [
        (old.id, "REJECTED", False, "USER"),
        (new_id, "PENDING", False, "USER"),
    ]


@pytest.mark.parametrize("status", ["ACTIVE", "PENDING"])
def test_current_username_cannot_be_registered_again(env, status):
    _factory, make_user, _admin_client, _created = env
    name = _name()
    make_user(username=name, status=status)

    response = _register(TestClient(main.app), name)

    assert response.status_code == 400
    assert response.json()["detail"] == "Username already exists"


@pytest.mark.parametrize(
    ("first", "second", "allowed"),
    [
        ("REJECTED", "PENDING", True),
        ("REJECTED", "ACTIVE", True),
        ("REJECTED", "REJECTED", True),
        ("ACTIVE", "ACTIVE", False),
        ("PENDING", "PENDING", False),
        ("ACTIVE", "PENDING", False),
    ],
)
def test_partial_unique_index(env, first, second, allowed):
    _factory, make_user, _admin_client, _created = env
    name = _name()
    make_user(username=name, status=first)

    if allowed:
        make_user(username=name, status=second)
    else:
        with pytest.raises(IntegrityError):
            make_user(username=name, status=second)


def test_pending_account_is_used_when_rejected_row_shares_username(env):
    factory, make_user, _admin_client, _created = env
    name = _name()
    make_user(username=name, status="REJECTED", password="old rejected password")
    pending = make_user(username=name, status="PENDING")
    client = TestClient(main.app)

    response = _login(client, name)

    assert response.status_code == 403
    assert response.json()["detail"] == "Account pending administrator approval"
    assert len(_audits(factory, "login_blocked_pending", pending.id)) == 1
    # The old rejected password is not considered at all.
    assert _login(client, name, "old rejected password").status_code == 401


def test_active_account_is_used_when_rejected_row_shares_username(env):
    factory, make_user, _admin_client, _created = env
    name = _name()
    rejected = make_user(username=name, status="REJECTED")
    active = make_user(username=name, status="ACTIVE")

    response = _login(TestClient(main.app), name)

    assert response.status_code == 200
    with factory() as session:
        owners = session.scalars(
            select(Session.user_id).where(Session.user_id.in_([rejected.id, active.id]))
        ).all()
    assert owners == [active.id]


def test_rejected_account_never_authenticates(env):
    factory, make_user, _admin_client, _created = env
    rejected = make_user(status="REJECTED")

    response = _login(TestClient(main.app), rejected.username)

    assert response.status_code == 403
    assert response.json()["detail"] == "Account has been rejected"
    assert "lawchat_session" not in response.headers.get("set-cookie", "")
    assert len(_audits(factory, "login_blocked_rejected", rejected.id)) == 1


@pytest.mark.parametrize("status", ["PENDING", "REJECTED"])
def test_totp_cannot_bypass_status(env, status):
    factory, make_user, _admin_client, _created = env
    user = make_user(status=status, totp=True)
    code = pyotp.TOTP(user.totp_secret).now()

    response = TestClient(main.app).post(
        "/api/v1/auth/totp",
        json={"username": user.username, "password": PASSWORD, "code": code},
    )

    assert response.status_code == 403
    assert "lawchat_session" not in response.headers.get("set-cookie", "")
    with factory() as session:
        assert session.get(User, user.id).totp_last_step is None


def test_pending_user_with_rejected_namesake_can_be_approved(env):
    _factory, make_user, admin_client, _created = env
    name = _name()
    make_user(username=name, status="REJECTED")
    pending = make_user(username=name, status="PENDING")
    _admin, admin_http = admin_client()

    response = admin_http.post(f"/api/v1/admin/users/{pending.id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "ACTIVE"


def test_approving_cannot_create_a_username_conflict(env):
    factory, make_user, admin_client, _created = env
    name = _name()
    rejected = make_user(username=name, status="REJECTED")
    make_user(username=name, status="PENDING")
    _admin, admin_http = admin_client()

    response = admin_http.post(f"/api/v1/admin/users/{rejected.id}/approve")

    assert response.status_code == 409
    with factory() as session:
        assert session.get(User, rejected.id).status == "REJECTED"
        assert sorted(
            session.scalars(select(User.status).where(User.username == name)).all()
        ) == ["PENDING", "REJECTED"]


def test_admin_metrics_exclude_rejected_users(env):
    _factory, make_user, admin_client, _created = env
    make_user(status="ACTIVE")
    make_user(status="PENDING")
    make_user(status="REJECTED")
    _admin, admin_http = admin_client()

    rows = admin_http.get("/api/v1/admin/users").json()
    metrics = user_metrics(rows)
    by_status = {
        status: sum(1 for r in rows if r["role"] == "USER" and r["status"] == status)
        for status in ("ACTIVE", "PENDING", "REJECTED")
    }

    assert metrics["users"] == by_status["ACTIVE"] + by_status["PENDING"]
    assert metrics["active"] == by_status["ACTIVE"]
    assert metrics["pending"] == by_status["PENDING"]
    assert metrics["rejected"] == by_status["REJECTED"] >= 1


# ---------------------------------------------------------------------------
# Public frontend client against the real API: Vietnamese messages
# ---------------------------------------------------------------------------

@pytest.fixture
def public_api():
    api = LawChatAPI()
    api.client.close()
    api.client = TestClient(main.app)
    return api


def _api_error(call):
    with pytest.raises(APIError) as exc_info:
        call()
    return str(exc_info.value)


def test_public_client_shows_vietnamese_login_errors(env, public_api):
    _factory, make_user, _admin_client, _created = env
    rejected = make_user(status="REJECTED")
    pending = make_user(status="PENDING")

    assert _api_error(lambda: public_api.login(rejected.username, PASSWORD)) == (
        "Tài khoản đã bị từ chối."
    )
    assert _api_error(lambda: public_api.login(pending.username, PASSWORD)) == (
        "Tài khoản đang chờ quản trị viên phê duyệt."
    )
    assert _api_error(lambda: public_api.login(pending.username, "wrong password")) == (
        "Tên đăng nhập hoặc mật khẩu không chính xác."
    )


def test_public_client_shows_vietnamese_registration_errors(env, public_api):
    _factory, make_user, _admin_client, _created = env
    taken = make_user(status="ACTIVE", username=_name())

    assert _api_error(
        lambda: public_api.register(taken.username, PASSWORD, PASSWORD)
    ) == "Tên đăng nhập đã tồn tại."
    assert _api_error(
        lambda: public_api.register("bad name!", "short", "short")
    ) == (
        "Tên đăng nhập phải có từ 3 đến 50 ký tự, chỉ gồm chữ cái không dấu, "
        "chữ số hoặc dấu gạch dưới (_). Mật khẩu phải có từ 8 đến 128 ký tự. "
        "Mật khẩu nhập lại phải có từ 8 đến 128 ký tự."
    )
    assert _api_error(
        lambda: public_api.register(_name(), PASSWORD, PASSWORD + "x")
    ) == "Mật khẩu nhập lại không khớp."
