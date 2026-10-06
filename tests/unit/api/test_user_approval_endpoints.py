"""Login blocking and admin approval endpoints, without a database."""

import asyncio
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.schemas import ChatBody
from auth.rate_limit import LoginRateLimiter
from auth.service import (
    AccountNotActiveError,
    AdminAccountProtectedError,
    InvalidStatusTransitionError,
    UserNotFoundError,
)

ADMIN = SimpleNamespace(id=uuid.uuid4(), role="ADMIN", username="admin")


class _SpyLimiter(LoginRateLimiter):
    def __init__(self):
        super().__init__()
        self.failures = 0
        self.successes = 0

    def record_failure(self, keys):
        self.failures += 1
        return super().record_failure(keys)

    def record_success(self, username):
        self.successes += 1
        super().record_success(username)


class _FakeSession:
    def rollback(self):
        pass


@pytest.fixture
def env(monkeypatch):
    @contextmanager
    def _session():
        yield _FakeSession()

    audits = []
    limiter = _SpyLimiter()
    monkeypatch.setattr(main.app.state, "session_factory", _session, raising=False)
    monkeypatch.setattr(main, "login_rate_limiter", limiter)
    monkeypatch.setattr(
        main, "write_audit_log", lambda db, **kwargs: audits.append(kwargs)
    )
    main.app.dependency_overrides[main.require_admin] = lambda: ADMIN
    try:
        yield TestClient(main.app), audits, limiter
    finally:
        main.app.dependency_overrides.pop(main.require_admin, None)


def _blocked(status):
    user_id = uuid.uuid4()

    def _authenticate(db, username, password):
        raise AccountNotActiveError(user_id, status)

    return user_id, _authenticate


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        ("PENDING", "Account pending administrator approval"),
        ("REJECTED", "Account has been rejected"),
    ],
)
def test_unapproved_login_is_blocked_and_audited(env, monkeypatch, status, detail):
    client, audits, limiter = env
    user_id, authenticate = _blocked(status)
    monkeypatch.setattr(main, "authenticate_user", authenticate)

    response = client.post(
        "/api/v1/auth/login", json={"username": "someone", "password": "pw"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == detail
    assert "lawchat_session" not in response.headers.get("set-cookie", "")
    assert audits == [
        {
            "event": f"login_blocked_{status.lower()}",
            "user_id": user_id,
            "ip_address": "testclient",
            "user_agent": "testclient",
        }
    ]
    # Correct password: not a failed attempt, and not a success that resets it.
    assert (limiter.failures, limiter.successes) == (0, 0)


def test_unapproved_totp_login_is_blocked_before_code_is_consumed(env, monkeypatch):
    client, audits, limiter = env
    _user_id, authenticate = _blocked("PENDING")
    monkeypatch.setattr(main, "authenticate_user", authenticate)

    def _no_totp(*args, **kwargs):
        raise AssertionError("TOTP code must not be consumed")

    monkeypatch.setattr(main, "verify_user_totp", _no_totp)

    response = client.post(
        "/api/v1/auth/totp",
        json={"username": "someone", "password": "pw", "code": "123456"},
    )

    assert response.status_code == 403
    assert [a["event"] for a in audits] == ["login_blocked_pending"]
    assert (limiter.failures, limiter.successes) == (0, 0)


def test_wrong_password_still_counts_as_failure(env, monkeypatch):
    client, audits, limiter = env
    monkeypatch.setattr(main, "authenticate_user", lambda db, u, p: None)

    response = client.post(
        "/api/v1/auth/login", json={"username": "someone", "password": "pw"}
    )

    assert response.status_code == 401
    assert [a["event"] for a in audits] == ["login_failed"]
    assert limiter.failures == 1


@pytest.mark.parametrize(
    ("method", "path", "service_name"),
    [
        ("post", "/approve", "approve_user"),
        ("post", "/reject", "reject_user"),
        ("delete", "", "delete_user"),
    ],
)
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (UserNotFoundError("User not found"), 404),
        (AdminAccountProtectedError("protected"), 403),
        (InvalidStatusTransitionError("already"), 409),
    ],
)
def test_admin_action_errors(env, monkeypatch, method, path, service_name, error, status_code):
    client, audits, _limiter = env
    target_id = uuid.uuid4()

    def _raise(db, target, **kwargs):
        raise error

    monkeypatch.setattr(main, service_name, _raise)

    response = getattr(client, method)(f"/api/v1/admin/users/{target_id}{path}")

    assert response.status_code == status_code
    if status_code == 403:
        [audit] = audits
        assert audit["event"] == "admin_action_denied"
        assert audit["user_id"] == target_id
        assert audit["actor_user_id"] == ADMIN.id
    else:
        assert audits == []


@pytest.mark.parametrize(
    ("path", "service_name", "status"),
    [("/approve", "approve_user", "ACTIVE"), ("/reject", "reject_user", "REJECTED")],
)
def test_admin_status_actions_pass_actor(env, monkeypatch, path, service_name, status):
    client, _audits, _limiter = env
    target_id = uuid.uuid4()
    calls = []

    def _action(db, target, *, actor_id, ip_address, user_agent):
        calls.append((target, actor_id))
        return SimpleNamespace(
            id=target, username="bob", role="USER",
            status=status, is_active=status == "ACTIVE",
        )

    monkeypatch.setattr(main, service_name, _action)

    response = client.post(f"/api/v1/admin/users/{target_id}{path}")

    assert response.status_code == 200
    assert response.json()["status"] == status
    assert calls == [(target_id, ADMIN.id)]


def test_admin_delete_returns_204(env, monkeypatch):
    client, _audits, _limiter = env
    monkeypatch.setattr(main, "delete_user", lambda db, target, **kw: {})

    response = client.delete(f"/api/v1/admin/users/{uuid.uuid4()}")

    assert response.status_code == 204


def test_non_admin_cannot_use_admin_actions(env):
    client, _audits, _limiter = env
    main.app.dependency_overrides.pop(main.require_admin)
    main.app.dependency_overrides[main.get_current_user] = lambda: SimpleNamespace(
        id=uuid.uuid4(), role="USER", totp_enabled=False, totp_secret=None
    )
    try:
        response = client.post(f"/api/v1/admin/users/{uuid.uuid4()}/approve")
    finally:
        main.app.dependency_overrides.pop(main.get_current_user, None)

    assert response.status_code == 403


def test_stream_ends_cleanly_when_user_is_deleted_mid_stream(monkeypatch):
    """Deleting the user cascades the conversation away; the stream reports
    chat.failed instead of crashing, and quota settlement still runs."""
    settled = []

    @contextmanager
    def _session():
        yield None

    class _Repository:
        def create_user_message(self, *args, **kwargs):
            return SimpleNamespace(id=uuid.uuid4()), True

        def list_messages(self, *args):
            return []

        def save_assistant_reply(self, *args, **kwargs):
            raise main.ChatNotFoundError("conversation not found")

    async def _pipeline(*args, event_queue=None, **kwargs):
        await event_queue.put(("retrieval.started", {}))
        return SimpleNamespace()

    monkeypatch.setattr(main.app.state, "session_factory", _session, raising=False)
    monkeypatch.setattr(main, "_reserve_quota", lambda user_id: 8000)
    monkeypatch.setattr(
        main, "settle_tokens",
        lambda db, user_id, *, reserved, actual: settled.append((reserved, actual)),
    )
    monkeypatch.setattr(main, "_execute_answer_pipeline", _pipeline)
    monkeypatch.setattr(main, "_pipeline_token_usage", lambda pipeline: 500)
    monkeypatch.setattr(
        main, "_compact_payload_from_pipeline",
        lambda *a, **k: {"answer": "ok", "citations": [], "status": "VERIFIED"},
    )

    async def _run():
        body = ChatBody(
            conversation_id=uuid.uuid4(), client_message_id="m1", message="Câu hỏi"
        )
        response = await main.chat_stream(
            body, SimpleNamespace(id=uuid.uuid4()), _Repository(), "local", None, None
        )
        return "".join([chunk async for chunk in response.body_iterator])

    events = asyncio.run(_run())

    assert "chat.failed" in events
    assert len(settled) == 1
