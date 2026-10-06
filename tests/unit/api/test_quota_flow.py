"""Token quota is reserved before a chat run and settled however the run ends."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

import api.main as main
from api.schemas import ChatBody

RESERVED = 8000


class _Repository:
    def __init__(self, *, created=True, reply=None):
        self.created = created
        self.reply = reply

    def create_user_message(self, *args, **kwargs):
        return SimpleNamespace(id=uuid4()), self.created

    def find_assistant_reply(self, conversation_id, client_message_id):
        return self.reply

    def list_messages(self, *args):
        return []

    def save_assistant_reply(self, *args, **kwargs):
        return SimpleNamespace(id=uuid4())


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


PIPELINE = (
    SimpleNamespace(telemetry=_usage(100, 20)),
    None,
    None,
    SimpleNamespace(telemetry=(_usage(1000, 300),), semantic_observations=()),
)


@pytest.fixture
def settled(monkeypatch):
    calls = []

    @contextmanager
    def _session():
        yield None

    monkeypatch.setattr(main.app.state, "session_factory", _session, raising=False)
    monkeypatch.setattr(main, "_reserve_quota", lambda user_id: RESERVED)
    monkeypatch.setattr(
        main,
        "settle_tokens",
        lambda db, user_id, *, reserved, actual: calls.append((reserved, actual)),
    )
    monkeypatch.setattr(
        main,
        "_compact_payload_from_pipeline",
        lambda *args, **kwargs: {"answer": "ok", "citations": [], "status": "VERIFIED"},
    )
    return calls


def _body():
    return ChatBody(conversation_id=uuid4(), client_message_id="m1", message="Câu hỏi")


def _chat(repository):
    return main.chat(
        SimpleNamespace(id=uuid4()), _body(), repository, "local", None, None
    )


async def _stream(repository, *, stop_after=None):
    response = await main.chat_stream(
        _body(), SimpleNamespace(id=uuid4()), repository, "local", None, None
    )
    chunks = []
    iterator = response.body_iterator
    async for chunk in iterator:
        chunks.append(chunk)
        if stop_after is not None and stop_after in chunk:
            await iterator.aclose()
            break
    return "".join(chunks)


def _pipeline_returning(value=None, error=None):
    async def _pipeline(*args, event_queue=None, **kwargs):
        # Like the real pipeline, report progress so the stream loop wakes up.
        if event_queue is not None:
            await event_queue.put(("retrieval.started", {}))
        if error is not None:
            raise error
        return value

    return _pipeline


def test_chat_success_charges_actual_usage(settled, monkeypatch):
    monkeypatch.setattr(main, "_execute_answer_pipeline", _pipeline_returning(PIPELINE))

    asyncio.run(_chat(_Repository()))

    assert settled == [(RESERVED, 1420)]


def test_chat_failure_keeps_reservation(settled, monkeypatch):
    monkeypatch.setattr(
        main, "_execute_answer_pipeline", _pipeline_returning(error=TimeoutError())
    )

    with pytest.raises(HTTPException):
        asyncio.run(_chat(_Repository()))

    assert settled == [(RESERVED, RESERVED)]


def test_chat_replay_refunds_reservation(settled):
    reply = SimpleNamespace(
        id=uuid4(), status="COMPLETED", content="", citations=[], claims=[],
        limitations=[], as_of=None, request_id="r", semantic_status=None,
        coverage_status=None, confidence=None, error_code=None,
    )

    asyncio.run(_chat(_Repository(created=False, reply=reply)))

    assert settled == [(RESERVED, 0)]


def test_stream_failure_keeps_reservation(settled, monkeypatch):
    monkeypatch.setattr(
        main, "_execute_answer_pipeline", _pipeline_returning(error=RuntimeError("boom"))
    )

    events = asyncio.run(_stream(_Repository()))

    assert "chat.failed" in events
    assert settled == [(RESERVED, RESERVED)]


def test_stream_success_charges_actual_usage(settled, monkeypatch):
    monkeypatch.setattr(main, "_execute_answer_pipeline", _pipeline_returning(PIPELINE))

    events = asyncio.run(_stream(_Repository()))

    assert "chat.completed" in events
    assert settled == [(RESERVED, 1420)]


def test_stream_disconnect_mid_pipeline_keeps_reservation(settled, monkeypatch):
    async def _blocking_pipeline(*args, event_queue=None, **kwargs):
        await event_queue.put(("retrieval.started", {}))
        await asyncio.Event().wait()

    monkeypatch.setattr(main, "_execute_answer_pipeline", _blocking_pipeline)

    asyncio.run(_stream(_Repository(), stop_after="retrieval.started"))

    assert settled == [(RESERVED, RESERVED)]


def test_stream_unknown_conversation_refunds_reservation(settled):
    class _Missing(_Repository):
        def create_user_message(self, *args, **kwargs):
            raise main.ChatNotFoundError("conversation not found")

    events = asyncio.run(_stream(_Missing()))

    assert "CONVERSATION_NOT_FOUND" in events
    assert settled == [(RESERVED, 0)]
