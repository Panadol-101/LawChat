"""Regression tests for the chat -> retrieval pipeline (review findings)."""

import asyncio
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import api.main as main
from api.main import AnswerBody, _execute_answer_pipeline
from api.schemas import ChatBody
from rag import QueryRewrite
from rag.runtime import RAGExecutionGate
from retrieval import RetrievalResponse


class _RecordingRetrieval:
    def __init__(self):
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return RetrievalResponse(
            query=request.query,
            semantic_query=request.query,
            as_of=date(2026, 1, 1),
            results=(),
            searched_candidates=0,
            rejected_candidates=0,
        )


class _RecordingGeneration:
    def __init__(self):
        self.request = None

    async def answer_async(self, request):
        self.request = request
        return SimpleNamespace(
            status=SimpleNamespace(value="REFUSED"),
            semantic_status=None,
            coverage_status=None,
            attempts=1,
            telemetry=(),
        )


def _runtime(generation):
    return SimpleNamespace(
        execution_gate=RAGExecutionGate(
            max_concurrency=1,
            max_queue=1,
            queue_timeout_seconds=5,
            request_timeout_seconds=30,
        ),
        issue_decomposer=None,
        context_builder=SimpleNamespace(
            build=lambda retrieval: SimpleNamespace(evidence=(), used_tokens=0)
        ),
        generation_service=generation,
    )


def test_follow_up_question_carries_history_entities_into_retrieval():
    retrieval = _RecordingRetrieval()
    generation = _RecordingGeneration()
    current = "Vậy điều đó áp dụng thế nào?"
    history = [
        SimpleNamespace(role="user", content="Điều 36 Nghị định 123/2024/NĐ-CP quy định gì?"),
        SimpleNamespace(role="assistant", content="Điều 36 quy định về ..."),
        # list_messages() already contains the just-saved current message.
        SimpleNamespace(role="user", content=current),
    ]

    asyncio.run(_execute_answer_pipeline(
        AnswerBody(query=current),
        retrieval,
        _runtime(generation),
        history=history,
    ))

    assert "123/2024/NĐ-CP" in retrieval.requests[0].query
    # The answer is still phrased against what the user actually asked.
    assert generation.request.question == current


def test_standalone_question_is_not_rewritten_with_history():
    retrieval = _RecordingRetrieval()
    current = "Thời giờ làm việc bình thường tối đa bao nhiêu giờ?"
    history = [
        SimpleNamespace(role="user", content="Điều 321 Bộ luật Hình sự quy định gì?"),
        SimpleNamespace(role="assistant", content="Điều 321 quy định tội đánh bạc."),
        SimpleNamespace(role="user", content=current),
    ]

    asyncio.run(_execute_answer_pipeline(
        AnswerBody(query=current),
        retrieval,
        _runtime(_RecordingGeneration()),
        history=history,
    ))

    assert retrieval.requests[0].query == current


class _RecordingRewriter:
    def __init__(self, standalone):
        self.standalone = standalone
        self.request = None

    async def rewrite_async(self, request):
        self.request = request
        if self.standalone is None:
            return None
        return QueryRewrite(self.standalone, request.question, rewritten=True)


def test_follow_up_is_rewritten_by_llm_for_retrieval_and_generation():
    retrieval = _RecordingRetrieval()
    generation = _RecordingGeneration()
    current = "vậy nếu từ 5 tỷ đồng trở lên thì phạt như thế nào"
    standalone = "Đánh bạc trên không gian mạng với số tiền từ 5 tỷ đồng trở lên bị xử phạt như thế nào?"
    rewriter = _RecordingRewriter(standalone)
    runtime = _runtime(generation)
    runtime.query_rewriter = rewriter
    history = [
        SimpleNamespace(role="user", content="Đánh cờ bạc trên không gian mạng bị xử phạt thế nào?"),
        SimpleNamespace(role="assistant", content="Hành vi đánh bạc trái phép ..."),
        SimpleNamespace(role="user", content=current),
    ]

    pipeline = asyncio.run(_execute_answer_pipeline(
        AnswerBody(query=current), retrieval, runtime, history=history,
    ))

    # The current message is not repeated as history.
    assert [role for role, _ in rewriter.request.history] == ["user", "assistant"]
    assert retrieval.requests[0].query == standalone
    assert generation.request.question == standalone
    assert generation.request.original_question == current
    assert pipeline[10].standalone_question == standalone


def test_failed_llm_rewrite_falls_back_to_reference_heuristic():
    retrieval = _RecordingRetrieval()
    generation = _RecordingGeneration()
    current = "Vậy điều đó áp dụng thế nào?"
    runtime = _runtime(generation)
    runtime.query_rewriter = _RecordingRewriter(None)
    history = [
        SimpleNamespace(role="user", content="Điều 36 Nghị định 123/2024/NĐ-CP quy định gì?"),
        SimpleNamespace(role="assistant", content="Điều 36 quy định về ..."),
        SimpleNamespace(role="user", content=current),
    ]

    asyncio.run(_execute_answer_pipeline(
        AnswerBody(query=current), retrieval, runtime, history=history,
    ))

    assert "123/2024/NĐ-CP" in retrieval.requests[0].query
    assert generation.request.question == current
    assert generation.request.original_question is None


class _ReplayRepository:
    def __init__(self):
        self.saved_replies = []

    def create_user_message(self, *args, **kwargs):
        return SimpleNamespace(id=uuid4()), False

    def find_assistant_reply(self, conversation_id, client_message_id):
        return None

    def save_assistant_reply(self, *args, **kwargs):
        self.saved_replies.append(kwargs)


def test_stream_retry_of_known_message_does_not_crash_or_overwrite(monkeypatch):
    @contextmanager
    def _session():
        yield None

    monkeypatch.setattr(main, "_check_user_quota", lambda db, user_id: None)
    monkeypatch.setattr(main.app.state, "session_factory", _session, raising=False)
    repository = _ReplayRepository()

    async def _collect():
        response = await main.chat_stream(
            ChatBody(conversation_id=uuid4(), client_message_id="m1", message="Câu hỏi"),
            SimpleNamespace(id=uuid4()),
            repository,
            "default",
            _RecordingRetrieval(),
            _runtime(_RecordingGeneration()),
        )
        return "".join([chunk async for chunk in response.body_iterator])

    events = asyncio.run(_collect())

    assert "MESSAGE_IN_PROGRESS" in events
    assert repository.saved_replies == []


def test_token_usage_counts_decomposition_generation_and_semantic_review():
    def usage(prompt, completion):
        return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)

    issue_plan = SimpleNamespace(telemetry=usage(100, 20))
    result = SimpleNamespace(
        telemetry=(usage(1000, 300), usage(900, 250)),
        semantic_observations=(
            SimpleNamespace(review=SimpleNamespace(telemetry=usage(500, 50))),
            SimpleNamespace(review=SimpleNamespace(telemetry=None)),  # cache hit
        ),
    )

    assert main._pipeline_token_usage((issue_plan, None, None, result)) == 3120
