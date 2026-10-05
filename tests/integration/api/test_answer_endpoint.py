from __future__ import annotations

import re
import time
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_rag_runtime,
    get_retrieval_service,
    require_admin,
)
from rag import (
    FakeLegalAnswerGenerator,
    GeneratedAnswer,
    GeneratedClaim,
    GroundedRAGService,
    RAGContextBuilder,
    RAGExecutionGate,
    RAGRuntime,
    IssuePlan,
    IssueResolution,
    SupportingQuote,
)
from retrieval import (
    LegalCitation,
    RetrievalResponse,
    RetrievedLegalChunk,
    LegalIssue,
)


AS_OF = date(2026, 9, 1)


@pytest.fixture(autouse=True)
def _reference_date(monkeypatch):
    # The API only accepts as_of equal to the data reference date.
    monkeypatch.setenv("LAWCHAT_LEGAL_CUTOFF_DATE", AS_OF.isoformat())


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class FakeRetrievalService:
    def __init__(self, *, results=(), error: Exception | None = None) -> None:
        self.results = tuple(results)
        self.error = error
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return RetrievalResponse(
            query=request.query,
            semantic_query=request.query,
            as_of=request.as_of or AS_OF,
            results=self.results,
            searched_candidates=len(self.results),
            rejected_candidates=0,
            retrieval_sources=("dense", "sparse"),
        )


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    app.dependency_overrides[require_admin] = lambda: SimpleNamespace(role="ADMIN")
    yield
    app.dependency_overrides.clear()


def _result() -> RetrievedLegalChunk:
    text = "Khoản 1 Điều 5 quy định người sử dụng lao động phải báo cáo định kỳ."
    return RetrievedLegalChunk(
        point_id="point-1",
        chunk_id="doc-1::article_5::clause_1",
        score=0.9,
        text=text,
        context_text=text,
        chunk_type="clause",
        citation=LegalCitation(
            document_id="doc-1",
            title="Văn bản thử nghiệm",
            document_number="01/2025/QH15",
            article="5",
            clause="1",
            point=None,
            source_url="https://example.test/doc-1",
            as_of=AS_OF,
            status="EFFECTIVE",
        ),
    )


def _answer(evidence_ids=("E1",)) -> GeneratedAnswer:
    text = "Khoản 1 Điều 5 của 01/2025/QH15 quy định nghĩa vụ báo cáo định kỳ."
    return GeneratedAnswer(
        answer=text,
        claims=(GeneratedClaim(text, tuple(evidence_ids)),),
        limitations=(),
        confidence="high",
    )


def _client(retrieval_service, generator, *, gate=None) -> TestClient:
    runtime = RAGRuntime(
        context_builder=RAGContextBuilder(WordCounter()),
        generation_service=GroundedRAGService(generator),
        execution_gate=gate or RAGExecutionGate(
            max_concurrency=1,
            max_queue=4,
            queue_timeout_seconds=1,
            request_timeout_seconds=10,
        ),
    )
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_rag_runtime] = lambda: runtime
    return TestClient(app)


def _request(client: TestClient):
    return client.post(
        "/api/v1/answer",
        json={
            "query": "Khoản 1 Điều 5 quy định gì?",
            "as_of": AS_OF.isoformat(),
            "limit": 5,
            "candidate_limit": 20,
            "response_mode": "verbose",
        },
    )


def test_answer_endpoint_returns_verified_grounded_response_over_http():
    retrieval = FakeRetrievalService(results=(_result(),))
    generator = FakeLegalAnswerGenerator([_answer(), _answer()])
    client = _client(retrieval, generator)

    first = _request(client)
    second = _request(client)

    assert first.status_code == 200
    payload = first.json()
    assert payload["status"] == "VERIFIED"
    assert payload["attempts"] == 1
    assert payload["verification"] == {"status": "VERIFIED", "issues": []}
    assert payload["mode"] == "verbose"
    assert payload["claims"][0]["evidence_ids"] == ["E1"]
    assert payload["answer"].endswith("[E1]")
    assert payload["evidence"][0]["evidence_id"] == "E1"
    assert payload["evidence"][0]["document_number"] == "01/2025/QH15"
    assert payload["retrieval"]["sources"] == ["dense", "sparse"]
    assert payload["context"]["evidence_count"] == 1
    assert payload["timing"]["total"] >= 0
    assert payload["timing"]["queue_wait"] >= 0
    assert payload["timing"]["processing"] >= 0
    assert [item["stage"] for item in payload["timing"]["stages"]] == [
        "structural_preflight", "generation", "structural_verification"
    ]
    assert second.status_code == 200
    assert len(generator.requests) == 2
    assert len(retrieval.requests) == 2


def test_answer_endpoint_refuses_empty_evidence_without_calling_llm():
    retrieval = FakeRetrievalService()
    generator = FakeLegalAnswerGenerator([_answer()])
    client = _client(retrieval, generator)

    response = _request(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "REFUSED"
    assert payload["attempts"] == 0
    assert payload["claims"] == []
    assert payload["verification"]["issues"][0]["code"] == "INSUFFICIENT_EVIDENCE"
    assert generator.requests == []


def test_prediction_safety_gate_skips_retrieval_and_generation():
    retrieval = FakeRetrievalService(results=(_result(),))
    generator = FakeLegalAnswerGenerator([_answer()])
    client = _client(retrieval, generator)

    response = client.post(
        "/api/v1/answer",
        json={
            "query": "Hãy dự đoán chính xác ngày cơ quan nhà nước giải quyết xong hồ sơ của tôi.",
            "as_of": AS_OF.isoformat(),
            "response_mode": "verbose",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REFUSED"
    assert body["attempts"] == 0
    assert body["verification"]["issues"][0]["code"] == (
        "CASE_SPECIFIC_PREDICTION_UNSUPPORTED"
    )
    assert retrieval.requests == []
    assert generator.requests == []


def test_answer_endpoint_repairs_once_before_returning_verified():
    retrieval = FakeRetrievalService(results=(_result(),))
    generator = FakeLegalAnswerGenerator([_answer(("E404",)), _answer()])
    client = _client(retrieval, generator)

    response = _request(client)

    assert response.status_code == 200
    assert response.json()["status"] == "VERIFIED"
    assert response.json()["attempts"] == 2
    assert len(generator.requests) == 2
    assert generator.requests[1].previous_answer == _answer(("E404",))
    assert generator.requests[1].repair_instructions


def test_answer_endpoint_hides_provider_exception_and_fails_closed():
    secret = "provider-secret-must-not-leak"
    retrieval = FakeRetrievalService(results=(_result(),))
    generator = FakeLegalAnswerGenerator([RuntimeError(secret)])
    client = _client(retrieval, generator)

    response = _request(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "REFUSED"
    assert payload["attempts"] == 1
    assert payload["verification"]["issues"][0]["code"] == "GENERATION_FAILED"
    assert secret not in response.text


def test_answer_endpoint_rejects_bad_limits_and_historical_as_of():
    retrieval = FakeRetrievalService()
    client = _client(retrieval, FakeLegalAnswerGenerator([_answer()]))

    bad_limit = client.post(
        "/api/v1/answer",
        json={"query": "test", "limit": 10, "candidate_limit": 5},
    )
    historical = client.post(
        "/api/v1/answer",
        json={"query": "Quy định là gì?", "as_of": "2015-01-01"},
    )

    assert bad_limit.status_code == 422
    assert historical.status_code == 422
    assert "pháp luật đang có hiệu lực" in historical.json()["detail"]


def test_answer_endpoint_defaults_to_compact_typed_response():
    retrieval = FakeRetrievalService(results=(_result(),))
    client = _client(retrieval, FakeLegalAnswerGenerator([_answer()]))

    response = client.post(
        "/api/v1/answer",
        json={"query": "Khoản 1 Điều 5 quy định gì?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "compact"
    assert payload["status"] == "VERIFIED"
    assert payload["citations"][0]["evidence_id"] == "E1"
    assert "verification" not in payload
    assert "evidence" not in payload
    assert "retrieval" not in payload


def test_answer_endpoint_enforces_total_request_deadline():
    class SlowRetrieval(FakeRetrievalService):
        def retrieve(self, request):
            time.sleep(0.05)
            return super().retrieve(request)

    gate = RAGExecutionGate(
        max_concurrency=1,
        max_queue=0,
        queue_timeout_seconds=0.01,
        request_timeout_seconds=0.01,
    )
    client = _client(
        SlowRetrieval(results=(_result(),)),
        FakeLegalAnswerGenerator([_answer()]),
        gate=gate,
    )

    response = _request(client)

    assert response.status_code == 504
    assert response.json()["detail"] == "RAG request deadline exceeded"


def test_answer_endpoint_exposes_shadow_flag_and_review_usage():
    from rag.semantic import SemanticVerifier, SemanticReview, ClaimCheck
    from rag.generator import GenerationTelemetry

    class Judge:
        async def generate_async(self, request):
            return SemanticReview(checks=(ClaimCheck(0, 'INSUFFICIENT_EVIDENCE', 'Chưa đủ căn cứ.'),),
                telemetry=GenerationTelemetry(.1, .05, 100, 20))

    generated = _answer()
    generated = replace(
        generated,
        claims=(replace(
            generated.claims[0],
            supporting_quotes=(SupportingQuote("E1", _result().text),),
        ),),
    )
    runtime = RAGRuntime(
        context_builder=RAGContextBuilder(WordCounter()),
        generation_service=GroundedRAGService(FakeLegalAnswerGenerator([generated]),
            semantic_verifier=SemanticVerifier(Judge()), semantic_mode='shadow'),
    )
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService(results=(_result(),))
    app.dependency_overrides[get_rag_runtime] = lambda: runtime
    response = _request(TestClient(app))
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'VERIFIED'
    assert body['semantic_mode'] == 'shadow'
    assert body['semantic_status'] == 'FLAGGED'
    assert len(body['semantic_observations']) == 1
    assert body['timing']['semantic_attempts'][0]['prompt_tokens'] == 100
    assert body['attempts'] == 1


def test_answer_endpoint_decomposes_and_retrieves_every_legal_issue():
    class Decomposer:
        async def decompose_async(self, request):
            return IssuePlan((
                LegalIssue("I1", "Nghĩa vụ báo cáo?", "nghĩa vụ báo cáo"),
                LegalIssue("I2", "Thời hạn báo cáo?", "thời hạn báo cáo"),
            ))

    retrieval = FakeRetrievalService(results=(_result(),))
    generated = _answer()
    generated = replace(
        generated,
        claims=(replace(
            generated.claims[0],
            issue_ids=("I1", "I2"),
            supporting_quotes=(SupportingQuote("E1", _result().text),),
        ),),
        issue_resolutions=(
            IssueResolution("I1", "ANSWERED", (0,), "Claim 0 trả lời nghĩa vụ."),
            IssueResolution("I2", "ANSWERED", (0,), "Claim 0 trả lời thời hạn."),
        ),
    )
    runtime = RAGRuntime(
        context_builder=RAGContextBuilder(WordCounter()),
        generation_service=GroundedRAGService(FakeLegalAnswerGenerator([generated])),
        issue_decomposer=Decomposer(),
    )
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_rag_runtime] = lambda: runtime

    response = _request(TestClient(app))

    assert response.status_code == 200
    body = response.json()
    assert [item["issue_id"] for item in body["legal_issues"]] == ["I1", "I2"]
    assert [item["issue_id"] for item in body["issue_resolutions"]] == ["I1", "I2"]
    assert [item.query for item in retrieval.requests] == [
        "Khoản 1 Điều 5 quy định gì?",
        "nghĩa vụ báo cáo",
        "thời hạn báo cáo",
    ]
