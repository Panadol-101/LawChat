import json
from datetime import date

import pytest

from rag import (
    InvalidStructuredResponseError,
    IssueDecompositionRequest,
    IssuePlan,
    OpenAICompatibleIssueDecomposer,
    OpenAICompatibleSettings,
    retrieve_issue_plan,
)
from retrieval import (
    LegalCitation,
    LegalIssue,
    LegalQueryParser,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedLegalChunk,
    UnsupportedAsOfDate,
)
from retrieval.query_parser import ensure_supported_as_of


def _decomposer(content):
    return OpenAICompatibleIssueDecomposer(
        OpenAICompatibleSettings("http://test", "", "gemma"),
        transport=lambda *args: {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
        },
    )


def test_issue_decomposer_requires_ordered_self_contained_issues():
    content = """{"issues":[
      {"issue_id":"I1","question":"Có trả thiếu lương không?","search_query":"người sử dụng lao động trả lương không đầy đủ"},
      {"issue_id":"I2","question":"Có được làm ban đêm không?","search_query":"lao động chưa thành niên làm việc ban đêm"}
    ]}"""
    plan = _decomposer(content).generate(IssueDecompositionRequest("Hai vấn đề"))
    assert [item.issue_id for item in plan.issues] == ["I1", "I2"]
    assert plan.issues[1].search_query == "lao động chưa thành niên làm việc ban đêm"


def test_issue_decomposer_collapses_generic_impact_after_specific_relationship():
    content = """{"issues":[
      {"issue_id":"I1","question":"74/2025/QH15 thay thế văn bản nào?","search_query":"74/2025/QH15 thay thế văn bản nào"},
      {"issue_id":"I2","question":"74/2025/QH15 tác động đến văn bản nào?","search_query":"74/2025/QH15 tác động đến văn bản nào"}
    ]}"""

    plan = _decomposer(content).generate(IssueDecompositionRequest("Quan hệ"))

    assert [(item.issue_id, item.search_query) for item in plan.issues] == [
        ("I1", "74/2025/QH15 thay thế văn bản nào")
    ]


@pytest.mark.parametrize(
    "content",
    [
        '{"issues":[]}',
        '{"issues":[{"issue_id":"I2","question":"x","search_query":"x"}]}',
        '{"issues":[{"issue_id":"I1","question":"","search_query":"x"}]}',
        '{"issues":[{"issue_id":"I1","question":"x","search_query":"x","extra":1}]}',
    ],
)
def test_issue_decomposer_rejects_incomplete_or_malformed_plan(content):
    with pytest.raises(InvalidStructuredResponseError):
        _decomposer(content).generate(IssueDecompositionRequest("Câu hỏi"))


def _result(chunk_id, issue_text):
    citation = LegalCitation(
        document_id="doc", title="Văn bản", document_number="01/2026/QH15",
        article="1", clause=None, point=None, source_url="https://example.test",
        as_of=date(2026, 6, 17), status="EFFECTIVE",
    )
    return RetrievedLegalChunk(
        point_id=chunk_id, chunk_id=chunk_id, score=1.0, text=issue_text,
        context_text=issue_text, chunk_type="article", citation=citation,
    )


def test_issue_aware_retrieval_runs_each_query_and_round_robins_with_deduplication():
    class Service:
        def __init__(self):
            self.queries = []

        def retrieve(self, request):
            self.queries.append(request.query)
            common = _result("common", "Nguồn chung")
            unique = _result(request.query, request.query)
            return RetrievalResponse(
                request.query, request.query, date(2026, 6, 17),
                (common, unique), 2, 0, retrieval_sources=("dense",),
            )

    service = Service()
    plan = IssuePlan((LegalIssue("I1", "Một", "q1"), LegalIssue("I2", "Hai", "q2")))
    response = retrieve_issue_plan(service, RetrievalRequest("original"), plan)
    assert service.queries == ["original", "q1", "q2"]
    assert response.query == "original"
    assert [item.chunk_id for item in response.results] == [
        "common", "q1", "q2", "original"
    ]
    assert response.results[0].metadata["issue_ids"] == ["I1", "I2"]
    assert response.results[0].metadata["issue_scores"] == {"I1": 1.0, "I2": 1.0}
    assert response.results[0].metadata["issue_ranks"] == {"I1": 1, "I2": 1}
    assert response.retrieval_trace == {
        "base": ("common", "original"),
        "I1": ("common", "q1"),
        "I2": ("common", "q2"),
    }
    assert "I1.total" in response.timings
    assert "I2.total" in response.timings
    assert response.legal_issues == plan.issues


def test_single_issue_retrieval_uses_original_query_without_llm_rewrite():
    class Service:
        def __init__(self):
            self.queries = []

        def retrieve(self, request):
            self.queries.append(request.query)
            return RetrievalResponse(
                request.query, request.query, date(2026, 6, 17),
                (_result("base", "Nguồn gốc"),), 1, 0,
            )

    service = Service()
    plan = IssuePlan((LegalIssue("I1", "Một", "query đã viết lại"),))

    response = retrieve_issue_plan(service, RetrievalRequest("query nguyên bản"), plan)

    assert service.queries == ["query nguyên bản"]
    assert response.results[0].metadata["issue_ids"] == ["I1"]
    assert response.retrieval_trace == {"base": ("base",)}


def test_reference_date_is_cutoff_and_other_as_of_dates_are_rejected(monkeypatch):
    monkeypatch.setenv("LAWCHAT_LEGAL_CUTOFF_DATE", "2026-07-31")
    parser = LegalQueryParser()
    assert parser.parse("Quy định là gì?").as_of == date(2026, 7, 31)
    # A date in the question is a case fact, not a request for historical law.
    assert parser.parse("Tại ngày 30/07/2015 quy định là gì?").as_of == date(2026, 7, 31)
    ensure_supported_as_of(None)
    ensure_supported_as_of(date(2026, 7, 31))
    with pytest.raises(UnsupportedAsOfDate, match="31/07/2026"):
        ensure_supported_as_of(date(2015, 1, 1))


def test_versioned_benchmark_parser_can_ignore_serving_cutoff(monkeypatch):
    monkeypatch.setenv("LAWCHAT_LEGAL_CUTOFF_DATE", "2026-07-31")
    parser = LegalQueryParser(
        today=lambda: date(2026, 8, 30),
        enforce_configured_cutoff=False,
    )

    parsed = parser.parse("Quy định là gì?", as_of=date(2026, 8, 30))

    assert parsed.as_of == date(2026, 8, 30)


def test_single_issue_heuristic_handles_law_names_and_conditional_clauses():
    from rag.issues import _looks_single_issue

    assert _looks_single_issue("Đánh bạc trên không gian mạng bị xử phạt thế nào?")
    assert _looks_single_issue("Luật Hôn nhân và gia đình quy định tuổi kết hôn ra sao?")
    assert not _looks_single_issue(
        "Công ty nợ lương 3 tháng, tôi nghỉ ngang có phải bồi thường không?"
    )
    assert not _looks_single_issue("Nếu bị sa thải trái luật thì được bồi thường gì?")


def test_shared_context_is_carried_into_every_issue_search_query():
    plan = _decomposer(json.dumps({
        "shared_context": "hợp đồng lao động 2 năm bị chấm dứt trước hạn",
        "issues": [
            {"issue_id": "I1", "question": "Bồi thường?", "search_query": "bồi thường khi chấm dứt trái luật"},
            {"issue_id": "I2", "question": "Trợ cấp?", "search_query": "điều kiện hưởng trợ cấp thất nghiệp"},
        ],
    }, ensure_ascii=False)).generate(IssueDecompositionRequest("q"))

    assert all(
        "hợp đồng lao động 2 năm bị chấm dứt trước hạn" in issue.search_query
        for issue in plan.issues
    )


def test_facade_accepts_single_issue_plan_and_degrades_on_errors():
    import asyncio

    from rag.issues import IssueDecomposer

    class OneIssue:
        calls = 0

        async def generate_async(self, request):
            OneIssue.calls += 1
            return IssuePlan((LegalIssue("I1", request.question, request.question),))

    class Broken:
        async def generate_async(self, request):
            raise InvalidStructuredResponseError("bad json")

    question = "Nếu bị sa thải trái luật thì được bồi thường gì?"
    plan = asyncio.run(IssueDecomposer(OneIssue()).decompose_async(IssueDecompositionRequest(question)))
    assert len(plan.issues) == 1 and OneIssue.calls == 1

    fallback = asyncio.run(IssueDecomposer(Broken()).decompose_async(IssueDecompositionRequest(question)))
    assert [issue.search_query for issue in fallback.issues] == [question]
