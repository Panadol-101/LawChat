"""Regression tests for the question -> retrieval flow (review findings)."""

from datetime import date

from retrieval import LegalQueryParser, RetrievalRequest, RetrievalResponse
from retrieval.temporal import TemporalRetrievalRouter


class _CurrentTarget:
    def __init__(self):
        self.request = None

    def retrieve(self, request):
        self.request = request
        return RetrievalResponse(
            query=request.query,
            semantic_query=request.query,
            as_of=date(2026, 1, 1),
            results=(),
            searched_candidates=1,
            rejected_candidates=0,
        )


def test_question_with_case_date_is_routed_to_current_law():
    current = _CurrentTarget()
    router = TemporalRetrievalRouter(current)

    router.retrieve(
        RetrievalRequest(
            query="Tôi nghỉ việc vào ngày 1/3/2024 thì có được trợ cấp thôi việc không?"
        )
    )

    assert current.request is not None


def test_question_with_two_case_dates_is_not_rejected():
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    parsed = parser.parse("Tôi làm từ 01/01/2020 đến 01/01/2025 thì được nghỉ phép bao nhiêu ngày?")

    assert parsed.as_of == date(2026, 1, 1)


def test_money_and_rate_phrases_do_not_become_hard_filters():
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    money = parser.parse("Vay khoản 50 triệu không trả có bị đi tù không?")
    rate = parser.parse("Lãi suất 200/ngày có hợp pháp không?")

    assert money.referenced_clauses == ()
    assert rate.document_numbers == ()


def test_year_specific_law_alias_does_not_also_match_generic_alias():
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    parsed = parser.parse("BLLĐ 2012 quy định thời giờ làm việc thế nào?")

    assert parsed.document_numbers == ("10/2012/QH13",)


def test_semantic_query_stays_a_string_with_rewriter_v2(monkeypatch):
    monkeypatch.setenv("LAWCHAT_FLAG_SEMANTIC_REWRITER_V2", "true")
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    parsed = parser.parse("Sa thải người lao động trái luật bị xử lý thế nào?")

    assert isinstance(parsed.semantic_query, str)
