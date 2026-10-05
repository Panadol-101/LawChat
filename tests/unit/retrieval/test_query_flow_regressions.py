"""Regression tests for the question -> retrieval flow (review findings)."""

from datetime import date

from retrieval import LegalQueryParser, RetrievalRequest
from retrieval.temporal import TemporalIntent, TemporalPolicy


def test_question_with_case_date_is_answered_under_current_law():
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))
    query = "Tôi nghỉ việc vào ngày 1/3/2024 thì có được trợ cấp thôi việc không?"

    parsed = parser.parse(query)
    decision = TemporalPolicy().decide(parsed, RetrievalRequest(query=query))

    assert parsed.as_of == date(2026, 1, 1)
    assert decision.intent is TemporalIntent.CURRENT_LAW


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


def test_law_aliases_resolve_old_names_to_current_law_without_duplicates():
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    old_name = parser.parse("BLLĐ 2012 quy định thời giờ làm việc thế nào?")
    current = parser.parse("Luật Doanh nghiệp 2020 quy định gì về vốn điều lệ?")

    # The replaced law is kept (its status is disclosed) and the law in force
    # is searched too, because LawChat answers from current law only.
    assert old_name.document_numbers == ("10/2012/QH13", "45/2019/QH14")
    assert current.document_numbers == ("59/2020/QH14",)


def test_semantic_query_stays_a_string_with_rewriter_v2(monkeypatch):
    monkeypatch.setenv("LAWCHAT_FLAG_SEMANTIC_REWRITER_V2", "true")
    parser = LegalQueryParser(today=lambda: date(2026, 1, 1))

    parsed = parser.parse("Sa thải người lao động trái luật bị xử lý thế nào?")

    assert isinstance(parsed.semantic_query, str)
