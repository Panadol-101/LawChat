import json
from datetime import date
from pathlib import Path

from retrieval import (
    ALL_LEGAL_STATUSES,
    LegalQueryParser,
    RetrievalRequest,
    TemporalIntent,
    TemporalPolicy,
)


def _decide(query: str, **request_options):
    parser = LegalQueryParser(
        today=lambda: date(2026, 8, 31), enforce_configured_cutoff=False
    )
    request = RetrievalRequest(query=query, **request_options)
    parsed = parser.parse(query, as_of=request.as_of)
    return TemporalPolicy().decide(parsed, request)


def test_current_law_defaults_to_active_statuses():
    decision = _decide("Hiện nay điều kiện cấp phép là gì?")

    assert decision.intent is TemporalIntent.CURRENT_LAW
    assert decision.allowed_statuses == ("EFFECTIVE", "PARTIALLY_EFFECTIVE")


def test_exact_document_preserves_inactive_statuses():
    decision = _decide("Văn bản 12/2013/QĐ-UBND quy định gì?")

    assert decision.preserve_explicit_documents
    assert decision.allowed_statuses == ALL_LEGAL_STATUSES


def test_status_lookup_uses_all_statuses_and_is_not_misclassified_as_historical():
    decision = _decide(
        "Văn bản 12/2013/QĐ-UBND từng quy định nội dung này hiện còn hiệu lực không?",
        as_of=date(2026, 8, 30),
    )

    assert decision.intent is TemporalIntent.STATUS_LOOKUP
    assert decision.allowed_statuses == ALL_LEGAL_STATUSES
    assert decision.expand_replacements


def test_dated_question_is_current_law_without_historical_warning():
    decision = _decide("Vào ngày 01/05/2015, quy định về nguyên tắc áp dụng là gì?")

    assert decision.intent is TemporalIntent.CURRENT_LAW
    assert decision.warnings == ()


def test_explicit_current_wording_wins_over_date_for_current_snapshot_queries():
    decision = _decide(
        "Quy định hiện hành tại ngày 01/05/2015 là gì?",
        as_of=date(2015, 5, 1),
    )
    assert decision.intent is TemporalIntent.CURRENT_LAW


def test_resolved_current_intent_survives_decomposed_query_without_current_wording():
    decision = _decide(
        "quy định về thụ lý vụ án",
        as_of=date(2026, 8, 30),
        resolved_temporal_intent="current_law",
    )

    assert decision.intent is TemporalIntent.CURRENT_LAW


def test_explicit_status_override_wins_over_policy_defaults():
    decision = _decide(
        "Văn bản 12/2013/QĐ-UBND còn hiệu lực không?",
        statuses=("EXPIRED",),
    )

    assert decision.allowed_statuses == ("EXPIRED",)


def test_all_15_expired_status_benchmark_queries_parse_as_exact_status_lookups():
    fixture = json.loads(
        Path("tests/fixtures/legal_retrieval_benchmark_bge_m3_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cases = [
        item for item in fixture["cases"] if item["category"] == "expired_status"
    ]
    parser = LegalQueryParser(
        today=lambda: date(2026, 8, 31), enforce_configured_cutoff=False
    )

    assert len(cases) == 15
    for case in cases:
        parsed = parser.parse(
            case["query"],
            as_of=date.fromisoformat(case["request"]["as_of"]),
        )
        decision = TemporalPolicy().decide(
            parsed,
            RetrievalRequest(
                query=case["query"],
                as_of=date.fromisoformat(case["request"]["as_of"]),
                statuses=tuple(case["request"]["statuses"]),
            ),
        )

        assert parsed.document_numbers
        assert parsed.document_numbers[0].replace(" ", "") == case["expected"][
            "document_numbers"
        ][0].replace(" ", "")
        assert "hiện còn hiệu lực không" not in parsed.semantic_query.casefold()
        assert decision.intent is TemporalIntent.STATUS_LOOKUP
        assert decision.allowed_statuses == ("EXPIRED",)
