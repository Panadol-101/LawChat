from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_retrieval import (
    CATEGORY_TARGETS,
    _evaluate_v2_cases,
    _round_robin_rows,
)


FIXTURE = Path("tests/fixtures/legal_retrieval_benchmark_bge_m3_v1.json")


def test_benchmark_bge_m3_v1_has_200_unique_cases_and_requested_distribution():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = payload["cases"]

    assert payload["schema_version"] == "2.0"
    assert payload["benchmark_version"] == "2.0.0"
    assert payload["corpus_release"] == "chunker-v4-bge-m3-20260830"
    assert payload["qdrant_collection"] == "legal_chunks_bge_m3_1024_v1"
    assert payload["bm25_index"] == "legal_bm25_bge_m3_v1"
    assert len(cases) == 200
    assert len({case["case_id"] for case in cases}) == 200
    assert Counter(case["category"] for case in cases) == Counter(CATEGORY_TARGETS)
    assert CATEGORY_TARGETS["exact_document_number"] == 10
    assert CATEGORY_TARGETS["exact_provision"] == 10


def test_answerable_cases_have_ground_truth_and_unanswerable_cases_do_not():
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

    for case in cases:
        assert case["query"].strip()
        assert case["source"]["ground_truth_method"]
        if case["answerable"]:
            assert case["expected"].get("document_ids")
            assert case["expected"].get("statuses")
            if case["expected"].get("chunk_ids"):
                relevance = case["expected"].get("relevance")
                assert relevance
                assert 3 in relevance.values()
                assert all(relevance[chunk_id] >= 2 for chunk_id in case["expected"]["chunk_ids"])
        else:
            assert case["category"] == "insufficient_evidence"
            assert case["expected"]["answerability"] == "insufficient"
            assert not case["expected"]["chunk_ids"]


def test_v2_evaluator_reports_strict_document_structure_and_status_hits():
    case = {
        "case_id": "example",
        "category": "exact_provision",
        "query": "example",
        "answerable": True,
        "expected": {
            "chunk_ids": ["doc::article_5::clause_2"],
            "document_ids": ["doc"],
            "articles": ["5"],
            "clauses": ["2"],
            "statuses": ["EFFECTIVE"],
        },
    }
    result = SimpleNamespace(
        chunk_id="doc::article_5::clause_2",
        citation=SimpleNamespace(
            document_id="doc",
            article="5",
            clause="2",
            point=None,
            status="EFFECTIVE",
        ),
    )
    responses = {"example": SimpleNamespace(results=(result,))}

    report = _evaluate_v2_cases([case], responses, k=10)

    assert report["strict_hit_rate_at_k"] == 1.0
    assert report["chunk_hit_rate_at_k"] == 1.0
    assert report["document_hit_rate_at_k"] == 1.0
    assert report["structure_accuracy_at_k"] == 1.0
    assert report["correct_legal_provision_retrieval_rate"] == 1.0
    assert report["status_accuracy_at_k"] == 1.0


def test_round_robin_curated_rows_allows_variable_v4_chunk_counts():
    rows = [
        [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}],
        [{"id": "b1"}],
        [{"id": "c1"}, {"id": "c2"}],
    ]

    assert [item["id"] for item in _round_robin_rows(rows)] == [
        "a1",
        "b1",
        "c1",
        "a2",
        "c2",
        "a3",
    ]


def test_v2_evaluator_reports_expired_status_metadata_coverage_and_selection():
    case = {
        "case_id": "expired",
        "category": "expired_status",
        "query": "Văn bản 12/2013/QĐ-UBND còn hiệu lực không?",
        "answerable": True,
        "expected": {
            "chunk_ids": ["expected::article_3"],
            "document_ids": ["expected"],
            "statuses": ["EXPIRED"],
        },
    }
    expected_seed = SimpleNamespace(
        document=SimpleNamespace(document_id="expected", status="EXPIRED")
    )
    response = SimpleNamespace(
        results=(),
        seed_resolutions=(expected_seed,),
        status_resolution=SimpleNamespace(
            document_id="expected",
            status="EXPIRED",
        ),
    )

    report = _evaluate_v2_cases([case], {"expired": response}, k=10)

    assert report["status_metadata_accuracy"] == 1.0
    assert report["status_selection_accuracy"] == 1.0
