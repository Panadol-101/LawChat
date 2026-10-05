from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    relevance_grades: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty")
        if any(value < 0 for value in self.relevance_grades.values()):
            raise ValueError("relevance grades must be >= 0")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    cases: int
    recall_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float


def evaluate_rankings(
    cases: Sequence[RetrievalCase],
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int,
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("cases must not be empty")
    if k <= 0:
        raise ValueError("k must be > 0")
    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for case in cases:
        ranked = list(rankings.get(case.case_id, ()))[:k]
        relevant = case.relevant_chunk_ids
        matched = [chunk_id in relevant for chunk_id in ranked]
        matched_count = sum(matched)
        recalls.append(matched_count / len(relevant))
        hits.append(float(matched_count > 0))
        first_rank = next(
            (index for index, is_relevant in enumerate(matched, start=1) if is_relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        gains = [
            case.relevance_grades.get(chunk_id, int(chunk_id in relevant))
            for chunk_id in ranked
        ]
        dcg = sum(
            (2**gain - 1) / log2(index + 1)
            for index, gain in enumerate(gains, start=1)
            if gain > 0
        )
        ideal_gains = sorted(
            case.relevance_grades.values()
            if case.relevance_grades
            else (1 for _chunk_id in relevant),
            reverse=True,
        )[:k]
        ideal_dcg = sum(
            (2**gain - 1) / log2(index + 1)
            for index, gain in enumerate(ideal_gains, start=1)
        )
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    count = len(cases)
    return RetrievalMetrics(
        cases=count,
        recall_at_k=sum(recalls) / count,
        hit_rate_at_k=sum(hits) / count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
        ndcg_at_k=sum(ndcgs) / count,
    )


@dataclass(frozen=True, slots=True)
class GenerationCaseResult:
    case_id: str
    answerable: bool
    status: str
    attempts: int
    expected_document_ids: tuple[str, ...]
    cited_document_ids: tuple[str, ...]
    verification_codes: tuple[str, ...]
    latency_seconds: float
    semantic_mode: str = "off"
    semantic_status: str = "DISABLED"
    coverage_status: str = "NOT_CHECKED"
    semantic_observations: tuple[dict, ...] = ()
    generation_telemetry: tuple[dict, ...] = ()
    answer: dict | None = None
    evidence: tuple[dict, ...] = ()
    query: str = ""
    retrieval_answerable: bool | None = None
    expected_refusal_reason: str | None = None
    legal_issues: tuple[dict, ...] = ()
    retrieval_trace: dict[str, tuple[str, ...]] = field(default_factory=dict)
    context_dropped_chunk_ids: tuple[str, ...] = ()


def summarize_generation_results(
    results: list[GenerationCaseResult],
) -> dict:
    answerable = [item for item in results if item.answerable]
    unanswerable = [item for item in results if not item.answerable]
    false_refusals = [item for item in answerable if item.status == "REFUSED"]
    unsafe_answers = [item for item in unanswerable if item.status == "VERIFIED"]
    repaired = [item for item in results if item.attempts == 2]
    failures = [
        item
        for item in results
        if any(
            code.startswith("GENERATION_")
            or code in {"PROVIDER_UNAVAILABLE", "INVALID_STRUCTURED_RESPONSE"}
            for code in item.verification_codes
        )
    ]
    citation_total = sum(len(item.cited_document_ids) for item in answerable)
    citation_hits = sum(
        sum(
            document_id in set(item.expected_document_ids)
            for document_id in item.cited_document_ids
        )
        for item in answerable
    )
    latencies = sorted(item.latency_seconds for item in results)
    total = len(results)
    return {
        "total_cases": total,
        "semantic_flagged_cases": sum(item.semantic_status == "FLAGGED" for item in results),
        "semantic_error_cases": sum(item.semantic_status == "ERROR" for item in results),
        "semantic_checked_cases": sum(item.semantic_status in {"PASSED", "FLAGGED", "ERROR"} for item in results),
        "coverage_complete_cases": sum(item.coverage_status == "COMPLETE" for item in results),
        "coverage_incomplete_cases": sum(item.coverage_status == "INCOMPLETE" for item in results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "expected_refusal_cases": sum(
            item.expected_refusal_reason is not None for item in results
        ),
        "verified_rate": _ratio(
            sum(item.status == "VERIFIED" for item in results), total
        ),
        "answerable_verified_rate": _ratio(
            sum(item.status == "VERIFIED" for item in answerable), len(answerable)
        ),
        "false_refusal_rate": _ratio(len(false_refusals), len(answerable)),
        "unsafe_answer_rate": _ratio(len(unsafe_answers), len(unanswerable)),
        "repair_rate": _ratio(len(repaired), total),
        "generation_failure_rate": _ratio(len(failures), total),
        "grounding_document_precision": _ratio(citation_hits, citation_total),
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": latencies[-1] if latencies else 0.0,
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


@dataclass(frozen=True, slots=True)
class HistoricalGateObservation:
    case_id: str
    expected_content_kind: str
    actual_content_kind: str | None
    expected_version_id: str | None
    cited_version_ids: tuple[str, ...] = ()
    historical_available: bool = True
    response_status: str = "VERIFIED"


@dataclass(frozen=True, slots=True)
class HistoricalGateReport:
    total: int
    current_as_historical: int
    historical_as_current: int
    incorrect_version_citations: int
    unavailable_not_refused: int

    @property
    def passed(self) -> bool:
        return all(
            value == 0
            for value in (
                self.current_as_historical,
                self.historical_as_current,
                self.incorrect_version_citations,
                self.unavailable_not_refused,
            )
        )


def evaluate_historical_gate(
    observations: Iterable[HistoricalGateObservation],
) -> HistoricalGateReport:
    rows = tuple(observations)
    current_as_historical = sum(
        row.expected_content_kind == "historical"
        and row.actual_content_kind == "current"
        for row in rows
    )
    historical_as_current = sum(
        row.expected_content_kind == "current"
        and row.actual_content_kind == "historical"
        for row in rows
    )
    incorrect_citations = sum(
        bool(row.expected_version_id)
        and (
            not row.cited_version_ids
            or any(item != row.expected_version_id for item in row.cited_version_ids)
        )
        for row in rows
        if row.historical_available
    )
    unavailable_not_refused = sum(
        not row.historical_available and row.response_status != "REFUSED"
        for row in rows
    )
    return HistoricalGateReport(
        total=len(rows),
        current_as_historical=current_as_historical,
        historical_as_current=historical_as_current,
        incorrect_version_citations=incorrect_citations,
        unavailable_not_refused=unavailable_not_refused,
    )
