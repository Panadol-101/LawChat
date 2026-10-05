from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

from database import DatabaseSettings, create_db_engine
from evaluation import (
    GenerationCaseResult,
    HistoricalGateObservation,
    evaluate_historical_gate,
    summarize_generation_results,
)
from indexing import QdrantSettings, SentenceTransformerEmbedder
from rag import (
    Evidence,
    GenerationRequest,
    GeneratedAnswer,
    GeneratedClaim,
    GroundedRAGService,
    GroundingVerifier,
    IssueDecompositionRequest,
    OpenAICompatibleLegalAnswerGenerator,
    OpenAICompatibleSettings,
    PackedContext,
    create_rag_runtime,
    retrieve_issue_plan,
)
from rag.generator import SupportingQuote
from rag.semantic import (
    JUDGE_PROMPT_VERSION,
    OpenAICompatibleClaimJudge,
    SemanticVerifier,
)
from scripts import dispatch
from retrieval import LegalQueryParser, RetrievalRequest
from retrieval.runtime import create_hybrid_retrieval_service


def generation_main() -> None:
    args = _parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if args.category:
        cases = [item for item in cases if item.get("category") in args.category]
    if args.sample_size is not None:
        cases = _stratified_sample(cases, args.sample_size)
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    completed = _load_completed(args.output) if args.resume else {}
    if args.rerun_refused:
        completed = {
            case_id: item
            for case_id, item in completed.items()
            if item.get("status") != "REFUSED"
            and not (
                item.get("expected_refusal_reason") is not None
                and item.get("status") == "VERIFIED"
            )
        }

    engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()
    runtime = create_rag_runtime(enforce_configured_cutoff=False)
    retrieval_service = create_hybrid_retrieval_service(
        engine,
        client,
        qdrant_settings,
        SentenceTransformerEmbedder.from_env(batch_size=1),
        query_parser=LegalQueryParser(enforce_configured_cutoff=False),
    )
    try:
        for index, case in enumerate(cases, start=1):
            if case["case_id"] in completed:
                continue
            started = time.perf_counter()
            request_options = {
                "query": case["query"],
                "limit": args.limit,
                "candidate_limit": args.candidate_limit,
                "max_candidate_limit": max(500, args.candidate_limit),
            }
            if case.get("request", {}).get("as_of"):
                request_options["as_of"] = date.fromisoformat(
                    case["request"]["as_of"]
                )
            if case.get("request", {}).get("statuses"):
                request_options["statuses"] = tuple(
                    case["request"]["statuses"]
                )
            base_request = RetrievalRequest(**request_options)
            issue_plan = None
            if args.decompose_issues:
                issue_plan = runtime.issue_decomposer.generate(
                    IssueDecompositionRequest(case["query"])
                )
                retrieval = retrieve_issue_plan(
                    retrieval_service, base_request, issue_plan
                )
            else:
                retrieval = retrieval_service.retrieve(base_request)
            context = runtime.context_builder.build(retrieval)
            result = asyncio.run(
                runtime.generation_service.answer_async(
                    GenerationRequest.from_retrieval(retrieval, context)
                )
            )
            evidence_by_id = {
                item.evidence_id: item for item in context.evidence
            }
            cited_documents = tuple(
                dict.fromkeys(
                    evidence_by_id[evidence_id].document_id
                    for claim in result.answer.claims
                    for evidence_id in claim.evidence_ids
                    if evidence_id in evidence_by_id
                )
            )
            retrieval_answerable = case.get("answerable", True)
            expected_refusal_reason = None
            rag_answerable = retrieval_answerable
            if not retrieval_answerable:
                expected_refusal_reason = "FIXTURE_INSUFFICIENT_EVIDENCE"
            case_result = GenerationCaseResult(
                case_id=case["case_id"],
                answerable=rag_answerable,
                status=result.status.value,
                attempts=result.attempts,
                expected_document_ids=tuple(
                    case.get("expected", {}).get("document_ids", ())
                ),
                cited_document_ids=cited_documents,
                verification_codes=tuple(
                    issue.code.value for issue in result.verification.issues
                ),
                latency_seconds=round(time.perf_counter() - started, 3),
                semantic_mode=result.semantic_mode,
                semantic_status=result.semantic_status,
                coverage_status=result.coverage_status,
                semantic_observations=tuple(asdict(item) for item in result.semantic_observations),
                generation_telemetry=tuple(asdict(item) for item in result.telemetry),
                answer=result.answer.to_dict(),
                evidence=tuple({"evidence_id": item.evidence_id, "document_id": item.document_id, "text": item.text} for item in context.evidence),
                query=case["query"],
                retrieval_answerable=retrieval_answerable,
                expected_refusal_reason=expected_refusal_reason,
                legal_issues=tuple(
                    asdict(item) for item in retrieval.legal_issues
                ),
                retrieval_trace={
                    key: tuple(value)
                    for key, value in retrieval.retrieval_trace.items()
                },
                context_dropped_chunk_ids=context.dropped_chunk_ids,
            )
            completed[case_result.case_id] = asdict(case_result)
            _write_report(args.output, fixture, completed)
            print(
                f"[{index}/{len(cases)}] {case_result.case_id}: "
                f"{case_result.status} {case_result.latency_seconds:.1f}s",
                flush=True,
            )
    finally:
        client.close()
        engine.dispose()

    report = _write_report(args.output, fixture, completed)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    metrics = report["metrics"]
    if (
        args.max_false_refusal_rate is not None
        and metrics["false_refusal_rate"] is not None
        and metrics["false_refusal_rate"] > args.max_false_refusal_rate
    ):
        raise SystemExit(1)
    if (
        args.max_unsafe_answer_rate is not None
        and metrics["unsafe_answer_rate"] is not None
        and metrics["unsafe_answer_rate"] > args.max_unsafe_answer_rate
    ):
        raise SystemExit(1)


def _write_report(path: Path, fixture: dict, completed: dict) -> dict:
    results = [GenerationCaseResult(**item) for item in completed.values()]
    report = {
        "benchmark_version": fixture.get("benchmark_version"),
        "corpus_release": fixture.get("corpus_release"),
        "metrics": summarize_generation_results(results),
        "results": list(completed.values()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _load_completed(path: Path) -> dict:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in report.get("results", ())}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded LLM generation on curated legal cases."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/legal_retrieval_benchmark_bge_m3_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/RAG_GENERATION_EVAL.json"),
    )
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Deterministic round-robin sample across benchmark categories.",
    )
    parser.add_argument("--category", action="append")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-refused",
        action="store_true",
        help="With --resume, discard and rerun previously refused cases.",
    )
    parser.add_argument(
        "--decompose-issues",
        action="store_true",
        help="Run issue decomposition and coverage review for completeness evaluation.",
    )
    parser.add_argument("--max-false-refusal-rate", type=float)
    parser.add_argument("--max-unsafe-answer-rate", type=float, default=0.0)
    return parser.parse_args()


def _stratified_sample(cases: list[dict], sample_size: int) -> list[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    groups: dict[str, deque[dict]] = defaultdict(deque)
    for case in cases:
        groups[case.get("category", "uncategorized")].append(case)
    selected: list[dict] = []
    categories = sorted(groups)
    while len(selected) < min(sample_size, len(cases)):
        added = False
        for category in categories:
            if groups[category] and len(selected) < sample_size:
                selected.append(groups[category].popleft())
                added = True
        if not added:
            break
    return selected


def case_request(case):
    from retrieval import LegalCitation

    citation = LegalCitation(
        document_id="synthetic",
        title="Văn bản giả lập, không phải pháp luật thực tế",
        document_number="01/2026/QH15",
        article="1",
        clause=None,
        point=None,
        source_url="https://example.test/synthetic",
        as_of=date(2026, 9, 6),
        status="EFFECTIVE",
    )
    evidence = Evidence(
        "E1", "synthetic", "synthetic::1", case["evidence"],
        citation, "queried_document", 0,
    )
    request = GenerationRequest(
        case["question"],
        PackedContext((evidence,), "[E1]\n" + evidence.text, 0, 8000, ()),
        citation.as_of,
    )
    answer = GeneratedAnswer(
        "",
        (GeneratedClaim(
            case["claim"], ("E1",),
            (SupportingQuote("E1", evidence.text),),
        ),),
        (),
        "high",
    )
    return request, answer


def summarize_entailment(rows):
    positive = [row for row in rows if row["expected_verdict"] == "SUPPORTED"]
    negative = [row for row in rows if row["expected_verdict"] != "SUPPORTED"]
    valid_negative = [row for row in negative if not row["review"]["error"]]
    latencies = sorted(row["review"]["total_seconds"] for row in rows)

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    telemetry = [row["review"]["telemetry"] for row in rows if row["review"]["telemetry"]]
    return {
        "cases": len(rows),
        "review_errors": sum(bool(row["review"]["error"]) for row in rows),
        "baseline_structural_acceptance_rate": ratio(
            sum(row["structural_status"] == "VERIFIED" for row in rows), len(rows)
        ),
        "exact_verdict_accuracy": ratio(
            sum(row["actual_verdict"] == row["expected_verdict"] and not row["review"]["error"] for row in rows),
            len(rows),
        ),
        "unsafe_claim_detection_rate": ratio(
            sum(not row["passed"] for row in valid_negative), len(negative)
        ),
        "false_flag_rate": ratio(sum(not row["passed"] for row in positive), len(positive)),
        "unsafe_claims_accepted": sum(row["passed"] for row in negative),
        "prompt_tokens": sum(item["prompt_tokens"] or 0 for item in telemetry),
        "completion_tokens": sum(item["completion_tokens"] or 0 for item in telemetry),
        "usage_missing_calls": sum(
            not row["review"]["telemetry"]
            or row["review"]["telemetry"]["prompt_tokens"] is None
            or row["review"]["telemetry"]["completion_tokens"] is None
            for row in rows
        ),
        "latency_seconds": {
            "p50": _entailment_percentile(latencies, 0.5),
            "p95": _entailment_percentile(latencies, 0.95),
            "p99": _entailment_percentile(latencies, 0.99),
        },
    }


def _entailment_percentile(values, quantile):
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


async def _evaluate_entailment(args) -> None:
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = fixture["cases"][:args.limit] if args.limit else fixture["cases"]
    settings = OpenAICompatibleSettings(
        args.base_url, "ollama", args.model, 60, 3072,
        response_format="json_object", allow_fenced_json=True,
    )
    verifier = SemanticVerifier(OpenAICompatibleClaimJudge(settings))
    rows = []
    report = {
        "scope": fixture["scope"], "fixture_version": fixture["version"],
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "judge_prompt_version": JUDGE_PROMPT_VERSION, "model": settings.model,
        "created_at": datetime.now(UTC).isoformat(), "selected_cases": len(cases),
        "mode": "shadow", "complete": False, "results": rows,
    }
    for case in cases:
        request, answer = case_request(case)
        review = await verifier.review_async(request, answer)
        rows.append({
            "case_id": case["case_id"], "category": case["category"],
            "expected_verdict": case["expected_verdict"],
            "actual_verdict": review.checks[0].verdict if len(review.checks) == 1 else None,
            "structural_status": GroundingVerifier().verify(request, answer).status.value,
            "passed": review.passed, "review": asdict(review),
        })
        report["metrics"] = summarize_entailment(rows)
        report["complete"] = len(rows) == len(cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
        if review.error:
            raise RuntimeError("Judge transport/contract failure; report saved")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def entailment_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic claim entailment.")
    parser.add_argument("--fixture", type=Path, default=Path("tests/fixtures/claim_entailment_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/CLAIM_ENTAILMENT.json"))
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    models = args.models or ["gemma4:31b-cloud"]
    base_output = args.output
    for model in models:
        args.model = model
        if len(models) > 1:
            safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
            args.output = base_output.with_name(f"{base_output.stem}.{safe_model}{base_output.suffix}")
        asyncio.run(_evaluate_entailment(args))


def historical_main() -> None:
    parser = argparse.ArgumentParser(description="Enforce the historical pilot gate.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    observations = [
        HistoricalGateObservation(**{
            **row, "cited_version_ids": tuple(row.get("cited_version_ids", ()))
        })
        for row in payload.get("observations", payload)
    ]
    report = evaluate_historical_gate(observations)
    print(json.dumps({**asdict(report), "passed": report.passed}, indent=2))
    if not report.passed:
        raise SystemExit(1)


async def _semantic_repair() -> None:
    fixture = json.loads(Path("tests/fixtures/claim_entailment_v1.json").read_text())
    case = next(item for item in fixture["cases"] if item["case_id"] == "negation_bad")
    request, bad = case_request(case)
    request = replace(request, question="Doanh nghiệp có phải báo cáo định kỳ không?")
    settings = OpenAICompatibleSettings(
        "http://127.0.0.1:11434/v1", "ollama", "gemma4:31b-cloud", 60, 4096,
        response_format="json_object", allow_fenced_json=True,
    )
    real = OpenAICompatibleLegalAnswerGenerator(settings)

    class InjectOnce:
        def __init__(self):
            self.calls = 0

        async def generate_async(self, request):
            self.calls += 1
            return bad if self.calls == 1 else await real.generate_async(request)

    service = GroundedRAGService(
        InjectOnce(),
        semantic_verifier=SemanticVerifier(OpenAICompatibleClaimJudge(settings)),
        semantic_mode="enforce",
    )
    result = await service.answer_async(request)
    output = Path("reports/SEMANTIC_REPAIR.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "scope": "Synthetic fault injection; first bad answer is injected.",
        "model": settings.model, "injected_claim": bad.claims[0].text,
        "status": result.status.value, "semantic_status": result.semantic_status,
        "result": asdict(result),
    }, ensure_ascii=False, indent=2) + "\n")
    if not (
        len(result.semantic_observations) == 2
        and not result.semantic_observations[0].review.passed
        and result.semantic_observations[1].review.passed
        and result.status.value == "VERIFIED" and result.attempts == 2
    ):
        raise SystemExit(1)


def semantic_repair_main() -> None:
    asyncio.run(_semantic_repair())


def main() -> None:
    dispatch(
        "Evaluate LawChat RAG quality.",
        {
            "generation": generation_main,
            "entailment": entailment_main,
            "historical": historical_main,
            "semantic-repair": semantic_repair_main,
        },
    )


if __name__ == "__main__":
    main()
