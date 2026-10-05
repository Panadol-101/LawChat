from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from qdrant_client import models as qdrant_models

from database import DatabaseSettings, create_db_engine
from evaluation import RetrievalCase, evaluate_rankings
from indexing import (
    DEFAULT_EMBEDDING_MODEL,
    QdrantSettings,
    SentenceTransformerEmbedder,
)
from indexing.qdrant_index import POINT_NAMESPACE
from retrieval import (
    DEFAULT_RERANKER_MODEL,
    LegalQueryParser,
    RerankerSettings,
    RetrievalRequest,
    SentenceTransformerCrossEncoderReranker,
    DenseSearcher,
)
from retrieval.runtime import create_hybrid_retrieval_service
from scripts import dispatch


CATEGORY_TARGETS = {
    "exact_document_number": 10,
    "exact_provision": 10,
    "semantic": 60,
    "scenario": 35,
    "temporal_current": 20,
    "temporal_historical": 20,
    "expired_status": 15,
    "legal_relationship": 10,
    "confusable": 10,
    "insufficient_evidence": 10,
}


def _round_robin_rows(
    rows_by_document: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    max_rows = max((len(rows) for rows in rows_by_document), default=0)
    return [
        rows[chunk_index]
        for chunk_index in range(max_rows)
        for rows in rows_by_document
        if chunk_index < len(rows)
    ]


def hybrid_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate production hybrid retrieval on curated legal cases."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/dense_retrieval_eval.json"),
    )
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument(
        "--reranker-model",
        default=os.getenv(
            "RERANKER_MODEL",
            DEFAULT_RERANKER_MODEL,
        ),
    )
    parser.add_argument(
        "--reranker-candidates",
        type=int,
        default=int(os.getenv("RERANKER_CANDIDATES", "8")),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=int(os.getenv("RERANKER_BATCH_SIZE", "1")),
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=int(os.getenv("RERANKER_MAX_LENGTH", "1536")),
    )
    parser.add_argument(
        "--reranker-device",
        default=os.getenv("RERANKER_DEVICE", "cpu"),
    )
    parser.add_argument(
        "--reranker-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=(
            os.getenv("RERANKER_LOCAL_FILES_ONLY", "false").casefold()
            in {"1", "true", "yes", "on"}
        ),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Evaluate only this benchmark category; may be repeated.",
    )
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--raw-semantic-query",
        action="store_true",
        help="Disable semantic-query cleaning for an ablation run.",
    )
    parser.add_argument("--min-hit-rate", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_PATH", "data/.cache/huggingface"),
    )
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    schema_v2 = fixture.get("schema_version") == "2.0"
    raw_cases = fixture["cases"]
    if args.categories:
        raw_cases = [item for item in raw_cases if item.get("category") in args.categories]
    if args.limit_cases is not None:
        raw_cases = raw_cases[: args.limit_cases]
    if not raw_cases:
        raise SystemExit("No benchmark cases selected")

    if schema_v2:
        retrieval_cases = [
            RetrievalCase(
                case_id=item["case_id"],
                query=item["query"],
                relevant_chunk_ids=frozenset(
                    chunk_id
                    for chunk_id, grade in item["expected"].get("relevance", {}).items()
                    if grade >= 2
                ) or frozenset(item["expected"].get("chunk_ids", ())),
                relevance_grades=item["expected"].get("relevance", {}),
            )
            for item in raw_cases
            if item.get("answerable", True) and item["expected"].get("chunk_ids")
        ]
        runnable_cases = [item for item in raw_cases if item.get("answerable", True)]
    else:
        retrieval_cases = [
            RetrievalCase(
                case_id=item["case_id"],
                query=item["query"],
                relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
            )
            for item in raw_cases
        ]
        runnable_cases = raw_cases
    engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()
    try:
        reranker_settings = RerankerSettings(
            enabled=args.reranker,
            model_name=args.reranker_model,
            candidate_limit=args.reranker_candidates,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            cache_dir=args.cache_dir,
            device=args.reranker_device,
            local_files_only=args.reranker_local_files_only,
        )
        service = create_hybrid_retrieval_service(
            engine,
            client,
            qdrant_settings,
            SentenceTransformerEmbedder.from_env(
                model_name=args.model,
                cache_dir=args.cache_dir,
                device=args.device,
                batch_size=1,
            ),
            query_parser=LegalQueryParser(
                clean_semantic_query=not args.raw_semantic_query,
                # Versioned fixtures evaluate their recorded as_of values.
                # The user-facing API keeps the configured cutoff enabled.
                enforce_configured_cutoff=False,
            ),
            reranker=(
                SentenceTransformerCrossEncoderReranker(reranker_settings)
                if args.reranker
                else None
            ),
            reranker_settings=reranker_settings,
        )
        checkpoint = _load_checkpoint(args.output) if args.resume and args.output else {}
        responses = dict(checkpoint.get("responses", {}))
        rankings = dict(checkpoint.get("rankings", {}))
        diagnostics = dict(checkpoint.get("diagnostics", {}))
        for index, item in enumerate(runnable_cases, start=1):
            if item["case_id"] in responses:
                continue
            request_data: dict[str, Any] = {
                "query": item["query"],
                "limit": args.k,
                "candidate_limit": args.candidate_limit,
                "max_candidate_limit": max(args.candidate_limit, 500),
            }
            if schema_v2:
                options = item.get("request", {})
                if options.get("as_of"):
                    request_data["as_of"] = date.fromisoformat(options["as_of"])
                if options.get("statuses"):
                    request_data["statuses"] = tuple(options["statuses"])
            response = service.retrieve(RetrievalRequest(**request_data))
            responses[item["case_id"]] = _response_summary(response)
            rankings[item["case_id"]] = [result.chunk_id for result in response.results]
            diagnostics[item["case_id"]] = {
                "sources": list(response.retrieval_sources),
                "warnings": list(response.warnings),
                "searched_candidates": response.searched_candidates,
                "rejected_candidates": response.rejected_candidates,
                "timings": response.timings,
            }
            print(f"[{index}/{len(runnable_cases)}] {item['case_id']}", flush=True)
            if args.output and index % args.checkpoint_every == 0:
                _write_checkpoint(args.output, responses, rankings, diagnostics)
        metrics = evaluate_rankings(retrieval_cases, rankings, k=args.k) if retrieval_cases else None
        benchmark_diagnostics = (
            _evaluate_v2_cases(raw_cases, responses, k=args.k) if schema_v2 else None
        )
        if args.summary_only and benchmark_diagnostics:
            benchmark_diagnostics = {
                key: value
                for key, value in benchmark_diagnostics.items()
                if key != "ranks"
            }
        report = {
            "metrics": asdict(metrics) if metrics else None,
            "benchmark": benchmark_diagnostics,
        }
        if not args.summary_only:
            report.update(
                {
                    "rankings": rankings,
                    "diagnostics": diagnostics,
                }
            )
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
        else:
            print(rendered, end="")
        if args.min_hit_rate is not None and (not metrics or metrics.hit_rate_at_k < args.min_hit_rate):
            raise SystemExit(1)
        if args.min_mrr is not None and (not metrics or metrics.mean_reciprocal_rank < args.min_mrr):
            raise SystemExit(1)
    finally:
        client.close()
        engine.dispose()


def _first_rank(values: list[bool]) -> int | None:
    return next((index for index, matched in enumerate(values, start=1) if matched), None)


def _evaluate_v2_cases(raw_cases, responses, *, k: int) -> dict[str, Any]:
    category_totals: Counter[str] = Counter()
    category_hits: Counter[str] = Counter()
    ranks: dict[str, dict[str, int | None]] = {}
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    status_metadata_total = 0
    status_metadata_hits = 0
    status_selection_hits = 0
    provision_total = 0
    provision_hits = 0

    for case in raw_cases:
        category = case["category"]
        if not case.get("answerable", True):
            excluded["insufficient_evidence"] += 1
            continue
        expected = case["expected"]
        response = responses[case["case_id"]]
        results = (
            response["results"][:k]
            if isinstance(response, dict)
            else list(response.results)[:k]
        )
        relevance = expected.get("relevance", {})
        expected_chunks = {
            chunk_id for chunk_id, grade in relevance.items() if grade >= 2
        } or set(expected.get("chunk_ids", ()))
        expected_documents = set(expected.get("document_ids", ()))
        expected_statuses = set(expected.get("statuses", ()))
        expected_articles = set(expected.get("articles", ()))
        expected_clauses = set(expected.get("clauses", ()))
        expected_points = set(expected.get("points", ()))

        if category == "expired_status":
            status_metadata_total += 1
            seed_resolutions = (
                response.get("seed_resolutions", ())
                if isinstance(response, dict)
                else getattr(response, "seed_resolutions", ())
            )
            if any(
                _document_field(item, "document_id") in expected_documents
                and (
                    not expected_statuses
                    or _document_field(item, "status") in expected_statuses
                )
                for item in seed_resolutions
            ):
                status_metadata_hits += 1
            status_resolution = (
                response.get("status_resolution")
                if isinstance(response, dict)
                else getattr(response, "status_resolution", None)
            )
            if (
                status_resolution is not None
                and _field(status_resolution, "document_id") in expected_documents
                and (
                    not expected_statuses
                    or _field(status_resolution, "status") in expected_statuses
                )
            ):
                status_selection_hits += 1

        chunk_matches = [_field(item, "chunk_id") in expected_chunks for item in results]
        document_matches = [_citation_field(item, "document_id") in expected_documents for item in results]
        status_matches = [not expected_statuses or _citation_field(item, "status") in expected_statuses for item in results]
        structure_matches = [
            (not expected_articles or _citation_field(item, "article") in expected_articles)
            and (not expected_clauses or _citation_field(item, "clause") in expected_clauses)
            and (not expected_points or _citation_field(item, "point") in expected_points)
            for item in results
        ]
        base_matches = chunk_matches if expected_chunks else document_matches
        strict_matches = [
            base and status and structure
            for base, status, structure in zip(base_matches, status_matches, structure_matches)
        ]
        has_provision = bool(expected_articles or expected_clauses or expected_points)
        if has_provision:
            provision_total += 1
            provision_hits += any(
                doc and structure
                for doc, structure in zip(document_matches, structure_matches)
            )
        case_ranks = {
            "strict": _first_rank(strict_matches),
            "chunk": _first_rank(chunk_matches) if expected_chunks else None,
            "document": _first_rank(document_matches) if expected_documents else None,
            "structure": _first_rank([doc and structure for doc, structure in zip(document_matches, structure_matches)]),
            "status": _first_rank([doc and status for doc, status in zip(document_matches, status_matches)]),
        }
        ranks[case["case_id"]] = case_ranks
        category_totals[category] += 1
        totals["strict"] += 1
        if case_ranks["strict"] is not None:
            category_hits[category] += 1
            hits["strict"] += 1
        for key in ("chunk", "document", "structure", "status"):
            if (key == "chunk" and not expected_chunks) or (key != "chunk" and not expected_documents):
                continue
            totals[key] += 1
            if case_ranks[key] is not None:
                hits[key] += 1

    def rate(key: str) -> float | None:
        return hits[key] / totals[key] if totals[key] else None

    return {
        "selected_cases": len(raw_cases),
        "evaluated_answerable_cases": totals["strict"],
        "excluded_cases": dict(excluded),
        "strict_hit_rate_at_k": rate("strict"),
        "chunk_hit_rate_at_k": rate("chunk"),
        "document_hit_rate_at_k": rate("document"),
        "structure_accuracy_at_k": rate("structure"),
        "correct_legal_provision_retrieval_rate": (
            provision_hits / provision_total if provision_total else None
        ),
        "status_accuracy_at_k": rate("status"),
        "status_metadata_accuracy": (
            status_metadata_hits / status_metadata_total
            if status_metadata_total
            else None
        ),
        "status_selection_accuracy": (
            status_selection_hits / status_metadata_total
            if status_metadata_total
            else None
        ),
        "category_hit_rate_at_k": {
            category: category_hits[category] / total
            for category, total in sorted(category_totals.items())
        },
        "ranks": ranks,
    }


def _response_summary(response) -> dict[str, Any]:
    return {
        "results": [
            {
                "chunk_id": item.chunk_id,
                "citation": {
                    "document_id": item.citation.document_id,
                    "status": item.citation.status,
                    "article": item.citation.article,
                    "clause": item.citation.clause,
                    "point": item.citation.point,
                },
            }
            for item in response.results
        ],
        "seed_resolutions": [
            {
                "document_id": item.document.document_id,
                "status": item.document.status,
            }
            for item in response.seed_resolutions
        ],
        "status_resolution": (
            asdict(response.status_resolution)
            if response.status_resolution is not None
            else None
        ),
    }


def _write_checkpoint(path, responses, rankings, diagnostics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "partial": True,
                "responses": responses,
                "rankings": rankings,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if value.get("partial") else {}


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name)


def _citation_field(value, name):
    citation = _field(value, "citation")
    return _field(citation, name)


def _document_field(value, name):
    document = value if isinstance(value, dict) else value.document
    return _field(document, name)


def dense_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval quality on curated legal cases."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/dense_retrieval_eval.json"),
    )
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_PATH", "data/.cache/huggingface"),
    )
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.85)
    parser.add_argument("--min-mrr", type=float, default=0.70)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    corpus = fixture["corpus"]
    cases = [
        RetrievalCase(
            case_id=item["case_id"],
            query=item["query"],
            relevant_chunk_ids=frozenset(item["relevant_chunk_ids"]),
        )
        for item in fixture["cases"]
    ]
    embedder = SentenceTransformerEmbedder.from_env(
        model_name=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    settings = QdrantSettings.from_env()
    client = settings.create_client()
    collection = "legal_dense_bge_m3_eval"
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qdrant_models.VectorParams(
                size=embedder.dimension,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        vectors = embedder.encode_documents([item["text"] for item in corpus])
        ids = [
            str(uuid.uuid5(POINT_NAMESPACE, item["chunk_id"]))
            for item in corpus
        ]
        client.upsert(
            collection_name=collection,
            wait=True,
            points=qdrant_models.Batch(
                ids=ids,
                vectors=vectors.tolist(),
                payloads=[{"chunk_id": item["chunk_id"]} for item in corpus],
            ),
        )
        searcher = DenseSearcher(client, embedder, collection=collection)
        rankings = {
            case.case_id: [
                result.payload["chunk_id"]
                for result in searcher.search(case.query, limit=args.k)
            ]
            for case in cases
        }
        metrics = evaluate_rankings(cases, rankings, k=args.k)
        print(
            json.dumps(
                {"metrics": asdict(metrics), "rankings": rankings},
                ensure_ascii=False,
                indent=2,
            )
        )
        if (
            metrics.hit_rate_at_k < args.min_hit_rate
            or metrics.mean_reciprocal_rank < args.min_mrr
        ):
            raise SystemExit(1)
    finally:
        client.close()


def main() -> None:
    dispatch(
        "Evaluate LawChat retrieval.",
        {"dense": dense_main, "hybrid": hybrid_main},
    )


if __name__ == "__main__":
    main()
