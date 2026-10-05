from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from chunking.tokenizers import HuggingFaceTokenizerCounter
from database import DatabaseSettings, create_db_engine
from indexing import QdrantSettings, SentenceTransformerEmbedder
from rag import (
    GenerationRequest,
    GroundedRAGService,
    OpenAICompatibleLegalAnswerGenerator,
    OpenAICompatibleIssueDecomposer,
    OpenAICompatibleSettings,
    IssueDecompositionRequest,
    retrieve_issue_plan,
    RAGContextBuilder,
    TokenBudget,
)
from rag.runtime import create_generation_service
from retrieval import LegalQueryParser, RetrievalRequest
from retrieval.runtime import create_hybrid_retrieval_service
from cli.query import dense_main, hybrid_main
from scripts import dispatch


def answer_main() -> None:
    args = _parse_args()
    engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()
    try:
        retrieval_service = create_hybrid_retrieval_service(
            engine,
            client,
            qdrant_settings,
            SentenceTransformerEmbedder.from_env(batch_size=1),
        )

        llm_settings = OpenAICompatibleSettings.from_env()
        parsed_original = LegalQueryParser().parse(args.query, as_of=args.as_of)
        decomposition_started = time.perf_counter()
        issue_plan = OpenAICompatibleIssueDecomposer(llm_settings).generate(
            IssueDecompositionRequest(args.query)
        )
        decomposition_seconds = time.perf_counter() - decomposition_started

        retrieval_started = time.perf_counter()
        retrieval = retrieve_issue_plan(
            retrieval_service,
            RetrievalRequest(
                query=args.query,
                as_of=parsed_original.as_of,
                limit=args.limit,
                candidate_limit=args.candidate_limit,
                max_candidate_limit=args.max_candidate_limit,
            ),
            issue_plan,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started

        tokenizer = HuggingFaceTokenizerCounter(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            revision=os.getenv("EMBEDDING_MODEL_REVISION") or None,
            cache_dir=os.getenv(
                "EMBEDDING_CACHE_PATH",
                "data/.cache/huggingface",
            ),
            local_files_only=_env_flag("EMBEDDING_LOCAL_FILES_ONLY"),
        )
        context = RAGContextBuilder(
            tokenizer,
            budget=TokenBudget(
                model_context_window=args.context_window,
                system_prompt_tokens=args.system_prompt_tokens,
                answer_reserve_tokens=args.answer_reserve_tokens,
            ),
        ).build(retrieval)

        request = GenerationRequest.from_retrieval(retrieval, context)
        generation_service = create_generation_service(llm_settings)
        generation_started = time.perf_counter()
        result = generation_service.answer(request)
        generation_seconds = time.perf_counter() - generation_started

        report: dict[str, Any] = {
            "query": retrieval.query,
            "as_of": retrieval.as_of.isoformat(),
            "status": result.status.value,
            "attempts": result.attempts,
            "legal_issues": [asdict(item) for item in issue_plan.issues],
            "issue_resolutions": [item.to_dict() for item in result.answer.issue_resolutions],
            "issue_decomposition_telemetry": (
                asdict(issue_plan.telemetry) if issue_plan.telemetry else None
            ),
            "semantic_mode": result.semantic_mode,
            "semantic_observations": [asdict(item) for item in result.semantic_observations],
            "generation_telemetry": [asdict(item) for item in result.telemetry],
            "timing_seconds": {
                "issue_decomposition": round(decomposition_seconds, 3),
                "retrieval": round(retrieval_seconds, 3),
                "generation_and_verification": round(generation_seconds, 3),
                "total": round(retrieval_seconds + generation_seconds, 3),
            },
            "retrieval": {
                "sources": list(retrieval.retrieval_sources),
                "searched_candidates": retrieval.searched_candidates,
                "rejected_candidates": retrieval.rejected_candidates,
                "warnings": list(retrieval.warnings),
            },
            "context": {
                "evidence_count": len(context.evidence),
                "used_tokens": context.used_tokens,
                "token_budget": context.token_budget,
                "dropped_chunk_ids": list(context.dropped_chunk_ids),
            },
            "answer": result.answer.to_dict(),
            "verification_issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "claim_index": issue.claim_index,
                }
                for issue in result.verification.issues
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "chunk_id": item.chunk_id,
                    "role": item.role,
                    "citation": asdict(item.citation),
                }
                for item in context.evidence
            ],
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
        else:
            print(rendered)
        if result.status.value != "VERIFIED":
            raise SystemExit(2)
    finally:
        client.close()
        engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one legal question through full-corpus hybrid retrieval, "
            "LLM generation, and deterministic grounding verification."
        )
    )
    parser.add_argument("query")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-candidate-limit", type=int, default=500)
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.getenv("LAWCHAT_LLM_CONTEXT_WINDOW", "8192")),
    )
    parser.add_argument("--system-prompt-tokens", type=int, default=1500)
    parser.add_argument("--answer-reserve-tokens", type=int, default=1500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").casefold() in {"1", "true", "yes", "on"}


def main() -> None:
    dispatch(
        "Run manual dense, hybrid/legal, or full-answer queries.",
        {
            "dense": dense_main,
            "hybrid": hybrid_main,
            "legal": hybrid_main,
            "answer": answer_main,
        },
    )


if __name__ == "__main__":
    main()
