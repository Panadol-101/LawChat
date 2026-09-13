from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date

from lawchat.database import DatabaseSettings, LegalStatus, create_db_engine
from lawchat.indexing import (
    DEFAULT_EMBEDDING_MODEL,
    QdrantSettings,
    SentenceTransformerEmbedder,
)
from lawchat.retrieval import (
    DenseSearchFilter,
    DenseSearcher,
    RetrievalRequest,
)
from lawchat.retrieval.runtime import create_hybrid_retrieval_service


def hybrid_main() -> None:
    args = _parse_args()
    engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()
    try:
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
        )
        options = {
            "query": args.query,
            "as_of": args.as_of,
            "limit": args.limit,
            "candidate_limit": args.candidate_limit,
            "max_candidate_limit": args.max_candidate_limit,
            "document_numbers": tuple(args.document_number),
            "document_types": tuple(args.document_type),
            "authorities": tuple(args.authority),
            "legal_fields": tuple(args.legal_field),
        }
        if args.status:
            options["statuses"] = tuple(args.status)
        response = service.retrieve(RetrievalRequest(**options))
        print(json.dumps(asdict(response), ensure_ascii=False, indent=2, default=str))
    finally:
        client.close()
        engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Dense + BM25 legal search with temporal validation."
    )
    parser.add_argument("query")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-candidate-limit", type=int, default=500)
    parser.add_argument("--document-number", action="append", default=[])
    parser.add_argument("--document-type", action="append", default=[])
    parser.add_argument("--authority", action="append", default=[])
    parser.add_argument("--legal-field", action="append", default=[])
    parser.add_argument(
        "--status",
        action="append",
        choices=[item.value for item in LegalStatus],
        default=[],
    )
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_PATH", "data/.cache/huggingface"),
    )
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    return parser.parse_args()


def dense_main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search in Qdrant.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--document-number", action="append", default=[])
    parser.add_argument("--status", action="append", default=[])
    parser.add_argument("--legal-field", action="append", default=[])
    parser.add_argument("--collection")
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_PATH", "data/.cache/huggingface"),
    )
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "auto"))
    args = parser.parse_args()

    settings = QdrantSettings.from_env()
    client = settings.create_client()
    searcher = DenseSearcher(
        client,
        SentenceTransformerEmbedder.from_env(
            model_name=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=1,
        ),
        collection=args.collection or settings.alias,
    )
    try:
        results = searcher.search(
            args.query,
            limit=args.limit,
            filters=DenseSearchFilter(
                document_numbers=tuple(args.document_number),
                statuses=tuple(args.status),
                legal_fields=tuple(args.legal_field),
            ),
        )
        print(
            json.dumps(
                [
                    {"rank": rank, "score": result.score, **result.payload}
                    for rank, result in enumerate(results, start=1)
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        client.close()
