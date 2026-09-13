from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

from lawchat.database import DatabaseSettings, create_db_engine
from lawchat.indexing import (
    BM25Settings,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    PostgresChunkSource,
    PostgresLexicalSource,
    QdrantDenseIndex,
    QdrantSettings,
    SentenceTransformerEmbedder,
    TantivyBM25Index,
)
from scripts import dispatch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def dense_main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed current legal chunks and upsert them into Qdrant."
    )
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--model-revision",
        default=os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_REVISION),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_PATH", "data/.cache/huggingface"),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE", "auto"),
        help="PyTorch device: auto, cpu, cuda, cuda:N, mps or xpu:N.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("EMBEDDING_MAX_LENGTH", "1200")),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(
            os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("EMBEDDING_LOCAL_FILES_ONLY")
        or _env_flag("HF_HUB_OFFLINE"),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
    )
    parser.add_argument("--fetch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--version-scope",
        choices=("current", "historical"),
        default="current",
    )
    parser.add_argument("--promote-alias", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--vectors-in-memory", action="store_true")
    args = parser.parse_args()

    db_engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()
    embedder = SentenceTransformerEmbedder(
        model_name=args.model,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        batch_size=args.embedding_batch_size,
        device=args.device,
        max_length=args.max_length,
        expected_dimension=args.dimension,
        local_files_only=args.local_files_only,
    )
    index = QdrantDenseIndex(
        client,
        embedder,
        PostgresChunkSource(
            db_engine,
            fetch_size=args.fetch_size,
            version_scope=args.version_scope,
        ),
        collection=qdrant_settings.collection,
        alias=qdrant_settings.alias,
        on_disk=not args.vectors_in_memory,
        scalar_quantization=not args.no_quantization,
    )

    def show_progress(report) -> None:
        print(
            json.dumps(
                {
                    "collection": report.collection,
                    "indexed_points": report.indexed_points,
                    "batches": report.batches,
                    "last_chunk_id": report.last_chunk_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        report = index.build(
            limit=args.limit,
            force=args.force,
            promote_alias=args.promote_alias,
            progress=show_progress,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    finally:
        client.close()
        db_engine.dispose()


def sparse_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a resumable Tantivy BM25 index from PostgreSQL."
    )
    parser.add_argument("--fetch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--index-name")
    parser.add_argument("--promote-alias", action="store_true")
    parser.add_argument(
        "--version-scope",
        choices=("current", "historical"),
        default="current",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    settings = BM25Settings.from_env()
    if args.index_name:
        settings = replace(settings, index_name=args.index_name)
    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        index = TantivyBM25Index(
            PostgresLexicalSource(
                engine,
                fetch_size=args.fetch_size,
                version_scope=args.version_scope,
            ),
            settings,
        )

        def show_progress(report) -> None:
            print(
                json.dumps(
                    {
                        "index_name": report.index_name,
                        "indexed_points": report.indexed_points,
                        "expected_points": report.expected_points,
                        "batches": report.batches,
                        "last_database_id": report.last_database_id,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        report = index.build(
            limit=args.limit,
            promote_alias=args.promote_alias,
            progress=show_progress,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


def main() -> None:
    dispatch(
        "Build LawChat search indexes.",
        {"dense": dense_main, "sparse": sparse_main},
    )


if __name__ == "__main__":
    main()
