from __future__ import annotations

import argparse
import json
from pathlib import Path

from lawchat.database import DatabaseSettings, create_db_engine
from lawchat.ingestion import PostgresMetadataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load legal Parquet metadata into PostgreSQL."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/huggingface/data/metadata.parquet"),
    )
    parser.add_argument(
        "--structured",
        type=Path,
        default=Path("data/processed/content_structured.parquet"),
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/chunks/chunks.parquet"),
    )
    parser.add_argument(
        "--relationships",
        type=Path,
        default=Path("data/raw/huggingface/data/relationships.parquet"),
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--dataset-revision", default="local-snapshot")
    parser.add_argument("--dataset-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--version-mode",
        choices=("current", "historical"),
        default="current",
        help=(
            "Historical mode requires source_revision/content_valid_from on "
            "versions and version_source_revision on chunks."
        ),
    )
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--skip-relationships", action="store_true")
    args = parser.parse_args()

    settings = DatabaseSettings.from_env()
    engine = create_db_engine(settings)
    kwargs = {
        "batch_size": args.batch_size,
        "dataset_revision": args.dataset_revision,
        "dry_run": args.dry_run,
        "version_mode": args.version_mode,
    }
    if args.dataset_url:
        kwargs["dataset_url"] = args.dataset_url
    loader = PostgresMetadataLoader(engine, **kwargs)
    try:
        report = loader.load(
            metadata_path=args.metadata,
            structured_path=None if args.skip_structured else args.structured,
            chunks_path=None if args.skip_chunks else args.chunks,
            relationships_path=(
                None if args.skip_relationships else args.relationships
            ),
            limit_documents=args.limit_documents,
        )
    finally:
        engine.dispose()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
