from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lawchat.chunking import (
    ChunkingConfig,
    HuggingFaceTokenizerCounter,
    LegalChunker,
)
from lawchat.chunking.tokenizers import BGE_M3_TOKENIZER_REVISION
from lawchat.chunking.utils import configure_token_counter


DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"


OUTPUT_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("parent_chunk_id", pa.string()),
        ("parent_type", pa.string()),
        ("chunk_type", pa.string()),
        ("strategy", pa.string()),
        ("is_indexable", pa.bool_()),
        ("structure_type", pa.string()),
        ("part", pa.string()),
        ("chapter", pa.string()),
        ("section", pa.string()),
        ("appendix", pa.string()),
        ("article", pa.string()),
        ("clause", pa.string()),
        ("point", pa.string()),
        ("ordinal", pa.int32()),
        ("text", pa.large_string()),
        ("retrieval_text", pa.large_string()),
        ("approx_token_count", pa.int32()),
        ("overlap_text", pa.large_string()),
        ("overlap_from_chunk_id", pa.string()),
        ("overlap_token_count", pa.int32()),
        ("document_title", pa.string()),
        ("document_number", pa.string()),
        ("document_type", pa.string()),
        ("authority", pa.string()),
        ("issued_date", pa.string()),
        ("effective_date", pa.string()),
        ("expiry_date", pa.string()),
        ("legal_status", pa.string()),
        ("legal_field", pa.string()),
        ("parse_quality", pa.string()),
        ("quality_flags", pa.large_string()),
        ("chunker_version", pa.string()),
    ]
)


PREFERRED_METADATA_COLUMNS = [
    "id",
    "title",
    "so_ky_hieu",
    "loai_van_ban",
    "co_quan_ban_hanh",
    "ngay_ban_hanh",
    "ngay_co_hieu_luc",
    "ngay_het_hieu_luc",
    "tinh_trang_hieu_luc",
    "linh_vuc",
]


def load_metadata_index(
    metadata_path: Path,
) -> dict[str, dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    parquet = pq.ParquetFile(
        metadata_path
    )

    available = set(
        parquet.schema_arrow.names
    )

    if "id" not in available:
        raise ValueError(
            "metadata.parquet must contain 'id'"
        )

    columns = [
        column
        for column in PREFERRED_METADATA_COLUMNS
        if column in available
    ]

    table = pq.read_table(
        metadata_path,
        columns=columns,
    )

    index: dict[str, dict[str, Any]] = {}

    for row in table.to_pylist():
        doc_id = row.get("id")

        if doc_id is None:
            continue

        index[str(doc_id)] = row

    return index


def validate_structured_schema(
    parquet: pq.ParquetFile,
) -> None:
    required = {
        "id",
        "parser_status",
        "structure_json",
    }

    available = set(
        parquet.schema_arrow.names
    )

    missing = required - available

    if missing:
        raise ValueError(
            "Missing structured columns: "
            + ", ".join(sorted(missing))
        )


def rows_to_table(
    rows: list[dict[str, Any]],
) -> pa.Table:
    arrays = []

    for field in OUTPUT_SCHEMA:
        arrays.append(
            pa.array(
                [
                    row.get(field.name)
                    for row in rows
                ],
                type=field.type,
            )
        )

    return pa.Table.from_arrays(
        arrays,
        schema=OUTPUT_SCHEMA,
    )


def build_chunks_dataset(
    *,
    structured_path: Path,
    metadata_path: Path,
    output_path: Path,
    batch_size: int,
    limit: int | None,
    config: ChunkingConfig,
) -> None:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0"
        )

    if not structured_path.exists():
        raise FileNotFoundError(
            structured_path
        )

    metadata_index = load_metadata_index(
        metadata_path
    )

    parquet = pq.ParquetFile(
        structured_path
    )
    validate_structured_schema(
        parquet
    )

    chunker = LegalChunker(
        config
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    writer = pq.ParquetWriter(
        temp_path,
        OUTPUT_SCHEMA,
        compression="zstd",
    )

    processed_docs = 0
    chunk_rows = 0

    parser_statuses: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    chunk_types: Counter[str] = Counter()
    parse_qualities: Counter[str] = Counter()
    seen_chunk_ids: set[str] = set()
    missing_metadata_docs = 0

    try:
        structured_columns = [
            "id",
            "parser_status",
            "structure_json",
        ]
        available_columns = set(parquet.schema_arrow.names)
        for optional in ("parse_quality", "quality_flags"):
            if optional in available_columns:
                structured_columns.append(optional)

        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=structured_columns,
        ):
            input_rows = batch.to_pylist()

            if limit is not None:
                remaining = (
                    limit - processed_docs
                )

                if remaining <= 0:
                    break

                input_rows = input_rows[
                    :remaining
                ]

            output_rows: list[
                dict[str, Any]
            ] = []

            for row in input_rows:
                processed_docs += 1

                doc_id = str(
                    row["id"]
                )

                parser_status = str(
                    row.get("parser_status")
                    or ""
                )

                parser_statuses[
                    parser_status
                ] += 1

                if parser_status != "ok":
                    continue

                raw_json = row.get(
                    "structure_json"
                )

                if not raw_json:
                    continue

                parsed_document = json.loads(
                    raw_json
                )

                metadata = metadata_index.get(doc_id)
                if metadata is None:
                    missing_metadata_docs += 1
                    metadata = {"id": doc_id}

                chunks = chunker.chunk_document(
                    parsed_document,
                    metadata,
                )

                parse_quality = str(row.get("parse_quality") or "unknown")
                quality_flags = row.get("quality_flags") or "[]"
                parse_qualities[parse_quality] += 1

                for chunk in chunks:
                    chunk["parse_quality"] = parse_quality
                    chunk["quality_flags"] = quality_flags
                    chunk_id = str(chunk["chunk_id"])
                    if chunk_id in seen_chunk_ids:
                        raise ValueError(f"Global duplicate chunk_id: {chunk_id}")
                    seen_chunk_ids.add(chunk_id)

                output_rows.extend(
                    chunks
                )

                strategies.update(
                    chunk["strategy"]
                    for chunk in chunks
                )

                chunk_types.update(
                    chunk["chunk_type"]
                    for chunk in chunks
                )

            if output_rows:
                table = rows_to_table(
                    output_rows
                )

                writer.write_table(
                    table
                )

                chunk_rows += table.num_rows

            print(
                (
                    f"\rDocuments: {processed_docs:,}"
                    f" | Chunk rows: {chunk_rows:,}"
                ),
                end="",
                flush=True,
            )

            if (
                limit is not None
                and processed_docs >= limit
            ):
                break

    except Exception:
        writer.close()

        if temp_path.exists():
            temp_path.unlink()

        raise

    else:
        writer.close()
        temp_path.replace(
            output_path
        )

    print()
    print(
        f"Saved to: {output_path}"
    )

    print("\nParser statuses:")
    for key, value in sorted(
        parser_statuses.items()
    ):
        print(
            f"  {key:<28} {value:>10,}"
        )

    print("\nStrategies:")
    for key, value in sorted(
        strategies.items()
    ):
        print(
            f"  {key:<28} {value:>10,}"
        )

    print("\nChunk types:")
    for key, value in sorted(
        chunk_types.items()
    ):
        print(
            f"  {key:<28} {value:>10,}"
        )

    print("\nParse quality:")
    for key, value in sorted(parse_qualities.items()):
        print(f"  {key:<28} {value:>10,}")

    print(f"\nDocuments missing metadata: {missing_metadata_docs:,}")


def main() -> None:
    cli = argparse.ArgumentParser(
        description=(
            "Build structure-aware chunks.parquet "
            "for Vietnamese Legal RAG."
        )
    )

    cli.add_argument(
        "--structured",
        type=Path,
        default=Path(
            "data/processed/"
            "content_structured.parquet"
        ),
    )

    cli.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/raw/huggingface/data/"
            "metadata.parquet"
        ),
    )

    cli.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/chunks/chunks.parquet"
        ),
    )

    cli.add_argument(
        "--batch-size",
        type=int,
        default=250,
    )

    cli.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    cli.add_argument(
        "--target-min-tokens",
        type=int,
        default=600,
    )

    cli.add_argument(
        "--target-tokens",
        type=int,
        default=700,
    )

    cli.add_argument(
        "--target-max-tokens",
        type=int,
        default=800,
    )

    cli.add_argument(
        "--max-child-tokens",
        type=int,
        default=1200,
    )

    cli.add_argument(
        "--freeform-target-tokens",
        type=int,
        default=700,
    )

    cli.add_argument(
        "--freeform-max-tokens",
        type=int,
        default=1200,
    )

    cli.add_argument(
        "--max-indexable-tokens",
        type=int,
        default=1200,
        help="Hard limit for final retrieval_text including metadata.",
    )

    cli.add_argument("--overlap-ratio", type=float, default=0.125)
    cli.add_argument("--max-overlap-ratio", type=float, default=0.20)
    cli.add_argument("--min-overlap-tokens", type=int, default=60)
    cli.add_argument("--overlap-min-source-tokens", type=int, default=600)
    cli.add_argument(
        "--disable-semantic-overlap",
        action="store_true",
        help="Disable same-parent semantic overlap.",
    )

    cli.add_argument(
        "--min-tail-tokens",
        type=int,
        default=100,
    )

    cli.add_argument(
        "--include-temporal-metadata-in-retrieval-text",
        action="store_true",
    )
    cli.add_argument(
        "--tokenizer-model",
        default=DEFAULT_TOKENIZER_MODEL,
        help="Hugging Face tokenizer used for exact token budgets.",
    )
    cli.add_argument(
        "--tokenizer-cache-dir",
        default="data/.cache/huggingface",
    )
    cli.add_argument(
        "--tokenizer-revision",
        default=BGE_M3_TOKENIZER_REVISION,
        help="Pinned tokenizer revision for reproducible chunk boundaries.",
    )
    cli.add_argument(
        "--approximate-token-counting",
        action="store_true",
        help="Use regex approximation instead of the model tokenizer.",
    )

    args = cli.parse_args()

    if not args.approximate_token_counting:
        token_counter = HuggingFaceTokenizerCounter(
            model_name=args.tokenizer_model,
            revision=args.tokenizer_revision,
            cache_dir=args.tokenizer_cache_dir,
            local_files_only=(
                os.getenv("HF_HUB_OFFLINE", "0") == "1"
            ),
        )
        # Force tokenizer initialization before starting the atomic build.
        token_counter.count_tokens("kiểm tra tokenizer")
        configure_token_counter(token_counter)
        print(
            "Exact token counter: "
            f"{args.tokenizer_model} (chunk limit "
            f"{args.max_indexable_tokens})"
        )

    config = ChunkingConfig(
        target_min_tokens=args.target_min_tokens,
        target_tokens=args.target_tokens,
        target_max_tokens=args.target_max_tokens,
        max_child_tokens=args.max_child_tokens,
        freeform_target_tokens=args.freeform_target_tokens,
        freeform_max_tokens=args.freeform_max_tokens,
        max_indexable_tokens=args.max_indexable_tokens,
        enable_semantic_overlap=not args.disable_semantic_overlap,
        overlap_ratio=args.overlap_ratio,
        max_overlap_ratio=args.max_overlap_ratio,
        min_overlap_tokens=args.min_overlap_tokens,
        overlap_min_source_tokens=args.overlap_min_source_tokens,
        min_tail_tokens=args.min_tail_tokens,
        include_temporal_metadata_in_retrieval_text=(
            args.include_temporal_metadata_in_retrieval_text
        ),
    )

    build_chunks_dataset(
        structured_path=args.structured,
        metadata_path=args.metadata,
        output_path=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
        config=config,
    )


if __name__ == "__main__":
    main()
