from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from parsing.legal_structure_parser import LegalStructureParser


OUTPUT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_clean_status", pa.string()),
        ("source_cleaner_version", pa.string()),
        ("parser_status", pa.string()),
        ("parser_error", pa.string()),
        ("parse_quality", pa.string()),
        ("quality_flags", pa.large_string()),
        ("structure_type", pa.string()),
        ("parser_version", pa.string()),
        ("structure_json", pa.large_string()),
        ("part_count", pa.int32()),
        ("chapter_count", pa.int32()),
        ("section_count", pa.int32()),
        ("article_count", pa.int32()),
        ("clause_count", pa.int32()),
        ("point_count", pa.int32()),
        ("appendix_count", pa.int32()),
        ("table_count", pa.int32()),
        ("warning_count", pa.int32()),
        ("source_text_length", pa.int64()),
    ]
)


def _minimal_empty_structure(
    parser: LegalStructureParser,
    doc_id: str,
) -> dict:
    return parser.parse_document(
        "",
        doc_id=doc_id,
    )


def process_batch(
    batch: pa.RecordBatch,
    parser: LegalStructureParser,
) -> pa.Table:
    ids = batch.column(
        batch.schema.get_field_index("id")
    ).to_pylist()

    contents = batch.column(
        batch.schema.get_field_index("content_clean")
    ).to_pylist()

    statuses = batch.column(
        batch.schema.get_field_index("clean_status")
    ).to_pylist()
    cleaner_version_index = batch.schema.get_field_index("cleaner_version")
    cleaner_versions = (
        batch.column(cleaner_version_index).to_pylist()
        if cleaner_version_index >= 0
        else ["unknown"] * batch.num_rows
    )

    output: dict[str, list] = {
        name: []
        for name in OUTPUT_SCHEMA.names
    }

    for doc_id, content, clean_status, cleaner_version in zip(
        ids,
        contents,
        statuses,
        cleaner_versions,
    ):
        doc_id_str = "" if doc_id is None else str(doc_id)
        content = content or ""
        clean_status = clean_status or ""

        parser_status = "ok"
        parser_error: str | None = None

        try:
            if clean_status == "ok" and content.strip():
                parsed = parser.parse_document(
                    content,
                    doc_id=doc_id_str,
                )
            else:
                parsed = _minimal_empty_structure(
                    parser,
                    doc_id_str,
                )
                parser_status = "skipped_no_text"

        except Exception as exc:
            parsed = _minimal_empty_structure(
                parser,
                doc_id_str,
            )
            parser_status = (
                "error:"
                + type(exc).__name__
            )
            parser_error = f"{type(exc).__name__}: {exc}"[:2000]

        stats = parsed["stats"]
        quality_flags = assess_parse_quality(parsed, len(content))
        parse_quality = (
            "quarantine"
            if any(flag.startswith("severe:") for flag in quality_flags)
            else "review"
            if quality_flags
            else "ok"
        )

        output["id"].append(doc_id_str)
        output["source_clean_status"].append(clean_status)
        output["source_cleaner_version"].append(
            cleaner_version or "unknown"
        )
        output["parser_status"].append(parser_status)
        output["parser_error"].append(parser_error)
        output["parse_quality"].append(parse_quality)
        output["quality_flags"].append(
            json.dumps(quality_flags, ensure_ascii=False)
        )
        output["structure_type"].append(
            parsed["structure_type"]
        )
        output["parser_version"].append(parsed["parser_version"])
        output["structure_json"].append(
            json.dumps(
                parsed,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        output["part_count"].append(stats["parts"])
        output["chapter_count"].append(stats["chapters"])
        output["section_count"].append(stats["sections"])
        output["article_count"].append(stats["articles"])
        output["clause_count"].append(stats["clauses"])
        output["point_count"].append(stats["points"])
        output["appendix_count"].append(stats["appendices"])
        output["table_count"].append(stats.get("tables", 0))
        output["warning_count"].append(
            len(parsed["warnings"])
        )
        output["source_text_length"].append(
            len(content)
        )

    arrays = [
        pa.array(
            output[field.name],
            type=field.type,
        )
        for field in OUTPUT_SCHEMA
    ]

    return pa.Table.from_arrays(
        arrays,
        schema=OUTPUT_SCHEMA,
    )


def assess_parse_quality(
    parsed: dict,
    source_text_length: int,
) -> list[str]:
    """Return auditable flags without silently discarding the document."""
    flags: list[str] = []
    preamble_length = len(parsed.get("preamble") or "")

    if preamble_length > 50_000:
        flags.append("severe:oversized_preamble")

    largest_node = 0

    def visit(nodes: list[dict]) -> None:
        nonlocal largest_node
        for node in nodes:
            node_text_length = len(node.get("text") or "")
            largest_node = max(largest_node, node_text_length)
            if node.get("type") == "article" and node_text_length > 50_000:
                flags.append("severe:oversized_article_text")
            visit(node.get("children") or [])

    visit(parsed.get("body") or [])

    if (
        source_text_length > 5_000
        and largest_node > source_text_length * 0.80
    ):
        flags.append("severe:single_node_dominates_document")

    warnings = parsed.get("warnings") or []
    if len(warnings) >= 20:
        flags.append("severe:many_parser_warnings")
    elif warnings:
        flags.append("parser_warnings")

    return list(dict.fromkeys(flags))


def validate_input_schema(
    parquet_file: pq.ParquetFile,
) -> None:
    required = {
        "id",
        "content_clean",
        "clean_status",
    }

    available = set(
        parquet_file.schema_arrow.names
    )

    missing = required - available

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )


def build_structured_dataset(
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int = 500,
    limit: int | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0"
        )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    parquet_file = pq.ParquetFile(input_path)
    validate_input_schema(parquet_file)

    parser = LegalStructureParser()

    writer = pq.ParquetWriter(
        temp_path,
        OUTPUT_SCHEMA,
        compression="zstd",
    )

    processed = 0
    status_counter: Counter[str] = Counter()
    structure_counter: Counter[str] = Counter()

    try:
        input_columns = [
            "id",
            "content_clean",
            "clean_status",
        ]
        if "cleaner_version" in parquet_file.schema_arrow.names:
            input_columns.append("cleaner_version")

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=input_columns,
        ):
            if limit is not None:
                remaining = limit - processed

                if remaining <= 0:
                    break

                if batch.num_rows > remaining:
                    batch = batch.slice(
                        0,
                        remaining,
                    )

            table = process_batch(
                batch,
                parser,
            )

            writer.write_table(table)

            status_counter.update(
                table["parser_status"].to_pylist()
            )
            structure_counter.update(
                table["structure_type"].to_pylist()
            )

            processed += table.num_rows

            print(
                f"\rProcessed: {processed:,}",
                end="",
                flush=True,
            )

    except Exception:
        writer.close()

        if temp_path.exists():
            temp_path.unlink()

        raise

    else:
        writer.close()
        temp_path.replace(output_path)

    print()
    print(f"Saved to: {output_path}")

    print("\nParser status:")
    for key, value in sorted(
        status_counter.items()
    ):
        print(f"  {key:<28} {value:>10,}")

    print("\nStructure types:")
    for key, value in sorted(
        structure_counter.items()
    ):
        print(f"  {key:<28} {value:>10,}")


def main() -> None:
    cli = argparse.ArgumentParser(
        description=(
            "Parse cleaned Vietnamese legal documents "
            "into hierarchical structures."
        )
    )

    cli.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/content_clean.parquet"
        ),
    )

    cli.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/content_structured.parquet"
        ),
    )

    cli.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )

    cli.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Parse only the first N rows for testing.",
    )

    args = cli.parse_args()

    build_structured_dataset(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
