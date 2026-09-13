from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree
from lxml import html as lxml_html

from lawchat.preprocessing.html_cleaner import (
    CLEANER_VERSION,
    FILLER_PLACEHOLDER,
    TABLE_END_MARKER,
    TABLE_HEADERS_MARKER,
    TABLE_ROW_MARKER,
    TABLE_START_MARKER,
    clean_html,
)


OUTPUT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("content_clean", pa.string()),
        ("original_html_length", pa.int64()),
        ("clean_text_length", pa.int64()),
        ("clean_status", pa.string()),
        ("clean_error", pa.string()),
        ("cleaner_version", pa.string()),
    ]
)


def _extract_visible_text(content_html: str) -> str | None:
    """
    Extract visible text from raw HTML only for diagnostic classification.

    Returns:
        str:
            Visible text if lxml can parse the HTML.
        None:
            If the HTML cannot be parsed even after wrapping it.

    This is intentionally separate from clean_html(). It is used only when
    clean_html() returns an empty string so that we can distinguish genuinely
    empty HTML from a possible cleaner edge case.
    """
    try:
        root = lxml_html.fromstring(content_html)
    except (etree.ParserError, ValueError):
        try:
            root = lxml_html.fromstring(f"<div>{content_html}</div>")
        except Exception:
            return None

    for node in root.xpath(
        "//script|//style|//noscript|//iframe|//svg|//canvas|//object|//embed|//head"
    ):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    bodies = root.xpath("//body")
    content_root = bodies[0] if bodies else root
    return content_root.text_content().strip()


def _has_meaningful_text(text: str) -> bool:
    """
    Return True when text contains at least one Unicode letter or digit.

    This lets us treat punctuation-only fragments such as '--', ':', or '.'
    as non-meaningful instead of flagging them as a cleaner failure.
    """
    without_placeholders = text.replace(FILLER_PLACEHOLDER, "")
    for marker in (
        TABLE_START_MARKER,
        TABLE_HEADERS_MARKER,
        TABLE_ROW_MARKER,
        TABLE_END_MARKER,
    ):
        without_placeholders = without_placeholders.replace(marker, "")
    return any(char.isalnum() for char in without_placeholders)


def classify_clean_status(
    content_html: str | None,
    content_clean: str,
) -> str:
    """
    Classify the result of cleaning in a way that is useful for later RAG
    ingestion and data-quality auditing.

    Status values:
        ok
            Clean text was produced successfully.

        source_null
            content_html is NULL in the source dataset.

        source_blank
            content_html exists but is an empty/whitespace-only string.

        no_visible_text
            HTML exists and parses, but contains no visible text. Example:
            <html><body></body></html>.

        no_meaningful_text
            HTML has visible text, but it contains only punctuation/symbols.

        cleaner_empty
            HTML contains meaningful visible text, but clean_html() returned
            an empty string. These rows should be investigated because they
            may reveal an unsupported HTML structure or cleaner bug.

        parse_error
            clean_html() returned empty and the raw HTML could not be parsed
            for diagnostic classification.
    """
    if content_clean:
        return (
            "ok"
            if _has_meaningful_text(content_clean)
            else "no_meaningful_text"
        )

    if content_html is None:
        return "source_null"

    if not content_html.strip():
        return "source_blank"

    visible_text = _extract_visible_text(content_html)

    if visible_text is None:
        return "parse_error"

    if not visible_text:
        return "no_visible_text"

    if not _has_meaningful_text(visible_text):
        return "no_meaningful_text"

    return "cleaner_empty"


def process_batch(batch: pa.RecordBatch) -> pa.Table:
    ids = batch.column(
        batch.schema.get_field_index("id")
    ).to_pylist()

    contents = batch.column(
        batch.schema.get_field_index("content_html")
    ).to_pylist()

    result_ids: list[str] = []
    clean_contents: list[str] = []
    html_lengths: list[int] = []
    text_lengths: list[int] = []
    statuses: list[str] = []
    errors: list[str | None] = []
    versions: list[str] = []

    for doc_id, content_html in zip(ids, contents):
        try:
            clean = clean_html(content_html)
            status = classify_clean_status(
                content_html=content_html,
                content_clean=clean,
            )
            error = None
        except Exception as exc:
            # A cleaning exception is different from an empty source.
            # Keep the document in the output so it can be audited later.
            clean = ""
            status = "clean_error"
            error = f"{type(exc).__name__}: {exc}"[:2000]

        result_ids.append(
            "" if doc_id is None else str(doc_id)
        )

        clean_contents.append(clean)

        html_lengths.append(
            len(content_html)
            if content_html
            else 0
        )

        text_lengths.append(len(clean))
        statuses.append(status)
        errors.append(error)
        versions.append(CLEANER_VERSION)

    return pa.Table.from_arrays(
        [
            pa.array(
                result_ids,
                type=pa.string(),
            ),
            pa.array(
                clean_contents,
                type=pa.string(),
            ),
            pa.array(
                html_lengths,
                type=pa.int64(),
            ),
            pa.array(
                text_lengths,
                type=pa.int64(),
            ),
            pa.array(
                statuses,
                type=pa.string(),
            ),
            pa.array(
                errors,
                type=pa.string(),
            ),
            pa.array(
                versions,
                type=pa.string(),
            ),
        ],
        schema=OUTPUT_SCHEMA,
    )


def build_clean_dataset(
    input_path: Path,
    output_path: Path,
    batch_size: int = 1000,
    limit: int | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input parquet does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write to a temporary file first. If cleaning fails halfway through,
    # the previous valid content_clean.parquet is not destroyed.
    temp_output_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    if temp_output_path.exists():
        temp_output_path.unlink()

    parquet_file = pq.ParquetFile(input_path)

    required_columns = {"id", "content_html"}
    available_columns = set(parquet_file.schema_arrow.names)
    missing_columns = required_columns - available_columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Input parquet is missing required columns: {missing}"
        )

    writer = pq.ParquetWriter(
        temp_output_path,
        OUTPUT_SCHEMA,
        compression="zstd",
    )

    processed = 0
    status_counts: Counter[str] = Counter()
    completed = False

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[
                "id",
                "content_html",
            ],
        ):
            if limit is not None:
                remaining = limit - processed
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)

            table = process_batch(batch)
            writer.write_table(table)

            batch_statuses = (
                table.column("clean_status")
                .combine_chunks()
                .to_pylist()
            )
            status_counts.update(batch_statuses)

            processed += table.num_rows

            print(
                f"\rProcessed: {processed:,}",
                end="",
                flush=True,
            )

            if limit is not None and processed >= limit:
                break

        completed = True

    finally:
        writer.close()

        if not completed:
            temp_output_path.unlink(missing_ok=True)

    # Atomic replacement only after the whole dataset has been written.
    temp_output_path.replace(output_path)

    print()
    print(f"Saved to: {output_path}")
    print("\nClean status summary:")

    for status, count in sorted(status_counts.items()):
        percentage = (
            (count / processed) * 100
            if processed
            else 0.0
        )
        print(
            f"  {status:<20} {count:>10,} "
            f"({percentage:6.2f}%)"
        )

    suspicious_statuses = {
        "cleaner_empty",
        "parse_error",
        "clean_error",
    }
    suspicious_count = sum(
        status_counts.get(status, 0)
        for status in suspicious_statuses
    )

    if suspicious_count:
        print(
            "\nWARNING: "
            f"{suspicious_count:,} row(s) require investigation "
            "(cleaner_empty / parse_error / clean_error)."
        )
    else:
        print(
            "\nNo suspicious cleaner failures detected."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legal-document HTML into structure-preserving clean text."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/huggingface/data/content.parquet"
        ),
        help="Input parquet containing id and content_html.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/content_clean.parquet"
        ),
        help="Output parquet for normalized clean content.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows processed per parquet batch.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows for reproducible QA runs.",
    )

    args = parser.parse_args()

    build_clean_dataset(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
