#!/usr/bin/env python3
"""
Inspect and Extract LawChat Chunks
=================================
Công cụ trích xuất và phân tích toàn diện kho chunks:
1. Thống kê chỉ số phân bố (Tokens, Strategy, Chunk types, Overlap metrics, Buckets)
2. Trích xuất các chunk bé nhất (Smallest chunks)
3. Trích xuất các chunk lớn nhất (Largest chunks)
4. Trích xuất các chunk sát nhau (Adjacent / Consecutive chunks)
5. Trích xuất các chunk overlap (Overlapping chunks & context link)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import polars as pl
import pyarrow.parquet as pq


AUDIT_BUCKETS = (
    ("0-100", 0, 100),
    ("101-300", 101, 300),
    ("301-599", 301, 599),
    ("600-800", 600, 800),
    ("801-1000", 801, 1000),
    ("1001-1200", 1001, 1200),
    (">1200", 1201, None),
)

AUDIT_COLUMNS = (
    "document_number", "chunk_id", "chunk_type", "strategy",
    "approx_token_count", "article", "clause", "point",
    "parent_chunk_id", "text", "retrieval_text",
)


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def summarize(path: Path) -> dict[str, Any]:
    rows = pq.read_table(
        path, columns=["is_indexable", "approx_token_count"]
    ).to_pylist()
    token_counts = sorted(
        int(row["approx_token_count"] or 0)
        for row in rows
        if row["is_indexable"]
    )
    buckets = {
        label: sum(
            1 for value in token_counts
            if value >= low and (high is None or value <= high)
        )
        for label, low, high in AUDIT_BUCKETS
    }
    return {
        "path": str(path),
        "total_chunks": len(rows),
        "indexable_chunks": len(token_counts),
        "parent_chunks": len(rows) - len(token_counts),
        "tokens": {
            "min": min(token_counts, default=0),
            "p10": percentile(token_counts, 0.10),
            "p25": percentile(token_counts, 0.25),
            "p50": percentile(token_counts, 0.50),
            "p75": percentile(token_counts, 0.75),
            "p90": percentile(token_counts, 0.90),
            "p95": percentile(token_counts, 0.95),
            "p99": percentile(token_counts, 0.99),
            "max": max(token_counts, default=0),
            "mean": mean(token_counts) if token_counts else 0.0,
            "buckets": buckets,
        },
    }


def preview(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def audit_rows(
    path: Path, seed: int, sample_size: int
) -> dict[str, list[dict[str, Any]]]:
    rows = [
        row for row in pq.read_table(
            path, columns=list(AUDIT_COLUMNS) + ["is_indexable"]
        ).to_pylist()
        if row["is_indexable"]
    ]
    rng = random.Random(seed)

    def compact(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **{key: row.get(key) for key in AUDIT_COLUMNS[:-2]},
                "text_preview": preview(row.get("text")),
                "retrieval_text_preview": preview(row.get("retrieval_text")),
            }
            for row in items
        ]

    shortest = sorted(rows, key=lambda row: int(row["approx_token_count"] or 0))[:20]
    longest = sorted(
        rows, key=lambda row: int(row["approx_token_count"] or 0), reverse=True
    )[:20]
    packed = [row for row in rows if row.get("chunk_type") == "clause_group"][:20]
    point_heavy = [
        row for row in rows
        if row.get("chunk_type") in {"point_group", "point", "point_fragment"}
    ][:20]
    random_rows = rng.sample(rows, min(sample_size, len(rows))) if rows else []
    return {
        "shortest": compact(shortest),
        "longest": compact(longest),
        "packed_clauses": compact(packed),
        "point_heavy": compact(point_heavy),
        "random": compact(random_rows),
    }


def format_separator(title: str = "", char: str = "=", length: int = 80) -> str:
    if not title:
        return char * length
    prefix = f" {title} "
    total_padding = max(0, length - len(prefix))
    left = total_padding // 2
    right = total_padding - left
    return f"{char * left}{prefix}{char * right}"


def truncate(text: str | None, limit: int = 250) -> str:
    if not text:
        return "<EMPTY>"
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def get_token_buckets(df_indexable: pl.LazyFrame) -> dict[str, int]:
    buckets = [
        ("0-100", 0, 100),
        ("101-300", 101, 300),
        ("301-400", 301, 400),
        ("401-450", 401, 450),
        ("451-600", 451, 600),
        ("601-800", 601, 800),
        ("801-1200", 801, 1200),
        (">1200", 1201, None),
    ]
    exprs = []
    for label, low, high in buckets:
        if high is not None:
            exprs.append(
                ((pl.col("approx_token_count") >= low) & (pl.col("approx_token_count") <= high))
                .sum()
                .alias(label)
            )
        else:
            exprs.append((pl.col("approx_token_count") >= low).sum().alias(label))

    res = df_indexable.select(exprs).collect().to_dicts()[0]
    return {k: int(v) for k, v in res.items() if v > 0 or k in ["0-100", "101-300", "301-400", "401-450", ">1200"]}


def compute_statistics(parquet_path: Path) -> dict[str, Any]:
    df = pl.scan_parquet(str(parquet_path))

    # Basic totals
    counts = df.select([
        pl.len().alias("total_chunks"),
        pl.col("is_indexable").sum().alias("indexable_chunks"),
        (pl.col("is_indexable").not_()).sum().alias("parent_chunks"),
        pl.col("overlap_from_chunk_id").is_not_null().sum().alias("overlap_chunks_count"),
        pl.col("doc_id").n_unique().alias("unique_documents"),
    ]).collect().to_dicts()[0]

    # Indexable token statistics
    df_idx = df.filter(pl.col("is_indexable"))
    token_stats = df_idx.select([
        pl.col("approx_token_count").min().alias("min"),
        pl.col("approx_token_count").quantile(0.10).alias("p10"),
        pl.col("approx_token_count").quantile(0.25).alias("p25"),
        pl.col("approx_token_count").median().alias("p50"),
        pl.col("approx_token_count").quantile(0.75).alias("p75"),
        pl.col("approx_token_count").quantile(0.90).alias("p90"),
        pl.col("approx_token_count").quantile(0.95).alias("p95"),
        pl.col("approx_token_count").quantile(0.99).alias("p99"),
        pl.col("approx_token_count").max().alias("max"),
        pl.col("approx_token_count").mean().alias("mean"),
        pl.col("approx_token_count").std().alias("std"),
    ]).collect().to_dicts()[0]

    # Overlap statistics
    df_ov = df.filter(pl.col("overlap_token_count") > 0)
    overlap_stats = df_ov.select([
        pl.col("overlap_token_count").min().alias("min"),
        pl.col("overlap_token_count").median().alias("median"),
        pl.col("overlap_token_count").mean().alias("mean"),
        pl.col("overlap_token_count").max().alias("max"),
    ]).collect().to_dicts()[0]

    # Group by Strategy
    strategy_dist = (
        df.group_by("strategy")
        .agg([
            pl.len().alias("total"),
            pl.col("is_indexable").sum().alias("indexable"),
        ])
        .sort("total", descending=True)
        .collect()
        .to_dicts()
    )

    # Group by Chunk Type (Top 10)
    chunk_type_dist = (
        df.group_by("chunk_type")
        .agg([
            pl.len().alias("total"),
            pl.col("is_indexable").sum().alias("indexable"),
        ])
        .sort("total", descending=True)
        .limit(10)
        .collect()
        .to_dicts()
    )

    token_buckets = get_token_buckets(df_idx)

    return {
        "counts": counts,
        "token_distribution": {k: round(v, 2) if isinstance(v, float) else v for k, v in token_stats.items()},
        "token_buckets": token_buckets,
        "overlap_metrics": {k: round(v, 2) if isinstance(v, float) else v for k, v in overlap_stats.items()},
        "strategy_distribution": strategy_dist,
        "chunk_type_distribution": chunk_type_dist,
    }


def extract_smallest(parquet_path: Path, n: int = 5, indexable_only: bool = True) -> list[dict[str, Any]]:
    df = pl.scan_parquet(str(parquet_path))
    if indexable_only:
        df = df.filter(pl.col("is_indexable"))
    cols = [
        "chunk_id", "doc_id", "document_number", "chunk_type", "strategy",
        "article", "clause", "point", "approx_token_count", "text", "retrieval_text"
    ]
    return (
        df.select(cols)
        .sort("approx_token_count", descending=False)
        .limit(n)
        .collect()
        .to_dicts()
    )


def extract_largest(parquet_path: Path, n: int = 5, indexable_only: bool = True) -> list[dict[str, Any]]:
    df = pl.scan_parquet(str(parquet_path))
    if indexable_only:
        df = df.filter(pl.col("is_indexable"))
    cols = [
        "chunk_id", "doc_id", "document_number", "chunk_type", "strategy",
        "article", "clause", "point", "approx_token_count", "text", "retrieval_text"
    ]
    return (
        df.select(cols)
        .sort("approx_token_count", descending=True)
        .limit(n)
        .collect()
        .to_dicts()
    )


def extract_adjacent(
    parquet_path: Path,
    doc_id: str | None = None,
    parent_chunk_id: str | None = None,
    limit_chunks: int = 6,
) -> dict[str, Any]:
    df = pl.scan_parquet(str(parquet_path))

    if parent_chunk_id is not None:
        target_parent = parent_chunk_id
    elif doc_id is not None:
        # Get first parent with multiple children in that doc
        target_parent_row = (
            df.filter((pl.col("doc_id") == doc_id) & pl.col("parent_chunk_id").is_not_null() & pl.col("is_indexable"))
            .group_by("parent_chunk_id")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= 2)
            .select("parent_chunk_id")
            .limit(1)
            .collect()
        )
        if target_parent_row.height > 0:
            target_parent = target_parent_row.item()
        else:
            target_parent = None
    else:
        # Automatically select a clean, representative legal article with consecutive clauses
        sample_parent = (
            df.filter(
                (pl.col("strategy") == "article")
                & (pl.col("chunk_type") == "clause")
                & pl.col("parent_chunk_id").is_not_null()
                & pl.col("is_indexable")
            )
            .group_by("parent_chunk_id")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= limit_chunks)
            .select("parent_chunk_id")
            .limit(1)
            .collect()
        )
        target_parent = sample_parent.item() if sample_parent.height > 0 else None

    cols = [
        "chunk_id", "doc_id", "document_number", "document_title", "parent_chunk_id",
        "ordinal", "chunk_type", "strategy", "is_indexable", "article", "clause", "point",
        "approx_token_count", "text", "retrieval_text"
    ]

    if target_parent:
        # Fetch parent metadata
        parent_row = (
            df.filter(pl.col("chunk_id") == target_parent)
            .select(cols)
            .limit(1)
            .collect()
            .to_dicts()
        )
        parent_info = parent_row[0] if parent_row else None

        # Fetch child adjacent chunks
        child_rows = (
            df.filter(pl.col("parent_chunk_id") == target_parent)
            .select(cols)
            .sort("chunk_id", descending=False)
            .limit(limit_chunks)
            .collect()
            .to_dicts()
        )
        return {
            "mode": "parent_children",
            "parent_id": target_parent,
            "parent_chunk": parent_info,
            "adjacent_children": child_rows,
        }
    else:
        # Fallback to document chunks
        filter_expr = (pl.col("doc_id") == doc_id) if doc_id else pl.col("is_indexable")
        rows = (
            df.filter(filter_expr)
            .select(cols)
            .sort("ordinal", descending=False)
            .limit(limit_chunks)
            .collect()
            .to_dicts()
        )
        return {
            "mode": "document_sequence",
            "parent_id": None,
            "parent_chunk": None,
            "adjacent_children": rows,
        }


def extract_overlapping(parquet_path: Path, n: int = 3) -> list[dict[str, Any]]:
    df = pl.scan_parquet(str(parquet_path))
    overlap_chunks = (
        df.filter(
            pl.col("overlap_from_chunk_id").is_not_null()
            & (pl.col("overlap_token_count") > 0)
            & pl.col("is_indexable")
        )
        .select([
            "chunk_id", "doc_id", "document_number", "chunk_type", "strategy",
            "approx_token_count", "overlap_from_chunk_id", "overlap_token_count",
            "overlap_text", "text", "retrieval_text"
        ])
        .limit(n)
        .collect()
        .to_dicts()
    )

    # For each overlap chunk, also fetch the previous source chunk
    enriched = []
    for chunk in overlap_chunks:
        from_id = chunk["overlap_from_chunk_id"]
        source = (
            df.filter(pl.col("chunk_id") == from_id)
            .select(["chunk_id", "chunk_type", "approx_token_count", "text"])
            .limit(1)
            .collect()
            .to_dicts()
        )
        enriched.append({
            "target_chunk": chunk,
            "source_chunk": source[0] if source else None,
        })
    return enriched


def print_stats_table(stats: dict[str, Any]) -> None:
    print(format_separator("1. TỔNG QUAN CHỈ SỐ CHUNK (OVERALL METRICS)", "="))
    c = stats["counts"]
    print(f"• Tổng số chunks:              {c['total_chunks']:,}")
    print(f"• Số chunk tìm kiếm (Indexable):{c['indexable_chunks']:,} ({c['indexable_chunks']/c['total_chunks']*100:.1f}%)")
    print(f"• Số chunk ngữ cảnh cha (Parent):{c['parent_chunks']:,} ({c['parent_chunks']/c['total_chunks']*100:.1f}%)")
    print(f"• Số lượng văn bản (Documents): {c['unique_documents']:,}")
    print(f"• Số chunk có Overlap:         {c['overlap_chunks_count']:,} ({c['overlap_chunks_count']/c['total_chunks']*100:.2f}%)")

    print("\n" + format_separator("PHÂN BỐ SỐ TOKEN (INDEXABLE CHUNKS)", "-"))
    t = stats["token_distribution"]
    print(f"  Min: {t['min']} tok | P10: {t['p10']} tok | P25: {t['p25']} tok | Median (P50): {t['p50']} tok")
    print(f"  P75: {t['p75']} tok | P90: {t['p90']} tok | P95: {t['p95']} tok | P99: {t['p99']} tok | Max: {t['max']} tok")
    print(f"  Mean: {t['mean']} ± {t['std']} tokens")

    print("\n" + format_separator("PHÂN KHOẢNG TOKEN BUCKETS", "-"))
    for b_label, b_cnt in stats["token_buckets"].items():
        bar = "█" * int(b_cnt / c["indexable_chunks"] * 40)
        print(f"  [{b_label:>8} tokens]: {b_cnt:>9,} chunks ({b_cnt/c['indexable_chunks']*100:>5.1f}%) | {bar}")

    print("\n" + format_separator("PHÂN BỐ THEO CHIẾN LƯỢC (STRATEGY)", "-"))
    for s in stats["strategy_distribution"]:
        print(f"  • {s['strategy']:<20}: Total={s['total']:>9,} | Indexable={s['indexable']:>9,}")

    print("\n" + format_separator("CHỈ SỐ OVERLAP (KHI CÓ GỐI ĐẦU)", "-"))
    o = stats["overlap_metrics"]
    print(f"  Min: {o['min']} tok | Median: {o['median']} tok | Mean: {o['mean']} tok | Max: {o['max']} tok")


def print_chunk_item(idx: int, item: dict[str, Any], tag: str = "") -> None:
    badge = f"[{tag} #{idx}]" if tag else f"[#{idx}]"
    print(f"\n{badge} Chunk ID: {item.get('chunk_id')}")
    print(f"  • Doc ID: {item.get('doc_id')} | Số hiệu: {item.get('document_number')}")
    print(f"  • Loại: {item.get('chunk_type')} | Chiến lược: {item.get('strategy')}")
    coords = []
    if item.get("article"): coords.append(f"Điều {item['article']}")
    if item.get("clause"): coords.append(f"Khoản {item['clause']}")
    if item.get("point"): coords.append(f"Điểm {item['point']}")
    if coords:
        print(f"  • Vị trí: {' > '.join(coords)}")
    print(f"  • Số Token (Approx): {item.get('approx_token_count')}")
    print(f"  • Raw Text: {truncate(item.get('text'), 180)}")
    print(f"  • Retrieval Text Preview: {truncate(item.get('retrieval_text'), 220)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LawChat Chunk Extraction & Analysis Tool")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/chunks.parquet"), help="Đường dẫn file parquet")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ: Thống kê, Smallest, Largest, Adjacent, Overlap")
    parser.add_argument("--stats", action="store_true", help="In bảng thống kê chỉ số chunks")
    parser.add_argument("--smallest", type=int, default=0, metavar="N", help="Trích xuất N chunk bé nhất")
    parser.add_argument("--largest", type=int, default=0, metavar="N", help="Trích xuất N chunk lớn nhất")
    parser.add_argument("--adjacent", type=int, default=0, metavar="N", help="Trích xuất N chunk sát nhau trong cùng tài liệu / Điều cha")
    parser.add_argument("--doc-id", type=str, default=None, help="Chỉ định Doc ID cụ thể để trích xuất chunk sát nhau")
    parser.add_argument("--parent-id", type=str, default=None, help="Chỉ định Parent Chunk ID cụ thể để trích xuất các con sát nhau")
    parser.add_argument("--overlap", type=int, default=0, metavar="N", help="Trích xuất N chunk có overlap và liên kết nguồn")
    parser.add_argument("--json", action="store_true", help="Xuất kết quả dưới dạng JSON")
    parser.add_argument("--gate", action="store_true", help="Run the release quality gate")
    parser.add_argument("--old", type=Path, help="Optional previous release for comparison")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # If no flags specified, default to --all
    if not (args.stats or args.smallest or args.largest or args.adjacent or args.overlap or args.all):
        args.all = True

    if not args.chunks.exists():
        print(f"Lỗi: Không tìm thấy file {args.chunks}", file=sys.stderr)
        sys.exit(1)

    if args.gate:
        report: dict[str, Any] = {
            "new": summarize(args.chunks),
            "audit": audit_rows(args.chunks, args.seed, args.sample_size),
        }
        if args.old:
            report["old"] = summarize(args.old)
        if report["new"]["tokens"]["buckets"][">1200"]:
            raise ValueError("Chunk audit failed: indexable chunks exceed 1200 tokens")
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return

    result_json: dict[str, Any] = {}

    # 1. Statistics
    if args.all or args.stats:
        stats = compute_statistics(args.chunks)
        result_json["statistics"] = stats
        if not args.json:
            print_stats_table(stats)

    # 2. Smallest
    smallest_n = args.smallest if args.smallest > 0 else (5 if args.all else 0)
    if smallest_n > 0:
        smallest = extract_smallest(args.chunks, n=smallest_n)
        result_json["smallest_chunks"] = smallest
        if not args.json:
            print("\n" + format_separator(f"2. TOP {smallest_n} CHUNK BÉ NHẤT (SMALLEST CHUNKS)", "="))
            for i, it in enumerate(smallest, 1):
                print_chunk_item(i, it, tag="BÉ NHẤT")

    # 3. Largest
    largest_n = args.largest if args.largest > 0 else (5 if args.all else 0)
    if largest_n > 0:
        largest = extract_largest(args.chunks, n=largest_n)
        result_json["largest_chunks"] = largest
        if not args.json:
            print("\n" + format_separator(f"3. TOP {largest_n} CHUNK LỚN NHẤT (LARGEST CHUNKS)", "="))
            for i, it in enumerate(largest, 1):
                print_chunk_item(i, it, tag="LỚN NHẤT")

    # 4. Adjacent
    adj_n = args.adjacent if args.adjacent > 0 else (6 if args.all else 0)
    if adj_n > 0:
        adj_res = extract_adjacent(
            args.chunks,
            doc_id=args.doc_id,
            parent_chunk_id=args.parent_id,
            limit_chunks=adj_n,
        )
        result_json["adjacent_chunks"] = adj_res
        if not args.json:
            parent = adj_res.get("parent_chunk")
            children = adj_res.get("adjacent_children", [])
            print("\n" + format_separator(f"4. CÁC CHUNK SÁT NHAU / LIÊN TIẾP (ADJACENT CHUNKS)", "="))
            if parent:
                print(f"• Chunk cha (Parent): {parent.get('chunk_id')}")
                print(f"• Văn bản:           {parent.get('document_number')} - {parent.get('document_title')}")
                coords = []
                if parent.get("article"): coords.append(f"Điều {parent['article']}")
                if parent.get("clause"): coords.append(f"Khoản {parent['clause']}")
                if coords:
                    print(f"• Phạm vi cha:       {' > '.join(coords)}")
                print(f"• Raw Text cha:      \"{truncate(parent.get('text'), 180)}\"")
                print(f"\n--- Danh sách {len(children)} chunk con sát nhau (Indexable Leaf Chunks) ---")
            else:
                print(f"Danh sách {len(children)} chunk liên tiếp trong tài liệu:")

            for i, it in enumerate(children, 1):
                role = "Indexable [Lá]" if it.get("is_indexable") else "Parent [Cha]"
                coords = []
                if it.get("article"): coords.append(f"Điều {it['article']}")
                if it.get("clause"): coords.append(f"Khoản {it['clause']}")
                if it.get("point"): coords.append(f"Điểm {it['point']}")
                loc_str = f" | Vị trí: {' > '.join(coords)}" if coords else ""

                print(f"\n[Sát nhau #{i}] Chunk ID: {it.get('chunk_id')}")
                print(f"  • Vai trò: {role} | Loại: {it.get('chunk_type')}{loc_str}")
                print(f"  • Tokens: {it.get('approx_token_count')}")
                print(f"  • Text: \"{truncate(it.get('text'), 160)}\"")

    # 5. Overlap
    ov_n = args.overlap if args.overlap > 0 else (3 if args.all else 0)
    if ov_n > 0:
        overlapping = extract_overlapping(args.chunks, n=ov_n)
        result_json["overlapping_chunks"] = overlapping
        if not args.json:
            print("\n" + format_separator(f"5. CÁC CHUNK OVERLAP (GỐI ĐẦU NGỮ CẢNH)", "="))
            for i, pair in enumerate(overlapping, 1):
                tgt = pair["target_chunk"]
                src = pair.get("source_chunk")
                print(f"\n[Cặp Overlap #{i}]")
                print(f"  ├── Chunk hiện tại:  {tgt['chunk_id']} ({tgt['approx_token_count']} tok)")
                print(f"  ├── Nguồn gối đầu:   {tgt['overlap_from_chunk_id']} ({src['approx_token_count'] if src else '?'} tok)")
                print(f"  ├── Token Overlap:   {tgt['overlap_token_count']} tokens")
                print(f"  ├── Đoạn Overlap:    \"{truncate(tgt['overlap_text'], 150)}\"")
                print(f"  └── Text hiện tại:   \"{truncate(tgt['text'], 150)}\"")

    if args.json:
        print(json.dumps(result_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
