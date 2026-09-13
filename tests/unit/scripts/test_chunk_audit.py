from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check_chunks import audit_rows, summarize


def _write_chunks(path):
    pq.write_table(
        pa.Table.from_pylist([
            {
                "document_number": "01/2026/QH",
                "chunk_id": "doc::article_1",
                "chunk_type": "article",
                "strategy": "article",
                "is_indexable": True,
                "approx_token_count": 700,
                "article": "1",
                "clause": None,
                "point": None,
                "parent_chunk_id": None,
                "text": "Nội dung Điều 1",
                "retrieval_text": "Văn bản: Luật thử nghiệm\nNội dung Điều 1",
            },
            {
                "document_number": "01/2026/QH",
                "chunk_id": "doc::article_2",
                "chunk_type": "article_parent",
                "strategy": "article",
                "is_indexable": False,
                "approx_token_count": 0,
                "article": "2",
                "clause": None,
                "point": None,
                "parent_chunk_id": None,
                "text": "Nội dung Điều 2",
                "retrieval_text": "",
            },
            {
                "document_number": "01/2026/QH",
                "chunk_id": "doc::article_2::clauses_1-2",
                "chunk_type": "clause_group",
                "strategy": "article",
                "is_indexable": True,
                "approx_token_count": 1200,
                "article": "2",
                "clause": "1-2",
                "point": None,
                "parent_chunk_id": "doc::article_2",
                "text": "Khoản 1 và Khoản 2",
                "retrieval_text": "Điều 2\nKhoản 1-2\nNội dung",
            },
        ]),
        path,
    )


def test_chunk_audit_reports_distribution_and_samples(tmp_path):
    path = tmp_path / "chunks.parquet"
    _write_chunks(path)

    report = summarize(path)
    audit = audit_rows(path, seed=1, sample_size=100)

    assert report["total_chunks"] == 3
    assert report["indexable_chunks"] == 2
    assert report["parent_chunks"] == 1
    assert report["tokens"]["max"] == 1200
    assert report["tokens"]["buckets"]["600-800"] == 1
    assert report["tokens"]["buckets"]["1001-1200"] == 1
    assert audit["packed_clauses"][0]["clause"] == "1-2"
    assert len(audit["random"]) == 2
