from __future__ import annotations

import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine, text

from lawchat.ingestion import PostgresMetadataLoader


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_loader_replaces_stale_chunks_and_resolves_dependencies(tmp_path):
    suffix = uuid.uuid4().hex
    first_id = f"loader-{suffix}-1"
    second_id = f"loader-{suffix}-2"
    metadata_path = tmp_path / "metadata.parquet"
    structured_path = tmp_path / "structured.parquet"
    chunks_path = tmp_path / "chunks.parquet"
    relationships_path = tmp_path / "relationships.parquet"

    metadata_rows = []
    for external_id, number in ((first_id, "01/TEST"), (second_id, "02/TEST")):
        metadata_rows.append(
            {
                "id": external_id,
                "title": f"Văn bản {number}",
                "so_ky_hieu": number,
                "ngay_ban_hanh": "01/01/2026",
                "loai_van_ban": "Nghị định",
                "ngay_co_hieu_luc": "10/01/2026",
                "ngay_het_hieu_luc": None,
                "nguon_thu_thap": "integration-test",
                "nganh": "Kiểm thử",
                "linh_vuc": "Kiểm thử",
                "co_quan_ban_hanh": "Cơ quan kiểm thử",
                "chuc_danh": None,
                "nguoi_ky": None,
                "pham_vi": "Trung ương",
                "thong_tin_ap_dung": None,
                "tinh_trang_hieu_luc": "Còn hiệu lực",
            }
        )
    _write(metadata_path, metadata_rows)
    _write(
        structured_path,
        [
            {
                "id": external_id,
                "parser_status": "ok",
                "parse_quality": "good",
                "quality_flags": "[]",
                "structure_type": "article_based",
                "parser_version": "test-1",
                "structure_json": f'{{"doc_id":"{external_id}"}}',
            }
            for external_id in (first_id, second_id)
        ],
    )
    chunk_rows = [
            {
                "chunk_id": f"{first_id}::article_1",
                "doc_id": first_id,
                "parent_chunk_id": None,
                "parent_type": None,
                "chunk_type": "article_parent",
                "strategy": "article",
                "is_indexable": False,
                "structure_type": "article_based",
                "part": None,
                "chapter": None,
                "section": None,
                "appendix": None,
                "article": "1",
                "clause": None,
                "point": None,
                "ordinal": 0,
                "text": "Điều 1. Phạm vi.",
                "retrieval_text": None,
                "approx_token_count": 0,
                "overlap_from_chunk_id": None,
                "overlap_token_count": 0,
                "parse_quality": "good",
                "quality_flags": "[]",
                "chunker_version": "test-1",
            },
            {
                "chunk_id": f"{first_id}::article_1::clause_1",
                "doc_id": first_id,
                "parent_chunk_id": f"{first_id}::article_1",
                "parent_type": "article",
                "chunk_type": "clause",
                "strategy": "article",
                "is_indexable": True,
                "structure_type": "article_based",
                "part": None,
                "chapter": None,
                "section": None,
                "appendix": None,
                "article": "1",
                "clause": "1",
                "point": None,
                "ordinal": 1,
                "text": "1. Nội dung kiểm thử.",
                "retrieval_text": "Điều 1\n1. Nội dung kiểm thử.",
                "approx_token_count": 10,
                "overlap_from_chunk_id": None,
                "overlap_token_count": 0,
                "parse_quality": "good",
                "quality_flags": "[]",
                "chunker_version": "test-1",
            },
        ]
    _write(chunks_path, chunk_rows)
    _write(
        relationships_path,
        [
            {
                "doc_id": first_id,
                "other_doc_id": second_id,
                "relationship": "Thay thế",
            }
        ],
    )

    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    run_ids: list[str] = []
    try:
        loader = PostgresMetadataLoader(
            engine,
            batch_size=10,
            dataset_revision="integration-test",
        )
        first_report = loader.load(
            metadata_path=metadata_path,
            structured_path=structured_path,
            chunks_path=chunks_path,
            relationships_path=relationships_path,
        )
        run_ids.append(first_report.run_id)

        # The next complete generation no longer contains the clause. It must
        # be deleted only after the replacement generation has fully loaded.
        _write(chunks_path, chunk_rows[:1])
        second_report = PostgresMetadataLoader(
            engine,
            batch_size=10,
            dataset_revision="integration-test",
        ).load(
            metadata_path=metadata_path,
            structured_path=structured_path,
            chunks_path=chunks_path,
            relationships_path=relationships_path,
        )
        run_ids.append(second_report.run_id)
        assert second_report.counters["stale_chunks_deleted"] == 1

        with engine.connect() as connection:
            counts = connection.execute(
                text(
                    "SELECT "
                    "(SELECT count(*) FROM documents WHERE external_id IN (:a,:b)), "
                    "(SELECT count(*) FROM document_versions v JOIN documents d "
                    " ON d.id=v.document_id WHERE d.external_id IN (:a,:b)), "
                    "(SELECT count(*) FROM chunks c JOIN documents d "
                    " ON d.id=c.document_id WHERE d.external_id IN (:a,:b)), "
                    "(SELECT count(*) FROM document_relationships r JOIN documents d "
                    " ON d.id=r.source_document_id WHERE d.external_id=:a)"
                ),
                {"a": first_id, "b": second_id},
            ).one()
            assert tuple(counts) == (2, 2, 1, 1)
            target_id = connection.execute(
                text(
                    "SELECT target_document_id FROM document_relationships r "
                    "JOIN documents d ON d.id=r.source_document_id "
                    "WHERE d.external_id=:external_id"
                ),
                {"external_id": first_id},
            ).scalar_one()
            assert target_id is not None
    finally:
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM documents WHERE external_id IN (:a,:b)"),
                {"a": first_id, "b": second_id},
            )
            connection.execute(
                text("DELETE FROM crawl_runs WHERE id=ANY(CAST(:ids AS uuid[]))"),
                {"ids": run_ids},
            )
        engine.dispose()
