from __future__ import annotations

import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine, text

from lawchat.ingestion import PostgresMetadataLoader
from lawchat.indexing import PostgresChunkSource


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is required")


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _metadata(external_id: str) -> dict:
    return {
        "id": external_id, "title": "Historical test", "so_ky_hieu": "01/HIST",
        "ngay_ban_hanh": "01/01/2010", "loai_van_ban": "Luật",
        "ngay_co_hieu_luc": "01/07/2010", "ngay_het_hieu_luc": None,
        "nguon_thu_thap": "test", "nganh": None, "linh_vuc": "test",
        "co_quan_ban_hanh": "test", "chuc_danh": None, "nguoi_ky": None,
        "pham_vi": None, "thong_tin_ap_dung": None,
        "tinh_trang_hieu_luc": "Còn hiệu lực",
    }


def _structured(external_id: str, revision: str, start: str, end: str | None) -> dict:
    return {
        "id": external_id, "parser_status": "ok", "parse_quality": "good",
        "quality_flags": "[]", "structure_type": "article_based",
        "parser_version": "test", "structure_json": f'{{"revision":"{revision}"}}',
        "source_revision": revision, "source_url": f"https://official.test/{revision}",
        "content_valid_from": start, "content_valid_to": end,
    }


def _chunk(external_id: str, revision: str, ordinal: int) -> dict:
    return {
        "chunk_id": f"{external_id}::{revision}", "doc_id": external_id,
        "version_source_revision": revision, "parent_chunk_id": None,
        "parent_type": None, "chunk_type": "article", "strategy": "article",
        "is_indexable": True, "structure_type": "article_based", "part": None,
        "chapter": None, "section": None, "appendix": None, "article": "1",
        "clause": None, "point": None, "ordinal": ordinal,
        "text": f"Nội dung {revision}", "retrieval_text": f"Nội dung {revision}",
        "approx_token_count": 3, "overlap_from_chunk_id": None,
        "overlap_token_count": 0, "parse_quality": "good", "quality_flags": "[]",
        "chunker_version": "test",
    }


def test_historical_import_attaches_chunks_to_versions_and_preserves_current(tmp_path):
    external_id = f"historical-{uuid.uuid4().hex}"
    metadata = tmp_path / "metadata.parquet"
    structured = tmp_path / "structured.parquet"
    chunks = tmp_path / "chunks.parquet"
    _write(metadata, [_metadata(external_id)])
    _write(structured, [_structured(external_id, "hist-v1", "2010-07-01", "2015-01-01")])
    _write(chunks, [_chunk(external_id, "hist-v1", 1)])
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    run_ids = []
    try:
        _write(structured, [_structured(external_id, "current-v3", "2020-01-01", None)])
        _write(chunks, [_chunk(external_id, "current-v3", 3)])
        report = PostgresMetadataLoader(engine, version_mode="current").load(
            metadata_path=metadata, structured_path=structured, chunks_path=chunks
        )
        run_ids.append(report.run_id)
        _write(structured, [_structured(external_id, "hist-v1", "2010-07-01", "2015-01-01")])
        _write(chunks, [_chunk(external_id, "hist-v1", 1)])
        report = PostgresMetadataLoader(engine, version_mode="historical").load(
            metadata_path=metadata, structured_path=structured, chunks_path=chunks
        )
        run_ids.append(report.run_id)
        _write(structured, [_structured(external_id, "hist-v2", "2015-01-01", "2020-01-01")])
        _write(chunks, [_chunk(external_id, "hist-v2", 2)])
        report = PostgresMetadataLoader(engine, version_mode="historical").load(
            metadata_path=metadata, structured_path=structured, chunks_path=chunks
        )
        run_ids.append(report.run_id)
        with engine.connect() as connection:
            rows = connection.execute(text(
                "SELECT v.source_revision, v.is_current, count(c.id) "
                "FROM document_versions v JOIN documents d ON d.id=v.document_id "
                "LEFT JOIN chunks c ON c.version_id=v.id WHERE d.external_id=:id "
                "GROUP BY v.id ORDER BY v.source_revision"
            ), {"id": external_id}).all()
        assert rows == [
            ("current-v3", True, 1),
            ("hist-v1", False, 1),
            ("hist-v2", False, 1),
        ]
        current_chunks = [
            item
            for batch in PostgresChunkSource(
                engine, version_scope="current"
            ).iter_batches(collection="current-test", force=True)
            for item in batch
            if item.payload["doc_id"] == external_id
        ]
        historical_chunks = [
            item
            for batch in PostgresChunkSource(
                engine, version_scope="historical"
            ).iter_batches(collection="historical-test", force=True)
            for item in batch
            if item.payload["doc_id"] == external_id
        ]
        assert len(current_chunks) == 1
        assert len(historical_chunks) == 3
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM documents WHERE external_id=:id"), {"id": external_id})
            connection.execute(text("DELETE FROM crawl_runs WHERE id=ANY(CAST(:ids AS uuid[]))"), {"ids": run_ids})
        engine.dispose()
