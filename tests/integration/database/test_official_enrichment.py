from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from lawchat.ingestion import OfficialEnrichmentLoader


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is required")


def test_official_enrichment_loads_all_supported_assertions(tmp_path):
    suffix = uuid.uuid4().hex
    source_id, target_id = f"official-source-{suffix}", f"official-target-{suffix}"
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    with engine.begin() as connection:
        rows = connection.execute(text(
            "INSERT INTO documents (external_id,title,status,source_url,metadata) VALUES "
            "(:source,'Source','UNKNOWN','https://aggregate.test/source','{}'),"
            "(:target,'Target','UNKNOWN','https://aggregate.test/target','{}') "
            "RETURNING external_id,id"
        ), {"source": source_id, "target": target_id}).all()
    ids = {row.external_id: str(row.id) for row in rows}
    official_url = "https://official.test/source"
    payload = {
        "documents": [{"external_id": source_id, "official_source_url": official_url,
                       "title": "Official source", "status": "EFFECTIVE"}],
        "effective_statuses": [{"document_external_id": source_id,
            "status": "EFFECTIVE", "valid_from": "2020-01-01", "valid_to": None,
            "reason": "official", "official_source_url": official_url}],
        "provision_statuses": [{"document_external_id": source_id,
            "provision_key": "article:1/clause:/point:", "article": "1",
            "status": "EFFECTIVE", "valid_from": "2020-01-01", "valid_to": None,
            "reason": "official", "official_source_url": official_url}],
        "relationships": [{"source_external_id": source_id,
            "target_external_id": target_id, "relationship_type": "REPLACES",
            "source_relationship": "Thay thế", "effective_from": "2020-01-01",
            "official_source_url": official_url}],
        "amendment_events": [{"source_external_id": source_id,
            "target_external_id": target_id, "event_type": "REPLACES",
            "effective_date": "2020-01-01", "official_source_url": official_url}],
        "provenance": [{"document_external_id": source_id,
            "entity_type": "DOCUMENT", "entity_id": ids[source_id],
            "source_kind": "OFFICIAL_PORTAL", "source_url": official_url,
            "retrieved_at": "2026-09-02T00:00:00Z"}],
    }
    path = tmp_path / "official.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        report = OfficialEnrichmentLoader(engine).load(path)
        assert report.counters == {
            "documents": 1, "effective_statuses": 1, "provision_statuses": 1,
            "relationships": 1, "amendment_events": 1, "provenance": 1,
        }
        with engine.connect() as connection:
            values = connection.execute(text(
                "SELECT d.source_url, d.metadata->>'trusted_snapshot', "
                "(SELECT count(*) FROM amendment_events a WHERE a.source_document_id=d.id), "
                "(SELECT count(*) FROM provenance_records p WHERE p.document_id=d.id) "
                "FROM documents d WHERE d.external_id=:id"
            ), {"id": source_id}).one()
        assert values == (official_url, "true", 1, 1)
    finally:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM documents WHERE external_id IN (:a,:b)"),
                               {"a": source_id, "b": target_id})
        engine.dispose()
