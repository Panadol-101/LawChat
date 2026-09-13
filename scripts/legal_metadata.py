from __future__ import annotations

import argparse
import json
from datetime import date
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from lawchat.database import (
    DatabaseSettings,
    Document,
    LegalStatus,
    ProvisionEffectiveStatus,
    create_db_engine,
    provision_key,
)
from lawchat.ingestion import OfficialEnrichmentLoader
from scripts import dispatch


DEFAULT_REVISION_METADATA = Path(
    "data/raw/huggingface/.cache/huggingface/download/data/metadata.parquet.metadata"
)
DEFAULT_METADATA_PARQUET = Path("data/raw/huggingface/data/metadata.parquet")


def enrich_vbpl_main() -> None:
    args = _parse_args()
    revision = args.dataset_revision or _dataset_revision()
    retrieved_at = args.retrieved_at or datetime.fromtimestamp(
        DEFAULT_METADATA_PARQUET.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    source_reference = args.source_reference or f"dataset://vbpl.vn/snapshot/{revision}"
    parameters = {
        "dataset_revision": revision,
        "source_reference": source_reference,
        "snapshot_date": args.snapshot_date,
        "retrieved_at": retrieved_at,
    }
    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        with engine.connect() as connection:
            before = _audit(connection, parameters)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "before": before, **parameters}, indent=2))
            return
        with engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(hashtext('lawchat_vbpl_enrichment'))"))
            connection.commit()
            try:
                with connection.begin():
                    graph = connection.execute(text(TRUST_GRAPH_SQL), parameters)
                    graph_dates = connection.execute(
                        text(DATE_GRAPH_EVENTS_SQL), parameters
                    )
                    amendments = connection.execute(text(AMENDMENT_EVENTS_SQL), parameters)
                    provisions = connection.execute(text(PROVISION_STATUS_SQL), parameters)
                    provenance = connection.execute(text(VERSION_PROVENANCE_SQL), parameters)
                    versions = connection.execute(text(HISTORICAL_AVAILABILITY_SQL), parameters)
                after = _audit(connection, parameters)
            finally:
                connection.execute(text("SELECT pg_advisory_unlock(hashtext('lawchat_vbpl_enrichment'))"))
                connection.commit()
        report = {
            "dry_run": False,
            "dataset_revision": revision,
            "source_reference": source_reference,
            "snapshot_date": args.snapshot_date,
            "retrieved_at": retrieved_at,
            "affected": {
                "graph_relationships": max(graph.rowcount, 0),
                "graph_effective_dates": max(graph_dates.rowcount, 0),
                "amendment_events": max(amendments.rowcount, 0),
                "provision_statuses": max(provisions.rowcount, 0),
                "version_provenance": max(provenance.rowcount, 0),
                "version_availability": max(versions.rowcount, 0),
            },
            "before": before,
            "after": after,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


def _audit(connection, parameters) -> dict:
    return dict(connection.execute(text(AUDIT_SQL), parameters).mappings().one())


def _dataset_revision() -> str:
    if not DEFAULT_REVISION_METADATA.exists():
        raise ValueError("--dataset-revision is required when cache metadata is absent")
    revision = DEFAULT_REVISION_METADATA.read_text(encoding="utf-8").splitlines()[0].strip()
    if not revision:
        raise ValueError("dataset revision metadata is empty")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich all current VBPL snapshot metadata without touching vectors."
    )
    parser.add_argument("--dataset-revision")
    parser.add_argument("--source-reference")
    parser.add_argument("--snapshot-date", default="2026-07-31")
    parser.add_argument("--retrieved-at")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


TRUST_GRAPH_SQL = """
WITH resolved AS (
  SELECT r.id AS relationship_id, v.id AS version_id
  FROM document_relationships r
  LEFT JOIN document_versions v
    ON v.document_id=r.source_document_id AND v.is_current
)
UPDATE document_relationships r SET
  source_version_id=resolved.version_id,
  metadata=r.metadata || jsonb_build_object(
    'source_kind', 'VBPL_SNAPSHOT',
    'dataset_revision', CAST(:dataset_revision AS text),
    'source_reference', CAST(:source_reference AS text),
    'is_trusted', true
  ),
  updated_at=now()
FROM resolved
WHERE resolved.relationship_id=r.id
  AND (
    r.source_version_id IS DISTINCT FROM resolved.version_id
    OR r.metadata->>'dataset_revision' IS DISTINCT FROM CAST(:dataset_revision AS text)
    OR COALESCE((r.metadata->>'is_trusted')::boolean, false)=false
  )
"""

DATE_GRAPH_EVENTS_SQL = """
UPDATE document_relationships r SET
  effective_from=d.effective_date,
  metadata=r.metadata || jsonb_build_object(
    'effective_from_basis', 'SOURCE_DOCUMENT_EFFECTIVE_DATE',
    'effective_from_dataset_revision', CAST(:dataset_revision AS text)
  ),
  updated_at=now()
FROM documents d
WHERE d.id=r.source_document_id
  AND r.relationship_type IN ('AMENDS','SUPPLEMENTS','REPEALS','REPLACES')
  AND d.effective_date IS NOT NULL
  AND (
    r.effective_from IS DISTINCT FROM d.effective_date
    OR r.metadata->>'effective_from_dataset_revision'
       IS DISTINCT FROM CAST(:dataset_revision AS text)
  )
"""

AMENDMENT_EVENTS_SQL = """
INSERT INTO amendment_events (
  event_key, source_document_id, target_document_id, target_external_id,
  source_version_id, source_relationship_id, event_type, effective_date,
  source_reference, metadata
)
SELECT 'vbpl-relationship:' || r.id::text,
       r.source_document_id, r.target_document_id, r.target_external_id,
       r.source_version_id, r.id, r.relationship_type, d.effective_date,
       CAST(:source_reference AS text),
       jsonb_build_object(
         'source_kind', 'VBPL_SNAPSHOT',
         'dataset_revision', CAST(:dataset_revision AS text),
         'date_basis', CASE WHEN d.effective_date IS NULL
                            THEN 'UNAVAILABLE'
                            ELSE 'SOURCE_DOCUMENT_EFFECTIVE_DATE' END,
         'is_trusted', true
       )
FROM document_relationships r
JOIN documents d ON d.id=r.source_document_id
WHERE r.relationship_type IN ('AMENDS','SUPPLEMENTS','REPEALS','REPLACES')
ON CONFLICT (source_relationship_id) DO UPDATE SET
  target_document_id=EXCLUDED.target_document_id,
  target_external_id=EXCLUDED.target_external_id,
  source_version_id=EXCLUDED.source_version_id,
  event_type=EXCLUDED.event_type,
  effective_date=EXCLUDED.effective_date,
  source_reference=EXCLUDED.source_reference,
  metadata=amendment_events.metadata || EXCLUDED.metadata,
  updated_at=now()
WHERE amendment_events.target_document_id IS DISTINCT FROM EXCLUDED.target_document_id
   OR amendment_events.target_external_id IS DISTINCT FROM EXCLUDED.target_external_id
   OR amendment_events.source_version_id IS DISTINCT FROM EXCLUDED.source_version_id
   OR amendment_events.event_type IS DISTINCT FROM EXCLUDED.event_type
   OR amendment_events.effective_date IS DISTINCT FROM EXCLUDED.effective_date
   OR amendment_events.source_reference IS DISTINCT FROM EXCLUDED.source_reference
   OR amendment_events.metadata->>'dataset_revision'
      IS DISTINCT FROM EXCLUDED.metadata->>'dataset_revision'
"""

PROVISION_STATUS_SQL = """
WITH provision_keys AS (
  SELECT DISTINCT c.document_id, v.id AS version_id,
    concat(
      'article:', lower(btrim(coalesce(c.metadata->>'article', a.article_number, ''))),
      '/clause:', lower(btrim(coalesce(c.metadata->>'clause', ''))),
      '/point:', lower(btrim(coalesce(c.metadata->>'point', '')))
    ) AS provision_key,
    nullif(coalesce(c.metadata->>'article', a.article_number), '') AS article,
    nullif(c.metadata->>'clause', '') AS clause,
    nullif(c.metadata->>'point', '') AS point
  FROM chunks c
  JOIN document_versions v ON v.id=c.version_id AND v.is_current
  JOIN documents d ON d.id=c.document_id
  LEFT JOIN articles a ON a.id=c.article_id
  WHERE c.is_indexable AND d.status='PARTIALLY_EFFECTIVE'
    AND nullif(coalesce(c.metadata->>'article', a.article_number), '') IS NOT NULL
)
INSERT INTO provision_effective_status (
  document_id, source_version_id, provision_key, article, clause, point,
  status, valid_from, valid_to, reason, source_url, metadata
)
SELECT p.document_id, p.version_id, p.provision_key, p.article, p.clause, p.point,
       'UNKNOWN', CAST(:snapshot_date AS date), NULL,
       'vbpl_snapshot_has_document_level_partial_status_only',
       CAST(:source_reference AS text),
       jsonb_build_object(
         'source_kind', 'VBPL_SNAPSHOT',
         'dataset_revision', CAST(:dataset_revision AS text),
         'resolution', 'PROVISION_STATUS_UNAVAILABLE',
         'fallback_scope', 'DOCUMENT',
         'is_trusted', true
       )
FROM provision_keys p
WHERE NOT EXISTS (
  SELECT 1 FROM provision_effective_status existing
  WHERE existing.document_id=p.document_id
    AND existing.provision_key=p.provision_key
    AND existing.metadata->>'dataset_revision'=CAST(:dataset_revision AS text)
)
"""

VERSION_PROVENANCE_SQL = """
INSERT INTO provenance_records (
  document_id, crawl_run_id, entity_type, entity_id, source_kind,
  source_reference, dataset_revision, publisher, retrieved_at,
  source_revision, content_hash, is_official, verification_status, metadata
)
SELECT v.document_id, v.crawl_run_id, 'DOCUMENT_VERSION', v.id,
       'VBPL_SNAPSHOT', CAST(:source_reference AS text),
       CAST(:dataset_revision AS text), 'Bộ Tư pháp - VBPL',
       CAST(:retrieved_at AS timestamptz), CAST(:dataset_revision AS text),
       v.content_hash, false, 'TRUSTED_SNAPSHOT',
       jsonb_build_object(
         'source_snapshot_period', '2026-07',
         'artifact_scope', 'PARQUET_DATASET',
         'is_trusted', true
       )
FROM document_versions v
WHERE v.is_current
ON CONFLICT ON CONSTRAINT uq_provenance_records_source DO UPDATE SET
  crawl_run_id=EXCLUDED.crawl_run_id,
  retrieved_at=EXCLUDED.retrieved_at,
  content_hash=EXCLUDED.content_hash,
  verification_status=EXCLUDED.verification_status,
  metadata=provenance_records.metadata || EXCLUDED.metadata,
  updated_at=now()
WHERE provenance_records.crawl_run_id IS DISTINCT FROM EXCLUDED.crawl_run_id
   OR provenance_records.retrieved_at IS DISTINCT FROM EXCLUDED.retrieved_at
   OR provenance_records.content_hash IS DISTINCT FROM EXCLUDED.content_hash
   OR provenance_records.verification_status IS DISTINCT FROM EXCLUDED.verification_status
   OR provenance_records.metadata IS DISTINCT FROM
      (provenance_records.metadata || EXCLUDED.metadata)
"""

HISTORICAL_AVAILABILITY_SQL = """
UPDATE document_versions SET
  metadata=metadata || jsonb_build_object(
    'dataset_revision', CAST(:dataset_revision AS text),
    'content_scope', 'CURRENT_SNAPSHOT_ONLY',
    'historical_content_available', false,
    'historical_unavailable_reason', 'NO_MULTIPLE_TEXTUAL_VERSIONS_IN_VBPL_SNAPSHOT'
  ),
  updated_at=now()
WHERE is_current
  AND metadata->>'historical_content_available' IS DISTINCT FROM 'false'
"""

AUDIT_SQL = """
SELECT
  (SELECT count(*) FROM documents WHERE status='PARTIALLY_EFFECTIVE') AS partial_documents,
  (SELECT count(*) FROM provision_effective_status
    WHERE metadata->>'dataset_revision'=CAST(:dataset_revision AS text)) AS provision_statuses,
  (SELECT count(*) FROM document_relationships) AS graph_relationships,
  (SELECT count(*) FROM document_relationships
    WHERE metadata->>'dataset_revision'=CAST(:dataset_revision AS text)
      AND COALESCE((metadata->>'is_trusted')::boolean, false)=true) AS trusted_graph_relationships,
  (SELECT count(*) FROM amendment_events
    WHERE metadata->>'dataset_revision'=CAST(:dataset_revision AS text)) AS amendment_events,
  (SELECT count(*) FROM document_relationships
    WHERE relationship_type IN ('AMENDS','SUPPLEMENTS','REPEALS','REPLACES')
      AND effective_from IS NOT NULL) AS dated_amendment_relationships,
  (SELECT count(*) FROM provenance_records
    WHERE dataset_revision=CAST(:dataset_revision AS text)) AS provenance_records,
  (SELECT count(*) FROM document_versions WHERE content_valid_period IS NOT NULL) AS historical_versions,
  (SELECT count(*) FROM document_versions WHERE NOT is_current) AS non_current_versions
"""


def load_official_main() -> None:
    parser = argparse.ArgumentParser(
        description="Load audited trusted provenance, status and graph data."
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        report = OfficialEnrichmentLoader(engine).load(args.input)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


def load_provision_status_main() -> None:
    parser = argparse.ArgumentParser(
        description="Load curated provision-level valid-time statuses."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("input must be a JSON array")
    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        with Session(engine) as session:
            objects = [_to_status(session, row) for row in rows]
            if args.dry_run:
                print(json.dumps({"validated": len(objects), "dry_run": True}))
                return
            session.add_all(objects)
            session.commit()
            print(json.dumps({"inserted": len(objects), "dry_run": False}))
    finally:
        engine.dispose()


def _to_status(session: Session, row: dict) -> ProvisionEffectiveStatus:
    external_id = str(row["document_id"])
    document = session.scalar(
        select(Document).where(Document.external_id == external_id)
    )
    if document is None:
        raise ValueError(f"Unknown document_id: {external_id}")
    status = str(row["status"]).upper()
    if status not in {item.value for item in LegalStatus}:
        raise ValueError(f"Unknown legal status: {status}")
    article = str(row["article"]).strip()
    clause = _optional(row.get("clause"))
    point = _optional(row.get("point"))
    return ProvisionEffectiveStatus(
        document_id=document.id,
        provision_key=provision_key(article, clause, point),
        article=article,
        clause=clause,
        point=point,
        status=status,
        valid_from=date.fromisoformat(row["valid_from"]),
        valid_to=date.fromisoformat(row["valid_to"]) if row.get("valid_to") else None,
        reason=row.get("reason"),
        source_url=row.get("source_url"),
        metadata_json={"source": row.get("source") or "curated"},
    )


def _optional(value) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def audit_provision_status_main() -> None:
    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        with engine.connect() as connection:
            row = connection.execute(text(PROVISION_AUDIT_SQL)).mappings().one()
        total = int(row["total_current_provisions"])
        curated = int(row["curated_provisions"])
        report = {
            **{key: int(value) for key, value in row.items()},
            "coverage_rate": curated / total if total else 0.0,
            "policy": (
                "Missing provision status is never inferred. Runtime falls back "
                "to document status and requires an explicit limitation."
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


PROVISION_AUDIT_SQL = """
WITH current_provisions AS (
  SELECT DISTINCT
    c.document_id,
    concat(
      'article:', lower(btrim(coalesce(c.metadata->>'article', a.article_number, ''))),
      '/clause:', lower(btrim(coalesce(c.metadata->>'clause', ''))),
      '/point:', lower(btrim(coalesce(c.metadata->>'point', '')))
    ) AS provision_key,
    d.status AS document_status
  FROM chunks c
  JOIN document_versions v ON v.id = c.version_id
  JOIN documents d ON d.id = c.document_id
  LEFT JOIN articles a ON a.id = c.article_id
  WHERE v.is_current AND c.is_indexable
    AND coalesce(c.metadata->>'article', a.article_number) IS NOT NULL
), curated AS (
  SELECT DISTINCT document_id, provision_key FROM provision_effective_status
)
SELECT
  count(*) AS total_current_provisions,
  count(*) FILTER (WHERE cp.document_status = 'PARTIALLY_EFFECTIVE')
    AS partially_effective_document_provisions,
  count(*) FILTER (WHERE curated.document_id IS NOT NULL) AS curated_provisions,
  count(*) FILTER (
    WHERE cp.document_status = 'PARTIALLY_EFFECTIVE' AND curated.document_id IS NULL
  ) AS high_priority_missing
FROM current_provisions cp
LEFT JOIN curated USING (document_id, provision_key)
"""


def main() -> None:
    dispatch(
        "Manage legal metadata and enrichment.",
        {
            "enrich-vbpl": enrich_vbpl_main,
            "load-official": load_official_main,
            "load-provision-status": load_provision_status_main,
            "audit-provision-status": audit_provision_status_main,
        },
    )


if __name__ == "__main__":
    main()
