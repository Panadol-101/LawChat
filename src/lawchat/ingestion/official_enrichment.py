from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text


@dataclass(slots=True)
class OfficialEnrichmentReport:
    counters: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {"counters": dict(sorted(self.counters.items()))}


class OfficialEnrichmentLoader:
    """Load trusted snapshot temporal and graph assertions from JSON."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load(self, path: Path) -> OfficialEnrichmentReport:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("official enrichment input must be a JSON object")
        report = OfficialEnrichmentReport()
        stages = (
            ("documents", DOCUMENT_OFFICIAL_SQL),
            ("effective_statuses", EFFECTIVE_STATUS_SQL),
            ("provision_statuses", PROVISION_STATUS_SQL),
            ("relationships", RELATIONSHIP_SQL),
            ("amendment_events", AMENDMENT_EVENT_SQL),
            ("provenance", PROVENANCE_SQL),
        )
        with self.engine.begin() as connection:
            for stage, statement in stages:
                for row in payload.get(stage, ()):
                    self._require_source(row)
                    parameters = _json_params(row)
                    if stage == "amendment_events":
                        parameters["event_key"] = row.get("event_key") or _event_key(row)
                    if stage == "effective_statuses":
                        connection.execute(text(DELETE_DERIVED_STATUS_SQL), parameters)
                    elif stage == "provision_statuses":
                        connection.execute(text(DELETE_DERIVED_PROVISION_STATUS_SQL), parameters)
                    result = connection.execute(text(statement), parameters)
                    if stage == "documents" and result.rowcount != 1:
                        raise ValueError(f"unknown document: {row.get('external_id')}")
                    report.counters[stage] += 1
        return report

    @staticmethod
    def _require_source(row: Any) -> None:
        if not isinstance(row, dict):
            raise ValueError("each enrichment row must be an object")
        reference = str(
            row.get("source_reference")
            or row.get("official_source_url")
            or row.get("source_url")
            or ""
        )
        if not reference:
            raise ValueError("trusted enrichment rows require a source reference")


def _event_key(row: dict[str, Any]) -> str:
    identity = "|".join(
        str(row.get(key) or "")
        for key in (
            "source_external_id", "target_external_id", "event_type",
            "effective_date", "source_article", "target_provision_key",
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _json_params(row: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "title": None, "document_number": None, "document_type": None,
        "authority": None, "issued_date": None, "effective_date": None,
        "expiry_date": None, "status": None, "valid_from": None,
        "valid_to": None, "reason": None, "source_revision": None,
        "provision_key": None, "article": None, "clause": None,
        "point": None, "relationship_type": None, "source_relationship": None,
        "target_external_id": None, "citation_text": None,
        "source_article": None, "target_article": None,
        "effective_from": None, "effective_to": None,
        "target_provision_key": None, "entity_type": None, "entity_id": None,
        "source_kind": None, "source_url": None, "publisher": None,
        "retrieved_at": None, "content_hash": None, "verification_status": None,
        "crawl_run_id": None, "dataset_revision": None,
    }
    result = {**defaults, **row}
    official_url = row.get("official_source_url") or row.get("source_url")
    source_reference = row.get("source_reference") or official_url
    result["official_source_url"] = official_url
    result["source_reference"] = source_reference
    result["dataset_revision"] = row.get("dataset_revision") or "unspecified-snapshot"
    result["metadata"] = json.dumps(
        {**(row.get("metadata") or {}), "is_trusted": True,
         "source_reference": source_reference},
        ensure_ascii=False,
    )
    return result


DOCUMENT_OFFICIAL_SQL = """
UPDATE documents SET
  title=COALESCE(:title, title), document_number=COALESCE(:document_number, document_number),
  document_type=COALESCE(:document_type, document_type), authority=COALESCE(:authority, authority),
  issued_date=COALESCE(CAST(:issued_date AS date), issued_date),
  effective_date=COALESCE(CAST(:effective_date AS date), effective_date),
  expiry_date=COALESCE(CAST(:expiry_date AS date), expiry_date),
  status=COALESCE(:status, status),
  source_url=COALESCE(:official_source_url, :source_reference, source_url),
  metadata=metadata || CAST(:metadata AS jsonb)
           || jsonb_build_object('trusted_snapshot', true),
  updated_at=now()
WHERE external_id=:external_id
"""

DELETE_DERIVED_STATUS_SQL = """
DELETE FROM effective_status x USING documents d
WHERE x.document_id=d.id AND d.external_id=:document_external_id
  AND x.valid_period && daterange(CAST(:valid_from AS date), CAST(:valid_to AS date), '[)')
  AND COALESCE((x.metadata->>'is_official')::boolean, false)=false
  AND COALESCE((x.metadata->>'is_trusted')::boolean, false)=false
"""

DELETE_DERIVED_PROVISION_STATUS_SQL = """
DELETE FROM provision_effective_status x USING documents d
WHERE x.document_id=d.id AND d.external_id=:document_external_id
  AND x.provision_key=:provision_key
  AND x.valid_period && daterange(CAST(:valid_from AS date), CAST(:valid_to AS date), '[)')
  AND COALESCE((x.metadata->>'is_official')::boolean, false)=false
  AND COALESCE((x.metadata->>'is_trusted')::boolean, false)=false
"""

EFFECTIVE_STATUS_SQL = """
INSERT INTO effective_status (document_id, source_version_id, status, valid_from,
 valid_to, reason, source_url, metadata)
SELECT d.id, v.id, CAST(:status AS varchar), CAST(:valid_from AS date), CAST(:valid_to AS date),
 CAST(:reason AS text), CAST(COALESCE(:official_source_url, :source_reference) AS text),
 CAST(:metadata AS jsonb)
FROM documents d LEFT JOIN document_versions v ON v.document_id=d.id
 AND v.source_revision=CAST(:source_revision AS varchar)
WHERE d.external_id=CAST(:document_external_id AS varchar) AND NOT EXISTS (
 SELECT 1 FROM effective_status x WHERE x.document_id=d.id
 AND x.status=CAST(:status AS varchar)
 AND x.valid_from=CAST(:valid_from AS date)
 AND x.valid_to IS NOT DISTINCT FROM CAST(:valid_to AS date)
 AND x.source_url=CAST(COALESCE(:official_source_url, :source_reference) AS text))
"""

PROVISION_STATUS_SQL = """
INSERT INTO provision_effective_status (document_id, source_version_id,
 provision_key, article, clause, point, status, valid_from, valid_to, reason,
 source_url, metadata)
SELECT d.id, v.id, CAST(:provision_key AS varchar), CAST(:article AS varchar),
 CAST(:clause AS varchar), CAST(:point AS varchar), CAST(:status AS varchar),
 CAST(:valid_from AS date), CAST(:valid_to AS date), CAST(:reason AS text),
 CAST(COALESCE(:official_source_url, :source_reference) AS text), CAST(:metadata AS jsonb)
FROM documents d LEFT JOIN document_versions v ON v.document_id=d.id
 AND v.source_revision=CAST(:source_revision AS varchar)
WHERE d.external_id=CAST(:document_external_id AS varchar) AND NOT EXISTS (
 SELECT 1 FROM provision_effective_status x WHERE x.document_id=d.id
 AND x.provision_key=CAST(:provision_key AS varchar)
 AND x.status=CAST(:status AS varchar)
 AND x.valid_from=CAST(:valid_from AS date)
 AND x.valid_to IS NOT DISTINCT FROM CAST(:valid_to AS date)
 AND x.source_url=CAST(COALESCE(:official_source_url, :source_reference) AS text))
"""

RELATIONSHIP_SQL = """
INSERT INTO document_relationships (source_document_id, target_document_id,
 source_version_id, relationship_type, source_relationship, target_external_id,
 citation_text, source_article, target_article, effective_from, effective_to, metadata)
SELECT source.id, target.id, version.id, :relationship_type, :source_relationship,
 :target_external_id, :citation_text, :source_article, :target_article,
 CAST(:effective_from AS date), CAST(:effective_to AS date), CAST(:metadata AS jsonb)
FROM documents source
LEFT JOIN documents target ON target.external_id=:target_external_id
LEFT JOIN document_versions version ON version.document_id=source.id
 AND version.source_revision=:source_revision
WHERE source.external_id=:source_external_id
ON CONFLICT ON CONSTRAINT uq_document_relationships_source_edge DO UPDATE SET
 target_document_id=EXCLUDED.target_document_id, source_version_id=EXCLUDED.source_version_id,
 relationship_type=EXCLUDED.relationship_type, citation_text=EXCLUDED.citation_text,
 source_article=EXCLUDED.source_article, target_article=EXCLUDED.target_article,
 effective_from=EXCLUDED.effective_from, effective_to=EXCLUDED.effective_to,
 metadata=document_relationships.metadata || EXCLUDED.metadata, updated_at=now()
"""

AMENDMENT_EVENT_SQL = """
INSERT INTO amendment_events (event_key, source_document_id, target_document_id,
 source_version_id, source_relationship_id, target_external_id, event_type,
 effective_date, source_article, target_provision_key, citation_text,
 source_reference, metadata)
SELECT :event_key, source.id, target.id, version.id, NULL, :target_external_id,
 :event_type,
 CAST(:effective_date AS date), :source_article, :target_provision_key,
 :citation_text, :source_reference, CAST(:metadata AS jsonb)
FROM documents source LEFT JOIN documents target ON target.external_id=:target_external_id
LEFT JOIN document_versions version ON version.document_id=source.id
 AND version.source_revision=:source_revision
WHERE source.external_id=:source_external_id
ON CONFLICT (event_key) DO UPDATE SET citation_text=EXCLUDED.citation_text,
 source_reference=EXCLUDED.source_reference,
 metadata=amendment_events.metadata || EXCLUDED.metadata, updated_at=now()
"""

PROVENANCE_SQL = """
INSERT INTO provenance_records (document_id, crawl_run_id, entity_type, entity_id,
 source_kind, source_reference, dataset_revision, publisher, retrieved_at,
 source_revision, content_hash, is_official, verification_status, metadata)
SELECT d.id, CAST(:crawl_run_id AS uuid), :entity_type, CAST(:entity_id AS uuid),
 :source_kind, :source_reference, :dataset_revision, :publisher,
 CAST(:retrieved_at AS timestamptz), COALESCE(:source_revision, ''),
 :content_hash, true, COALESCE(:verification_status, 'VERIFIED'), CAST(:metadata AS jsonb)
FROM documents d WHERE d.external_id=:document_external_id
ON CONFLICT ON CONSTRAINT uq_provenance_records_source DO UPDATE SET
 publisher=EXCLUDED.publisher, retrieved_at=EXCLUDED.retrieved_at,
 content_hash=EXCLUDED.content_hash, is_official=true,
 verification_status=EXCLUDED.verification_status,
 metadata=provenance_records.metadata || EXCLUDED.metadata, updated_at=now()
"""
