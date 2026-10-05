from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import Connection, Engine, text

from database.connection import DatabaseSettings, create_db_engine

logger = logging.getLogger(__name__)

RECONCILE_TARGETS_SQL = """
CREATE TEMP TABLE tmp_reconcile_targets ON COMMIT DROP AS
WITH repeal_events AS (
    -- 1. Events from amendment_events (REPLACES / REPEALS)
    SELECT 
        e.target_document_id AS document_id,
        COALESCE(e.effective_date, s.effective_date, s.issued_date) AS event_date,
        'amendment_event:' || e.event_type AS reason
    FROM amendment_events e
    JOIN documents s ON s.id = e.source_document_id
    WHERE e.event_type IN ('REPLACES', 'REPEALS')
      AND e.target_document_id IS NOT NULL
      AND COALESCE(e.effective_date, s.effective_date, s.issued_date) IS NOT NULL

    UNION ALL

    -- 2. Direct relationships (source repeals/replaces target)
    SELECT 
        r.target_document_id AS document_id,
        COALESCE(r.effective_from, s.effective_date, s.issued_date) AS event_date,
        'relationship:' || r.relationship_type AS reason
    FROM document_relationships r
    JOIN documents s ON s.id = r.source_document_id
    WHERE (
        r.relationship_type IN ('REPLACES', 'REPEALS')
        OR r.source_relationship IN ('Thay thế', 'Bãi bỏ', 'Văn bản hết hiệu lực', 'thay thế', 'bãi bỏ', 'văn bản hết hiệu lực')
    )
    AND r.target_document_id IS NOT NULL
    AND COALESCE(r.effective_from, s.effective_date, s.issued_date) IS NOT NULL

    UNION ALL

    -- 3. Inverted relationships (target repeals source in 'Văn bản quy định hết hiệu lực')
    SELECT 
        r.source_document_id AS document_id,
        COALESCE(t.effective_date, t.issued_date) AS event_date,
        'relationship:REPEALS_INVERTED' AS reason
    FROM document_relationships r
    JOIN documents t ON t.id = r.target_document_id
    WHERE lower(btrim(r.source_relationship)) = 'văn bản quy định hết hiệu lực'
      AND r.source_document_id IS NOT NULL
      AND COALESCE(t.effective_date, t.issued_date) IS NOT NULL

    UNION ALL

    -- 4. Expiry date passed
    SELECT 
        d.id AS document_id,
        d.expiry_date AS event_date,
        'expired_by_expiry_date' AS reason
    FROM documents d
    WHERE d.expiry_date IS NOT NULL
      AND d.expiry_date <= :as_of
      AND d.status IN ('EFFECTIVE', 'PARTIALLY_EFFECTIVE')
),
earliest_repeal AS (
    SELECT 
        re.document_id,
        min(re.event_date) AS repeal_date,
        string_agg(DISTINCT re.reason, ', ') AS reason_summary
    FROM repeal_events re
    JOIN documents d ON d.id = re.document_id
    WHERE re.event_date <= :as_of
      AND (d.effective_date IS NULL OR re.event_date >= d.effective_date)
      AND (d.issued_date IS NULL OR re.event_date >= d.issued_date)
    GROUP BY re.document_id
)
SELECT 
    d.id AS document_id,
    d.external_id,
    d.document_number,
    d.status AS old_status,
    er.repeal_date,
    er.reason_summary
FROM earliest_repeal er
JOIN documents d ON d.id = er.document_id
WHERE d.status IN ('EFFECTIVE', 'PARTIALLY_EFFECTIVE')
  AND er.repeal_date <= :as_of;
"""

AUDIT_TARGETS_SQL = """
SELECT old_status, count(*) AS count
FROM tmp_reconcile_targets
GROUP BY old_status;
"""

FETCH_EXTERNAL_IDS_SQL = """
SELECT external_id
FROM tmp_reconcile_targets
ORDER BY external_id;
"""

SAMPLE_TARGETS_SQL = """
SELECT external_id, document_number, old_status, repeal_date, reason_summary
FROM tmp_reconcile_targets
ORDER BY repeal_date DESC
LIMIT 10;
"""

UPDATE_DOCUMENTS_SQL = """
UPDATE documents d
SET status = 'EXPIRED',
    expiry_date = t.repeal_date,
    metadata = d.metadata || jsonb_build_object(
        'status_reconciled', true,
        'reconciled_at', now(),
        'previous_status', t.old_status,
        'reconcile_reason', t.reason_summary
    ),
    updated_at = now()
FROM tmp_reconcile_targets t
WHERE d.id = t.document_id;
"""

DELETE_FUTURE_ACTIVE_STATUS_SQL = """
DELETE FROM effective_status es
USING tmp_reconcile_targets t
WHERE es.document_id = t.document_id
  AND es.valid_from >= t.repeal_date;
"""

TRUNCATE_OVERLAPPING_STATUS_SQL = """
UPDATE effective_status es
SET valid_to = t.repeal_date,
    metadata = es.metadata || jsonb_build_object('truncated_by_reconciliation', true),
    updated_at = now()
FROM tmp_reconcile_targets t
WHERE es.document_id = t.document_id
  AND es.valid_from < t.repeal_date
  AND (es.valid_to IS NULL OR es.valid_to > t.repeal_date);
"""

INSERT_EXPIRED_STATUS_SQL = """
INSERT INTO effective_status (
    document_id, status, valid_from, valid_to, reason, metadata
)
SELECT 
    t.document_id,
    'EXPIRED',
    t.repeal_date,
    NULL,
    'reconciled_from_repeal_events',
    jsonb_build_object(
        'importer', 'status_reconciliation_v1',
        'previous_status', t.old_status,
        'repeal_date', t.repeal_date,
        'reconcile_reason', t.reason_summary
    )
FROM tmp_reconcile_targets t
WHERE NOT EXISTS (
    SELECT 1 FROM effective_status existing
    WHERE existing.document_id = t.document_id
      AND existing.status = 'EXPIRED'
      AND existing.valid_from = t.repeal_date
);
"""


@dataclass(slots=True)
class ReconciliationReport:
    as_of: str
    dry_run: bool
    total_reconciled: int = 0
    effective_to_expired: int = 0
    partially_effective_to_expired: int = 0
    effective_status_truncated: int = 0
    effective_status_deleted: int = 0
    effective_status_inserted: int = 0
    qdrant_points_updated: int = 0
    sample_documents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "dry_run": self.dry_run,
            "total_reconciled": self.total_reconciled,
            "effective_to_expired": self.effective_to_expired,
            "partially_effective_to_expired": self.partially_effective_to_expired,
            "effective_status_truncated": self.effective_status_truncated,
            "effective_status_deleted": self.effective_status_deleted,
            "effective_status_inserted": self.effective_status_inserted,
            "qdrant_points_updated": self.qdrant_points_updated,
            "sample_documents": self.sample_documents,
        }


class StatusReconciler:
    """Detects and reconciles stale EFFECTIVE and PARTIALLY_EFFECTIVE legal documents."""

    def __init__(
        self,
        engine: Engine,
        *,
        qdrant_client: QdrantClient | None = None,
        qdrant_collection: str = "legal_chunks_bge_m3_1024_v1",
    ) -> None:
        self.engine = engine
        self.qdrant_client = qdrant_client
        self.qdrant_collection = qdrant_collection

    def reconcile(
        self,
        *,
        as_of: date | None = None,
        dry_run: bool = False,
        sync_qdrant: bool = True,
        qdrant_batch_size: int = 500,
    ) -> ReconciliationReport:
        effective_as_of = as_of or date.today()
        report = ReconciliationReport(
            as_of=effective_as_of.isoformat(),
            dry_run=dry_run,
        )

        with self.engine.connect() as connection:
            connection.execute(
                text("SELECT pg_advisory_lock(hashtext('lawchat_status_reconciliation'))")
            )
            connection.commit()
            try:
                with connection.begin():
                    # 1. Build staging table of targets
                    connection.execute(
                        text(RECONCILE_TARGETS_SQL),
                        {"as_of": effective_as_of},
                    )

                    # 2. Audit target breakdown
                    audit_rows = connection.execute(text(AUDIT_TARGETS_SQL)).mappings().all()
                    for row in audit_rows:
                        status = row["old_status"]
                        count = int(row["count"])
                        if status == "EFFECTIVE":
                            report.effective_to_expired = count
                        elif status == "PARTIALLY_EFFECTIVE":
                            report.partially_effective_to_expired = count
                        report.total_reconciled += count

                    # 3. Collect samples
                    samples = connection.execute(text(SAMPLE_TARGETS_SQL)).mappings().all()
                    report.sample_documents = [
                        {
                            "external_id": str(r["external_id"]),
                            "document_number": r["document_number"],
                            "previous_status": r["old_status"],
                            "repeal_date": str(r["repeal_date"]),
                            "reason": r["reason_summary"],
                        }
                        for r in samples
                    ]

                    # 4. Fetch all external_ids for vector sync
                    external_ids = [
                        str(r[0])
                        for r in connection.execute(text(FETCH_EXTERNAL_IDS_SQL)).fetchall()
                    ]

                    if not dry_run and report.total_reconciled > 0:
                        # 5. Apply PostgreSQL updates
                        del_res = connection.execute(text(DELETE_FUTURE_ACTIVE_STATUS_SQL))
                        report.effective_status_deleted = max(int(del_res.rowcount), 0) if isinstance(getattr(del_res, "rowcount", None), int) else 0

                        trunc_res = connection.execute(text(TRUNCATE_OVERLAPPING_STATUS_SQL))
                        report.effective_status_truncated = max(int(trunc_res.rowcount), 0) if isinstance(getattr(trunc_res, "rowcount", None), int) else 0

                        ins_res = connection.execute(text(INSERT_EXPIRED_STATUS_SQL))
                        report.effective_status_inserted = max(int(ins_res.rowcount), 0) if isinstance(getattr(ins_res, "rowcount", None), int) else 0

                        doc_res = connection.execute(text(UPDATE_DOCUMENTS_SQL))
                        # doc_res.rowcount should equal report.total_reconciled

            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtext('lawchat_status_reconciliation'))")
                )
                connection.commit()

        # 6. Sync Qdrant vector payload
        if not dry_run and sync_qdrant and external_ids:
            report.qdrant_points_updated = self._sync_qdrant_payload(
                external_ids, batch_size=qdrant_batch_size
            )

        return report

    def sync_all_reconciled_to_qdrant(self, *, batch_size: int = 100) -> int:
        """Fetch all documents in PostgreSQL marked as status_reconciled and push status: EXPIRED to Qdrant."""
        with self.engine.connect() as connection:
            external_ids = [
                str(r[0])
                for r in connection.execute(
                    text("SELECT external_id FROM documents WHERE metadata->>'status_reconciled' = 'true'")
                ).fetchall()
            ]
        return self._sync_qdrant_payload(external_ids, batch_size=batch_size)

    def _sync_qdrant_payload(
        self,
        external_ids: Sequence[str],
        *,
        batch_size: int = 100,
    ) -> int:
        if self.qdrant_client is None:
            return 0
        try:
            if not self.qdrant_client.collection_exists(self.qdrant_collection):
                logger.warning(
                    "Qdrant collection %s does not exist, skipping payload sync",
                    self.qdrant_collection,
                )
                return 0
        except Exception as exc:
            logger.warning("Failed to check Qdrant collection: %s", exc)
            return 0

        updated_docs = 0
        total_batches = (len(external_ids) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(external_ids), batch_size), start=1):
            chunk_ids = external_ids[i : i + batch_size]
            try:
                self.qdrant_client.set_payload(
                    collection_name=self.qdrant_collection,
                    payload={"status": "EXPIRED"},
                    points=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_id",
                                match=qmodels.MatchAny(any=list(chunk_ids)),
                            )
                        ]
                    ),
                    wait=False,
                )
                updated_docs += len(chunk_ids)
                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    logger.info(
                        "Qdrant payload sync: batch %d/%d (%d/%d documents dispatched)",
                        batch_idx,
                        total_batches,
                        updated_docs,
                        len(external_ids),
                    )
            except Exception as exc:
                logger.error("Failed to update Qdrant batch starting at %d: %s", i, exc)

        return updated_docs


