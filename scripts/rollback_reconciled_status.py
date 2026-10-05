#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import text

from database import DatabaseSettings, create_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rollback")


def sync_qdrant_status(
    client: QdrantClient,
    collection_name: str,
    external_ids: Sequence[str],
    target_status: str,
    batch_size: int = 100,
) -> int:
    if not external_ids:
        return 0
    total = len(external_ids)
    batches = (total + batch_size - 1) // batch_size
    updated = 0
    for b_idx in range(batches):
        batch = external_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
        try:
            client.set_payload(
                collection_name=collection_name,
                payload={"status": target_status},
                points=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="doc_id",
                            match=qmodels.MatchAny(any=list(batch)),
                        )
                    ]
                ),
                wait=False,
            )
            updated += len(batch)
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == batches:
                logger.info(
                    "Qdrant payload sync [%s]: batch %d/%d (%d/%d documents dispatched)",
                    target_status,
                    b_idx + 1,
                    batches,
                    updated,
                    total,
                )
        except Exception as exc:
            logger.error("Failed to update Qdrant batch %d: %s", b_idx, exc)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback erroneous status reconciliation in PostgreSQL and Qdrant."
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("QDRANT_COLLECTION", "legal_chunks_bge_m3_1024_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())

    qclient = None
    try:
        qclient = QdrantClient(url=args.qdrant_url, timeout=60.0)
    except Exception as exc:
        logger.warning("Failed to connect to Qdrant: %s", exc)

    with engine.begin() as conn:
        logger.info("Scanning for documents marked by reconciliation...")
        rows = conn.execute(
            text(
                """
                SELECT external_id, metadata->>'previous_status' as prev_status
                FROM documents
                WHERE metadata->>'status_reconciled' = 'true'
                """
            )
        ).fetchall()

        effective_ids = [str(r[0]) for r in rows if r[1] == "EFFECTIVE"]
        partially_effective_ids = [str(r[0]) for r in rows if r[1] == "PARTIALLY_EFFECTIVE"]

        logger.info(
            "Found %d documents to restore (%d EFFECTIVE, %d PARTIALLY_EFFECTIVE).",
            len(rows),
            len(effective_ids),
            len(partially_effective_ids),
        )

        if not rows:
            logger.info("No documents need rollback. Exiting.")
            return

        # 1. Clean effective_status table
        del_res = conn.execute(
            text(
                """
                DELETE FROM effective_status
                WHERE metadata->>'importer' = 'status_reconciliation_v1'
                """
            )
        )
        logger.info("Deleted %d erroneous rows from effective_status.", del_res.rowcount)

        trunc_res = conn.execute(
            text(
                """
                UPDATE effective_status
                SET valid_to = NULL,
                    metadata = metadata - 'truncated_by_reconciliation',
                    updated_at = now()
                WHERE metadata->>'truncated_by_reconciliation' = 'true'
                """
            )
        )
        logger.info("Untruncated %d rows in effective_status.", trunc_res.rowcount)

        # 2. Restore documents table
        if effective_ids:
            eff_res = conn.execute(
                text(
                    """
                    UPDATE documents
                    SET status = 'EFFECTIVE',
                        expiry_date = NULL,
                        metadata = (metadata - 'status_reconciled' - 'reconcile_reason' - 'reconciled_at')
                                   || jsonb_build_object('status_rollback', true, 'rollback_at', now()),
                        updated_at = now()
                    WHERE metadata->>'status_reconciled' = 'true'
                      AND metadata->>'previous_status' = 'EFFECTIVE'
                    """
                )
            )
            logger.info("Restored %d documents to EFFECTIVE.", eff_res.rowcount)

        if partially_effective_ids:
            part_res = conn.execute(
                text(
                    """
                    UPDATE documents
                    SET status = 'PARTIALLY_EFFECTIVE',
                        expiry_date = NULL,
                        metadata = (metadata - 'status_reconciled' - 'reconcile_reason' - 'reconciled_at')
                                   || jsonb_build_object('status_rollback', true, 'rollback_at', now()),
                        updated_at = now()
                    WHERE metadata->>'status_reconciled' = 'true'
                      AND metadata->>'previous_status' = 'PARTIALLY_EFFECTIVE'
                    """
                )
            )
            logger.info("Restored %d documents to PARTIALLY_EFFECTIVE.", part_res.rowcount)

    # 3. Synchronize Qdrant
    if qclient is not None:
        logger.info("Synchronizing payload to Qdrant collection %s...", args.qdrant_collection)
        if effective_ids:
            synced_eff = sync_qdrant_status(
                qclient,
                args.qdrant_collection,
                effective_ids,
                "EFFECTIVE",
                batch_size=args.batch_size,
            )
            logger.info("Dispatched Qdrant sync for %d EFFECTIVE documents.", synced_eff)

        if partially_effective_ids:
            synced_part = sync_qdrant_status(
                qclient,
                args.qdrant_collection,
                partially_effective_ids,
                "PARTIALLY_EFFECTIVE",
                batch_size=args.batch_size,
            )
            logger.info("Dispatched Qdrant sync for %d PARTIALLY_EFFECTIVE documents.", synced_part)

    logger.info("Rollback and synchronization completed successfully!")


if __name__ == "__main__":
    main()
