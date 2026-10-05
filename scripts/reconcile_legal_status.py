#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date

from qdrant_client import QdrantClient

from database import DatabaseSettings, StatusReconciler, create_db_engine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Reconcile stale legal statuses (EFFECTIVE/PARTIALLY_EFFECTIVE -> EXPIRED)."
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Target date for status validity check (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate reconciliation and print targets without modifying data",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip updating Qdrant vector payloads",
    )
    parser.add_argument(
        "--sync-all-to-qdrant",
        action="store_true",
        help="Sync all previously reconciled documents in Postgres to Qdrant",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant service URL",
    )
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("QDRANT_COLLECTION", "legal_chunks_bge_m3_1024_v1"),
        help="Target Qdrant collection name",
    )
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())

    qdrant_client = None
    if not args.skip_qdrant and not args.dry_run:
        try:
            qdrant_client = QdrantClient(
                url=args.qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY") or None,
                timeout=60.0,
            )
        except Exception as exc:
            print(f"Warning: Failed to initialize QdrantClient ({exc}), proceeding without vector sync.")

    reconciler = StatusReconciler(
        engine,
        qdrant_client=qdrant_client,
        qdrant_collection=args.qdrant_collection,
    )

    try:
        if args.sync_all_to_qdrant:
            if qdrant_client is None:
                raise SystemExit("QdrantClient is required for --sync-all-to-qdrant")
            synced = reconciler.sync_all_reconciled_to_qdrant()
            print(json.dumps({"sync_all_to_qdrant": True, "documents_synced": synced}, indent=2))
        else:
            report = reconciler.reconcile(
                as_of=args.as_of,
                dry_run=args.dry_run,
                sync_qdrant=not args.skip_qdrant and qdrant_client is not None,
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
