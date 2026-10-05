#!/usr/bin/env python3
"""Report EXPIRED documents whose only repeal evidence is partial (read-only).

The source snapshot marks some documents EXPIRED from a REPLACES/REPEALS edge
that the replacing text shows to be partial, e.g. Luật BHXH 41/2024/QH15 is
"replaced" by 74/2025/QH15, whose text only ends "Luật Việc làm ... đã được
sửa đổi theo Luật số 41/2024/QH15". This script lists such documents for
human review; it never writes to the database.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

from database import DatabaseSettings, create_db_engine
from reconcile_status_with_evidence import SOURCE_TEXT_SQL, classify

logger = logging.getLogger("audit_expired")

EXPIRED_WITH_EDGES_SQL = """
SELECT
    t.id AS target_id,
    t.external_id AS target_external_id,
    t.document_number AS target_number,
    t.document_type AS target_type,
    t.expiry_date,
    s.id AS source_id,
    s.document_number AS source_number,
    r.relationship_type
FROM documents t
JOIN document_relationships r ON r.target_document_id = t.id
JOIN documents s ON s.id = r.source_document_id
WHERE t.status = 'EXPIRED'
  AND t.document_number IS NOT NULL
  AND r.relationship_type IN ('REPLACES', 'REPEALS')
  AND COALESCE(t.metadata->>'reconcile_reason', '') NOT LIKE 'evidence:%'
  AND (CAST(:types AS text[]) IS NULL OR t.document_type = ANY(CAST(:types AS text[])))
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=Path("reports/expired_status_audit.csv"))
    parser.add_argument(
        "--types",
        default="Luật,Bộ luật,Pháp lệnh,Nghị định,Nghị quyết,Thông tư,Thông tư liên tịch",
        help="Comma-separated document types to audit ('' for all)",
    )
    args = parser.parse_args()
    types = [item.strip() for item in args.types.split(",") if item.strip()] or None

    engine = create_db_engine(DatabaseSettings.from_env())
    verdicts: dict[str, dict] = {}
    with engine.connect() as connection:
        edges = connection.execute(text(EXPIRED_WITH_EDGES_SQL), {"types": types}).mappings().all()
        logger.info("expired documents with repeal edges: %d edges", len(edges))
        for edge in edges:
            chunks = connection.execute(
                text(SOURCE_TEXT_SQL),
                {"source_id": edge["source_id"], "needle": f"%{edge['target_number'].split('/')[0]}/%"},
            ).scalars().all()
            verdict, snippet = "no_evidence", ""
            for chunk in chunks:
                verdict, snippet = classify(chunk or "", edge["target_number"])
                if verdict == "whole":
                    break
            entry = verdicts.setdefault(str(edge["target_id"]), {
                "target_number": edge["target_number"],
                "target_external_id": edge["target_external_id"],
                "target_type": edge["target_type"],
                "expiry_date": edge["expiry_date"].isoformat() if edge["expiry_date"] else "",
                "verdicts": defaultdict(list),
            })
            entry["verdicts"][verdict].append((edge["source_number"], snippet))

    suspicious = [
        entry for entry in verdicts.values()
        if not entry["verdicts"]["whole"] and entry["verdicts"]["partial"]
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_number", "target_external_id", "target_type", "expiry_date", "partial_sources", "evidence"])
        for entry in sorted(suspicious, key=lambda item: (item["target_type"], item["target_number"])):
            sources = entry["verdicts"]["partial"]
            writer.writerow([
                entry["target_number"], entry["target_external_id"], entry["target_type"],
                entry["expiry_date"], "; ".join(source for source, _ in sources), sources[0][1][:400],
            ])
    logger.info(
        "documents: %d audited, %d with whole evidence, %d suspicious (partial only) -> %s",
        len(verdicts), sum(1 for e in verdicts.values() if e["verdicts"]["whole"]), len(suspicious), args.report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
