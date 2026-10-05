#!/usr/bin/env python3
"""Expire documents only where the replacing text says the whole document goes.

``document_relationships`` marks REPLACES/REPEALS edges without telling a
whole-document repeal from a partial one ("bãi bỏ các điều 223..." or
"Luật X đã được sửa đổi theo Luật số Y hết hiệu lực" are both stored as edges
to Y). A previous reconciliation that trusted every edge was rolled back.

This script reads the source document's own text and accepts an edge only
when a sentence there replaces or ends the target as a whole, e.g.
"Nghị định này thay thế Nghị định số 167/2013/NĐ-CP" or
"Nghị định số 167/2013/NĐ-CP ... hết hiệu lực". Edges without such a
sentence are left alone. Without ``--apply`` nothing is written.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import text

from database import DatabaseSettings, StatusReconciler, create_db_engine
from database.reconciliation import (
    DELETE_FUTURE_ACTIVE_STATUS_SQL,
    INSERT_EXPIRED_STATUS_SQL,
    TRUNCATE_OVERLAPPING_STATUS_SQL,
    UPDATE_DOCUMENTS_SQL,
)
from indexing import QdrantSettings

logger = logging.getLogger("reconcile_evidence")

CANDIDATE_EDGES_SQL = """
SELECT
    t.id AS target_id,
    t.external_id AS target_external_id,
    t.document_number AS target_number,
    t.status AS target_status,
    s.id AS source_id,
    s.document_number AS source_number,
    r.relationship_type,
    COALESCE(r.effective_from, s.effective_date, s.issued_date) AS event_date
FROM document_relationships r
JOIN documents s ON s.id = r.source_document_id
JOIN documents t ON t.id = r.target_document_id
WHERE r.relationship_type IN ('REPLACES', 'REPEALS')
  AND r.target_article IS NULL
  AND t.status IN ('EFFECTIVE', 'PARTIALLY_EFFECTIVE')
  AND t.document_number IS NOT NULL
  AND s.status <> 'NOT_YET_EFFECTIVE'
  AND COALESCE(r.effective_from, s.effective_date, s.issued_date) <= :as_of
  AND (t.effective_date IS NULL
       OR COALESCE(r.effective_from, s.effective_date, s.issued_date) >= t.effective_date)
  AND (s.issued_date IS NULL OR t.issued_date IS NULL OR s.issued_date >= t.issued_date)
"""

SOURCE_TEXT_SQL = """
SELECT text_content
FROM chunks
WHERE document_id = :source_id
  AND text_content ILIKE :needle
"""

_DOC_TYPE = (
    r"(?:Bộ\s+luật|Luật|Pháp\s+lệnh|Nghị\s+định|Nghị\s+quyết|"
    r"Thông\s+tư\s+liên\s+tịch|Thông\s+tư|Quyết\s+định|Chỉ\s+thị|Văn\s+bản)"
)
# Text just before the document mention that makes it a partial reference.
_PARTIAL_CONTEXT_RE = re.compile(
    r"(?:điều|khoản|điểm|chương|mục|phần|phụ\s+lục|quy\s+định\s+tại|nội\s+dung)"
    r"[^.;]{0,40}$",
    re.IGNORECASE,
)


_LIST_KIND = (
    r"(?:văn\s+bản|luật|pháp\s+lệnh|nghị\s+định|nghị\s+quyết|thông\s+tư(?:\s+liên\s+tịch)?|"
    r"quyết\s+định|chỉ\s+thị)"
)
# Header of an enumerated list of documents that end as a whole, e.g.
# "các nghị quyết sau đây hết hiệu lực:" or "thay thế các Quyết định sau:".
# "bãi bỏ nội dung tại các Thông tư sau:" deliberately does not match.
_LIST_HEADER_RE = re.compile(
    rf"(?:(?:các|những)\s+{_LIST_KIND}[^.:;]{{0,80}}?sau(?:\s+đây)?[^.:;]{{0,40}}?"
    rf"(?:hết\s+hiệu\s+lực|bị\s+bãi\s+bỏ|bị\s+thay\s+thế)[^.:;]{{0,60}}?"
    rf"|(?:thay\s+thế|bãi\s+bỏ(?:\s+toàn\s+bộ)?)\s+(?:các|những)\s+{_LIST_KIND}"
    rf"(?:\s+sau(?:\s+đây)?)?(?:[^.:;\d]{{0,40}})?)\s*:",
    re.IGNORECASE,
)
_NEXT_ARTICLE_RE = re.compile(r"\bĐiều\s+\d+\s*\.")
_ITEM_START_RE = re.compile(
    r"(?:^|[:;]|[a-zđ]\)|\d+\.|-|\+|\bvà)\s*$", re.IGNORECASE
)


def _item_piece_re() -> re.Pattern[str]:
    return re.compile(
        rf"\s*(?:(?:[a-zđ]\)|\d+\.?|-|\+)\s*)?{_DOC_TYPE}", re.IGNORECASE
    )


def _listed_as_whole(text_value: str, mention: str) -> str | None:
    """Snippet when the target is an item of a whole-document repeal list."""
    mention_re = re.compile(mention, re.IGNORECASE)
    for header in _LIST_HEADER_RE.finditer(text_value):
        body_start = header.end()
        next_article = _NEXT_ARTICLE_RE.search(text_value, body_start)
        body_end = min(next_article.start() if next_article else len(text_value), body_start + 4000)
        for match in mention_re.finditer(text_value, body_start, body_end):
            before = text_value[max(body_start, match.start() - 60):match.start()]
            if _PARTIAL_CONTEXT_RE.search(before):
                continue
            # The target must open a list item ("a) Nghị định số X",
            # "; Thông tư số X", "... và Nghị quyết số X") ...
            if not _ITEM_START_RE.search(text_value[body_start:match.start()][-15:] or ":"):
                continue
            # ... and the list must run unbroken from the header to it: every
            # sentence after the header opens with another listed document.
            raw_between = text_value[body_start:match.start()]
            # In an "a), b)" list, a numbered "3." opens the next clause.
            if re.match(r"\s*[a-zđ]\)", raw_between, re.IGNORECASE) and re.search(
                r"(?:^|\s)\d+\.\s", raw_between
            ):
                continue
            # Drop item numbers ("1. ", "2. ") so they do not end a sentence.
            between = re.sub(r"(?:(?<=\s)|^)\d+\.\s+", "", text_value[body_start:match.start()])
            if any(
                not _ITEM_PIECE_RE.match(piece)
                for piece in re.split(r"\.\s+", between)[1:]
                if piece.strip()
            ):
                continue
            return text_value[max(0, header.start() - 40):min(match.end() + 40, body_end)]
    return None


_ITEM_PIECE_RE = _item_piece_re()


def _number_pattern(document_number: str) -> str:
    return r"\s*/\s*".join(re.escape(part.strip()) for part in document_number.split("/"))


def classify(source_text: str, target_number: str) -> tuple[str, str]:
    """Return ("whole" | "partial" | "no_evidence", snippet)."""
    text_value = " ".join(source_text.split())
    number = _number_pattern(target_number)
    mention = rf"{_DOC_TYPE}(?:\s+[^\d.;]{{0,120}}?)?\s*(?:số\s*)?{number}(?![\w/])"
    patterns = (
        # "... thay thế / bãi bỏ (toàn bộ) Nghị định số X"
        re.compile(rf"(?:thay\s+thế|bãi\s+bỏ(?:\s+toàn\s+bộ)?)\s+(?:cho\s+)?{mention}", re.IGNORECASE),
        # "Nghị định số X ... hết hiệu lực", no other document number between
        re.compile(
            rf"{mention}(?:(?!\d{{1,4}}\s*/\s*(?:\d{{4}}\s*/\s*)?[A-ZĐ]{{2}})[^.;]){{0,400}}?hết\s+hiệu\s+lực",
            re.IGNORECASE,
        ),
    )
    saw_mention = False
    for pattern in patterns:
        for match in pattern.finditer(text_value):
            saw_mention = True
            before = text_value[max(0, match.start() - 60):match.start()]
            # "theo Luật số X", "bởi Nghị định số X": X is the amending act,
            # not the act that ends.
            if re.search(r"(?:theo|bởi|tại|của)\s*$", before, re.IGNORECASE):
                continue
            if _PARTIAL_CONTEXT_RE.search(before):
                continue
            snippet = text_value[max(0, match.start() - 80):match.end() + 40]
            return "whole", snippet
    listed = _listed_as_whole(text_value, mention)
    if listed is not None:
        return "whole", listed
    if saw_mention or re.search(number, text_value, re.IGNORECASE):
        index = re.search(number, text_value, re.IGNORECASE)
        start = index.start() if index else 0
        return "partial", text_value[max(0, start - 120):start + 120]
    return "no_evidence", ""


@dataclass
class Target:
    document_id: str
    external_id: str
    document_number: str
    old_status: str
    repeal_date: date
    reasons: list[str]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.fromisoformat(os.getenv("LAWCHAT_LEGAL_CUTOFF_DATE") or date.today().isoformat()),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/status_reconciliation_evidence.csv"))
    parser.add_argument("--apply", action="store_true", help="Write the accepted targets")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())
    rows_out: list[dict] = []
    targets: dict[str, Target] = {}
    counts: dict[str, int] = defaultdict(int)
    with engine.connect() as connection:
        edges = connection.execute(text(CANDIDATE_EDGES_SQL), {"as_of": args.as_of}).mappings().all()
        logger.info("candidate edges: %d", len(edges))
        for index, edge in enumerate(edges, start=1):
            chunks = connection.execute(
                text(SOURCE_TEXT_SQL),
                {"source_id": edge["source_id"], "needle": f"%{edge['target_number'].split('/')[0]}/%"},
            ).scalars().all()
            verdict, snippet = "no_evidence", ""
            for chunk in chunks:
                verdict, snippet = classify(chunk or "", edge["target_number"])
                if verdict == "whole":
                    break
            counts[verdict] += 1
            rows_out.append({
                "target_number": edge["target_number"],
                "target_external_id": edge["target_external_id"],
                "target_status": edge["target_status"],
                "source_number": edge["source_number"],
                "relationship": edge["relationship_type"],
                "event_date": edge["event_date"].isoformat(),
                "verdict": verdict,
                "evidence": snippet,
            })
            if verdict == "whole":
                key = str(edge["target_id"])
                reason = f"evidence:{edge['relationship_type']}:{edge['source_number']}"
                existing = targets.get(key)
                if existing is None:
                    targets[key] = Target(
                        key, edge["target_external_id"], edge["target_number"],
                        edge["target_status"], edge["event_date"], [reason],
                    )
                else:
                    existing.repeal_date = min(existing.repeal_date, edge["event_date"])
                    existing.reasons.append(reason)
            if index % 1000 == 0:
                logger.info("classified %d/%d edges %s", index, len(edges), dict(counts))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]) if rows_out else ["verdict"])
        writer.writeheader()
        writer.writerows(rows_out)
    logger.info("edge verdicts: %s", dict(counts))
    logger.info("documents to expire: %d (report: %s)", len(targets), args.report)

    if not args.apply or not targets:
        logger.info("dry run: nothing written")
        return 0

    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TEMP TABLE tmp_reconcile_targets ("
            " document_id uuid, external_id text, document_number text,"
            " old_status text, repeal_date date, reason_summary text) ON COMMIT DROP"
        ))
        connection.execute(
            text("INSERT INTO tmp_reconcile_targets VALUES "
                 "(CAST(:document_id AS uuid), :external_id, :document_number, :old_status, :repeal_date, :reason)"),
            [
                {
                    "document_id": item.document_id,
                    "external_id": item.external_id,
                    "document_number": item.document_number,
                    "old_status": item.old_status,
                    "repeal_date": item.repeal_date,
                    "reason": ", ".join(sorted(set(item.reasons)))[:1000],
                }
                for item in targets.values()
            ],
        )
        for statement in (
            DELETE_FUTURE_ACTIVE_STATUS_SQL,
            TRUNCATE_OVERLAPPING_STATUS_SQL,
            INSERT_EXPIRED_STATUS_SQL,
            UPDATE_DOCUMENTS_SQL,
        ):
            result = connection.execute(text(statement))
            logger.info("%s -> %s rows", statement.split()[0], result.rowcount)

    if not args.skip_qdrant:
        settings = QdrantSettings.from_env()
        reconciler = StatusReconciler(
            engine,
            qdrant_client=settings.create_client(),
            qdrant_collection=settings.collection,
        )
        dispatched = reconciler._sync_qdrant_payload(
            sorted(item.external_id for item in targets.values()), batch_size=200
        )
        logger.info("qdrant payload sync dispatched for %d documents", dispatched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
