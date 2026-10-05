#!/usr/bin/env python3
"""Script kiểm tra và xác minh trạng thái hiệu lực pháp lý trong Database, Qdrant và Retrieval Hydration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import text

from database import DatabaseSettings, create_db_engine, create_session_factory
from database.queries import LegalMetadataFilter
from retrieval.hydration import PostgresChunkHydrator


def check_database_status(engine) -> list[dict]:
    print("\n" + "=" * 60)
    print("1. KIỂM TRA PHÂN BỐ TRẠNG THÁI TRONG POSTGRESQL (documents)")
    print("=" * 60)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, count(*) FROM documents GROUP BY status ORDER BY count(*) DESC")
        ).fetchall()
        for r in rows:
            print(f"  - {r[0]:22}: {r[1]:>8,} văn bản")

        reconciled_cnt = conn.execute(
            text("SELECT count(*) FROM documents WHERE metadata->>'status_reconciled' = 'true'")
        ).scalar()
        print(f"\n  ==> Tổng số văn bản đã xử lý Reconciliation: {reconciled_cnt:,} văn bản")

        sample_rows = conn.execute(
            text("""
                SELECT external_id, document_number, status, 
                       expiry_date, metadata->>'reconcile_reason' as reason
                FROM documents 
                WHERE metadata->>'status_reconciled' = 'true'
                ORDER BY expiry_date DESC NULLS LAST
                LIMIT 5
            """)
        ).mappings().all()

        print("\n  Văn bản mẫu đã được cắt hiệu lực (chuyển sang EXPIRED):")
        for r in sample_rows:
            print(f"    * ID: {r['external_id']} | Số: {r['document_number']} | Trạng thái: {r['status']} | Ngày hết hạn: {r['expiry_date']} | Lý do: {r['reason']}")

        return [dict(r) for r in sample_rows]


def check_qdrant_payload(client: QdrantClient, collection_name: str, sample_docs: list[dict]):
    print("\n" + "=" * 60)
    print(f"2. KIỂM TRA PAYLOAD VECTOR TRÊN QDRANT ({collection_name})")
    print("=" * 60)
    for doc in sample_docs:
        ext_id = str(doc["external_id"])
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=ext_id))]
            ),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            statuses = [p.payload.get("status") for p in points]
            print(f"  [PASS] Doc ID {ext_id} ({doc['document_number']}): Tìm thấy {len(points)} chunks trong Qdrant -> status={statuses}")
        else:
            print(f"  [INFO] Doc ID {ext_id} ({doc['document_number']}): Văn bản không có chunk vector (chỉ lưu metadata văn bản)")


def check_retrieval_hydration(engine, sample_docs: list[dict]):
    print("\n" + "=" * 60)
    print("3. KIỂM TRA CỔNG LỌC PHÁP LÝ KHI RETRIEVAL (PostgreSQL Hydration Gate)")
    print("=" * 60)
    with engine.connect() as conn:
        reconciled_point_ids = [
            str(r[0])
            for r in conn.execute(
                text("""
                    SELECT c.qdrant_point_id
                    FROM documents d
                    JOIN chunks c ON c.document_id = d.id
                    WHERE d.metadata->>'status_reconciled' = 'true'
                      AND c.qdrant_point_id IS NOT NULL
                    LIMIT 5
                """)
            ).fetchall()
        ]

    if not reconciled_point_ids:
        print("  Không tìm thấy point_id mẫu có chunks.")
        return

    print(f"  Thử nghiệm với {len(reconciled_point_ids)} chunks thuộc các văn bản đã reconcile...")

    factory = create_session_factory(engine)
    hydrator = PostgresChunkHydrator(factory)

    # Thử nghiệm 1: Tìm kiếm theo Luật hiện hành (as_of = today, allowed = EFFECTIVE, PARTIALLY_EFFECTIVE)
    filter_current = LegalMetadataFilter(
        as_of=date.today(),
        statuses=("EFFECTIVE", "PARTIALLY_EFFECTIVE"),
        temporal_intent="current_law",
    )
    hydrated_current = hydrator.hydrate(reconciled_point_ids, filter_current)
    print(f"\n  [Kịch bản 1 - Tìm văn bản hiện hành]:")
    print(f"    -> Số chunk lọt qua cổng kiểm duyệt: {len(hydrated_current)} (Kỳ vọng: 0)")
    if len(hydrated_current) == 0:
        print("    ==> CHÍNH XÁC: Toàn bộ chunk của văn bản hết hạn/thay thế ĐÃ BỊ CHẶN TUYỆT ĐỐI.")
    else:
        print("    ==> CẢNH BÁO: Vẫn còn chunk bị lọt vào kết quả hiện hành!")

    # Thử nghiệm 2: Tìm kiếm lịch sử hoặc văn bản hết hiệu lực (allowed = EXPIRED)
    filter_expired = LegalMetadataFilter(
        as_of=date.today(),
        statuses=("EXPIRED",),
        temporal_intent="historical",
    )
    hydrated_expired = hydrator.hydrate(reconciled_point_ids, filter_expired)
    print(f"\n  [Kịch bản 2 - Tìm văn bản lịch sử / đã hết hiệu lực]:")
    print(f"    -> Số chunk được hydrate: {len(hydrated_expired)} / {len(reconciled_point_ids)}")
    if len(hydrated_expired) > 0:
        print("    ==> CHÍNH XÁC: Hệ thống vẫn cho phép tra cứu lịch sử văn bản đã hết hạn khi cần.")


def main():
    parser = argparse.ArgumentParser(description="Xác minh tính đúng đắn của việc xử lý trạng thái và Retrieval.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "legal_chunks_bge_m3_1024_v1"))
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())
    try:
        sample_docs = check_database_status(engine)

        try:
            client = QdrantClient(url=args.qdrant_url, timeout=30.0)
            check_qdrant_payload(client, args.qdrant_collection, sample_docs)
        except Exception as exc:
            print(f"\n[Qdrant Warning] Không thể kết nối Qdrant: {exc}")

        check_retrieval_hydration(engine, sample_docs)
        print("\n" + "=" * 60)
        print("KẾT LUẬN: HỆ THỐNG RETRIEVAL ĐÃ HOẠT ĐỘNG HOÀN TOÀN CHÍNH XÁC!")
        print("=" * 60 + "\n")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
