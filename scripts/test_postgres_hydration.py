#!/usr/bin/env python3
"""Script test độc lập phần truy xuất thông tin (Hydration) từ PostgreSQL.

Không tải embedding model hay reranker, chạy trực tiếp truy vấn SQL tới PostgreSQL
để kiểm tra xem PostgreSQL trích xuất (hydrate) thông tin văn bản, điều khoản,
nội dung và trạng thái hiệu lực như thế nào.

Cách dùng:
  uv run python scripts/test_postgres_hydration.py
  uv run python scripts/test_postgres_hydration.py --point-id <point_id>
  uv run python scripts/test_postgres_hydration.py --limit 3
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from database import (
    Chunk,
    DatabaseSettings,
    LegalMetadataFilter,
    MetadataQueries,
    create_db_engine,
)
from retrieval.hydration import LegalContextBuilder, PostgresChunkHydrator


def main():
    parser = argparse.ArgumentParser(description="Test PostgreSQL Hydration (Trích xuất thông tin văn bản)")
    parser.add_argument("--point-id", nargs="*", help="Danh sách point_id cần test")
    parser.add_argument("--limit", type=int, default=3, help="Số lượng chunk mẫu lấy từ DB nếu không truyền point-id")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="Ngày xét hiệu lực (YYYY-MM-DD)")
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())
    session_factory = sessionmaker(bind=engine)

    point_ids = args.point_id
    if not point_ids:
        print(f"🔍 Không có point-id cụ thể, đang lấy {args.limit} point_id mẫu từ PostgreSQL...")
        with session_factory() as session:
            rows = session.execute(
                select(Chunk.qdrant_point_id)
                .where(Chunk.qdrant_point_id.is_not(None))
                .limit(args.limit)
            ).scalars().all()
            point_ids = [str(pid) for pid in rows if pid]

    if not point_ids:
        print("❌ Không tìm thấy point_id nào trong database.")
        return

    print("=" * 80)
    print(f"📌 DANH SÁCH POINT_IDS CẦN HYDRATE ({len(point_ids)} điểm):")
    for pid in point_ids:
        print(f"   • {pid}")
    print(f"📅 Ngày xét hiệu lực: {args.as_of}")
    print("=" * 80)

    hydrator = PostgresChunkHydrator(session_factory)
    filters = LegalMetadataFilter(as_of=args.as_of)

    # 1. Đo thời gian query PostgreSQL
    t0 = time.perf_counter()
    chunks = hydrator.hydrate(point_ids, filters)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\n⚡ THỜI GIAN POSTGRESQL TRUY VẤN: {elapsed_ms:.2f} ms")
    print(f"📊 SỐ CHUNK ĐƯỢC HYDRATE THÀNH CÔNG: {len(chunks)}/{len(point_ids)}")

    rejections = hydrator.rejection_reasons(point_ids, filters)
    if rejections:
        print(f"⚠️ Ứng viên bị lọc bỏ do trạng thái hiệu lực: {rejections}")

    # 2. In chi tiết từng thông tin PostgreSQL đưa ra
    context_builder = LegalContextBuilder()
    print("\n" + "─" * 80)
    print("📋 CHI TIẾT THÔNG TIN POSTGRESQL ĐƯA RA (HYDRATED CHUNKS):")
    print("─" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[MỤC #{i}] Chunk ID: {chunk.chunk_id} | Point ID: {chunk.point_id}")
        print(f"  📜 Tên văn bản        : {chunk.title}")
        print(f"  🔢 Số hiệu            : {chunk.document_number or 'Không có số'}")
        print(f"  🏛️ Cơ quan ban hành   : {chunk.authority or 'N/A'}")
        print(f"  📑 Loại văn bản       : {chunk.document_type or 'N/A'}")
        print(f"  🌐 URL nguồn          : {chunk.source_url or 'N/A'}")
        print(f"  📌 Điều / Khoản / Điểm: Điều {chunk.article or '_'} | Khoản {chunk.clause or '_'} | Điểm {chunk.point or '_'}")
        print(f"  ⚖️ Hiệu lực pháp lý   : {chunk.status} (Hiệu lực: {chunk.valid_from} -> {chunk.valid_to or 'vô thời hạn'})")
        print(f"  📝 Độ dài văn bản     : {len(chunk.text)} ký tự")
        print(f"  📖 Nội dung trích xuất (Leaf text):")
        for line in chunk.text.strip().splitlines()[:5]:
            print(f"      {line}")
        if len(chunk.text.strip().splitlines()) > 5:
            print("      ...")

        if chunk.parent_text:
            print(f"  🌳 Ngữ cảnh cấp trên (Parent text): {len(chunk.parent_text)} ký tự")

        print("\n  📦 Định dạng ngữ cảnh gửi cho LLM (Context Builder):")
        context_str = context_builder.build(chunk)
        for line in context_str.splitlines()[:6]:
            print(f"      | {line}")
        print("      | ...")
        print("─" * 80)

    engine.dispose()
    print("\n✅ Hoàn thành kiểm tra PostgreSQL Hydration.")


if __name__ == "__main__":
    main()
