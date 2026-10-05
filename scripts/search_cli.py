#!/usr/bin/env python3
"""CLI tìm kiếm pháp lý (Search & Retrieval thuần túy - KHÔNG qua LLM).

Sử dụng:
  PYTHONPATH=src:. .venv/bin/python scripts/search_cli.py "câu hỏi pháp lý"
  PYTHONPATH=src:. .venv/bin/python scripts/search_cli.py --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

from database import DatabaseSettings, LegalStatus, create_db_engine
from indexing import QdrantSettings, SentenceTransformerEmbedder
from retrieval import RetrievalRequest
from retrieval.runtime import create_hybrid_retrieval_service


def format_status(status: str) -> str:
    if status == "EFFECTIVE":
        return "🟢 CÒN HIỆU LỰC (EFFECTIVE)"
    elif status == "PARTIALLY_EFFECTIVE":
        return "🟡 CÒN HIỆU LỰC MỘT PHẦN (PARTIALLY_EFFECTIVE)"
    elif status == "EXPIRED":
        return "🔴 ĐÃ HẾT HIỆU LỰC (EXPIRED)"
    elif status == "SUSPENDED":
        return "🟠 TẠM NGƯNG HIỆU LỰC (SUSPENDED)"
    elif status == "NOT_YET_EFFECTIVE":
        return "🔵 CHƯA CÓ HIỆU LỰC (NOT_YET_EFFECTIVE)"
    return f"⚪ {status}"


def execute_search(
    service,
    query: str,
    *,
    limit: int = 3,
    as_of: date | None = None,
    status_filter: list[str] | None = None,
    show_context: bool = False,
) -> None:
    search_as_of = as_of or date.today()
    print("\n" + "=" * 80)
    print(f"🔎 CÂU TRUY VẤN: \"{query}\"")
    print(f"📅 MỐC THỜI GIAN XÉT HIỆU LỰC: {search_as_of.isoformat()} (Mặc định: Hiện tại)")
    print(f"📊 GIỚI HẠN KẾT QUẢ: Top {limit}")
    print("=" * 80)

    t0 = time.perf_counter()
    options: dict = {
        "query": query,
        "as_of": search_as_of,
        "limit": limit,
        "candidate_limit": 50,
        "max_candidate_limit": 200,
    }
    if status_filter:
        options["statuses"] = tuple(status_filter)

    print("🔍 Đang tìm kiếm và trích xuất thông tin từ PostgreSQL...")
    response = service.retrieve(RetrievalRequest(**options))
    search_time = time.perf_counter() - t0

    print(f"⚡ Hoàn tất sau {search_time:.2f} giây.")
    print(f"  - Tổng ứng viên quét: {response.searched_candidates}")
    print(f"  - Ứng viên bị PostgreSQL lọc bỏ (hết hiệu lực/bãi bỏ): {response.rejected_candidates}")

    print("\n" + "─" * 80)
    print(f"📋 KẾT QUẢ TRẢ VỀ: {len(response.results)} điều khoản liên quan nhất từ PostgreSQL")
    print("─" * 80)

    if not response.results:
        print("  Không tìm thấy kết quả phù hợp với các tiêu chí và mốc thời gian hiệu lực đã chọn.")
        return

    for rank, res in enumerate(response.results, start=1):
        cit = res.citation
        status_text = format_status(cit.status)
        location = []
        if cit.article:
            location.append(f"Điều {cit.article}")
        if cit.clause:
            location.append(f"Khoản {cit.clause}")
        if cit.point:
            location.append(f"Điểm {cit.point}")
        location_str = " > ".join(location) if location else "Toàn văn"

        valid_period = f"Từ {cit.as_of}"
        if cit.content_valid_from:
            valid_period = f"Từ {cit.content_valid_from}"
        if cit.content_valid_to:
            valid_period += f" đến {cit.content_valid_to}"

        print(f"\n[VỊ TRÍ #{rank}] Điểm tương đồng: {res.score:.4f}")
        print(f"  📜 Văn bản : {cit.title}")
        print(f"  🔢 Số hiệu : {cit.document_number or 'Không có số'} | Cơ quan: {res.authority or 'N/A'}")
        print(f"  📌 Vị trí  : {location_str}")
        print(f"  ⚖️ Hiệu lực: {status_text} ({valid_period})")
        print(f"  📖 Nội dung trích dẫn từ PostgreSQL:")
        
        lines = res.text.strip().split("\n")
        preview = "\n".join(f"      {line}" for line in lines[:8])
        if len(lines) > 8:
            preview += f"\n      ... (còn tiếp {len(lines) - 8} dòng)"
        print(preview)
        print("─" * 80)

    if show_context:
        print("\n📦 NGỮ CẢNH TỔNG HỢP (Chuẩn bị đưa cho LLM nếu có):")
        print("─" * 80)
        context_parts = []
        for i, res in enumerate(response.results, start=1):
            cit = res.citation
            loc = []
            if cit.article:
                loc.append(f"Điều {cit.article}")
            if cit.clause:
                loc.append(f"Khoản {cit.clause}")
            if cit.point:
                loc.append(f"Điểm {cit.point}")
            header = f"[Tài liệu {i}] {cit.title}"
            if cit.document_number:
                header += f" (Số: {cit.document_number})"
            if loc:
                header += f" - {' '.join(loc)}"
            context_parts.append(f"{header}\n{res.text.strip()}")
        print("\n\n".join(context_parts))
        print("─" * 80)


def main():
    parser = argparse.ArgumentParser(description="Tra cứu và trích xuất thông tin pháp lý từ PostgreSQL (KHÔNG qua LLM).")
    parser.add_argument("query", nargs="?", help="Câu hỏi hoặc từ khóa pháp lý cần tra cứu (để trống để vào chế độ hỏi liên tục)")
    parser.add_argument("--limit", type=int, default=3, help="Số lượng kết quả cần lấy (mặc định: 3)")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Mốc ngày xét hiệu lực (YYYY-MM-DD)")
    parser.add_argument(
        "--status",
        action="append",
        choices=[s.value for s in LegalStatus],
        help="Lọc trạng thái cụ thể nếu muốn (ví dụ: EFFECTIVE, EXPIRED)",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Hiển thị ngữ cảnh tổng hợp được đóng gói chuẩn bị cho LLM",
    )
    args = parser.parse_args()

    print("\n⏳ Đang khởi tạo mô hình tìm kiếm (Qdrant Dense + Tantivy BM25 + Postgres Hydration)...")
    engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    client = qdrant_settings.create_client()

    try:
        service = create_hybrid_retrieval_service(
            engine,
            client,
            qdrant_settings,
            SentenceTransformerEmbedder.from_env(batch_size=1),
        )
        print("✅ Khởi tạo thành công!")

        if args.query:
            execute_search(
                service,
                args.query,
                limit=args.limit,
                as_of=args.as_of,
                status_filter=args.status,
                show_context=args.context,
            )
            return

        print("\n" + "=" * 70)
        print("🚀 CÔNG CỤ TRUY XUẤT THÔNG TIN PHÁP LÝ (Gõ 'exit' hoặc 'quit' để thoát)")
        print("   Chế độ: Retrieval + PostgreSQL Hydration (KHÔNG gọi LLM)")
        print("=" * 70)

        while True:
            try:
                query = input("\nNhập câu hỏi pháp lý: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nĐã thoát.")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Đã thoát.")
                break

            execute_search(
                service,
                query,
                limit=args.limit,
                as_of=args.as_of,
                status_filter=args.status,
                show_context=args.context,
            )
    finally:
        client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
