"""Test retrieval & database for key refused questions."""
import requests
import json
from database import DatabaseSettings, create_db_engine
from sqlalchemy import text

questions_to_test = [
    ("Hiến pháp 2013", 1, "Hiến pháp năm 2013 quy định quyền lực nhà nước thuộc về ai?"),
    ("Lao động 2019", 171, "Điều 13 Bộ luật Lao động 2019 định nghĩa hợp đồng lao động và hình thức hợp đồng lao động ra sao?"),
    ("Hình sự 2015", 141, "Điều 12 Bộ luật Hình sự 2015 quy định tuổi chịu trách nhiệm hình sự đối với từng nhóm tội danh như thế nào?"),
    ("Doanh nghiệp 2020", 156, "Điều 17 Luật Doanh nghiệp 2020 quy định những đối tượng nào không có quyền thành lập và quản lý doanh nghiệp?"),
    ("Dân sự 2015", 28, "Hợp đồng dân sự vô hiệu tuyệt đối và vô hiệu tương đối trong những trường hợp nào?"),
]

engine = create_db_engine(DatabaseSettings.from_env())

print("=== BẮT ĐẦU KIỂM TRA CHI TIẾT TỪNG TRƯỜNG HỢP REFUSED ===")

for law_name, stt, query in questions_to_test:
    print(f"\n" + "="*80)
    print(f"👉 [{law_name}] STT {stt}: \"{query}\"")
    
    # 1. Check what /api/v1/search returns
    try:
        resp = requests.post("http://127.0.0.1:8000/api/v1/search", json={"query": query, "limit": 3}, timeout=60)
        if resp.status_code == 200:
            search_data = resp.json()
            results = search_data.get("results", [])
            print(f"  [1] Retrieval /api/v1/search trả về {len(results)} kết quả:")
            for idx, r in enumerate(results, 1):
                cit = r.get("citation", {})
                title = cit.get("title", "")
                doc_num = cit.get("document_number", "")
                status = cit.get("status", "")
                article = cit.get("article", "")
                clause = cit.get("clause", "")
                print(f"      Top {idx}: [{doc_num}] - {title[:60]}... | Điều {article} | Trạng thái: {status}")
        else:
            print(f"  [1] Retrieval error: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [1] Retrieval exception: {e}")

    # 2. Check Postgres directly for the expected law document and provisions
    print("  [2] Tra cứu trực tiếp trong PostgreSQL:")
    with engine.connect() as conn:
        # Search document
        keywords = {
            "Hiến pháp 2013": "%Hiến pháp%",
            "Lao động 2019": "%Lao động%",
            "Hình sự 2015": "%Hình sự%",
            "Doanh nghiệp 2020": "%Doanh nghiệp%",
            "Dân sự 2015": "%Dân sự%",
        }
        kw = keywords.get(law_name, "%")
        docs = conn.execute(
            text("SELECT id, title, document_number, status FROM documents WHERE title ILIKE :kw ORDER BY id LIMIT 5;"),
            {"kw": kw}
        ).fetchall()
        print(f"      Tìm thấy {len(docs)} văn bản chứa từ khóa '{kw}':")
        for d in docs[:3]:
            print(f"        - ID: {d[0]} | Số hiệu: {d[1]} | Status: {d[3]} | Title: {d[2][:50]}...")
