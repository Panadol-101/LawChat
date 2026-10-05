"""Diagnose why 88 cases were REFUSED in the benchmark."""
import json
from collections import Counter, defaultdict

def main():
    json_path = "reports/BENCHMARK_200_F1_RESULTS.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["individual_results"]
    refused = [r for r in results if r.get("rag_status") == "REFUSED"]
    verified = [r for r in results if r.get("rag_status") == "VERIFIED"]

    print(f"Total evaluated: {len(results)}")
    print(f"Total VERIFIED: {len(verified)}")
    print(f"Total REFUSED: {len(refused)}")

    # Group refused by law type
    law_groups = defaultdict(list)
    for r in refused:
        q = r["question"]
        ref = r.get("reference_answer", "")
        text = (q + " " + ref).lower()
        if "lao động" in text:
            law = "Bộ luật Lao động 2019"
        elif "hình sự" in text:
            law = "Bộ luật Hình sự 2015"
        elif "doanh nghiệp" in text:
            law = "Luật Doanh nghiệp 2020"
        elif "hiến pháp" in text:
            law = "Hiến pháp 2013"
        elif "đầu tư" in text:
            law = "Luật Đầu tư 2020"
        elif "dân sự" in text:
            law = "Bộ luật Dân sự 2015"
        elif "đấu thầu" in text:
            law = "Luật Đấu thầu 2023"
        elif "giao dịch điện tử" in text:
            law = "Luật Giao dịch điện tử 2023"
        elif "tố tụng dân sự" in text:
            law = "Bộ luật Tố tụng dân sự 2015"
        elif "tố tụng hình sự" in text:
            law = "Bộ luật Tố tụng hình sự 2015"
        elif "dữ liệu cá nhân" in text:
            law = "Bảo vệ dữ liệu cá nhân"
        elif "hôn nhân" in text:
            law = "Hôn nhân và gia đình"
        else:
            law = "Lĩnh vực khác"
        law_groups[law].append(r)

    print("\n=== SỐ LƯỢNG CÂU REFUSED THEO TỪNG VĂN BẢN PHÁP LUẬT ===")
    for law, items in sorted(law_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {law:32}: {len(items):2d} câu")

    print("\n=== CHI TIẾT 10 CÂU BỊ TỪ CHỐI TIÊU BIỂU ===")
    for idx, r in enumerate(refused[:10], 1):
        print(f"\n[{idx}] STT {r.get('stt')} (Row {r.get('row_idx')}): {r.get('question')}")
        print(f"    - Đáp án mẫu: {r.get('reference_answer')[:120]}...")
        timing = r.get("timing", {})
        print(f"    - Retrieval time: {timing.get('retrieval')}s | Total: {timing.get('total')}s")

if __name__ == "__main__":
    main()
