"""
Benchmark script for LawChat using the 200 Legal Questions dataset.
Evaluates end-to-end RAG and Retrieval performance using F1 metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl
import requests

# Default paths
DATASET_PATH = Path("200_Cau_Hoi_Test_Chatbot_RAG_Phap_Luat (1).xlsx")
DEFAULT_REPORT_JSON = Path("reports/BENCHMARK_200_F1_RESULTS.json")
DEFAULT_REPORT_XLSX = Path("reports/BENCHMARK_200_F1_RESULTS.xlsx")
API_BASE_URL = os.getenv("LAWCHAT_API_BASE_URL", "http://127.0.0.1:8000")


def normalize_vietnamese_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation, strip extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Replace punctuation with spaces
    for p in string.punctuation:
        text = text.replace(p, " ")
    # Replace common formatting characters
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def compute_token_f1(prediction: str, reference: str) -> dict[str, float]:
    """
    Standard QA Token-level Precision, Recall, and F1 score.
    Tokens are normalized whitespace-separated words/syllables.
    """
    pred_norm = normalize_vietnamese_text(prediction)
    ref_norm = normalize_vietnamese_text(reference)

    pred_tokens = pred_norm.split()
    ref_tokens = ref_norm.split()

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    common_tokens = sum((pred_counts & ref_counts).values())
    if common_tokens == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = common_tokens / len(pred_tokens)
    recall = common_tokens / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extract_legal_citations(text: str) -> set[str]:
    """
    Extract legal articles, clauses, and document numbers from text.
    E.g. 'Điều 13', 'Khoản 1', '45/2019/QH14', 'Hiến pháp 2013', etc.
    """
    if not text:
        return set()

    found = set()
    # Patterns for articles and clauses
    articles = re.findall(r"(?:điều|đ)\s*(\d+[a-z]?)", text, flags=re.IGNORECASE)
    for a in articles:
        found.add(f"điều {a.lower()}")

    clauses = re.findall(r"(?:khoản|k)\s*(\d+)", text, flags=re.IGNORECASE)
    for c in clauses:
        found.add(f"khoản {c.lower()}")

    # Patterns for law document numbers e.g. 45/2019/QH14, 20/2023/QH15
    doc_numbers = re.findall(r"\b\d+/\d+/[A-Z0-9-]+\b", text)
    for d in doc_numbers:
        found.add(d.upper())

    # Named laws
    named_laws = [
        "hiến pháp",
        "bộ luật lao động",
        "bộ luật dân sự",
        "bộ luật hình sự",
        "bộ luật tố tụng dân sự",
        "bộ luật tố tụng hình sự",
        "luật doanh nghiệp",
        "luật đầu tư",
        "luật đất đai",
        "luật đấu thầu",
        "luật giao dịch điện tử",
        "luật hôn nhân và gia đình",
        "luật bảo vệ dữ liệu cá nhân",
    ]
    text_lower = text.lower()
    for law in named_laws:
        if law in text_lower:
            found.add(law)

    return found


def compute_citation_f1(pred_citations: set[str], ref_citations: set[str]) -> dict[str, float]:
    """Compute precision, recall, and F1 on extracted legal citations."""
    if not pred_citations and not ref_citations:
        return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
    if not pred_citations or not ref_citations:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

    common = len(pred_citations & ref_citations)
    precision = common / len(pred_citations)
    recall = common / len(ref_citations)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
    }


def load_dataset(file_path: Path) -> list[dict[str, Any]]:
    """Read questions and reference answers from Excel file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    # First row is header
    cases = []
    for row_idx, r in enumerate(rows[1:], start=2):
        if not r or len(r) < 3:
            continue
        stt = r[0]
        question = str(r[1]).strip() if r[1] is not None else ""
        answer = str(r[2]).strip() if r[2] is not None else ""

        if not question:
            continue

        cases.append({
            "row_idx": row_idx,
            "stt": stt,
            "question": question,
            "reference_answer": answer,
            "reference_citations": list(extract_legal_citations(answer)),
        })

    return cases


def query_answer_api(
    base_url: str,
    question: str,
    timeout: int = 180,
) -> dict[str, Any]:
    """Call POST /api/v1/answer endpoint."""
    url = f"{base_url.rstrip('/')}/api/v1/answer"
    payload = {"query": question, "response_mode": "compact"}

    started = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = round(time.perf_counter() - started, 3)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "status_code": 200,
                "rag_status": data.get("status"),
                "predicted_answer": data.get("answer", ""),
                "limitations": data.get("limitations", []),
                "citations": data.get("citations", []),
                "elapsed_seconds": elapsed,
                "timing": data.get("timing", {}),
            }
        else:
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text[:200],
                "elapsed_seconds": elapsed,
            }
    except Exception as exc:
        return {
            "success": False,
            "status_code": 0,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def query_search_api(
    base_url: str,
    question: str,
    limit: int = 5,
    timeout: int = 60,
) -> dict[str, Any]:
    """Call POST /api/v1/search endpoint to retrieve context."""
    url = f"{base_url.rstrip('/')}/api/v1/search"
    payload = {"query": question, "limit": limit}

    started = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = round(time.perf_counter() - started, 3)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            context_text = "\n\n".join(r.get("text", "") for r in results)
            citations = [r.get("citation", {}) for r in results]
            return {
                "success": True,
                "status_code": 200,
                "context_text": context_text,
                "results_count": len(results),
                "citations": citations,
                "elapsed_seconds": elapsed,
            }
        else:
            return {
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text[:200],
                "elapsed_seconds": elapsed,
            }
    except Exception as exc:
        return {
            "success": False,
            "status_code": 0,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }


def run_benchmark(
    cases: list[dict[str, Any]],
    mode: str = "rag",
    base_url: str = API_BASE_URL,
    output_json: Path = DEFAULT_REPORT_JSON,
    output_xlsx: Path = DEFAULT_REPORT_XLSX,
    resume: bool = True,
    rerun_refused: bool = False,
    delay: float = 0.5,
) -> dict[str, Any]:
    """Run benchmark over test cases and compute F1 metrics."""
    results: list[dict[str, Any]] = []
    completed_rows: set[int] = set()

    # Load existing checkpoint if resuming
    if resume and output_json.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                results = prev_data.get("individual_results", [])
                if rerun_refused:
                    completed_rows = {
                        r["row_idx"] for r in results
                        if r.get("success") and r.get("rag_status") != "REFUSED"
                    }
                else:
                    completed_rows = {r["row_idx"] for r in results if r.get("success")}
                print(f"Resuming: Loaded {len(completed_rows)} already completed test cases (rerun_refused={rerun_refused}).")
        except Exception as e:
            print(f"Warning: could not load existing checkpoint ({e}). Starting fresh.")
            results = []
            completed_rows = set()

    total_cases = len(cases)
    print(f"\nStarting Benchmark [{mode.upper()} Mode] on {total_cases} cases...")
    print(f"Target API: {base_url}\n")

    for i, case in enumerate(cases, start=1):
        row_idx = case["row_idx"]
        stt = case["stt"]
        q = case["question"]
        ref_ans = case["reference_answer"]
        ref_cits = set(case["reference_citations"])

        if row_idx in completed_rows:
            print(f"[{i}/{total_cases}] Row {row_idx} (STT {stt}) already done. Skipping.")
            continue

        print(f"[{i}/{total_cases}] Row {row_idx} (STT {stt}): {q[:60]}...")

        if mode == "rag":
            api_res = query_answer_api(base_url, q)
            pred_ans = api_res.get("predicted_answer", "")
            pred_cits = extract_legal_citations(pred_ans)
            # Token F1
            token_metrics = compute_token_f1(pred_ans, ref_ans)
            # Citation F1
            cit_metrics = compute_citation_f1(pred_cits, ref_cits)

            case_record = {
                "row_idx": row_idx,
                "stt": stt,
                "question": q,
                "reference_answer": ref_ans,
                "predicted_answer": pred_ans,
                "rag_status": api_res.get("rag_status"),
                "success": api_res.get("success", False),
                "error": api_res.get("error"),
                "elapsed_seconds": api_res.get("elapsed_seconds"),
                "token_f1": token_metrics["f1"],
                "token_precision": token_metrics["precision"],
                "token_recall": token_metrics["recall"],
                "citation_f1": cit_metrics["citation_f1"],
                "citation_precision": cit_metrics["citation_precision"],
                "citation_recall": cit_metrics["citation_recall"],
                "reference_citations": list(ref_cits),
                "predicted_citations": list(pred_cits),
                "timing": api_res.get("timing", {}),
            }
            print(
                f"    -> Status: {case_record['rag_status']} | "
                f"F1: {token_metrics['f1']:.3f} (P: {token_metrics['precision']:.3f}, R: {token_metrics['recall']:.3f}) | "
                f"Time: {api_res.get('elapsed_seconds')}s"
            )

        else:  # retrieval mode
            api_res = query_search_api(base_url, q)
            context = api_res.get("context_text", "")
            ctx_cits = extract_legal_citations(context)
            # Context Token F1 vs Reference Answer
            token_metrics = compute_token_f1(context, ref_ans)
            # Context Citation F1
            cit_metrics = compute_citation_f1(ctx_cits, ref_cits)

            case_record = {
                "row_idx": row_idx,
                "stt": stt,
                "question": q,
                "reference_answer": ref_ans,
                "retrieved_context_preview": context[:300] + ("..." if len(context) > 300 else ""),
                "results_count": api_res.get("results_count", 0),
                "success": api_res.get("success", False),
                "error": api_res.get("error"),
                "elapsed_seconds": api_res.get("elapsed_seconds"),
                "token_f1": token_metrics["f1"],
                "token_precision": token_metrics["precision"],
                "token_recall": token_metrics["recall"],
                "citation_f1": cit_metrics["citation_f1"],
                "citation_precision": cit_metrics["citation_precision"],
                "citation_recall": cit_metrics["citation_recall"],
                "reference_citations": list(ref_cits),
                "retrieved_citations": list(ctx_cits),
            }
            print(
                f"    -> Results: {case_record['results_count']} | "
                f"Context Recall: {token_metrics['recall']:.3f}, Context F1: {token_metrics['f1']:.3f} | "
                f"Time: {api_res.get('elapsed_seconds')}s"
            )

        # Update existing record or append
        existing_idx = next((idx for idx, r in enumerate(results) if r.get("row_idx") == row_idx), None)
        if existing_idx is not None:
            results[existing_idx] = case_record
        else:
            results.append(case_record)
        completed_rows.add(row_idx)

        # Save checkpoint after each item
        _save_checkpoint(results, total_cases, mode, output_json, output_xlsx)

        if delay > 0:
            time.sleep(delay)

    # Compute overall summary metrics
    summary = compute_summary_metrics(results, total_cases, mode)
    _save_checkpoint(results, total_cases, mode, output_json, output_xlsx, summary=summary)
    return summary


def compute_summary_metrics(
    results: list[dict[str, Any]],
    total_cases: int,
    mode: str,
) -> dict[str, Any]:
    """Calculate aggregate Macro/Micro Precision, Recall, F1 and status counts."""
    valid = [r for r in results if r.get("success")]
    if not valid:
        return {
            "total_cases": total_cases,
            "completed_cases": len(results),
            "successful_cases": 0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }

    macro_f1 = sum(r["token_f1"] for r in valid) / len(valid)
    macro_precision = sum(r["token_precision"] for r in valid) / len(valid)
    macro_recall = sum(r["token_recall"] for r in valid) / len(valid)

    citation_f1_list = [r["citation_f1"] for r in valid if r.get("reference_citations")]
    avg_citation_f1 = sum(citation_f1_list) / len(citation_f1_list) if citation_f1_list else 0.0

    avg_latency = sum(r["elapsed_seconds"] for r in valid) / len(valid)

    status_counts = Counter(r.get("rag_status", "UNKNOWN") for r in valid)

    return {
        "benchmark_timestamp": datetime.now().isoformat(),
        "mode": mode,
        "total_cases": total_cases,
        "completed_cases": len(results),
        "successful_cases": len(valid),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "avg_citation_f1": round(avg_citation_f1, 4),
        "avg_latency_seconds": round(avg_latency, 2),
        "rag_status_distribution": dict(status_counts),
    }


def _save_checkpoint(
    results: list[dict[str, Any]],
    total_cases: int,
    mode: str,
    output_json: Path,
    output_xlsx: Path,
    summary: dict[str, Any] | None = None,
) -> None:
    """Save progress and metrics to both JSON and Excel."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if summary is None:
        summary = compute_summary_metrics(results, total_cases, mode)

    payload = {
        "summary": summary,
        "individual_results": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Save to Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Benchmark Results"

    headers = [
        "STT",
        "Row",
        "Nội Dung Câu Hỏi",
        "Câu Trả Lời Chuẩn (Reference)",
        "Kết Quả Hệ Thống (Prediction / Context)",
        "Trạng Thái RAG",
        "Token F1",
        "Token Precision",
        "Token Recall",
        "Citation F1",
        "Thời Gian (s)",
    ]
    ws.append(headers)

    for r in results:
        pred = r.get("predicted_answer") or r.get("retrieved_context_preview", "")
        ws.append([
            r.get("stt"),
            r.get("row_idx"),
            r.get("question"),
            r.get("reference_answer"),
            pred,
            r.get("rag_status", "SEARCH"),
            r.get("token_f1"),
            r.get("token_precision"),
            r.get("token_recall"),
            r.get("citation_f1"),
            r.get("elapsed_seconds"),
        ])

    # Summary sheet
    ws_sum = wb.create_sheet(title="Summary Metrics")
    ws_sum.append(["Chỉ Số", "Giá Trị"])
    for k, v in summary.items():
        ws_sum.append([str(k), str(v)])

    wb.save(output_xlsx)


def main() -> None:
    parser = argparse.ArgumentParser(description="LawChat 200 Questions F1 Benchmark")
    parser.add_argument("--file", type=Path, default=DATASET_PATH, help="Path to Excel dataset")
    parser.add_argument("--mode", choices=["rag", "retrieval"], default="rag", help="Benchmark mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to test")
    parser.add_argument("--sample-indices", type=str, default=None, help="Comma-separated row indices or range e.g. 1,5,10 or 1-10")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--rerun-refused", action="store_true", help="Re-evaluate cases that previously had status REFUSED")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_JSON, help="Output JSON path")
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_REPORT_XLSX, help="Output Excel path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (s)")
    args = parser.parse_args()

    cases = load_dataset(args.file)
    print(f"Loaded {len(cases)} cases from {args.file}")

    if args.sample_indices:
        indices = set()
        for part in args.sample_indices.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                indices.update(range(int(start), int(end) + 1))
            elif part.isdigit():
                indices.add(int(part))
        cases = [c for c in cases if c["stt"] in indices or c["row_idx"] in indices]
        print(f"Filtered by indices: {len(cases)} cases selected.")
    elif args.limit:
        cases = cases[: args.limit]
        print(f"Limited to first {args.limit} cases.")

    summary = run_benchmark(
        cases=cases,
        mode=args.mode,
        output_json=args.output_json,
        output_xlsx=args.output_xlsx,
        resume=not args.no_resume,
        rerun_refused=args.rerun_refused,
        delay=args.delay,
    )

    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETED")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nDetailed reports saved to:\n  - JSON: {args.output_json}\n  - Excel: {args.output_xlsx}")


if __name__ == "__main__":
    main()
