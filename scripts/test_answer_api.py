"""Bounded concurrent load test for the LawChat answer endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from scripts import dispatch


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def run_level(client, url, payloads, concurrency, requests_per_user):
    semaphore = asyncio.Semaphore(concurrency)

    async def send(index: int) -> dict:
        async with semaphore:
            started = perf_counter()
            try:
                response = await client.post(url, json=payloads[index % len(payloads)])
                elapsed = perf_counter() - started
                body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                timing = body.get("timing", {}) if isinstance(body, dict) else {}
                return {
                    "status": response.status_code,
                    "seconds": elapsed,
                    "queue_wait": timing.get("queue_wait"),
                    "processing": timing.get("processing"),
                }
            except Exception as exc:
                return {"status": type(exc).__name__, "seconds": perf_counter() - started}

    total = concurrency * requests_per_user
    rows = await asyncio.gather(*(send(index) for index in range(total)))
    latencies = [row["seconds"] for row in rows]
    queue_wait = [row["queue_wait"] for row in rows if row.get("queue_wait") is not None]
    processing = [row["processing"] for row in rows if row.get("processing") is not None]
    return {
        "concurrency": concurrency,
        "requests": total,
        "statuses": dict(Counter(str(row["status"]) for row in rows)),
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "queue_wait_seconds": {
            "p50": percentile(queue_wait, 0.50),
            "p95": percentile(queue_wait, 0.95),
            "p99": percentile(queue_wait, 0.99),
        },
        "processing_seconds": {
            "p50": percentile(processing, 0.50),
            "p95": percentile(processing, 0.95),
            "p99": percentile(processing, 0.99),
        },
    }


async def cancellation_probe(client, url, payload, count, cancel_after):
    tasks = [asyncio.create_task(client.post(url, json=payload)) for _ in range(count)]
    await asyncio.sleep(cancel_after)
    for task in tasks:
        task.cancel()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    recovery_started = perf_counter()
    try:
        recovery = await client.post(url, json=payload)
        recovery_status = recovery.status_code
    except Exception as exc:
        recovery_status = type(exc).__name__
    return {
        "requested": count,
        "client_cancelled": sum(isinstance(item, asyncio.CancelledError) for item in outcomes),
        "cancel_after_seconds": cancel_after,
        "recovery_status": recovery_status,
        "recovery_seconds": perf_counter() - recovery_started,
    }


async def main_async(args) -> None:
    payloads = json.loads(args.payloads.read_text(encoding="utf-8"))
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("payload file must contain a non-empty JSON array")
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = []
        for concurrency in args.concurrency:
            result = await run_level(
                client, args.url, payloads, concurrency, args.requests_per_user
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        cancellation = (
            await cancellation_probe(
                client, args.url, payloads[0], args.cancel_count, args.cancel_after
            )
            if args.cancel_after is not None
            else None
        )
    report = {"url": args.url, "results": results, "cancellation_probe": cancellation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def load_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/answer")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/RAG_LOAD_TEST.json"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--requests-per-user", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument(
        "--cancel-after", type=float,
        help="Cancel in-flight client requests after this many seconds, then probe recovery.",
    )
    parser.add_argument("--cancel-count", type=int, default=2)
    args = parser.parse_args()
    if (any(value <= 0 for value in args.concurrency)
        or args.requests_per_user <= 0 or args.cancel_count <= 0
        or (args.cancel_after is not None and args.cancel_after <= 0)):
        parser.error("concurrency and requests-per-user must be positive")
    asyncio.run(main_async(args))


def smoke_main() -> None:
    parser = argparse.ArgumentParser(
        description="Save real API answers for human review."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/RAG_API_SMOKE.json"),
    )
    parser.add_argument("--as-of", default="2026-09-06")
    args = parser.parse_args()
    queries = [
        "Bộ luật Lao động 45/2019/QH14 quy định thế nào về sử dụng lao động chưa thành niên?",
        "Một người bị ép kết hôn nhưng không tự yêu cầu hủy kết hôn trái pháp luật thì người thân của họ có quyền yêu cầu hay không?",
        "Người sử dụng lao động trả lương không đầy đủ, yêu cầu người lao động chưa thành niên làm thêm ban đêm và không có sự đồng ý của người giám hộ. Hãy xác định riêng từng vấn đề pháp lý và căn cứ tương ứng.",
    ]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Live RAG smoke; no independently reviewed legal gold answers.",
        "selected_cases": len(queries),
        "complete": False,
        "results": [],
    }
    for query in queries:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/api/v1/answer",
            data=json.dumps(
                {
                    "query": query,
                    "as_of": args.as_of,
                    "response_mode": "verbose",
                    "limit": 5,
                    "candidate_limit": 50,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=195) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            body = {"http_status": exc.code, "error": exc.read().decode()}
        report["results"].append(body)
        report["complete"] = len(report["results"]) == len(queries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    key: body.get(key)
                    for key in (
                        "request_id",
                        "status",
                        "semantic_mode",
                        "semantic_status",
                        "attempts",
                        "timing",
                        "error",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if "error" in body:
            raise SystemExit(1)


def main() -> None:
    dispatch(
        "Smoke and load-test the answer API.",
        {"smoke": smoke_main, "load": load_main},
    )


if __name__ == "__main__":
    main()
