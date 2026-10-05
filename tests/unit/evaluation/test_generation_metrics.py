from evaluation import GenerationCaseResult, summarize_generation_results


def _case(case_id, *, answerable, status, cited=(), expected=(), latency=1.0, attempts=1, codes=()):
    return GenerationCaseResult(
        case_id=case_id,
        answerable=answerable,
        status=status,
        attempts=attempts,
        expected_document_ids=tuple(expected),
        cited_document_ids=tuple(cited),
        verification_codes=tuple(codes),
        latency_seconds=latency,
    )


def test_generation_metrics_separate_false_refusal_and_unsafe_answer():
    metrics = summarize_generation_results(
        [
            _case("ok", answerable=True, status="VERIFIED", cited=("d1",), expected=("d1",), latency=1),
            _case("refused", answerable=True, status="REFUSED", latency=2),
            _case("unsafe", answerable=False, status="VERIFIED", latency=3),
            _case("safe", answerable=False, status="REFUSED", latency=4, attempts=2),
        ]
    )

    assert metrics["false_refusal_rate"] == 0.5
    assert metrics["unsafe_answer_rate"] == 0.5
    assert metrics["grounding_document_precision"] == 1.0
    assert metrics["repair_rate"] == 0.25
    assert metrics["latency_seconds"]["p50"] == 2.5
    assert metrics["latency_seconds"]["p95"] == 3.8499999999999996
