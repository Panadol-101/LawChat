import pytest

from evaluation import RetrievalCase, evaluate_rankings


def test_retrieval_metrics_measure_rank_and_coverage():
    cases = [
        RetrievalCase("a", "query a", frozenset({"a1"})),
        RetrievalCase("b", "query b", frozenset({"b1", "b2"})),
    ]
    metrics = evaluate_rankings(
        cases,
        {"a": ["x", "a1"], "b": ["b2", "x"]},
        k=2,
    )

    assert metrics.hit_rate_at_k == 1.0
    assert metrics.recall_at_k == 0.75
    assert metrics.mean_reciprocal_rank == 0.75
    assert metrics.ndcg_at_k == pytest.approx(0.6220384732)


def test_retrieval_metrics_reject_invalid_suite():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_rankings([], {}, k=3)

    with pytest.raises(ValueError, match="k must be"):
        evaluate_rankings(
            [RetrievalCase("a", "q", frozenset({"x"}))], {}, k=0
        )


def test_ndcg_uses_graded_relevance_while_recall_uses_direct_evidence():
    case = RetrievalCase(
        "graded",
        "query",
        frozenset({"direct", "equivalent"}),
        relevance_grades={"direct": 3, "equivalent": 2, "context": 1},
    )

    metrics = evaluate_rankings(
        [case],
        {"graded": ["equivalent", "context", "direct"]},
        k=3,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert 0.0 < metrics.ndcg_at_k < 1.0
