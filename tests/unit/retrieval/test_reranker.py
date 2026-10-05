from datetime import date
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from retrieval import (
    HydratedLegalChunk,
    NoOpReranker,
    RerankerSettings,
    SentenceTransformerCrossEncoderReranker,
)


def _chunk(point_id: str) -> HydratedLegalChunk:
    return HydratedLegalChunk(
        point_id=point_id,
        chunk_id=f"chunk-{point_id}",
        chunk_type="clause",
        text="Nội dung",
        retrieval_text="Nội dung retrieval",
        chunk_metadata={},
        parent_chunk_id=None,
        parent_text=None,
        document_id="doc",
        title="Luật kiểm thử",
        document_number="01/2026/QH15",
        document_type="Luật",
        authority="Quốc hội",
        legal_field="Kiểm thử",
        source_url="https://example.test",
        article="1",
        clause="1",
        point=None,
        status="EFFECTIVE",
        valid_from=date(2026, 1, 1),
        valid_to=None,
    )


def test_noop_reranker_preserves_order():
    results = NoOpReranker().rerank("query", [_chunk("a"), _chunk("b")])

    assert [item.point_id for item in results] == ["a", "b"]
    assert [item.rank for item in results] == [1, 2]


def test_reranker_settings_validate_limits():
    with pytest.raises(ValueError, match="candidate_limit"):
        RerankerSettings(candidate_limit=0)
    with pytest.raises(ValueError, match="batch_size"):
        RerankerSettings(batch_size=0)
    with pytest.raises(ValueError, match="max_length"):
        RerankerSettings(max_length=2)


def test_sentence_transformer_cross_encoder_reranks_by_score(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def predict(self, pairs, **kwargs):
            assert pairs == [
                ("query", "Nội dung retrieval"),
                ("query", "Nội dung retrieval"),
            ]
            return np.asarray([0.1, 0.9], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    reranker = SentenceTransformerCrossEncoderReranker(
        RerankerSettings(device="cpu")
    )
    assert reranker.model.kwargs["max_length"] == 1536

    results = reranker.rerank("query", [_chunk("a"), _chunk("b")])

    assert [item.point_id for item in results] == ["b", "a"]
    assert [item.rank for item in results] == [1, 2]


def test_semantic_rewriter_appends_terms_and_respects_context():
    from retrieval import SemanticQueryRewriter

    rewriter = SemanticQueryRewriter()

    assert rewriter.rewrite("Ly hôn đơn phương cần thủ tục gì?") == "Ly hôn đơn phương cần thủ tục gì?"
    assert rewriter.rewrite("Đình công có hợp pháp không?") == "Đình công có hợp pháp không?"
    assert rewriter.rewrite("Công ty đơn phương cho tôi nghỉ việc") == (
        "Công ty đơn phương cho tôi nghỉ việc (đơn phương chấm dứt hợp đồng lao động)"
    )
    assert rewriter.rewrite("Bị sa thải trái luật").startswith("Bị sa thải trái luật")


def test_reranker_settings_reads_timeout_from_env(monkeypatch):
    from retrieval import RerankerSettings

    monkeypatch.setenv("RERANKER_TIMEOUT_SECONDS", "2.5")

    assert RerankerSettings.from_env().timeout_seconds == 2.5
