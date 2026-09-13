from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lawchat.indexing import SentenceTransformerEmbedder


class _FakeSentenceTransformer:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.max_seq_length = None
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 1024

    def eval(self):
        return self

    def encode_document(self, texts, **kwargs):
        self.calls.append(("document", list(texts), kwargs))
        values = np.zeros((len(texts), 1024), dtype=np.float32)
        values[:, 0] = 3
        values[:, 1] = 4
        return values

    def encode_query(self, texts, **kwargs):
        self.calls.append(("query", list(texts), kwargs))
        values = np.zeros((len(texts), 1024), dtype=np.float32)
        values[:, 2] = 5
        return values


def _install_fake_sentence_transformers(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )


def test_sentence_transformer_defaults_to_pinned_bge_m3():
    embedder = SentenceTransformerEmbedder(device="cpu")

    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.dimension == 1024
    assert embedder.max_length == 1200
    assert embedder._model is None


def test_sentence_transformer_reads_runtime_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "3")
    monkeypatch.setenv("EMBEDDING_MAX_LENGTH", "1536")
    monkeypatch.setenv("EMBEDDING_CACHE_PATH", "/tmp/lawchat-model-cache")

    embedder = SentenceTransformerEmbedder.from_env()

    assert embedder.device == "cpu"
    assert embedder.batch_size == 3
    assert embedder.max_length == 1536
    assert embedder.cache_dir == "/tmp/lawchat-model-cache"


def test_sentence_transformer_uses_document_and_query_encoders(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder(device="cpu", batch_size=2)

    documents = embedder.encode_documents(["Điều 1", "Điều 2"])
    queries = embedder.encode_queries(["Quy định nào áp dụng?"])

    assert documents.shape == (2, 1024)
    assert queries.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1)
    assert [item[0] for item in embedder._model.calls] == ["document", "query"]
    assert embedder._model.max_seq_length == 1200


def test_sentence_transformer_cuda_fails_fast_when_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=None),
        ),
    )
    embedder = SentenceTransformerEmbedder(device="cuda")

    with pytest.raises(RuntimeError, match="cannot access CUDA/ROCm"):
        _ = embedder.resolved_device


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_length": 2}, "max_length"),
        ({"expected_dimension": 0}, "expected_dimension"),
        ({"device": ""}, "device"),
    ],
)
def test_sentence_transformer_rejects_invalid_runtime_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SentenceTransformerEmbedder(**kwargs)
