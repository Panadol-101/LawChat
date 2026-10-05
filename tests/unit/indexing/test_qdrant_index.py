from __future__ import annotations

import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient

from indexing.qdrant_index import (
    IndexableChunk,
    QdrantDenseIndex,
    stable_point_id,
)
from retrieval import DenseSearchFilter, DenseSearcher


class KeywordEmbedder:
    model_name = "test-keywords"
    dimension = 4

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    @staticmethod
    def _encode(texts):
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vector = np.array(
                [
                    normalized.count("lao động") + normalized.count("nghỉ việc"),
                    normalized.count("kết hôn"),
                    normalized.count("hình sự"),
                    0.1,
                ],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


class CountingKeywordEmbedder(KeywordEmbedder):
    def __init__(self):
        self.query_calls = 0

    def encode_queries(self, texts):
        self.query_calls += 1
        return super().encode_queries(texts)


class MemorySource:
    def __init__(self, chunks):
        self.chunks = chunks
        self.marked = {}

    def iter_batches(self, *, collection, force=False, limit=None):
        pending = [
            chunk
            for chunk in self.chunks
            if force or self.marked.get(chunk.database_id) != collection
        ]
        if limit is not None:
            pending = pending[:limit]
        if pending:
            yield pending

    def mark_indexed(self, chunks, point_ids, *, collection):
        for chunk, point_id in zip(chunks, point_ids, strict=True):
            self.marked[chunk.database_id] = collection

    def count_expected(self):
        return len(self.chunks)

    def count_marked(self, *, collection):
        return sum(value == collection for value in self.marked.values())


def _chunk(chunk_id, text, *, status="EFFECTIVE", document_type=None):
    version_id = uuid.uuid5(uuid.NAMESPACE_URL, "version-1")
    return IndexableChunk(
        database_id=uuid.uuid5(uuid.NAMESPACE_URL, chunk_id),
        version_id=version_id,
        external_id=chunk_id,
        retrieval_text=text,
        payload={
            "chunk_id": chunk_id,
            "doc_id": chunk_id.split("::")[0],
            "status": status,
            **({"document_type": document_type} if document_type else {}),
        },
    )


def test_stable_point_id_is_repeatable_and_version_sensitive():
    first_version = uuid.uuid4()
    second_version = uuid.uuid4()

    assert stable_point_id(first_version, "doc::chunk") == stable_point_id(
        first_version, "doc::chunk"
    )
    assert stable_point_id(first_version, "doc::chunk") != stable_point_id(
        second_version, "doc::chunk"
    )


def test_index_and_search_are_idempotent_with_payload_filter():
    client = QdrantClient(":memory:")
    chunks = [
        _chunk("labor::36", "Người sử dụng lao động cho nghỉ việc"),
        _chunk("marriage::8", "Điều kiện kết hôn tự nguyện"),
        _chunk("criminal::12", "Tuổi chịu trách nhiệm hình sự", status="EXPIRED"),
    ]
    source = MemorySource(chunks)
    index = QdrantDenseIndex(
        client,
        KeywordEmbedder(),
        source,
        collection="legal-test",
        alias="legal-current",
        on_disk=False,
        scalar_quantization=False,
    )

    with pytest.warns(UserWarning, match="local Qdrant"):
        first = index.build(promote_alias=True)
    second = index.build()

    assert first.indexed_points == 3
    assert second.indexed_points == 0
    assert client.count("legal-test", exact=True).count == 3

    searcher = DenseSearcher(
        client, KeywordEmbedder(), collection="legal-current"
    )
    results = searcher.search("có được cho lao động nghỉ việc không", limit=2)
    assert results[0].payload["chunk_id"] == "labor::36"

    filtered = searcher.search(
        "tuổi chịu trách nhiệm hình sự",
        limit=3,
        filters=DenseSearchFilter(statuses=("EFFECTIVE",)),
    )
    assert all(item.payload["status"] == "EFFECTIVE" for item in filtered)
    client.close()


def test_search_rejects_empty_query():
    client = QdrantClient(":memory:")
    searcher = DenseSearcher(client, KeywordEmbedder(), collection="missing")
    with pytest.raises(ValueError, match="must not be empty"):
        searcher.search("   ")
    client.close()


def test_dense_search_excludes_translated_documents_by_default():
    client = QdrantClient(":memory:")
    source = MemorySource(
        [
            _chunk("vi::1", "lao động nghỉ việc", document_type="Bộ luật"),
            _chunk(
                "en::1",
                "lao động nghỉ việc",
                document_type="Bản dịch văn bản",
            ),
        ]
    )
    QdrantDenseIndex(
        client,
        KeywordEmbedder(),
        source,
        collection="translation-test",
        on_disk=False,
        scalar_quantization=False,
    ).build()
    results = DenseSearcher(
        client, KeywordEmbedder(), collection="translation-test"
    ).search("lao động nghỉ việc", filters=DenseSearchFilter())
    assert [item.payload["doc_id"] for item in results] == ["vi"]
    client.close()


def test_search_caches_normalized_query_embedding():
    client = QdrantClient(":memory:")
    source = MemorySource([_chunk("labor::36", "lao động nghỉ việc")])
    embedder = CountingKeywordEmbedder()
    index = QdrantDenseIndex(
        client,
        embedder,
        source,
        collection="cache-test",
        on_disk=False,
        scalar_quantization=False,
    )
    index.build()
    searcher = DenseSearcher(client, embedder, collection="cache-test")

    searcher.search("lao động nghỉ việc", limit=1)
    searcher.search("  lao động   nghỉ việc  ", limit=1)

    assert embedder.query_calls == 1
    client.close()


def test_partial_build_cannot_promote_production_alias():
    client = QdrantClient(":memory:")
    source = MemorySource([_chunk("labor::36", "lao động nghỉ việc")])
    index = QdrantDenseIndex(
        client,
        KeywordEmbedder(),
        source,
        collection="partial-test",
        alias="production",
        on_disk=False,
        scalar_quantization=False,
    )

    with pytest.raises(ValueError, match="partial"):
        index.build(limit=1, promote_alias=True)
    client.close()


def test_alias_requires_postgres_and_qdrant_completeness():
    client = QdrantClient(":memory:")
    source = MemorySource([_chunk("labor::36", "lao động nghỉ việc")])
    index = QdrantDenseIndex(
        client,
        KeywordEmbedder(),
        source,
        collection="incomplete-test",
        alias="production",
        on_disk=False,
        scalar_quantization=False,
    )
    index.ensure_collection()

    with pytest.raises(ValueError, match="completeness"):
        index.promote_alias()
    client.close()
