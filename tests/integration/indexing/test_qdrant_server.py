from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient

from lawchat.indexing.qdrant_index import IndexableChunk, QdrantDenseIndex
from lawchat.retrieval import DenseSearcher


QDRANT_TEST_URL = os.getenv("QDRANT_TEST_URL")
pytestmark = pytest.mark.skipif(
    not QDRANT_TEST_URL,
    reason="QDRANT_TEST_URL is required for Qdrant server integration tests",
)


class TinyEmbedder:
    model_name = "integration-test"
    dimension = 2

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    @staticmethod
    def _encode(texts):
        values = []
        for value in texts:
            vector = np.array(
                [1.0, 0.0] if "lao động" in value.casefold() else [0.0, 1.0],
                dtype=np.float32,
            )
            values.append(vector)
        return np.stack(values)


class TinySource:
    def __init__(self):
        version_id = uuid.uuid4()
        self.rows = [
            IndexableChunk(
                database_id=uuid.uuid4(),
                version_id=version_id,
                external_id="labor::36",
                retrieval_text="Quyền của người sử dụng lao động",
                payload={
                    "doc_id": "labor",
                    "chunk_id": "labor::36",
                    "status": "EFFECTIVE",
                },
            ),
            IndexableChunk(
                database_id=uuid.uuid4(),
                version_id=version_id,
                external_id="marriage::8",
                retrieval_text="Điều kiện kết hôn",
                payload={
                    "doc_id": "marriage",
                    "chunk_id": "marriage::8",
                    "status": "EFFECTIVE",
                },
            ),
        ]
        self.indexed = False

    def iter_batches(self, *, collection, force=False, limit=None):
        if force or not self.indexed:
            yield self.rows[:limit]

    def mark_indexed(self, chunks, point_ids, *, collection):
        self.indexed = True

    def count_expected(self):
        return len(self.rows)

    def count_marked(self, *, collection):
        return len(self.rows) if self.indexed else 0


def test_real_qdrant_server_collection_alias_upsert_and_search():
    client = QdrantClient(url=QDRANT_TEST_URL)
    suffix = uuid.uuid4().hex
    collection = f"lawchat_integration_{suffix}"
    alias = f"lawchat_integration_alias_{suffix}"
    try:
        index = QdrantDenseIndex(
            client,
            TinyEmbedder(),
            TinySource(),
            collection=collection,
            alias=alias,
            on_disk=False,
            scalar_quantization=False,
        )
        report = index.build(promote_alias=True)

        assert report.indexed_points == 2
        assert client.count(collection, exact=True).count == 2
        info = client.get_collection(collection)
        assert "doc_id" in info.payload_schema
        results = DenseSearcher(
            client, TinyEmbedder(), collection=alias
        ).search("quyền lao động", limit=1)
        assert results[0].payload["chunk_id"] == "labor::36"
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()
