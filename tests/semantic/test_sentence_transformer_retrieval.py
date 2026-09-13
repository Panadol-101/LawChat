from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from lawchat.evaluation import RetrievalCase, evaluate_rankings
from lawchat.indexing import SentenceTransformerEmbedder
from lawchat.retrieval import DenseSearcher


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BGE_M3_EVAL") != "1",
    reason="Set RUN_BGE_M3_EVAL=1 to run the real BGE-M3 quality gate",
)


def test_bge_m3_legal_retrieval_quality():
    fixture = json.loads(
        Path("tests/fixtures/dense_retrieval_eval.json").read_text(
            encoding="utf-8"
        )
    )
    corpus = fixture["corpus"]
    cases = [
        RetrievalCase(
            item["case_id"],
            item["query"],
            frozenset(item["relevant_chunk_ids"]),
        )
        for item in fixture["cases"]
    ]
    embedder = SentenceTransformerEmbedder.from_env(batch_size=4)
    client = QdrantClient(":memory:")
    collection = "bge-m3-quality"
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=embedder.dimension, distance=models.Distance.COSINE
        ),
    )
    vectors = embedder.encode_documents([item["text"] for item in corpus])
    client.upsert(
        collection_name=collection,
        wait=True,
        points=models.Batch(
            ids=[
                str(uuid.uuid5(uuid.NAMESPACE_URL, item["chunk_id"]))
                for item in corpus
            ],
            vectors=vectors.tolist(),
            payloads=[{"chunk_id": item["chunk_id"]} for item in corpus],
        ),
    )
    searcher = DenseSearcher(client, embedder, collection=collection)
    rankings = {
        case.case_id: [
            result.payload["chunk_id"]
            for result in searcher.search(case.query, limit=3)
        ]
        for case in cases
    }
    metrics = evaluate_rankings(cases, rankings, k=3)

    assert metrics.hit_rate_at_k >= 0.85
    assert metrics.mean_reciprocal_rank >= 0.70
    client.close()
