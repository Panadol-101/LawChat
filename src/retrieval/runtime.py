from __future__ import annotations

import os
from dataclasses import replace

from qdrant_client import QdrantClient
from sqlalchemy import Engine

from database import create_session_factory
from indexing import BM25Settings, EmbeddingBackend, QdrantSettings

from .dense import DenseSearcher
from .graph import PostgresLegalGraphResolver
from .hybrid import HybridRetrievalService
from .hydration import PostgresChunkHydrator
from .query_parser import LegalQueryParser
from .reranker import (
    SentenceTransformerCrossEncoderReranker,
    LegalReranker,
    RerankerSettings,
)
from .temporal import TemporalRetrievalRouter
from .sparse import SparseIndexUnavailable, TantivySparseSearcher


def create_hybrid_retrieval_service(
    engine: Engine,
    qdrant_client: QdrantClient,
    qdrant_settings: QdrantSettings,
    embedder: EmbeddingBackend,
    *,
    bm25_settings: BM25Settings | None = None,
    query_parser: LegalQueryParser | None = None,
    reranker: LegalReranker | None = None,
    reranker_settings: RerankerSettings | None = None,
) -> TemporalRetrievalRouter:
    try:
        sparse_searcher = TantivySparseSearcher.from_settings(bm25_settings)
    except SparseIndexUnavailable:
        sparse_searcher = None
    session_factory = create_session_factory(engine)
    resolved_reranker_settings = reranker_settings or RerankerSettings.from_env()
    if reranker is None and resolved_reranker_settings.enabled:
        reranker = SentenceTransformerCrossEncoderReranker(
            resolved_reranker_settings
        )
    legal_graph = PostgresLegalGraphResolver(session_factory)
    current = HybridRetrievalService(
        DenseSearcher(
            qdrant_client,
            embedder,
            collection=qdrant_settings.alias,
        ),
        sparse_searcher,
        PostgresChunkHydrator(session_factory),
        physical_collection=qdrant_settings.collection,
        query_parser=query_parser,
        graph_resolver=legal_graph,
        seed_resolver=legal_graph,
        reranker=reranker,
        reranker_settings=resolved_reranker_settings,
    )
    historical = None
    if os.getenv("HISTORICAL_INDEX_ENABLED", "false").casefold() in {
        "1", "true", "yes", "on"
    }:
        historical_qdrant = replace(
            qdrant_settings,
            collection=os.getenv(
                "HISTORICAL_QDRANT_COLLECTION",
                "legal_chunks_historical_bge_m3_1024_v1",
            ),
            alias=os.getenv(
                "HISTORICAL_QDRANT_ALIAS", "legal_chunks_historical"
            ),
        )
        historical_bm25 = None
        historical_bm25_settings = replace(
            bm25_settings or BM25Settings.from_env(),
            index_name=os.getenv(
                "HISTORICAL_BM25_INDEX_NAME", "legal_bm25_historical_v1"
            ),
            alias=os.getenv(
                "HISTORICAL_BM25_ALIAS", "legal_bm25_historical"
            ),
            source_collection=historical_qdrant.collection,
        )
        try:
            historical_bm25 = TantivySparseSearcher.from_settings(
                historical_bm25_settings
            )
        except SparseIndexUnavailable:
            pass
        historical = HybridRetrievalService(
            DenseSearcher(
                qdrant_client,
                embedder,
                collection=historical_qdrant.alias,
            ),
            historical_bm25,
            PostgresChunkHydrator(session_factory),
            physical_collection=historical_qdrant.collection,
            query_parser=query_parser,
            graph_resolver=legal_graph,
            seed_resolver=legal_graph,
            reranker=reranker,
            reranker_settings=resolved_reranker_settings,
            content_scope="historical",
            historical_content_enabled=True,
        )
    return TemporalRetrievalRouter(
        current,
        historical,
        query_parser=query_parser,
    )
