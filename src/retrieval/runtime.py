from __future__ import annotations

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
) -> HybridRetrievalService:
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
    return current
