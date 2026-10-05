"""Dense and lexical indexing components."""

from .bm25_index import (
    BM25BuildReport,
    BM25Manifest,
    BM25Settings,
    LexicalChunk,
    PostgresLexicalSource,
    TantivyBM25Index,
    normalize_legal_identifier,
)
from .embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EmbeddingBackend,
    SentenceTransformerEmbedder,
)
from .qdrant_index import (
    DenseIndexReport,
    PostgresChunkSource,
    QdrantDenseIndex,
    QdrantSettings,
    stable_point_id,
)

__all__ = [
    "BM25BuildReport",
    "BM25Manifest",
    "BM25Settings",
    "DenseIndexReport",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "EmbeddingBackend",
    "SentenceTransformerEmbedder",
    "LexicalChunk",
    "PostgresChunkSource",
    "PostgresLexicalSource",
    "QdrantDenseIndex",
    "QdrantSettings",
    "TantivyBM25Index",
    "normalize_legal_identifier",
    "stable_point_id",
]
