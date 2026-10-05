"""Retrieval strategies."""

from .dense import DenseSearchFilter, DenseSearchResult, DenseSearcher
from .hydration import ChunkHydrator, LegalContextBuilder, PostgresChunkHydrator
from .graph import (
    DocumentStatusResolution,
    LegalDocumentRole,
    LegalGraphEdge,
    LegalGraphDocument,
    LegalGraphExpansion,
    LegalGraphResolver,
    LegalSeedResolver,
    PostgresLegalGraphResolver,
    SeedResolution,
)
from .fusion import FusedCandidate, RetrievalCandidate, RRFSettings, reciprocal_rank_fusion
from .hybrid import HybridRetrievalService, HybridSearchUnavailable
from .models import (
    HydratedLegalChunk,
    LegalCitation,
    LegalIssue,
    ParsedLegalQuery,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedLegalChunk,
)
from .query_parser import AmbiguousTemporalQuery, LegalDataCutoffExceeded, LegalQueryParser
from .temporal import (
    ALL_LEGAL_STATUSES,
    TemporalDecision,
    TemporalIntent,
    TemporalPolicy,
    TemporalRetrievalRouter,
)
from .reranker import (
    DEFAULT_RERANKER_MODEL,
    SentenceTransformerCrossEncoderReranker,
    LegalReranker,
    LEGAL_SYNONYMS,
    NoOpReranker,
    RerankResult,
    RerankerSettings,
    SemanticQueryRewriter,
)
from .service import LegalRetrievalService
from .sparse import (
    SparseIndexUnavailable,
    SparseSearchFilter,
    TantivySparseSearcher,
)

__all__ = [
    "AmbiguousTemporalQuery",
    "LegalDataCutoffExceeded",
    "ALL_LEGAL_STATUSES",
    "ChunkHydrator",
    "DenseSearchFilter",
    "DenseSearchResult",
    "DenseSearcher",
    "DocumentStatusResolution",
    "DEFAULT_RERANKER_MODEL",
    "SentenceTransformerCrossEncoderReranker",
    "FusedCandidate",
    "HydratedLegalChunk",
    "HybridRetrievalService",
    "HybridSearchUnavailable",
    "LegalCitation",
    "LegalIssue",
    "LegalContextBuilder",
    "LegalDocumentRole",
    "LegalGraphEdge",
    "LegalGraphDocument",
    "LegalGraphExpansion",
    "LegalGraphResolver",
    "LegalSeedResolver",
    "LegalQueryParser",
    "LegalReranker",
    "LegalRetrievalService",
    "ParsedLegalQuery",
    "PostgresChunkHydrator",
    "PostgresLegalGraphResolver",
    "NoOpReranker",
    "RRFSettings",
    "RerankResult",
    "RerankerSettings",
    "LEGAL_SYNONYMS",
    "SemanticQueryRewriter",
    "SeedResolution",
    "RetrievalCandidate",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievedLegalChunk",
    "SparseIndexUnavailable",
    "SparseSearchFilter",
    "TantivySparseSearcher",
    "TemporalDecision",
    "TemporalIntent",
    "TemporalPolicy",
    "TemporalRetrievalRouter",
    "reciprocal_rank_fusion",
]
