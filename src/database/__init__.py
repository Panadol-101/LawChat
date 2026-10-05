"""PostgreSQL metadata store for deterministic legal filtering."""

from .connection import DatabaseSettings, create_db_engine, create_session_factory
from .models import (
    Article,
    AmendmentEvent,
    Base,
    Chunk,
    ChunkVectorRef,
    ChatConversation,
    ChatMessage,
    ChatProject,
    CrawlRecord,
    CrawlRun,
    Document,
    DocumentRelationship,
    DocumentSource,
    DocumentVersion,
    EffectiveStatus,
    LegalStatus,
    ProvisionEffectiveStatus,
    ProvenanceRecord,
    RelationshipType,
)
from .queries import LegalMetadataFilter, MetadataQueries
from .provisions import provision_key
from .reconciliation import ReconciliationReport, StatusReconciler

__all__ = [
    "Article",
    "AmendmentEvent",
    "Base",
    "Chunk",
    "ChunkVectorRef",
    "ChatConversation",
    "ChatMessage",
    "ChatProject",
    "CrawlRecord",
    "CrawlRun",
    "DatabaseSettings",
    "Document",
    "DocumentRelationship",
    "DocumentSource",
    "DocumentVersion",
    "EffectiveStatus",
    "LegalMetadataFilter",
    "LegalStatus",
    "MetadataQueries",
    "ProvisionEffectiveStatus",
    "ProvenanceRecord",
    "ReconciliationReport",
    "RelationshipType",
    "StatusReconciler",
    "create_db_engine",
    "create_session_factory",
    "provision_key",
]
