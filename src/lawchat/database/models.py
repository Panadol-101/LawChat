from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import DATERANGE, JSONB, UUID
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class LegalStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    NOT_YET_EFFECTIVE = "NOT_YET_EFFECTIVE"
    EFFECTIVE = "EFFECTIVE"
    PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
    SUSPENDED = "SUSPENDED"
    EXPIRED = "EXPIRED"
    REPEALED = "REPEALED"
    UNKNOWN = "UNKNOWN"


class RelationshipType(str, enum.Enum):
    AMENDS = "AMENDS"
    SUPPLEMENTS = "SUPPLEMENTS"
    REPEALS = "REPEALS"
    REPLACES = "REPLACES"
    GUIDES = "GUIDES"
    IMPLEMENTS = "IMPLEMENTS"
    CITES = "CITES"
    RELATED_TO = "RELATED_TO"


LEGAL_STATUS_VALUES = ", ".join(
    f"'{item.value}'" for item in LegalStatus
)
RELATIONSHIP_TYPE_VALUES = ", ".join(
    f"'{item.value}'" for item in RelationshipType
)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Document(TimestampMixin, Base):
    """Stable legal identity, independent of crawled content snapshots."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "expiry_date IS NULL OR effective_date IS NULL "
            "OR expiry_date >= effective_date",
            name="ck_documents_date_order",
        ),
        CheckConstraint(
            f"status IN ({LEGAL_STATUS_VALUES})",
            name="ck_documents_status",
        ),
        Index("idx_documents_effective_date", "effective_date"),
        Index("idx_documents_expiry_date", "expiry_date"),
        Index("idx_documents_document_number", "document_number"),
        Index(
            "idx_documents_filter_covering",
            "document_type",
            "authority",
            "legal_field",
            "effective_date",
            "expiry_date",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    external_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_number: Mapped[str | None] = mapped_column(String(255))
    document_type: Mapped[str | None] = mapped_column(String(100))
    authority: Mapped[str | None] = mapped_column(String(255))
    issued_date: Mapped[date | None] = mapped_column(Date)
    effective_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=LegalStatus.UNKNOWN.value,
        server_default=LegalStatus.UNKNOWN.value,
    )
    legal_field: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    effective_statuses: Mapped[list[EffectiveStatus]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class CrawlRun(TimestampMixin, Base):
    __tablename__ = "crawl_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'PARTIAL', 'FAILED')",
            name="ck_crawl_runs_status",
        ),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="ck_crawl_runs_time_order",
        ),
        Index("idx_crawl_runs_source_started", "source", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="RUNNING", server_default="RUNNING"
    )
    crawler_version: Mapped[str] = mapped_column(String(100), nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    error_message: Mapped[str | None] = mapped_column(Text)


class DocumentVersion(TimestampMixin, Base):
    """Immutable content snapshot observed during a crawl."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "version_number", name="uq_document_versions_number"
        ),
        UniqueConstraint(
            "document_id", "content_hash", name="uq_document_versions_hash"
        ),
        UniqueConstraint("id", "document_id", name="uq_document_versions_id_doc"),
        CheckConstraint("version_number > 0", name="ck_document_versions_number"),
        CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_sha256",
        ),
        Index("idx_document_versions_document", "document_id", "version_number"),
        Index("idx_document_versions_fetched_at", "fetched_at"),
        Index(
            "uq_document_versions_source_revision",
            "document_id",
            "source_revision",
            unique=True,
            postgresql_where=text("source_revision IS NOT NULL"),
        ),
        CheckConstraint(
            "content_valid_to IS NULL OR content_valid_from IS NULL "
            "OR content_valid_to > content_valid_from",
            name="ck_document_versions_content_date_order",
        ),
        ExcludeConstraint(
            ("document_id", "="),
            ("content_valid_period", "&&"),
            name="ex_document_versions_content_no_overlap",
            using="gist",
            where=text("content_valid_period IS NOT NULL"),
        ),
        Index(
            "idx_document_versions_content_period",
            "content_valid_period",
            postgresql_using="gist",
        ),
        Index(
            "uq_document_versions_current",
            "document_id",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    crawl_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("crawl_runs.id", ondelete="SET NULL")
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_revision: Mapped[str | None] = mapped_column(String(255))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_valid_from: Mapped[date | None] = mapped_column(Date)
    content_valid_to: Mapped[date | None] = mapped_column(Date)
    content_valid_period: Mapped[Any | None] = mapped_column(
        DATERANGE,
        Computed(
            "CASE WHEN content_valid_from IS NULL THEN NULL "
            "ELSE daterange(content_valid_from, content_valid_to, '[)') END",
            persisted=True,
        ),
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    raw_storage_uri: Mapped[str | None] = mapped_column(Text)
    normalized_storage_uri: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(String(100))
    chunker_version: Mapped[str | None] = mapped_column(String(100))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    document: Mapped[Document] = relationship(back_populates="versions")
    articles: Mapped[list[Article]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class Article(TimestampMixin, Base):
    __tablename__ = "articles"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            ondelete="CASCADE",
            name="fk_articles_version_document",
        ),
        UniqueConstraint("id", "version_id", name="uq_articles_id_version"),
        UniqueConstraint(
            "version_id", "article_key", name="uq_articles_version_key"
        ),
        CheckConstraint("ordinal >= 0", name="ck_articles_ordinal"),
        Index("idx_articles_document_number", "document_id", "article_number"),
        Index("idx_articles_version_ordinal", "version_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    parent_article_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("articles.id", ondelete="SET NULL")
    )
    article_key: Mapped[str] = mapped_column(String(500), nullable=False)
    article_number: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    path: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    text_content: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    version: Mapped[DocumentVersion] = relationship(back_populates="articles")


class Chunk(TimestampMixin, Base):
    """Chunk metadata and Vector DB pointer; embeddings stay in Qdrant."""

    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "document_id"],
            ["document_versions.id", "document_versions.document_id"],
            ondelete="CASCADE",
            name="fk_chunks_version_document",
        ),
        ForeignKeyConstraint(
            ["article_id", "version_id"],
            ["articles.id", "articles.version_id"],
            ondelete="CASCADE",
            name="fk_chunks_article_version",
        ),
        UniqueConstraint("id", "version_id", name="uq_chunks_id_version"),
        UniqueConstraint("version_id", "external_id", name="uq_chunks_version_external"),
        CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal"),
        CheckConstraint("approx_token_count >= 0", name="ck_chunks_tokens"),
        CheckConstraint(
            "(qdrant_collection IS NULL) = (qdrant_point_id IS NULL)",
            name="ck_chunks_qdrant_reference",
        ),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_article", "article_id"),
        Index("idx_chunks_parent", "parent_chunk_id"),
        Index("idx_chunks_version_indexable", "version_id", "is_indexable"),
        Index(
            "idx_chunks_qdrant_point",
            "qdrant_collection",
            "qdrant_point_id",
            unique=True,
            postgresql_where=text("qdrant_point_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    article_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    external_id: Mapped[str] = mapped_column(String(500), nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    structure_type: Mapped[str | None] = mapped_column(String(100))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    is_indexable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_text: Mapped[str | None] = mapped_column(Text)
    approx_token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    qdrant_collection: Mapped[str | None] = mapped_column(String(255))
    qdrant_point_id: Mapped[str | None] = mapped_column(String(255))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class ChunkVectorRef(TimestampMixin, Base):
    """Release-aware pointer from one chunk to one vector collection."""

    __tablename__ = "chunk_vector_refs"
    __table_args__ = (
        UniqueConstraint(
            "collection", "point_id", name="uq_chunk_vector_refs_point"
        ),
        Index("idx_chunk_vector_refs_release", "index_release", "collection"),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    collection: Mapped[str] = mapped_column(String(255), primary_key=True)
    point_id: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    index_release: Mapped[str | None] = mapped_column(String(255))


class EffectiveStatus(TimestampMixin, Base):
    """Non-overlapping valid-time history for one legal document."""

    __tablename__ = "effective_status"
    __table_args__ = (
        CheckConstraint(
            f"status IN ({LEGAL_STATUS_VALUES})", name="ck_effective_status_value"
        ),
        CheckConstraint(
            "valid_to IS NULL OR valid_to > valid_from",
            name="ck_effective_status_date_order",
        ),
        ExcludeConstraint(
            ("document_id", "="),
            ("valid_period", "&&"),
            name="ex_effective_status_no_overlap",
            using="gist",
        ),
        Index("idx_effective_status_lookup", "document_id", "valid_from", "valid_to"),
        Index("idx_effective_status_period", "valid_period", postgresql_using="gist"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)
    valid_period: Mapped[Any] = mapped_column(
        DATERANGE,
        Computed("daterange(valid_from, valid_to, '[)')", persisted=True),
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    document: Mapped[Document] = relationship(back_populates="effective_statuses")


class ProvisionEffectiveStatus(TimestampMixin, Base):
    """Optional valid-time status for one normalized legal provision."""

    __tablename__ = "provision_effective_status"
    __table_args__ = (
        CheckConstraint(
            f"status IN ({LEGAL_STATUS_VALUES})",
            name="ck_provision_effective_status_value",
        ),
        CheckConstraint(
            "valid_to IS NULL OR valid_to > valid_from",
            name="ck_provision_effective_status_date_order",
        ),
        ExcludeConstraint(
            ("document_id", "="),
            ("provision_key", "="),
            ("valid_period", "&&"),
            name="ex_provision_effective_status_no_overlap",
            using="gist",
        ),
        Index(
            "idx_provision_effective_status_lookup",
            "document_id",
            "provision_key",
            "valid_from",
            "valid_to",
        ),
        Index(
            "idx_provision_effective_status_period",
            "valid_period",
            postgresql_using="gist",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
    )
    provision_key: Mapped[str] = mapped_column(String(500), nullable=False)
    article: Mapped[str | None] = mapped_column(String(100))
    clause: Mapped[str | None] = mapped_column(String(100))
    point: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date)
    valid_period: Mapped[Any] = mapped_column(
        DATERANGE,
        Computed("daterange(valid_from, valid_to, '[)')", persisted=True),
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )


class DocumentRelationship(TimestampMixin, Base):
    __tablename__ = "document_relationships"
    __table_args__ = (
        CheckConstraint(
            f"relationship_type IN ({RELATIONSHIP_TYPE_VALUES})",
            name="ck_document_relationships_type",
        ),
        CheckConstraint(
            "target_document_id IS NOT NULL OR target_external_id IS NOT NULL "
            "OR target_document_number IS NOT NULL",
            name="ck_document_relationships_target",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL "
            "OR effective_to > effective_from",
            name="ck_document_relationships_date_order",
        ),
        Index("idx_relationships_source_type", "source_document_id", "relationship_type"),
        Index("idx_relationships_target_type", "target_document_id", "relationship_type"),
        UniqueConstraint(
            "source_document_id",
            "target_external_id",
            "source_relationship",
            name="uq_document_relationships_source_edge",
        ),
        Index("idx_relationships_unresolved", "target_external_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    source_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    target_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_relationship: Mapped[str] = mapped_column(String(255), nullable=False)
    target_external_id: Mapped[str | None] = mapped_column(String(255))
    target_document_number: Mapped[str | None] = mapped_column(String(255))
    citation_text: Mapped[str | None] = mapped_column(Text)
    source_article: Mapped[str | None] = mapped_column(String(100))
    target_article: Mapped[str | None] = mapped_column(String(100))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class AmendmentEvent(TimestampMixin, Base):
    """Dated official event that changes a document or provision."""

    __tablename__ = "amendment_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('AMENDS', 'SUPPLEMENTS', 'REPEALS', 'REPLACES', "
            "'SUSPENDS', 'RESTORES')",
            name="ck_amendment_events_type",
        ),
        UniqueConstraint(
            "source_document_id", "target_document_id", "event_type",
            "effective_date", "source_article", "target_provision_key",
            name="uq_amendment_events_identity",
        ),
        Index("idx_amendment_events_target_date", "target_document_id", "effective_date"),
        UniqueConstraint("event_key", name="uq_amendment_events_event_key"),
        UniqueConstraint(
            "source_relationship_id",
            name="uq_amendment_events_source_relationship",
        ),
        CheckConstraint(
            "target_document_id IS NOT NULL OR target_external_id IS NOT NULL",
            name="ck_amendment_events_target",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    event_key: Mapped[str] = mapped_column(String(255), nullable=False)
    source_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    target_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    target_external_id: Mapped[str | None] = mapped_column(String(255))
    source_relationship_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_relationships.id", ondelete="CASCADE"),
    )
    source_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date)
    source_article: Mapped[str | None] = mapped_column(String(100))
    target_provision_key: Mapped[str | None] = mapped_column(String(500))
    citation_text: Mapped[str | None] = mapped_column(Text)
    source_reference: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class ProvenanceRecord(TimestampMixin, Base):
    """Append-only source assertion for legal metadata and content."""

    __tablename__ = "provenance_records"
    __table_args__ = (
        CheckConstraint(
            "entity_type IN ('DOCUMENT', 'DOCUMENT_VERSION', 'EFFECTIVE_STATUS', "
            "'PROVISION_STATUS', 'RELATIONSHIP', 'AMENDMENT_EVENT')",
            name="ck_provenance_records_entity_type",
        ),
        CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_provenance_records_sha256",
        ),
        UniqueConstraint(
            "entity_type", "entity_id", "source_reference", "source_revision",
            name="uq_provenance_records_source",
        ),
        Index("idx_provenance_records_entity", "entity_type", "entity_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    crawl_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("crawl_runs.id", ondelete="SET NULL")
    )
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_reference: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    publisher: Mapped[str | None] = mapped_column(String(255))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    content_hash: Mapped[str | None] = mapped_column(String(64))
    is_official: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    verification_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="UNVERIFIED"
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class CrawlRecord(TimestampMixin, Base):
    __tablename__ = "crawl_records"
    __table_args__ = (
        UniqueConstraint("crawl_run_id", "source_url", name="uq_crawl_records_run_url"),
        CheckConstraint(
            "http_status IS NULL OR (http_status >= 100 AND http_status <= 599)",
            name="ck_crawl_records_http",
        ),
        CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_crawl_records_sha256",
        ),
        Index("idx_crawl_records_document", "document_id", "fetched_at"),
        Index("idx_crawl_records_hash", "content_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    crawl_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("crawl_runs.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL")
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    http_status: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    etag: Mapped[str | None] = mapped_column(String(255))
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )


class ChatProject(TimestampMixin, Base):
    """User-facing grouping for conversations, isolated from the legal corpus."""

    __tablename__ = "chat_projects"
    __table_args__ = (
        Index("idx_chat_projects_workspace_updated", "workspace_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    workspace_id: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    share_token: Mapped[str | None] = mapped_column(String(255), unique=True)
    is_shared: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )

    conversations: Mapped[list[ChatConversation]] = relationship(
        back_populates="project"
    )


class ChatConversation(TimestampMixin, Base):
    """Persistent conversation metadata owned by one logical workspace."""

    __tablename__ = "chat_conversations"
    __table_args__ = (
        Index(
            "idx_chat_conversations_workspace_updated",
            "workspace_id", "pinned", "updated_at",
        ),
        Index("idx_chat_conversations_project", "project_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    workspace_id: Mapped[str] = mapped_column(String(100), nullable=False)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_projects.id", ondelete="SET NULL"),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    pinned: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    share_token: Mapped[str | None] = mapped_column(String(255), unique=True)
    is_shared: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )

    project: Mapped[ChatProject | None] = relationship(
        back_populates="conversations"
    )
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class ChatMessage(TimestampMixin, Base):
    """One durable user or assistant turn and its verification metadata."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')", name="ck_chat_messages_role"
        ),
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'REFUSED', 'FAILED')",
            name="ck_chat_messages_status",
        ),
        UniqueConstraint(
            "conversation_id", "client_message_id",
            name="uq_chat_messages_client_message",
        ),
        Index("idx_chat_messages_conversation_created", "conversation_id", "created_at"),
        Index("idx_chat_messages_request", "request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    client_message_id: Mapped[str | None] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    claims: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    limitations: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    as_of: Mapped[date | None] = mapped_column(Date)
    request_id: Mapped[str | None] = mapped_column(String(100))
    semantic_status: Mapped[str | None] = mapped_column(String(32))
    coverage_status: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[str | None] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(100))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")


class DocumentSource(TimestampMixin, Base):
    """A human-readable source URL discovered for one legal document."""

    __tablename__ = "document_sources"
    __table_args__ = (
        CheckConstraint(
            "verification_status IN ('CANDIDATE', 'VERIFIED', 'REJECTED')",
            name="ck_document_sources_verification_status",
        ),
        UniqueConstraint("document_id", "url", name="uq_document_sources_document_url"),
        Index(
            "idx_document_sources_resolution",
            "document_id", "verification_status", "is_active", "verified_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="MCP_GOOGLE", server_default="MCP_GOOGLE"
    )
    verification_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="CANDIDATE", server_default="CANDIDATE"
    )
    match_score: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict,
        server_default=text("'{}'::jsonb")
    )
