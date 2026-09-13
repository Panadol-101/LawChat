"""add historical content validity and provenance

Revision ID: 2f5a1c7d9b01
Revises: e84b7c1a920d
Create Date: 2026-09-02 10:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "2f5a1c7d9b01"
down_revision: Union[str, Sequence[str], None] = "e84b7c1a920d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_versions", sa.Column("content_valid_from", sa.Date(), nullable=True)
    )
    op.add_column(
        "document_versions", sa.Column("content_valid_to", sa.Date(), nullable=True)
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "content_valid_period",
            postgresql.DATERANGE(),
            sa.Computed(
                "CASE WHEN content_valid_from IS NULL THEN NULL "
                "ELSE daterange(content_valid_from, content_valid_to, '[)') END",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "ck_document_versions_content_date_order",
        "document_versions",
        "content_valid_to IS NULL OR content_valid_from IS NULL "
        "OR content_valid_to > content_valid_from",
    )
    op.create_exclude_constraint(
        "ex_document_versions_content_no_overlap",
        "document_versions",
        ("document_id", "="),
        ("content_valid_period", "&&"),
        using="gist",
        where=sa.text("content_valid_period IS NOT NULL"),
    )
    op.create_index(
        "idx_document_versions_content_period",
        "document_versions",
        ["content_valid_period"],
        postgresql_using="gist",
    )
    op.create_index(
        "uq_document_versions_source_revision",
        "document_versions",
        ["document_id", "source_revision"],
        unique=True,
        postgresql_where=sa.text("source_revision IS NOT NULL"),
    )

    op.create_table(
        "chunk_vector_refs",
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("collection", sa.String(length=255), nullable=False),
        sa.Column("point_id", sa.String(length=255), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("index_release", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", "collection"),
        sa.UniqueConstraint("collection", "point_id", name="uq_chunk_vector_refs_point"),
    )
    op.create_index(
        "idx_chunk_vector_refs_release",
        "chunk_vector_refs",
        ["index_release", "collection"],
    )
    op.execute(
        "INSERT INTO chunk_vector_refs (chunk_id, collection, point_id) "
        "SELECT id, qdrant_collection, qdrant_point_id FROM chunks "
        "WHERE qdrant_collection IS NOT NULL AND qdrant_point_id IS NOT NULL "
        "ON CONFLICT DO NOTHING"
    )

    op.create_table(
        "amendment_events",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("event_key", sa.String(length=255), nullable=False),
        sa.Column("source_document_id", sa.UUID(), nullable=False),
        sa.Column("target_document_id", sa.UUID(), nullable=True),
        sa.Column("target_external_id", sa.String(length=255), nullable=True),
        sa.Column("source_relationship_id", sa.UUID(), nullable=True),
        sa.Column("source_version_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("source_article", sa.String(length=100), nullable=True),
        sa.Column("target_provision_key", sa.String(length=500), nullable=True),
        sa.Column("citation_text", sa.Text(), nullable=True),
        sa.Column("source_reference", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('AMENDS', 'SUPPLEMENTS', 'REPEALS', 'REPLACES', "
            "'SUSPENDS', 'RESTORES')",
            name="ck_amendment_events_type",
        ),
        sa.CheckConstraint(
            "target_document_id IS NOT NULL OR target_external_id IS NOT NULL",
            name="ck_amendment_events_target",
        ),
        sa.ForeignKeyConstraint(["source_document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_relationship_id"], ["document_relationships.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_version_id"], ["document_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_key", name="uq_amendment_events_event_key"),
        sa.UniqueConstraint(
            "source_relationship_id",
            name="uq_amendment_events_source_relationship",
        ),
        sa.UniqueConstraint(
            "source_document_id", "target_document_id", "event_type",
            "effective_date", "source_article", "target_provision_key",
            name="uq_amendment_events_identity",
        ),
    )
    op.create_index(
        "idx_amendment_events_target_date",
        "amendment_events",
        ["target_document_id", "effective_date"],
    )

    op.create_table(
        "provenance_records",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("crawl_run_id", sa.UUID(), nullable=True),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_reference", sa.Text(), nullable=False),
        sa.Column("dataset_revision", sa.String(length=255), nullable=False),
        sa.Column("publisher", sa.String(length=255), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_revision", sa.String(length=255), server_default="", nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("is_official", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("verification_status", sa.String(length=32), server_default="UNVERIFIED", nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "entity_type IN ('DOCUMENT', 'DOCUMENT_VERSION', 'EFFECTIVE_STATUS', "
            "'PROVISION_STATUS', 'RELATIONSHIP', 'AMENDMENT_EVENT')",
            name="ck_provenance_records_entity_type",
        ),
        sa.CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_provenance_records_sha256",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["crawl_run_id"], ["crawl_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_type", "entity_id", "source_reference", "source_revision",
            name="uq_provenance_records_source",
        ),
    )
    op.create_index(
        "idx_provenance_records_entity",
        "provenance_records",
        ["entity_type", "entity_id"],
    )


def downgrade() -> None:
    op.drop_table("provenance_records")
    op.drop_table("amendment_events")
    op.drop_table("chunk_vector_refs")
    op.drop_index("uq_document_versions_source_revision", table_name="document_versions")
    op.drop_index("idx_document_versions_content_period", table_name="document_versions")
    op.execute(
        "ALTER TABLE document_versions "
        "DROP CONSTRAINT ex_document_versions_content_no_overlap"
    )
    op.drop_constraint(
        "ck_document_versions_content_date_order",
        "document_versions",
        type_="check",
    )
    op.drop_column("document_versions", "content_valid_period")
    op.drop_column("document_versions", "content_valid_to")
    op.drop_column("document_versions", "content_valid_from")
