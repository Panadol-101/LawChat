"""add resolved human-readable document sources

Revision ID: 7c7e8f9a0b11
Revises: 6b6d7c8e9f10
Create Date: 2026-09-08 17:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "7c7e8f9a0b11"
down_revision: Union[str, Sequence[str], None] = "6b6d7c8e9f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_sources",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("source_kind", sa.String(length=32), server_default="MCP_GOOGLE", nullable=False),
        sa.Column("verification_status", sa.String(length=20), server_default="CANDIDATE", nullable=False),
        sa.Column("match_score", sa.Integer(), server_default="0", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "verification_status IN ('CANDIDATE', 'VERIFIED', 'REJECTED')",
            name="ck_document_sources_verification_status",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "url", name="uq_document_sources_document_url"),
    )
    op.create_index(
        "idx_document_sources_resolution", "document_sources",
        ["document_id", "verification_status", "is_active", "verified_at"],
    )


def downgrade() -> None:
    op.drop_table("document_sources")
