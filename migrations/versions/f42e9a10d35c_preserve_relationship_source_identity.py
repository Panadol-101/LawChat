"""preserve relationship source identity

Revision ID: f42e9a10d35c
Revises: b978056a67b6
Create Date: 2026-08-25 03:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f42e9a10d35c"
down_revision: Union[str, Sequence[str], None] = "b978056a67b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_relationships",
        sa.Column("source_relationship", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "document_relationships",
        sa.Column("target_external_id", sa.String(length=255), nullable=True),
    )
    op.execute(
        "UPDATE document_relationships "
        "SET source_relationship = relationship_type"
    )
    op.alter_column(
        "document_relationships", "source_relationship", nullable=False
    )
    op.drop_constraint(
        "ck_document_relationships_target",
        "document_relationships",
        type_="check",
    )
    op.create_check_constraint(
        "ck_document_relationships_target",
        "document_relationships",
        "target_document_id IS NOT NULL OR target_external_id IS NOT NULL "
        "OR target_document_number IS NOT NULL",
    )
    op.drop_index(
        "idx_relationships_unresolved",
        table_name="document_relationships",
    )
    op.create_index(
        "idx_relationships_unresolved",
        "document_relationships",
        ["target_external_id"],
    )
    op.create_unique_constraint(
        "uq_document_relationships_source_edge",
        "document_relationships",
        ["source_document_id", "target_external_id", "source_relationship"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_document_relationships_source_edge",
        "document_relationships",
        type_="unique",
    )
    op.drop_index(
        "idx_relationships_unresolved",
        table_name="document_relationships",
    )
    op.create_index(
        "idx_relationships_unresolved",
        "document_relationships",
        ["target_document_number"],
    )
    op.drop_constraint(
        "ck_document_relationships_target",
        "document_relationships",
        type_="check",
    )
    op.create_check_constraint(
        "ck_document_relationships_target",
        "document_relationships",
        "target_document_id IS NOT NULL OR target_document_number IS NOT NULL",
    )
    op.drop_column("document_relationships", "target_external_id")
    op.drop_column("document_relationships", "source_relationship")
