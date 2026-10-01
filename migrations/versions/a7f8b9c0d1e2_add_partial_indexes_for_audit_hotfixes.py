"""add partial indexes for audit hotfixes

Revision ID: a7f8b9c0d1e2
Revises: 6b6d7c8e9f10
Create Date: 2026-10-02 10:00:00

This migration supports the Phase 1 hotfixes by adding partial indexes that
allow fast lookups for:

* ``provision_effective_status`` rows that are still UNKNOWN on documents
  flagged as ``PARTIALLY_EFFECTIVE``. Used by the E1+E2 fail-closed patch
  in :mod:`src.database.queries`.
* ``amendment_events`` keyed by ``(target_document_id, target_provision_key)``
  to make the E9 amendment lookup cheap inside the hot retrieval path.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7f8b9c0d1e2"
down_revision: Union[str, Sequence[str], None] = "6b6d7c8e9f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # E1+E2: fast filter for "this PARTIALLY doc has no resolved provision".
    op.create_index(
        "idx_provision_status_unknown_partial",
        "provision_effective_status",
        ["document_id"],
        postgresql_where=sa.text("status = 'UNKNOWN'"),
    )
    # E9: amendment lookup by target document and provision key.
    op.create_index(
        "idx_amendment_events_target_key",
        "amendment_events",
        ["target_document_id", "target_provision_key"],
    )
    # E9: amendment lookup by source document when resolving "văn bản nào
    # sửa đổi văn bản đang xét".
    op.create_index(
        "idx_amendment_events_source_effective_date",
        "amendment_events",
        ["source_document_id", "effective_date"],
        postgresql_where=sa.text("effective_date IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "idx_amendment_events_source_effective_date",
        table_name="amendment_events",
    )
    op.drop_index(
        "idx_amendment_events_target_key",
        table_name="amendment_events",
    )
    op.drop_index(
        "idx_provision_status_unknown_partial",
        table_name="provision_effective_status",
    )