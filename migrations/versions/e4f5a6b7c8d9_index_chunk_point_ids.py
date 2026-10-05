"""index chunk point ids for fast retrieval

Revision ID: e4f5a6b7c8d9
Revises: 9b1c2d3e4f50
Create Date: 2026-10-02 09:00:00
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, Sequence[str], None] = "9b1c2d3e4f50"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_chunk_vector_refs_point_id "
            "ON chunk_vector_refs(point_id)"
        )
    )
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_point_id "
            "ON chunks(qdrant_point_id) WHERE qdrant_point_id IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.execute(text("DROP INDEX IF EXISTS idx_chunk_vector_refs_point_id"))
    op.execute(text("DROP INDEX IF EXISTS idx_chunks_qdrant_point_id"))
