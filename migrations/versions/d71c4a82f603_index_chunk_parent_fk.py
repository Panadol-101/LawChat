"""index chunk parent foreign key

Revision ID: d71c4a82f603
Revises: a13d8c7ef921
Create Date: 2026-08-25 16:40:00
"""
from typing import Sequence, Union

from alembic import op


revision: str = "d71c4a82f603"
down_revision: Union[str, Sequence[str], None] = "a13d8c7ef921"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "idx_chunks_parent", "chunks", ["parent_chunk_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_chunks_parent", table_name="chunks")
