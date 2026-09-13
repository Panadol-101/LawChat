"""allow multi-value legal fields

Revision ID: a13d8c7ef921
Revises: f42e9a10d35c
Create Date: 2026-08-25 03:30:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a13d8c7ef921"
down_revision: Union[str, Sequence[str], None] = "f42e9a10d35c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "documents",
        "legal_field",
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Refuse to truncate real data silently. PostgreSQL will reject downgrade
    # if a legal field no longer fits the old limit.
    op.alter_column(
        "documents",
        "legal_field",
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
