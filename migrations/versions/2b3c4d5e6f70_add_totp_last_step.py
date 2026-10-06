"""add totp_last_step to users

Stores the time-step of the last accepted TOTP code so a code cannot be
replayed inside its validity window.

Revision ID: 2b3c4d5e6f70
Revises: 1a2b3c4d5e6f
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2b3c4d5e6f70"
down_revision: Union[str, Sequence[str], None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("totp_last_step", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "totp_last_step")
