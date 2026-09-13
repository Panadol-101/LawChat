"""add provision-level valid-time status

Revision ID: e84b7c1a920d
Revises: d71c4a82f603
Create Date: 2026-09-01 12:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e84b7c1a920d"
down_revision: Union[str, Sequence[str], None] = "d71c4a82f603"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "provision_effective_status",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("source_version_id", sa.UUID(), nullable=True),
        sa.Column("provision_key", sa.String(length=500), nullable=False),
        sa.Column("article", sa.String(length=100), nullable=True),
        sa.Column("clause", sa.String(length=100), nullable=True),
        sa.Column("point", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column(
            "valid_period",
            postgresql.DATERANGE(),
            sa.Computed("daterange(valid_from, valid_to, '[)')", persisted=True),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('DRAFT', 'NOT_YET_EFFECTIVE', 'EFFECTIVE', "
            "'PARTIALLY_EFFECTIVE', 'SUSPENDED', 'EXPIRED', 'REPEALED', 'UNKNOWN')",
            name="ck_provision_effective_status_value",
        ),
        sa.CheckConstraint(
            "valid_to IS NULL OR valid_to > valid_from",
            name="ck_provision_effective_status_date_order",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_version_id"],
            ["document_versions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        postgresql.ExcludeConstraint(
            ("document_id", "="),
            ("provision_key", "="),
            ("valid_period", "&&"),
            name="ex_provision_effective_status_no_overlap",
            using="gist",
        ),
    )
    op.create_index(
        "idx_provision_effective_status_lookup",
        "provision_effective_status",
        ["document_id", "provision_key", "valid_from", "valid_to"],
    )
    op.create_index(
        "idx_provision_effective_status_period",
        "provision_effective_status",
        ["valid_period"],
        postgresql_using="gist",
    )


def downgrade() -> None:
    op.drop_table("provision_effective_status")
