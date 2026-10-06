"""add user approval status and audit actor/details

New registrations start PENDING and need administrator approval. Existing
accounts are backfilled as ACTIVE (is_active is left unchanged).

audit_logs gains actor_user_id (the administrator who acted; user_id stays the
target account) and details, which keeps the target's username after the
account is deleted.

Revision ID: 3c4d5e6f7081
Revises: 2b3c4d5e6f70
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "3c4d5e6f7081"
down_revision: Union[str, Sequence[str], None] = "2b3c4d5e6f70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The server default backfills every existing row as ACTIVE.
    op.add_column(
        "users",
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="ACTIVE",
        ),
    )
    op.create_check_constraint(
        "ck_users_status",
        "users",
        "status IN ('PENDING', 'ACTIVE', 'REJECTED')",
    )
    op.create_check_constraint(
        "ck_users_inactive_unless_active",
        "users",
        "status = 'ACTIVE' OR is_active = false",
    )
    op.create_index(
        "ix_users_pending_created_at",
        "users",
        ["created_at"],
        postgresql_where=sa.text("status = 'PENDING'"),
    )

    op.add_column(
        "audit_logs",
        sa.Column("actor_user_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_audit_logs_actor_user_id",
        "audit_logs",
        "users",
        ["actor_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_audit_logs_actor_user_id",
        "audit_logs",
        ["actor_user_id"],
    )
    op.add_column(
        "audit_logs",
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_audit_logs_event_created_at",
        "audit_logs",
        ["event", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_event_created_at", table_name="audit_logs")
    op.drop_column("audit_logs", "details")
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_constraint(
        "fk_audit_logs_actor_user_id",
        "audit_logs",
        type_="foreignkey",
    )
    op.drop_column("audit_logs", "actor_user_id")

    # PENDING/REJECTED users keep is_active = false, so they stay locked out.
    op.drop_index("ix_users_pending_created_at", table_name="users")
    op.drop_constraint("ck_users_inactive_unless_active", "users", type_="check")
    op.drop_constraint("ck_users_status", "users", type_="check")
    op.drop_column("users", "status")
