"""add authentication

Revision ID: de2ba91103cb
Revises: 7c7e8f9a0b11
Create Date: 2026-09-22 22:56:23.409901
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "de2ba91103cb"
down_revision: Union[str, Sequence[str], None] = "7c7e8f9a0b11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # =========================
    # USERS
    # =========================
    op.create_table(
        "users",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "username",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "password_hash",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.String(50),
            nullable=False,
            server_default="ADMIN",
        ),
        sa.Column(
            "totp_secret",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "totp_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "username",
            name="uq_users_username",
        ),
    )

    # =========================
    # SESSIONS
    # =========================
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "token_hash",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "revoked_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_sessions_user_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "token_hash",
            name="uq_sessions_token_hash",
        ),
    )

    # =========================
    # AUDIT LOGS
    # =========================
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "user_id",
            sa.UUID(),
            nullable=True,
        ),
        sa.Column(
            "event",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "ip_address",
            sa.String(45),
            nullable=True,
        ),
        sa.Column(
            "user_agent",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_audit_logs_user_id",
            ondelete="SET NULL",
        ),
    )

    # =========================
    # INDEXES
    # =========================
    op.create_index(
        "ix_sessions_user_id",
        "sessions",
        ["user_id"],
    )

    op.create_index(
        "ix_sessions_expires_at",
        "sessions",
        ["expires_at"],
    )

    op.create_index(
        "ix_audit_logs_user_id",
        "audit_logs",
        ["user_id"],
    )

    op.create_index(
        "ix_audit_logs_created_at",
        "audit_logs",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_audit_logs_created_at",
        table_name="audit_logs",
    )

    op.drop_index(
        "ix_audit_logs_user_id",
        table_name="audit_logs",
    )

    op.drop_index(
        "ix_sessions_expires_at",
        table_name="sessions",
    )

    op.drop_index(
        "ix_sessions_user_id",
        table_name="sessions",
    )

    op.drop_table("audit_logs")
    op.drop_table("sessions")
    op.drop_table("users")
