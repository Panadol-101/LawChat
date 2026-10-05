"""add user ownership and usage quota

Revision ID: 50e6d3b75a6a
Revises: de2ba91103cb
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "50e6d3b75a6a"
down_revision: Union[str, Sequence[str], None] = "de2ba91103cb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New users default to USER.
    # Existing users keep their current role.
    op.alter_column(
        "users",
        "role",
        server_default="USER",
    )

    # Add ownership to chat projects.
    op.add_column(
        "chat_projects",
        sa.Column(
            "user_id",
            sa.UUID(),
            nullable=True,
        ),
    )

    # Existing database currently has exactly one user: admin.
    op.execute(
        """
        UPDATE chat_projects
        SET user_id = (
            SELECT id
            FROM users
            WHERE username = 'admin'
            LIMIT 1
        )
        WHERE user_id IS NULL
        """
    )

    # Ownership is mandatory after backfill.
    op.alter_column(
        "chat_projects",
        "user_id",
        nullable=False,
    )

    op.create_foreign_key(
        "fk_chat_projects_user_id_users",
        "chat_projects",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_index(
        "ix_chat_projects_user_id",
        "chat_projects",
        ["user_id"],
    )

    # Per-user token quota.
    op.create_table(
        "user_usage",
        sa.Column(
            "id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "token_limit",
            sa.BigInteger(),
            nullable=False,
            server_default="100000",
        ),
        sa.Column(
            "tokens_used",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "period_start",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "period_end",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_usage_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            name="uq_user_usage_user_id",
        ),
    )

    op.create_index(
        "ix_user_usage_user_id",
        "user_usage",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_usage_user_id",
        table_name="user_usage",
    )

    op.drop_table("user_usage")

    op.drop_index(
        "ix_chat_projects_user_id",
        table_name="chat_projects",
    )

    op.drop_constraint(
        "fk_chat_projects_user_id_users",
        "chat_projects",
        type_="foreignkey",
    )

    op.drop_column(
        "chat_projects",
        "user_id",
    )

    op.alter_column(
        "users",
        "role",
        server_default="ADMIN",
    )
