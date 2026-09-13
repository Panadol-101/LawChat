"""add chat projects, conversations and messages

Revision ID: 6b6d7c8e9f10
Revises: 2f5a1c7d9b01
Create Date: 2026-09-08 12:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "6b6d7c8e9f10"
down_revision: Union[str, Sequence[str], None] = "2f5a1c7d9b01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_projects",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("share_token", sa.String(length=255), nullable=True),
        sa.Column("is_shared", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("share_token"),
    )
    op.create_index(
        "idx_chat_projects_workspace_updated", "chat_projects",
        ["workspace_id", "updated_at"],
    )

    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("workspace_id", sa.String(length=100), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("pinned", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("share_token", sa.String(length=255), nullable=True),
        sa.Column("is_shared", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["chat_projects.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("share_token"),
    )
    op.create_index(
        "idx_chat_conversations_workspace_updated", "chat_conversations",
        ["workspace_id", "pinned", "updated_at"],
    )
    op.create_index(
        "idx_chat_conversations_project", "chat_conversations", ["project_id"]
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("client_message_id", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("content", sa.Text(), server_default="", nullable=False),
        sa.Column("citations", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("claims", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("limitations", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=True),
        sa.Column("request_id", sa.String(length=100), nullable=True),
        sa.Column("semantic_status", sa.String(length=32), nullable=True),
        sa.Column("coverage_status", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.String(length=20), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'REFUSED', 'FAILED')",
            name="ck_chat_messages_status",
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["chat_conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "client_message_id", name="uq_chat_messages_client_message"),
    )
    op.create_index(
        "idx_chat_messages_conversation_created", "chat_messages",
        ["conversation_id", "created_at"],
    )
    op.create_index("idx_chat_messages_request", "chat_messages", ["request_id"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_conversations")
    op.drop_table("chat_projects")
