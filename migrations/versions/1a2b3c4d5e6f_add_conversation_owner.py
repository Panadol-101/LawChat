"""add owner to chat conversations

Conversations without a project used to be reachable by every logged-in user.
Ownership now lives on the conversation itself.

Revision ID: 1a2b3c4d5e6f
Revises: e4f5a6b7c8d9
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = "e4f5a6b7c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_conversations",
        sa.Column("user_id", sa.UUID(), nullable=True),
    )

    # Conversations in a project belong to the project owner.
    op.execute(
        """
        UPDATE chat_conversations AS c
        SET user_id = p.user_id
        FROM chat_projects AS p
        WHERE c.project_id = p.id
          AND c.user_id IS NULL
        """
    )

    # Orphaned conversations (no project) go to the admin account.
    op.execute(
        """
        UPDATE chat_conversations
        SET user_id = (
            SELECT id
            FROM users
            WHERE username = 'admin'
            LIMIT 1
        )
        WHERE user_id IS NULL
        """
    )

    remaining = op.get_bind().execute(
        sa.text("SELECT count(*) FROM chat_conversations WHERE user_id IS NULL")
    ).scalar_one()
    if remaining:
        raise RuntimeError(
            f"{remaining} chat_conversations have no owner and no 'admin' user "
            "exists to adopt them; create the admin user and rerun the migration."
        )

    op.alter_column("chat_conversations", "user_id", nullable=False)

    op.create_foreign_key(
        "fk_chat_conversations_user_id_users",
        "chat_conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_index(
        "ix_chat_conversations_user_workspace_updated",
        "chat_conversations",
        ["user_id", "workspace_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chat_conversations_user_workspace_updated",
        table_name="chat_conversations",
    )
    op.drop_constraint(
        "fk_chat_conversations_user_id_users",
        "chat_conversations",
        type_="foreignkey",
    )
    op.drop_column("chat_conversations", "user_id")
