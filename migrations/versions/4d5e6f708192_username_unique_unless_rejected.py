"""username unique only among non-rejected users

REJECTED accounts are kept for audit/history but no longer reserve their
username, so the name can be registered again. ACTIVE and PENDING usernames
stay unique through a partial unique index.

Revision ID: 4d5e6f708192
Revises: 3c4d5e6f7081
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4d5e6f708192"
down_revision: Union[str, Sequence[str], None] = "3c4d5e6f7081"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("uq_users_username", "users", type_="unique")
    op.create_index(
        "uq_users_username_not_rejected",
        "users",
        ["username"],
        unique=True,
        postgresql_where=sa.text("status <> 'REJECTED'"),
    )


def downgrade() -> None:
    duplicates = op.get_bind().execute(
        sa.text(
            """
            SELECT username, count(*) AS cnt
            FROM users
            GROUP BY username
            HAVING count(*) > 1
            ORDER BY username
            """
        )
    ).all()
    if duplicates:
        names = ", ".join(f"{row.username} ({row.cnt})" for row in duplicates)
        # Users are never changed or deleted here; resolve the duplicates first.
        raise RuntimeError(
            "Cannot restore the unconditional unique constraint on users.username: "
            f"these usernames are shared by a REJECTED account and a newer one: {names}. "
            "Delete or rename the old REJECTED accounts, then rerun the downgrade."
        )

    op.drop_index("uq_users_username_not_rejected", table_name="users")
    op.create_unique_constraint("uq_users_username", "users", ["username"])
