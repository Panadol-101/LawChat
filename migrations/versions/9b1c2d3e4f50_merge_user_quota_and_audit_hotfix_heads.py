"""merge heads: user ownership quota and partial indexes for audit hotfixes

Revision ID: 9b1c2d3e4f50
Revises:
    50e6d3b75a6a (add user ownership and usage quota)
    a7f8b9c0d1e2 (add partial indexes for audit hotfixes)

Create Date: 2026-10-02 03:00:00

This merge revision unifies two parallel heads that diverged from
``6b6d7c8e9f10``. It performs no schema changes; its sole purpose is to
restore a single linear migration graph so that ``alembic upgrade head``
can resolve a unique head on a fresh database while still replaying both
branches (the user/quota branch and the audit-hotfix partial indexes
branch) without dropping either set of changes.
"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "9b1c2d3e4f50"
down_revision: Union[str, Sequence[str], None] = (
    "50e6d3b75a6a",
    "a7f8b9c0d1e2",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No-op merge revision: both parent heads must already be applied.
    pass


def downgrade() -> None:
    # No-op merge revision: downgrading simply un-merges the graph.
    pass