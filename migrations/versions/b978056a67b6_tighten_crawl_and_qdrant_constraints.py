"""tighten crawl and qdrant constraints

Revision ID: b978056a67b6
Revises: c9b59054446e
Create Date: 2026-08-25 02:27:26.007766
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b978056a67b6'
down_revision: Union[str, Sequence[str], None] = 'c9b59054446e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        'ck_chunks_qdrant_reference',
        'chunks',
        '(qdrant_collection IS NULL) = (qdrant_point_id IS NULL)',
    )
    op.drop_constraint(
        'ck_crawl_records_http', 'crawl_records', type_='check'
    )
    op.alter_column('crawl_records', 'http_status',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.create_check_constraint(
        'ck_crawl_records_http',
        'crawl_records',
        'http_status IS NULL OR (http_status >= 100 AND http_status <= 599)',
    )
    op.create_check_constraint(
        'ck_crawl_records_sha256',
        'crawl_records',
        "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint('ck_crawl_records_sha256', 'crawl_records', type_='check')
    op.drop_constraint('ck_crawl_records_http', 'crawl_records', type_='check')
    op.alter_column('crawl_records', 'http_status',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.create_check_constraint(
        'ck_crawl_records_http',
        'crawl_records',
        'http_status >= 100 AND http_status <= 599',
    )
    op.drop_constraint('ck_chunks_qdrant_reference', 'chunks', type_='check')
