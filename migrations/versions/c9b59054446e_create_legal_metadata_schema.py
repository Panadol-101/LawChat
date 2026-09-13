"""create legal metadata schema

Revision ID: c9b59054446e
Revises:
Create Date: 2026-08-25 02:24:11.152565
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c9b59054446e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # UUID defaults and the mixed UUID/range GiST exclusion constraint.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table('crawl_runs',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('source', sa.String(length=100), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.String(length=16), server_default='RUNNING', nullable=False),
    sa.Column('crawler_version', sa.String(length=100), nullable=False),
    sa.Column('stats', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('RUNNING', 'SUCCEEDED', 'PARTIAL', 'FAILED')", name='ck_crawl_runs_status'),
    sa.CheckConstraint('finished_at IS NULL OR finished_at >= started_at', name='ck_crawl_runs_time_order'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_crawl_runs_source_started', 'crawl_runs', ['source', 'started_at'], unique=False)
    op.create_table('documents',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('external_id', sa.String(length=255), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('document_number', sa.String(length=255), nullable=True),
    sa.Column('document_type', sa.String(length=100), nullable=True),
    sa.Column('authority', sa.String(length=255), nullable=True),
    sa.Column('issued_date', sa.Date(), nullable=True),
    sa.Column('effective_date', sa.Date(), nullable=True),
    sa.Column('expiry_date', sa.Date(), nullable=True),
    sa.Column('status', sa.String(length=32), server_default='UNKNOWN', nullable=False),
    sa.Column('legal_field', sa.String(length=255), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('DRAFT', 'NOT_YET_EFFECTIVE', 'EFFECTIVE', 'PARTIALLY_EFFECTIVE', 'SUSPENDED', 'EXPIRED', 'REPEALED', 'UNKNOWN')", name='ck_documents_status'),
    sa.CheckConstraint('expiry_date IS NULL OR effective_date IS NULL OR expiry_date >= effective_date', name='ck_documents_date_order'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('external_id')
    )
    op.create_index('idx_documents_document_number', 'documents', ['document_number'], unique=False)
    op.create_index('idx_documents_effective_date', 'documents', ['effective_date'], unique=False)
    op.create_index('idx_documents_expiry_date', 'documents', ['expiry_date'], unique=False)
    op.create_index('idx_documents_filter_covering', 'documents', ['document_type', 'authority', 'legal_field', 'effective_date', 'expiry_date'], unique=False)
    op.create_table('document_versions',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('crawl_run_id', sa.UUID(), nullable=True),
    sa.Column('version_number', sa.Integer(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('source_revision', sa.String(length=255), nullable=True),
    sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_current', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('raw_storage_uri', sa.Text(), nullable=True),
    sa.Column('normalized_storage_uri', sa.Text(), nullable=True),
    sa.Column('parser_version', sa.String(length=100), nullable=True),
    sa.Column('chunker_version', sa.String(length=100), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name='ck_document_versions_sha256'),
    sa.CheckConstraint('version_number > 0', name='ck_document_versions_number'),
    sa.ForeignKeyConstraint(['crawl_run_id'], ['crawl_runs.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'content_hash', name='uq_document_versions_hash'),
    sa.UniqueConstraint('document_id', 'version_number', name='uq_document_versions_number'),
    sa.UniqueConstraint('id', 'document_id', name='uq_document_versions_id_doc')
    )
    op.create_index('idx_document_versions_document', 'document_versions', ['document_id', 'version_number'], unique=False)
    op.create_index('idx_document_versions_fetched_at', 'document_versions', ['fetched_at'], unique=False)
    op.create_index('uq_document_versions_current', 'document_versions', ['document_id'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('articles',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('version_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('parent_article_id', sa.UUID(), nullable=True),
    sa.Column('article_key', sa.String(length=500), nullable=False),
    sa.Column('article_number', sa.String(length=100), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('path', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('text_content', sa.Text(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ordinal >= 0', name='ck_articles_ordinal'),
    sa.ForeignKeyConstraint(['parent_article_id'], ['articles.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['version_id', 'document_id'], ['document_versions.id', 'document_versions.document_id'], name='fk_articles_version_document', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id', 'version_id', name='uq_articles_id_version'),
    sa.UniqueConstraint('version_id', 'article_key', name='uq_articles_version_key')
    )
    op.create_index('idx_articles_document_number', 'articles', ['document_id', 'article_number'], unique=False)
    op.create_index('idx_articles_version_ordinal', 'articles', ['version_id', 'ordinal'], unique=False)
    op.create_table('crawl_records',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('crawl_run_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=True),
    sa.Column('document_version_id', sa.UUID(), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('canonical_url', sa.Text(), nullable=True),
    sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('http_status', sa.Integer(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=True),
    sa.Column('etag', sa.String(length=255), nullable=True),
    sa.Column('last_modified', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('http_status >= 100 AND http_status <= 599', name='ck_crawl_records_http'),
    sa.ForeignKeyConstraint(['crawl_run_id'], ['crawl_runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['document_version_id'], ['document_versions.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('crawl_run_id', 'source_url', name='uq_crawl_records_run_url')
    )
    op.create_index('idx_crawl_records_document', 'crawl_records', ['document_id', 'fetched_at'], unique=False)
    op.create_index('idx_crawl_records_hash', 'crawl_records', ['content_hash'], unique=False)
    op.create_table('document_relationships',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('source_document_id', sa.UUID(), nullable=False),
    sa.Column('target_document_id', sa.UUID(), nullable=True),
    sa.Column('source_version_id', sa.UUID(), nullable=True),
    sa.Column('relationship_type', sa.String(length=32), nullable=False),
    sa.Column('target_document_number', sa.String(length=255), nullable=True),
    sa.Column('citation_text', sa.Text(), nullable=True),
    sa.Column('source_article', sa.String(length=100), nullable=True),
    sa.Column('target_article', sa.String(length=100), nullable=True),
    sa.Column('effective_from', sa.Date(), nullable=True),
    sa.Column('effective_to', sa.Date(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("relationship_type IN ('AMENDS', 'SUPPLEMENTS', 'REPEALS', 'REPLACES', 'GUIDES', 'IMPLEMENTS', 'CITES', 'RELATED_TO')", name='ck_document_relationships_type'),
    sa.CheckConstraint('effective_to IS NULL OR effective_from IS NULL OR effective_to > effective_from', name='ck_document_relationships_date_order'),
    sa.CheckConstraint('target_document_id IS NOT NULL OR target_document_number IS NOT NULL', name='ck_document_relationships_target'),
    sa.ForeignKeyConstraint(['source_document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_version_id'], ['document_versions.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['target_document_id'], ['documents.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_relationships_source_type', 'document_relationships', ['source_document_id', 'relationship_type'], unique=False)
    op.create_index('idx_relationships_target_type', 'document_relationships', ['target_document_id', 'relationship_type'], unique=False)
    op.create_index('idx_relationships_unresolved', 'document_relationships', ['target_document_number'], unique=False)
    op.create_table('effective_status',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('source_version_id', sa.UUID(), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('valid_period', postgresql.DATERANGE(), sa.Computed("daterange(valid_from, valid_to, '[)')", persisted=True), nullable=False),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    postgresql.ExcludeConstraint((sa.column('document_id'), '='), (sa.column('valid_period'), '&&'), using='gist', name='ex_effective_status_no_overlap'),
    sa.CheckConstraint("status IN ('DRAFT', 'NOT_YET_EFFECTIVE', 'EFFECTIVE', 'PARTIALLY_EFFECTIVE', 'SUSPENDED', 'EXPIRED', 'REPEALED', 'UNKNOWN')", name='ck_effective_status_value'),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name='ck_effective_status_date_order'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_version_id'], ['document_versions.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_effective_status_lookup', 'effective_status', ['document_id', 'valid_from', 'valid_to'], unique=False)
    op.create_index('idx_effective_status_period', 'effective_status', ['valid_period'], unique=False, postgresql_using='gist')
    op.create_table('chunks',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('version_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('article_id', sa.UUID(), nullable=True),
    sa.Column('parent_chunk_id', sa.UUID(), nullable=True),
    sa.Column('external_id', sa.String(length=500), nullable=False),
    sa.Column('chunk_type', sa.String(length=100), nullable=False),
    sa.Column('strategy', sa.String(length=100), nullable=False),
    sa.Column('structure_type', sa.String(length=100), nullable=True),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('is_indexable', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('text_content', sa.Text(), nullable=False),
    sa.Column('retrieval_text', sa.Text(), nullable=True),
    sa.Column('approx_token_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('qdrant_collection', sa.String(length=255), nullable=True),
    sa.Column('qdrant_point_id', sa.String(length=255), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('approx_token_count >= 0', name='ck_chunks_tokens'),
    sa.CheckConstraint('ordinal >= 0', name='ck_chunks_ordinal'),
    sa.ForeignKeyConstraint(['article_id', 'version_id'], ['articles.id', 'articles.version_id'], name='fk_chunks_article_version', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_chunk_id'], ['chunks.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['version_id', 'document_id'], ['document_versions.id', 'document_versions.document_id'], name='fk_chunks_version_document', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('id', 'version_id', name='uq_chunks_id_version'),
    sa.UniqueConstraint('version_id', 'external_id', name='uq_chunks_version_external')
    )
    op.create_index('idx_chunks_article', 'chunks', ['article_id'], unique=False)
    op.create_index('idx_chunks_document', 'chunks', ['document_id'], unique=False)
    op.create_index('idx_chunks_qdrant_point', 'chunks', ['qdrant_collection', 'qdrant_point_id'], unique=True, postgresql_where=sa.text('qdrant_point_id IS NOT NULL'))
    op.create_index('idx_chunks_version_indexable', 'chunks', ['version_id', 'is_indexable'], unique=False)


def downgrade() -> None:
    # Extensions may be shared by other schemas, so downgrade keeps them.
    op.drop_index('idx_chunks_version_indexable', table_name='chunks')
    op.drop_index('idx_chunks_qdrant_point', table_name='chunks', postgresql_where=sa.text('qdrant_point_id IS NOT NULL'))
    op.drop_index('idx_chunks_document', table_name='chunks')
    op.drop_index('idx_chunks_article', table_name='chunks')
    op.drop_table('chunks')
    op.drop_index('idx_effective_status_period', table_name='effective_status', postgresql_using='gist')
    op.drop_index('idx_effective_status_lookup', table_name='effective_status')
    op.drop_table('effective_status')
    op.drop_index('idx_relationships_unresolved', table_name='document_relationships')
    op.drop_index('idx_relationships_target_type', table_name='document_relationships')
    op.drop_index('idx_relationships_source_type', table_name='document_relationships')
    op.drop_table('document_relationships')
    op.drop_index('idx_crawl_records_hash', table_name='crawl_records')
    op.drop_index('idx_crawl_records_document', table_name='crawl_records')
    op.drop_table('crawl_records')
    op.drop_index('idx_articles_version_ordinal', table_name='articles')
    op.drop_index('idx_articles_document_number', table_name='articles')
    op.drop_table('articles')
    op.drop_index('uq_document_versions_current', table_name='document_versions', postgresql_where=sa.text('is_current'))
    op.drop_index('idx_document_versions_fetched_at', table_name='document_versions')
    op.drop_index('idx_document_versions_document', table_name='document_versions')
    op.drop_table('document_versions')
    op.drop_index('idx_documents_filter_covering', table_name='documents')
    op.drop_index('idx_documents_expiry_date', table_name='documents')
    op.drop_index('idx_documents_effective_date', table_name='documents')
    op.drop_index('idx_documents_document_number', table_name='documents')
    op.drop_table('documents')
    op.drop_index('idx_crawl_runs_source_started', table_name='crawl_runs')
    op.drop_table('crawl_runs')
