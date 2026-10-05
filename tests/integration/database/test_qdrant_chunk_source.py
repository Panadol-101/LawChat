from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from database import Chunk, Document, DocumentVersion
from indexing import PostgresChunkSource


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


def test_chunk_source_reads_current_indexable_payload_without_mutation():
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    collection = f"read-only-test-{uuid.uuid4().hex}"
    suffix = uuid.uuid4().hex
    document_external_id = f"chunk-source-{suffix}"
    try:
        with Session(engine) as session:
            document = Document(
                external_id=document_external_id,
                title="Văn bản kiểm thử chunk source",
                document_number=f"{suffix[:8]}/TEST",
                source_url=f"https://example.test/{suffix}",
                status="EFFECTIVE",
            )
            session.add(document)
            session.flush()
            version = DocumentVersion(
                document_id=document.id,
                version_number=1,
                content_hash=suffix.ljust(64, "0")[:64],
                source_url=document.source_url,
                is_current=True,
            )
            session.add(version)
            session.flush()
            session.add_all(
                [
                    Chunk(
                        version_id=version.id,
                        document_id=document.id,
                        external_id=f"{document_external_id}::article_{index}",
                        chunk_type="article",
                        strategy="article",
                        ordinal=index,
                        is_indexable=True,
                        text_content=f"Điều {index}. Nội dung kiểm thử.",
                        retrieval_text=f"Điều {index}. Nội dung kiểm thử.",
                        approx_token_count=8,
                        metadata_json={"article": str(index)},
                    )
                    for index in range(1, 4)
                ]
            )
            session.commit()

        batches = list(
            PostgresChunkSource(engine, fetch_size=2).iter_batches(
                collection=collection,
                limit=3,
            )
        )
        chunks = [chunk for batch in batches for chunk in batch]
        assert len(chunks) == 3
        assert all(chunk.retrieval_text for chunk in chunks)
        assert all(chunk.payload["doc_id"] for chunk in chunks)
        assert all(chunk.payload["chunk_id"] == chunk.external_id for chunk in chunks)
    finally:
        with engine.begin() as connection:
            document_id = connection.execute(
                Document.__table__.select()
                .with_only_columns(Document.id)
                .where(Document.external_id == document_external_id)
            ).scalar_one_or_none()
            if document_id is not None:
                connection.execute(
                    delete(Document).where(Document.id == document_id)
                )
        engine.dispose()
