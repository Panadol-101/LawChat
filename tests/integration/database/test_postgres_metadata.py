from __future__ import annotations

import hashlib
import os
import uuid
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import (
    Chunk,
    Document,
    DocumentVersion,
    EffectiveStatus,
    LegalMetadataFilter,
    LegalStatus,
    MetadataQueries,
    ProvisionEffectiveStatus,
    provision_key,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


@pytest.fixture
def session():
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    connection = engine.connect()
    transaction = connection.begin()
    db_session = Session(bind=connection, expire_on_commit=False)
    try:
        yield db_session
    finally:
        db_session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


def _legal_document(
    session: Session,
    *,
    status: str = LegalStatus.EFFECTIVE.value,
) -> tuple[Document, DocumentVersion, Chunk]:
    suffix = uuid.uuid4().hex
    document = Document(
        external_id=f"doc-{suffix}",
        title="Nghị định kiểm thử",
        document_number=f"123/{suffix[:6]}/NĐ-CP",
        document_type="Nghị định",
        authority="Chính phủ",
        effective_date=date(2026, 1, 1),
        status=status,
        legal_field="Doanh nghiệp",
        source_url=f"https://example.test/{suffix}",
    )
    session.add(document)
    session.flush()

    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        content_hash=hashlib.sha256(suffix.encode()).hexdigest(),
        source_url=document.source_url,
        is_current=True,
        content_valid_from=date(2026, 1, 1),
        content_valid_to=date(2027, 1, 1),
    )
    session.add(version)
    session.flush()

    chunk = Chunk(
        version_id=version.id,
        document_id=document.id,
        external_id=f"{document.external_id}::article_1",
        chunk_type="article",
        strategy="article",
        ordinal=0,
        text_content="Nội dung kiểm thử.",
        retrieval_text="Điều 1\nNội dung kiểm thử.",
        approx_token_count=8,
        qdrant_collection="legal-v1",
        qdrant_point_id=f"point-{suffix}",
        metadata_json={"article": "1"},
    )
    session.add_all(
        [
            chunk,
            EffectiveStatus(
                document_id=document.id,
                source_version_id=version.id,
                status=status,
                valid_from=date(2026, 1, 1),
                valid_to=date(2027, 1, 1),
            ),
        ]
    )
    session.flush()
    return document, version, chunk


def test_temporal_filter_returns_only_point_valid_on_requested_date(session):
    _, _, chunk = _legal_document(session)

    active = session.scalars(
        MetadataQueries.qdrant_point_ids(
            LegalMetadataFilter(as_of=date(2026, 6, 1)),
            collection="legal-v1",
        )
    ).all()
    expired = session.scalars(
        MetadataQueries.qdrant_point_ids(
            LegalMetadataFilter(as_of=date(2027, 1, 1)),
            collection="legal-v1",
        )
    ).all()

    assert chunk.qdrant_point_id in active
    assert chunk.qdrant_point_id not in expired


def test_database_rejects_overlapping_effective_periods(session):
    document, version, _ = _legal_document(session)

    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(
            EffectiveStatus(
                document_id=document.id,
                source_version_id=version.id,
                status=LegalStatus.SUSPENDED.value,
                valid_from=date(2026, 6, 1),
                valid_to=date(2026, 9, 1),
            )
        )
        session.flush()


def test_hydration_query_returns_text_and_rejects_expired_candidate(session):
    document, _, chunk = _legal_document(session)
    active_filter = LegalMetadataFilter(as_of=date(2026, 6, 1))

    active = session.execute(
        MetadataQueries.hydrate_qdrant_points(
            [chunk.qdrant_point_id],
            active_filter,
            collection="legal-v1",
        )
    ).mappings().one()
    expired = session.execute(
        MetadataQueries.hydrate_qdrant_points(
            [chunk.qdrant_point_id],
            LegalMetadataFilter(as_of=date(2027, 1, 1)),
            collection="legal-v1",
        )
    ).mappings().all()

    assert active["point_id"] == chunk.qdrant_point_id
    assert active["text"] == "Nội dung kiểm thử."
    assert active["document_id"] == document.external_id
    assert active["valid_from"] == date(2026, 1, 1)
    assert expired == []


def test_hydration_prefers_exact_provision_status_over_document_status(session):
    document, version, chunk = _legal_document(session)
    session.add(
        ProvisionEffectiveStatus(
            document_id=document.id,
            source_version_id=version.id,
            provision_key=provision_key("1"),
            article="1",
            status=LegalStatus.SUSPENDED.value,
            valid_from=date(2026, 2, 1),
            valid_to=date(2026, 12, 1),
        )
    )
    session.flush()

    row = session.execute(
        MetadataQueries.hydrate_qdrant_points(
            [chunk.qdrant_point_id],
            LegalMetadataFilter(
                as_of=date(2026, 6, 1),
                statuses=(LegalStatus.SUSPENDED.value,),
            ),
            collection="legal-v1",
        )
    ).mappings().one()

    assert row["status"] == "SUSPENDED"
    assert row["status_scope"] == "provision"
    assert row["valid_from"] == date(2026, 2, 1)


def test_current_law_rejects_unknown_partial_provision_but_historical_keeps_it(
    session,
):
    document, version, chunk = _legal_document(
        session,
        status=LegalStatus.PARTIALLY_EFFECTIVE.value,
    )
    session.add(
        ProvisionEffectiveStatus(
            document_id=document.id,
            source_version_id=version.id,
            provision_key=provision_key("1"),
            article="1",
            status=LegalStatus.UNKNOWN.value,
            valid_from=date(2026, 1, 1),
            valid_to=None,
        )
    )
    session.flush()

    current = session.execute(
        MetadataQueries.hydrate_qdrant_points(
            [chunk.qdrant_point_id],
            LegalMetadataFilter(
                as_of=date(2026, 6, 1),
                temporal_intent="current_law",
            ),
            collection="legal-v1",
        )
    ).mappings().all()
    rejected = session.scalars(
        MetadataQueries.unresolved_provision_point_ids(
            [chunk.qdrant_point_id],
            LegalMetadataFilter(
                as_of=date(2026, 6, 1),
                temporal_intent="current_law",
            ),
            collection="legal-v1",
        )
    ).all()
    historical = session.execute(
        MetadataQueries.hydrate_qdrant_points(
            [chunk.qdrant_point_id],
            LegalMetadataFilter(
                as_of=date(2026, 6, 1),
                statuses=(LegalStatus.PARTIALLY_EFFECTIVE.value,),
                content_scope="historical",
                temporal_intent="historical",
            ),
            collection="legal-v1",
        )
    ).mappings().one()

    assert current == []
    assert rejected == [chunk.qdrant_point_id]
    assert historical["status"] == LegalStatus.PARTIALLY_EFFECTIVE.value
    assert historical["status_scope"] == "document"
