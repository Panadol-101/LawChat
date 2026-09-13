from __future__ import annotations

import os
import uuid
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from lawchat.database import Document, DocumentRelationship, EffectiveStatus, LegalStatus
from lawchat.retrieval import PostgresLegalGraphResolver


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


@pytest.fixture
def graph_runtime():
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session, PostgresLegalGraphResolver(
            sessionmaker(bind=connection, expire_on_commit=False)
        )
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


def _document(session: Session, number: str, status: str) -> Document:
    suffix = uuid.uuid4().hex
    document = Document(
        external_id=f"graph-{suffix}",
        title=f"Văn bản {number}",
        document_number=number,
        status=status,
        source_url=f"https://example.test/{suffix}",
    )
    session.add(document)
    session.flush()
    session.add(
        EffectiveStatus(
            document_id=document.id,
            status=status,
            valid_from=date(2020, 1, 1),
        )
    )
    session.flush()
    return document


def test_seed_resolution_keeps_expired_document_and_graph_assigns_roles(graph_runtime):
    session, resolver = graph_runtime
    marker = uuid.uuid4().hex[:8].upper()
    old = _document(session, f"9911/2016/QĐ-TEST-{marker}", LegalStatus.EXPIRED.value)
    replacement = _document(
        session, f"9922/2025/QĐ-TEST-{marker}", LegalStatus.EFFECTIVE.value
    )
    session.add(
        DocumentRelationship(
            source_document_id=replacement.id,
            target_document_id=old.id,
            relationship_type="REPLACES",
            source_relationship="integration-test-replaces",
            target_external_id=old.external_id,
            target_document_number=old.document_number,
            effective_from=date(2025, 1, 1),
        )
    )
    session.flush()

    seeds = resolver.resolve_seeds(
        (old.document_number.replace("/", " /", 1),),
        as_of=date(2026, 1, 1),
        query=f"Văn bản {old.document_number} còn hiệu lực không?",
        preferred_statuses=("EXPIRED",),
    )
    expansion = resolver.expand(
        (old.document_number,),
        relationship_types=("REPLACES",),
        direction="incoming",
        as_of=date(2026, 1, 1),
    )

    assert seeds[0].document.document_id == old.external_id
    assert seeds[0].document.status == "EXPIRED"
    assert seeds[0].document.role == "queried_document"
    assert seeds[0].score >= 105
    assert expansion.document_ids == (replacement.external_id,)
    assert expansion.statuses == ("EFFECTIVE",)
    assert expansion.related_documents[0].role == "current_authority"
    assert expansion.edges[0].effective_from == date(2025, 1, 1)


def test_graph_excludes_relationship_outside_as_of(graph_runtime):
    session, resolver = graph_runtime
    marker = uuid.uuid4().hex[:8].upper()
    old = _document(session, f"9931/2016/QĐ-TEST-{marker}", LegalStatus.EXPIRED.value)
    replacement = _document(
        session,
        f"9932/2027/QĐ-TEST-{marker}",
        LegalStatus.NOT_YET_EFFECTIVE.value,
    )
    session.add(
        DocumentRelationship(
            source_document_id=replacement.id,
            target_document_id=old.id,
            relationship_type="REPLACES",
            source_relationship="integration-test-future-replaces",
            target_external_id=old.external_id,
            target_document_number=old.document_number,
            effective_from=date(2027, 1, 1),
        )
    )
    session.flush()

    expansion = resolver.expand(
        (old.document_number,),
        relationship_types=("REPLACES",),
        direction="incoming",
        as_of=date(2026, 1, 1),
    )

    assert expansion.document_ids == ()
    assert expansion.edges == ()
