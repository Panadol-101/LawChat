from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, aliased, sessionmaker

from lawchat.database import Document, DocumentRelationship, EffectiveStatus
from lawchat.database.queries import TRANSLATION_DOCUMENT_TYPES


class LegalDocumentRole(str, Enum):
    QUERIED_DOCUMENT = "queried_document"
    CURRENT_AUTHORITY = "current_authority"
    REPLACED_DOCUMENT = "replaced_document"
    REPEALING_DOCUMENT = "repealing_document"
    REPEALED_DOCUMENT = "repealed_document"
    AMENDING_DOCUMENT = "amending_document"
    AMENDED_DOCUMENT = "amended_document"
    SUPPLEMENTING_DOCUMENT = "supplementing_document"
    SUPPLEMENTED_DOCUMENT = "supplemented_document"
    GUIDANCE_DOCUMENT = "guidance_document"
    GUIDED_DOCUMENT = "guided_document"
    RELATED_DOCUMENT = "related_document"


@dataclass(frozen=True, slots=True)
class LegalGraphDocument:
    document_id: str
    document_number: str | None
    title: str
    status: str | None
    role: str
    relationship_type: str | None = None
    direction: str | None = None
    authority: str | None = None
    document_type: str | None = None
    issued_date: date | None = None
    valid_from: date | None = None
    valid_to: date | None = None


@dataclass(frozen=True, slots=True)
class SeedResolution:
    document: LegalGraphDocument
    score: float
    matched_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentStatusResolution:
    document_id: str
    document_number: str | None
    as_of: date
    status: str | None
    valid_from: date | None
    valid_to: date | None
    ambiguous: bool
    candidate_document_ids: tuple[str, ...] = ()
    related_documents: tuple[LegalGraphDocument, ...] = ()


@dataclass(frozen=True, slots=True)
class LegalGraphEdge:
    source_document_id: str
    target_document_id: str
    relationship_type: str
    source_document_number: str | None = None
    target_document_number: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    direction: str | None = None
    source_title: str | None = None
    target_title: str | None = None
    source_status: str | None = None
    target_status: str | None = None
    source_url: str | None = None
    citation_text: str | None = None
    is_official: bool = False


@dataclass(frozen=True, slots=True)
class LegalGraphExpansion:
    document_ids: tuple[str, ...] = ()
    document_numbers: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    edges: tuple[LegalGraphEdge, ...] = ()
    related_documents: tuple[LegalGraphDocument, ...] = ()


class LegalSeedResolver(Protocol):
    def resolve_seeds(
        self,
        document_numbers: tuple[str, ...],
        *,
        as_of: date,
        query: str,
        preferred_statuses: tuple[str, ...] = (),
    ) -> tuple[SeedResolution, ...]: ...


class LegalGraphResolver(Protocol):
    def expand(
        self,
        document_numbers: tuple[str, ...],
        *,
        relationship_types: tuple[str, ...],
        direction: str,
        as_of: date,
    ) -> LegalGraphExpansion: ...


class PostgresLegalGraphResolver:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def resolve_seeds(
        self,
        document_numbers: tuple[str, ...],
        *,
        as_of: date,
        query: str,
        preferred_statuses: tuple[str, ...] = (),
    ) -> tuple[SeedResolution, ...]:
        normalized = tuple(_normalize_document_number(item) for item in document_numbers)
        if not normalized:
            return ()
        status = aliased(EffectiveStatus, name="seed_status")
        statement = (
            select(
                Document.external_id.label("document_id"),
                Document.document_number,
                Document.title,
                func.coalesce(status.status, Document.status).label("status"),
                status.valid_from,
                status.valid_to,
                Document.authority,
                Document.document_type,
                Document.issued_date,
            )
            .outerjoin(
                status,
                and_(
                    status.document_id == Document.id,
                    status.valid_period.op("@>")(as_of),
                ),
            )
            .where(
                func.upper(func.replace(Document.document_number, " ", "")).in_(
                    normalized
                ),
                or_(
                    Document.document_type.is_(None),
                    Document.document_type.not_in(TRANSLATION_DOCUMENT_TYPES),
                ),
            )
            .order_by(Document.external_id)
        )
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()
        resolutions = [
            _to_seed_resolution(
                row,
                query=query,
                preferred_statuses=preferred_statuses,
            )
            for row in rows
        ]
        return tuple(
            sorted(
                resolutions,
                key=lambda item: (-item.score, item.document.document_id),
            )
        )

    def expand(
        self,
        document_numbers: tuple[str, ...],
        *,
        relationship_types: tuple[str, ...],
        direction: str,
        as_of: date,
    ) -> LegalGraphExpansion:
        if not document_numbers or not relationship_types:
            return LegalGraphExpansion()
        if direction not in {"incoming", "outgoing"}:
            raise ValueError("graph direction must be incoming or outgoing")

        source = aliased(Document, name="graph_source")
        target = aliased(Document, name="graph_target")
        source_status = aliased(EffectiveStatus, name="graph_source_status")
        target_status = aliased(EffectiveStatus, name="graph_target_status")
        query = (
            select(
                source.external_id.label("source_document_id"),
                source.document_number.label("source_document_number"),
                source.title.label("source_title"),
                func.coalesce(source_status.status, source.status).label("source_status"),
                target.external_id.label("target_document_id"),
                target.document_number.label("target_document_number"),
                target.title.label("target_title"),
                func.coalesce(target_status.status, target.status).label("target_status"),
                DocumentRelationship.relationship_type,
                DocumentRelationship.effective_from,
                DocumentRelationship.effective_to,
                DocumentRelationship.citation_text,
                source.source_url.label("source_url"),
                DocumentRelationship.metadata_json.label("relationship_metadata"),
            )
            .join(source, source.id == DocumentRelationship.source_document_id)
            .join(target, target.id == DocumentRelationship.target_document_id)
            .outerjoin(
                source_status,
                and_(
                    source_status.document_id == source.id,
                    source_status.valid_period.op("@>")(as_of),
                ),
            )
            .outerjoin(
                target_status,
                and_(
                    target_status.document_id == target.id,
                    target_status.valid_period.op("@>")(as_of),
                ),
            )
            .where(
                DocumentRelationship.relationship_type.in_(relationship_types),
                or_(
                    source.document_type.is_(None),
                    source.document_type.not_in(TRANSLATION_DOCUMENT_TYPES),
                ),
                or_(
                    target.document_type.is_(None),
                    target.document_type.not_in(TRANSLATION_DOCUMENT_TYPES),
                ),
                or_(
                    DocumentRelationship.effective_from.is_(None),
                    DocumentRelationship.effective_from <= as_of,
                ),
                or_(
                    DocumentRelationship.effective_to.is_(None),
                    DocumentRelationship.effective_to > as_of,
                ),
            )
        )
        normalized = tuple(_normalize_document_number(item) for item in document_numbers)
        if direction == "outgoing":
            query = query.where(
                func.upper(func.replace(source.document_number, " ", "")).in_(normalized)
            )
        else:
            query = query.where(
                func.upper(func.replace(target.document_number, " ", "")).in_(normalized)
            )

        with self.session_factory() as session:
            rows = session.execute(query).mappings().all()

        related_documents = tuple(
            _related_document(row, direction=direction) for row in rows
        )
        document_ids = tuple(dict.fromkeys(item.document_id for item in related_documents))
        document_numbers_result = tuple(
            dict.fromkeys(
                item.document_number for item in related_documents if item.document_number
            )
        )
        statuses = tuple(
            dict.fromkeys(item.status for item in related_documents if item.status)
        )
        edges = tuple(
            LegalGraphEdge(
                source_document_id=row["source_document_id"],
                target_document_id=row["target_document_id"],
                relationship_type=row["relationship_type"],
                source_document_number=row["source_document_number"],
                target_document_number=row["target_document_number"],
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                direction=direction,
                source_title=row["source_title"],
                target_title=row["target_title"],
                source_status=row["source_status"],
                target_status=row["target_status"],
                source_url=(row["relationship_metadata"] or {}).get(
                    "official_source_url"
                ) or (row["relationship_metadata"] or {}).get(
                    "source_reference"
                ) or row["source_url"],
                citation_text=row["citation_text"],
                is_official=bool(
                    (row["relationship_metadata"] or {}).get("is_official")
                    or (row["relationship_metadata"] or {}).get("is_trusted")
                ),
            )
            for row in rows
        )
        return LegalGraphExpansion(
            document_ids=document_ids,
            document_numbers=document_numbers_result,
            statuses=statuses,
            edges=edges,
            related_documents=related_documents,
        )


def _related_document(row, *, direction: str) -> LegalGraphDocument:
    relationship_type = row["relationship_type"]
    if direction == "outgoing":
        prefix = "target"
        role = {
            "REPLACES": LegalDocumentRole.REPLACED_DOCUMENT,
            "REPEALS": LegalDocumentRole.REPEALED_DOCUMENT,
            "AMENDS": LegalDocumentRole.AMENDED_DOCUMENT,
            "SUPPLEMENTS": LegalDocumentRole.SUPPLEMENTED_DOCUMENT,
            "GUIDES": LegalDocumentRole.GUIDED_DOCUMENT,
        }.get(relationship_type, LegalDocumentRole.RELATED_DOCUMENT)
    else:
        prefix = "source"
        role = {
            "REPLACES": LegalDocumentRole.CURRENT_AUTHORITY,
            "REPEALS": LegalDocumentRole.REPEALING_DOCUMENT,
            "AMENDS": LegalDocumentRole.AMENDING_DOCUMENT,
            "SUPPLEMENTS": LegalDocumentRole.SUPPLEMENTING_DOCUMENT,
            "GUIDES": LegalDocumentRole.GUIDANCE_DOCUMENT,
        }.get(relationship_type, LegalDocumentRole.RELATED_DOCUMENT)
    return LegalGraphDocument(
        document_id=row[f"{prefix}_document_id"],
        document_number=row[f"{prefix}_document_number"],
        title=row[f"{prefix}_title"],
        status=row[f"{prefix}_status"],
        role=role.value,
        relationship_type=relationship_type,
        direction=direction,
    )


def _normalize_document_number(value: str) -> str:
    return "".join(value.upper().split())


_TOKEN_RE = re.compile(r"[a-z0-9đ]+")
_SEED_STOPWORDS = {
    "ban", "bo", "cac", "cho", "co", "cua", "dan", "dong", "hien",
    "hoi", "khong", "mot", "nhan", "nhung", "phap", "quyet", "quy",
    "so", "thanh", "theo", "thi", "tinh", "tren", "trong", "trung",
    "uy", "van", "va", "ve", "viec",
}


def _seed_score(
    row,
    *,
    query: str,
    preferred_statuses: tuple[str, ...],
) -> tuple[float, tuple[str, ...]]:
    query_tokens = _significant_tokens(query)
    score = 100.0
    matched = ["document_number"]

    authority_tokens = _significant_tokens(row["authority"] or "")
    authority_overlap = query_tokens & authority_tokens
    if authority_overlap:
        score += 20.0 * len(authority_overlap) / max(1, len(authority_tokens))
        matched.append("authority")

    document_type_tokens = _significant_tokens(row["document_type"] or "")
    if document_type_tokens and document_type_tokens <= query_tokens:
        score += 15.0
        matched.append("document_type")

    if row["issued_date"] and str(row["issued_date"].year) in query:
        score += 10.0
        matched.append("issued_year")

    title_tokens = _significant_tokens(row["title"] or "")
    title_overlap = query_tokens & title_tokens
    if title_overlap:
        score += 10.0 * len(title_overlap) / max(1, len(title_tokens))
        matched.append("title")

    if preferred_statuses and row["status"] in preferred_statuses:
        score += 5.0
        matched.append("status")
    return score, tuple(matched)


def _to_seed_resolution(
    row,
    *,
    query: str,
    preferred_statuses: tuple[str, ...],
) -> SeedResolution:
    score, matched_fields = _seed_score(
        row,
        query=query,
        preferred_statuses=preferred_statuses,
    )
    return SeedResolution(
        document=LegalGraphDocument(
            document_id=row["document_id"],
            document_number=row["document_number"],
            title=row["title"],
            status=row["status"],
            role=LegalDocumentRole.QUERIED_DOCUMENT.value,
            authority=row["authority"],
            document_type=row["document_type"],
            issued_date=row["issued_date"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        ),
        score=score,
        matched_fields=matched_fields,
    )


def _significant_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        token
        for token in _TOKEN_RE.findall(normalized)
        if len(token) > 1 and token not in _SEED_STOPWORDS
    }
