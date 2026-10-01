from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, aliased, sessionmaker

from database import Document, DocumentRelationship, EffectiveStatus
from database.models import AmendmentEvent
from database.queries import TRANSLATION_DOCUMENT_TYPES


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
    # Audit fix G5: explicit roles for CITES, IMPLEMENTS, RELATED_TO edges.
    CITING_DOCUMENT = "citing_document"
    CITED_DOCUMENT = "cited_document"
    IMPLEMENTING_DOCUMENT = "implementing_document"
    IMPLEMENTED_DOCUMENT = "implemented_document"


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
    # Audit fix G3 (Phase 5): node summarisation. Populated by
    # ``PostgresLegalGraphResolver.summarize_node`` from a precomputed
    # ``document_metadata.summary`` column (added by migration
    # ``a7f8b9c0d1e3`` when it is run). Until then the field is ``None``
    # and callers must degrade gracefully.
    summary: str | None = None


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


@dataclass(frozen=True, slots=True)
class LegalAmendmentEdge:
    """Audit fix E9: surfacing fine-grained amendment events.

    These rows live in ``amendment_events`` but were never queried by the
    retrieval pipeline, so questions like "Điều X bị sửa đổi ngày Y bởi
    văn bản nào?" had no answer even though the data existed.
    """

    amendment_id: str
    event_type: str
    effective_date: date | None
    source_document_id: str | None
    target_document_id: str | None
    target_document_number: str | None
    source_article: str | None
    target_provision_key: str | None
    citation_text: str | None
    direction: str  # "outgoing" if doc is source, "incoming" if doc is target


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

    def expand_amendments(
        self,
        document_numbers: tuple[str, ...],
        *,
        as_of: date,
        max_results: int = 50,
    ) -> tuple[LegalAmendmentEdge, ...]: ...


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

    def cluster_by_legal_field(
        self,
        document_numbers: tuple[str, ...],
        *,
        as_of: date,
    ) -> dict[str, tuple[str, ...]]:
        """Audit fix G2 (Phase 5): group the seed documents by ``legal_field``.

        Used by questions of the form "tất cả văn bản về X" so the
        retrieval pipeline can fetch documents that share a legal field
        with the seed rather than relying solely on lexical overlap.

        Returns a mapping from ``legal_field`` -> tuple of document
        numbers. Documents without a known ``legal_field`` are grouped
        under the sentinel key ``"__unclassified__"``.
        """
        if not document_numbers:
            return {}
        normalized = tuple(_normalize_document_number(item) for item in document_numbers)
        if not normalized:
            return {}
        statement = select(
            Document.document_number,
            Document.legal_field,
        ).where(
            func.upper(func.replace(Document.document_number, " ", "")).in_(normalized),
        )
        with self.session_factory() as session:
            rows = session.execute(statement).all()
        clusters: dict[str, list[str]] = {}
        for row in rows:
            document_number = row.document_number
            legal_field = row.legal_field or "__unclassified__"
            if not document_number:
                continue
            clusters.setdefault(legal_field, []).append(document_number)
        return {key: tuple(values) for key, values in clusters.items()}

    def resolve_external_id(
        self,
        external_id: str,
        *,
        as_of: date,
    ) -> LegalGraphDocument | None:
        """Audit fix E5 (Phase 5): resolve a stray ``target_external_id``.

        Audit E5 found several ``target_external_id`` entries in
        ``document_relationships`` that did not have a corresponding
        ``target_document_id`` row in ``documents`` (typically foreign
        documents referenced in passing). Those edges used to vanish
        silently during graph expansion. We now surface the raw external
        ID as a stub ``LegalGraphDocument`` so the caller can decide how
        to handle it.
        """
        statement = select(Document).where(Document.external_id == external_id)
        with self.session_factory() as session:
            document = session.execute(statement).scalar_one_or_none()
        if document is None:
            return LegalGraphDocument(
                document_id=external_id,
                document_number=None,
                title="",
                status=None,
                role=LegalDocumentRole.RELATED_DOCUMENT.value,
                relationship_type=None,
                direction=None,
            )
        return LegalGraphDocument(
            document_id=document.external_id,
            document_number=document.document_number,
            title=document.title,
            status=document.status,
            role=LegalDocumentRole.QUERIED_DOCUMENT.value,
            relationship_type=None,
            direction=None,
            authority=document.authority,
            document_type=document.document_type,
            issued_date=document.issued_date,
        )

    def expand_transitive(
        self,
        document_numbers: tuple[str, ...],
        *,
        relationship_types: tuple[str, ...],
        direction: str,
        as_of: date,
        max_hops: int = 2,
        max_results: int = 50,
    ) -> LegalGraphExpansion:
        """Audit fix E7+E8 (Phase 4): multi-hop graph traversal.

        Walks the ``document_relationships`` graph starting from the given
        documents for up to ``max_hops`` steps. The implementation uses a
        bounded BFS that issues one SQL query per hop. This is safer than a
        recursive CTE for legal data because the corpus has heterogeneous
        edge types and depth limits the work per request.

        Direction semantics:
          - ``outgoing``: start at the given documents, walk the
            ``source -> target`` edge.
          - ``incoming``: start at the given documents, walk the
            ``target -> source`` edge.
        """
        if not document_numbers or not relationship_types:
            return LegalGraphExpansion()
        if direction not in {"incoming", "outgoing"}:
            raise ValueError("graph direction must be incoming or outgoing")
        if max_hops < 1:
            max_hops = 1
        if max_hops > 4:
            max_hops = 4

        normalized = tuple(_normalize_document_number(item) for item in document_numbers)
        if not normalized:
            return LegalGraphExpansion()

        visited: dict[str, LegalGraphDocument] = {}
        current_numbers = normalized
        for hop in range(max_hops):
            if not current_numbers:
                break
            hop_expansion = self.expand(
                current_numbers,
                relationship_types=relationship_types,
                direction=direction,
                as_of=as_of,
            )
            next_numbers: list[str] = []
            for document in hop_expansion.related_documents:
                key = document.document_number or document.document_id
                if key in visited:
                    continue
                visited[key] = document
                if document.document_number and document.document_number not in next_numbers:
                    next_numbers.append(document.document_number)
            current_numbers = tuple(next_numbers)
            if len(visited) >= max_results:
                break

        related = tuple(list(visited.values())[:max_results])
        return LegalGraphExpansion(
            document_ids=tuple(dict.fromkeys(item.document_id for item in related)),
            document_numbers=tuple(
                dict.fromkeys(item.document_number for item in related if item.document_number)
            ),
            statuses=(),
            edges=(),
            related_documents=related,
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

    def expand_amendments(
        self,
        document_numbers: tuple[str, ...],
        *,
        as_of: date,
        max_results: int = 50,
    ) -> tuple[LegalAmendmentEdge, ...]:
        """Audit fix E9: query ``amendment_events`` for the given documents.

        Returns rows ordered by ``effective_date`` desc. Both directions are
        returned: when ``document_number`` is the source (``outgoing``) or
        the target (``incoming``) of the amendment.
        """
        if not document_numbers:
            return ()
        normalized = tuple(_normalize_document_number(item) for item in document_numbers)
        if not normalized:
            return ()

        source_doc = aliased(Document, name="amend_source_doc")
        target_doc = aliased(Document, name="amend_target_doc")
        statement = (
            select(
                AmendmentEvent.id.label("amendment_id"),
                AmendmentEvent.event_type,
                AmendmentEvent.effective_date,
                AmendmentEvent.source_document_id,
                AmendmentEvent.target_document_id,
                target_doc.document_number.label("target_document_number"),
                AmendmentEvent.source_article,
                AmendmentEvent.target_provision_key,
                AmendmentEvent.citation_text,
                source_doc.document_number.label("source_document_number"),
            )
            .join(source_doc, source_doc.id == AmendmentEvent.source_document_id)
            .outerjoin(target_doc, target_doc.id == AmendmentEvent.target_document_id)
            .where(
                or_(
                    func.upper(func.replace(source_doc.document_number, " ", "")).in_(normalized),
                    func.upper(func.replace(target_doc.document_number, " ", "")).in_(normalized),
                ),
                or_(
                    AmendmentEvent.effective_date.is_(None),
                    AmendmentEvent.effective_date <= as_of,
                ),
            )
            .order_by(AmendmentEvent.effective_date.desc().nulls_last())
            .limit(max_results)
        )
        with self.session_factory() as session:
            rows = session.execute(statement).mappings().all()

        results: list[LegalAmendmentEdge] = []
        for row in rows:
            source_number = _normalize_document_number(row["source_document_number"] or "")
            target_number = _normalize_document_number(row["target_document_number"] or "")
            if source_number in normalized:
                direction = "outgoing"
            elif target_number in normalized:
                direction = "incoming"
            else:
                continue
            results.append(
                LegalAmendmentEdge(
                    amendment_id=str(row["amendment_id"]),
                    event_type=row["event_type"],
                    effective_date=row["effective_date"],
                    source_document_id=str(row["source_document_id"]) if row["source_document_id"] else None,
                    target_document_id=str(row["target_document_id"]) if row["target_document_id"] else None,
                    target_document_number=row["target_document_number"],
                    source_article=row["source_article"],
                    target_provision_key=row["target_provision_key"],
                    citation_text=row["citation_text"],
                    direction=direction,
                )
            )
        return tuple(results)


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
            # Audit fix G5: explicit roles for the previously-defaulted
            # relationship kinds. Without this every CITES/IMPLEMENTS/
            # RELATED_TO edge became RELATED_DOCUMENT and lost semantic
            # information during context packing.
            "CITES": LegalDocumentRole.CITED_DOCUMENT,
            "IMPLEMENTS": LegalDocumentRole.IMPLEMENTED_DOCUMENT,
            "RELATED_TO": LegalDocumentRole.RELATED_DOCUMENT,
        }.get(relationship_type, LegalDocumentRole.RELATED_DOCUMENT)
    else:
        prefix = "source"
        role = {
            "REPLACES": LegalDocumentRole.CURRENT_AUTHORITY,
            "REPEALS": LegalDocumentRole.REPEALING_DOCUMENT,
            "AMENDS": LegalDocumentRole.AMENDING_DOCUMENT,
            "SUPPLEMENTS": LegalDocumentRole.SUPPLEMENTING_DOCUMENT,
            "GUIDES": LegalDocumentRole.GUIDANCE_DOCUMENT,
            "CITES": LegalDocumentRole.CITING_DOCUMENT,
            "IMPLEMENTS": LegalDocumentRole.IMPLEMENTING_DOCUMENT,
            "RELATED_TO": LegalDocumentRole.RELATED_DOCUMENT,
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
