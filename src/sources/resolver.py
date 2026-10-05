from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from database.models import Document, DocumentSource


class DocumentNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class SourceResolution:
    url: str
    status: str
    document_number: str | None
    cached: bool = False


@dataclass(frozen=True, slots=True)
class MCPSourceSettings:
    url: str | None
    tool_name: str
    api_key: str | None
    timeout_seconds: float
    cache_days: int
    official_domains: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "MCPSourceSettings":
        domains = tuple(
            item.strip().casefold()
            for item in os.getenv(
                "LAWCHAT_SOURCE_OFFICIAL_DOMAINS",
                "vbpl.vn,congbao.chinhphu.vn,vanban.chinhphu.vn,moj.gov.vn",
            ).split(",")
            if item.strip()
        )
        return cls(
            url=os.getenv("LAWCHAT_SOURCE_MCP_URL") or None,
            tool_name=os.getenv("LAWCHAT_SOURCE_MCP_TOOL", "google_search"),
            api_key=os.getenv("LAWCHAT_SOURCE_MCP_API_KEY") or None,
            timeout_seconds=float(os.getenv("LAWCHAT_SOURCE_MCP_TIMEOUT_SECONDS", "20")),
            cache_days=int(os.getenv("LAWCHAT_SOURCE_CACHE_DAYS", "30")),
            official_domains=domains,
        )


class MCPGoogleSearchClient:
    def __init__(self, settings: MCPSourceSettings) -> None:
        self.settings = settings

    async def search(self, query: str) -> list[SearchCandidate]:
        if not self.settings.url:
            return []
        headers = {"Accept": "application/json, text/event-stream"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        timeout = httpx.Timeout(self.settings.timeout_seconds)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            async with streamable_http_client(
                self.settings.url, http_client=client
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.settings.timeout_seconds),
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        self.settings.tool_name,
                        {"query": query},
                    )
        if result.isError:
            return []
        payloads: list[Any] = []
        if result.structuredContent:
            payloads.append(result.structuredContent)
        for item in result.content:
            text_value = getattr(item, "text", None)
            if not text_value:
                continue
            try:
                payloads.append(json.loads(text_value))
            except json.JSONDecodeError:
                continue
        candidates: list[SearchCandidate] = []
        for payload in payloads:
            candidates.extend(_candidate_dicts(payload))
        return list({item.url: item for item in candidates}.values())


class SourceResolutionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: MCPSourceSettings,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.search_client = MCPGoogleSearchClient(settings)

    async def resolve(self, document_id: uuid.UUID | str) -> SourceResolution:
        document = self._document(document_id)
        cached = self._cached(document.id)
        if cached is not None:
            return SourceResolution(
                cached.url, cached.verification_status,
                document.document_number, cached=True,
            )
        if _is_human_readable_source(document.source_url):
            return SourceResolution(
                document.source_url, "CORPUS_SOURCE", document.document_number
            )

        query = _search_query(document)
        try:
            candidates = await self.search_client.search(query)
        except Exception:
            candidates = []
        ranked = sorted(
            (
                (_score_candidate(item, document, self.settings.official_domains), item)
                for item in candidates
                if _safe_public_url(item.url)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_host = (
            (urlparse(ranked[0][1].url).hostname or "").casefold()
            if ranked else ""
        )
        if (
            ranked
            and ranked[0][0] >= 90
            and _official_host(best_host, self.settings.official_domains)
        ):
            score, candidate = ranked[0]
            self._store(document, candidate, score)
            return SourceResolution(
                candidate.url, "VERIFIED", document.document_number
            )
        return SourceResolution(
            _google_fallback(query), "GOOGLE_SEARCH", document.document_number
        )

    def describe(self, document_id: uuid.UUID | str) -> dict[str, Any]:
        document = self._document(document_id)
        cached = self._cached(document.id)
        return {
            "document_id": str(document.id),
            "document_number": document.document_number,
            "title": document.title,
            "original_source_url": document.source_url,
            "resolved_url": cached.url if cached else None,
            "resolution_status": cached.verification_status if cached else "UNRESOLVED",
            "resolved_at": cached.verified_at if cached else None,
        }

    def _document(self, document_id: uuid.UUID | str) -> Document:
        with self.session_factory() as session:
            identifier = str(document_id).strip()
            try:
                internal_id = uuid.UUID(identifier)
            except ValueError:
                internal_id = None

            document = (
                session.get(Document, internal_id)
                if internal_id is not None
                else None
            )
            if document is None:
                document = session.scalar(
                    select(Document).where(Document.external_id == identifier)
                )
            if document is None:
                raise DocumentNotFoundError("document not found")
            return document

    def _cached(self, document_id: uuid.UUID) -> DocumentSource | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.cache_days)
        with self.session_factory() as session:
            return session.scalar(
                select(DocumentSource)
                .where(
                    DocumentSource.document_id == document_id,
                    DocumentSource.verification_status == "VERIFIED",
                    DocumentSource.is_active.is_(True),
                    DocumentSource.last_checked_at >= cutoff,
                )
                .order_by(DocumentSource.match_score.desc())
                .limit(1)
            )

    def _store(
        self, document: Document, candidate: SearchCandidate, score: int
    ) -> None:
        now = datetime.now(timezone.utc)
        domain = (urlparse(candidate.url).hostname or "").casefold()
        with self.session_factory() as session, session.begin():
            existing = session.scalar(select(DocumentSource).where(
                DocumentSource.document_id == document.id,
                DocumentSource.url == candidate.url,
            ))
            if existing is None:
                existing = DocumentSource(
                    document_id=document.id,
                    url=candidate.url,
                    domain=domain,
                )
                session.add(existing)
            existing.title = candidate.title or None
            existing.verification_status = "VERIFIED"
            existing.match_score = score
            existing.is_active = True
            existing.verified_at = now
            existing.last_checked_at = now
            existing.metadata_json = {
                "resolver": "mcp_google_search",
                "snippet": candidate.snippet[:1000],
                "document_number": document.document_number,
            }


def create_source_resolution_service(
    session_factory: sessionmaker[Session],
) -> SourceResolutionService:
    return SourceResolutionService(session_factory, MCPSourceSettings.from_env())


def _candidate_dicts(payload: Any) -> list[SearchCandidate]:
    found: list[SearchCandidate] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(_candidate_dicts(item))
    elif isinstance(payload, dict):
        url = payload.get("url") or payload.get("link") or payload.get("href")
        if isinstance(url, str):
            found.append(SearchCandidate(
                url=url,
                title=str(payload.get("title") or payload.get("name") or ""),
                snippet=str(
                    payload.get("snippet") or payload.get("description")
                    or payload.get("content") or ""
                ),
            ))
        for value in payload.values():
            if isinstance(value, (list, dict)):
                found.extend(_candidate_dicts(value))
    return found


def _search_query(document: Document) -> str:
    parts = []
    if document.document_number:
        parts.append(f'"{document.document_number}"')
    if document.authority:
        parts.append(f'"{document.authority}"')
    if document.title and not document.document_number:
        parts.append(f'"{document.title[:180]}"')
    parts.append("toàn văn")
    return " ".join(parts)


def _score_candidate(
    candidate: SearchCandidate,
    document: Document,
    official_domains: tuple[str, ...],
) -> int:
    combined = unquote(" ".join((candidate.title, candidate.snippet, candidate.url)))
    number_match = bool(
        document.document_number
        and _compact(document.document_number) in _compact(combined)
    )
    if not number_match:
        return 0
    host = (urlparse(candidate.url).hostname or "").casefold()
    title_tokens = _tokens(document.title)
    title_overlap = title_tokens & _tokens(combined)
    title_ratio = len(title_overlap) / len(title_tokens) if title_tokens else 0.0
    authority_tokens = _tokens(document.authority or "")
    authority_overlap = authority_tokens & _tokens(combined)
    authority_ratio = (
        len(authority_overlap) / len(authority_tokens) if authority_tokens else 1.0
    )
    # Local authorities frequently reuse the same document number. Do not
    # auto-redirect unless either the issuer or a substantial title fragment
    # distinguishes the result.
    if authority_tokens and authority_ratio < 0.5 and title_ratio < 0.35:
        return 0
    score = 70
    if _official_host(host, official_domains):
        score += 25
    if title_tokens:
        score += min(15, round(15 * title_ratio))
    if authority_ratio >= 0.5:
        score += 10
    return score


def _official_host(host: str, configured: tuple[str, ...]) -> bool:
    if host.endswith(".gov.vn"):
        return True
    return any(host == domain or host.endswith("." + domain) for domain in configured)


def _is_human_readable_source(url: str) -> bool:
    if not _safe_public_url(url):
        return False
    host = (urlparse(url).hostname or "").casefold()
    return not any(
        blocked in host for blocked in ("huggingface.co", "github.com", "raw.githubusercontent.com")
    )


def _safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.casefold()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return False
    return not any(
        blocked in host for blocked in ("huggingface.co", "github.com", "google.com")
    )


def _google_fallback(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(query)


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-zà-ỹđ]+", "", normalized)


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return {
        item for item in re.findall(r"[0-9a-zà-ỹđ]+", normalized)
        if len(item) > 2
    }
