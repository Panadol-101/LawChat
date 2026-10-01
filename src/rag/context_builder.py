from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from retrieval import (
    LegalCitation,
    LegalQueryParser,
    ParsedLegalQuery,
    RetrievalResponse,
    RetrievedLegalChunk,
)

from .context_models import Evidence, PackedContext, TokenBudget, TokenCounter


_ROLE_PRIORITY = {
    "queried_document": 3,
    "current_authority": 2,
}
_ACTIVE_STATUSES = {"EFFECTIVE", "PARTIALLY_EFFECTIVE"}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _Candidate:
    result: RetrievedLegalChunk
    role: str
    exact_provision: bool
    original_rank: int
    text: str
    dedup_text: str

    @property
    def priority(self) -> tuple[int, int, int, int, int, float, float, int]:
        reranker_score = self.result.source_scores.get("reranker")
        return (
            int(self.result.citation.status.upper() in _ACTIVE_STATUSES),
            int(self.exact_provision),
            int(bool(self.result.metadata.get("base_query_anchor"))),
            _ROLE_PRIORITY.get(self.role, 1),
            int(reranker_score is not None),
            reranker_score if reranker_score is not None else float("-inf"),
            self.result.score,
            -self.original_rank,
        )


class RAGContextBuilder:
    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        budget: TokenBudget | None = None,
        max_ancestor_tokens: int = 256,
        max_evidence_total: int = 12,
        max_evidence_per_issue: int = 3,
        # Audit fix W8 (Phase 5): the legacy cap of 2 graph-evidence
        # blocks meant multi-hop questions (e.g. "Nghị định hướng dẫn Điều
        # 12 của Luật") could only cite 2 surrounding relationships. The
        # new cap of 4 still fits the 12-evidence total budget while
        # preserving enough room for the primary 3 evidence per issue.
        max_graph_evidence: int = 4,
        query_parser: LegalQueryParser | None = None,
    ) -> None:
        if max_ancestor_tokens < 0:
            raise ValueError("max_ancestor_tokens must be >= 0")
        if min(max_evidence_total, max_evidence_per_issue) <= 0:
            raise ValueError("evidence limits must be > 0")
        if max_graph_evidence < 0:
            raise ValueError("max_graph_evidence must be >= 0")
        self.token_counter = token_counter
        self.budget = budget or TokenBudget()
        self.max_ancestor_tokens = max_ancestor_tokens
        self.max_evidence_total = max_evidence_total
        self.max_evidence_per_issue = max_evidence_per_issue
        self.max_graph_evidence = max_graph_evidence
        self.query_parser = query_parser or LegalQueryParser()

    def build(self, response: RetrievalResponse) -> PackedContext:
        parsed = self.query_parser.parse(response.query, as_of=response.as_of)
        roles = _document_roles(response)
        candidates = self._candidates(response, parsed, roles)
        ordered, duplicate_ids = _deduplicate_and_group(candidates)
        ordered, limited_ids = _issue_fair_order(
            ordered,
            tuple(item.issue_id for item in response.legal_issues),
            max_total=self.max_evidence_total,
            max_per_issue=self.max_evidence_per_issue,
        )

        evidence: list[Evidence] = []
        rendered_parts: list[str] = []
        dropped = [*duplicate_ids, *limited_ids]
        graph_issue_ids = _relationship_issue_ids(response, self.query_parser)
        if parsed.relationship_types:
            _append_graph_evidence(
                response,
                evidence,
                rendered_parts,
                token_counter=self.token_counter,
                token_budget=self.budget.evidence_tokens,
                max_graph_evidence=self.max_graph_evidence,
                max_evidence_total=self.max_evidence_total,
                issue_ids=graph_issue_ids,
            )
        current_group: tuple[str, str] | None = None
        text_index = 0
        for candidate in ordered:
            if len(evidence) >= self.max_evidence_total:
                dropped.append(candidate.result.chunk_id)
                continue
            evidence_id = f"E{text_index + 1}"
            group = _group_key(candidate.result)
            group_header = _group_header(candidate.result) if group != current_group else ""
            block = _render_evidence(evidence_id, candidate)
            proposed_parts = [*rendered_parts]
            if group_header:
                proposed_parts.append(group_header)
            proposed_parts.append(block)
            proposed = "\n\n".join(proposed_parts)
            used_tokens = self.token_counter.count_tokens(proposed)
            if used_tokens > self.budget.evidence_tokens:
                dropped.append(candidate.result.chunk_id)
                continue

            block_tokens = self.token_counter.count_tokens(block)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    document_id=candidate.result.citation.document_id,
                    chunk_id=candidate.result.chunk_id,
                    text=candidate.text,
                    citation=candidate.result.citation,
                    role=candidate.role,
                    token_count=block_tokens,
                    issue_ids=tuple(candidate.result.metadata.get("issue_ids", ())),
                )
            )
            text_index += 1
            rendered_parts = proposed_parts
            current_group = group

        # For non-relationship questions graph data remains supporting context
        # and cannot consume budget before textual evidence.
        if not parsed.relationship_types:
            _append_graph_evidence(
                response,
                evidence,
                rendered_parts,
                token_counter=self.token_counter,
                token_budget=self.budget.evidence_tokens,
                max_graph_evidence=self.max_graph_evidence,
                max_evidence_total=self.max_evidence_total,
                issue_ids=graph_issue_ids,
            )

        rendered = "\n\n".join(rendered_parts)
        used_tokens = self.token_counter.count_tokens(rendered) if rendered else 0
        return PackedContext(
            evidence=tuple(evidence),
            rendered_context=rendered,
            used_tokens=used_tokens,
            token_budget=self.budget.evidence_tokens,
            dropped_chunk_ids=tuple(dict.fromkeys(dropped)),
        )

    def _candidates(
        self,
        response: RetrievalResponse,
        parsed: ParsedLegalQuery,
        roles: dict[str, str],
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for rank, result in enumerate(response.results, start=1):
            leaf = _strip_known_overlap(result.text, result.metadata)
            if not leaf:
                continue
            parent = _ancestor_excerpt(
                result.parent_text,
                leaf,
                token_counter=self.token_counter,
                max_tokens=self.max_ancestor_tokens,
            )
            text = (
                f"Ngữ cảnh cấp trên:\n{parent}\n\nĐoạn trích:\n{leaf}"
                if parent
                else leaf
            )
            candidates.append(
                _Candidate(
                    result=result,
                    role=roles.get(result.citation.document_id, "retrieved_document"),
                    exact_provision=_matches_exact_provision(result, parsed),
                    original_rank=rank,
                    text=text,
                    dedup_text=leaf,
                )
            )
        return candidates


def _document_roles(response: RetrievalResponse) -> dict[str, str]:
    roles: dict[str, str] = {}
    for document in response.related_documents:
        roles[document.document_id] = document.role
    for document in response.seed_documents:
        roles[document.document_id] = document.role
    return roles


def _matches_exact_provision(
    result: RetrievedLegalChunk,
    parsed: ParsedLegalQuery,
) -> bool:
    citation = result.citation
    checks = (
        not parsed.document_numbers
        or _normalize_identifier(citation.document_number) in {
            _normalize_identifier(item) for item in parsed.document_numbers
        },
        not parsed.referenced_articles or citation.article in parsed.referenced_articles,
        not parsed.referenced_clauses or citation.clause in parsed.referenced_clauses,
        not parsed.referenced_points or citation.point in parsed.referenced_points,
    )
    has_reference = bool(
        parsed.referenced_articles
        or parsed.referenced_clauses
        or parsed.referenced_points
    )
    return has_reference and all(checks)


def _deduplicate_and_group(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], tuple[str, ...]]:
    priority_order = sorted(candidates, key=lambda item: item.priority, reverse=True)
    unique: list[_Candidate] = []
    dropped: list[str] = []
    for candidate in priority_order:
        duplicate_index = next(
            (
                index
                for index, kept in enumerate(unique)
                if _is_duplicate(candidate, kept)
            ),
            None,
        )
        if duplicate_index is not None:
            unique[duplicate_index] = _merge_issue_metadata(
                unique[duplicate_index], candidate
            )
            dropped.append(candidate.result.chunk_id)
            continue
        unique.append(candidate)

    groups: dict[tuple[str, str], list[_Candidate]] = {}
    for candidate in unique:
        groups.setdefault(_group_key(candidate.result), []).append(candidate)
    ordered_groups = sorted(
        groups.values(),
        key=lambda items: (
            max(item.priority for item in items),
            -min(item.original_rank for item in items),
        ),
        reverse=True,
    )
    ordered: list[_Candidate] = []
    for items in ordered_groups:
        ordered.extend(sorted(items, key=lambda item: item.priority, reverse=True))
    return ordered, tuple(dropped)


def _merge_issue_metadata(kept: _Candidate, duplicate: _Candidate) -> _Candidate:
    kept_metadata = kept.result.metadata
    duplicate_metadata = duplicate.result.metadata
    issue_ids = list(kept_metadata.get("issue_ids", ()))
    for issue_id in duplicate_metadata.get("issue_ids", ()):
        if issue_id not in issue_ids:
            issue_ids.append(issue_id)
    merged = {
        **kept_metadata,
        "issue_ids": issue_ids,
        "issue_scores": {
            **duplicate_metadata.get("issue_scores", {}),
            **kept_metadata.get("issue_scores", {}),
        },
        "issue_ranks": {
            **duplicate_metadata.get("issue_ranks", {}),
            **kept_metadata.get("issue_ranks", {}),
        },
    }
    return replace(kept, result=replace(kept.result, metadata=merged))


def _issue_fair_order(
    candidates: list[_Candidate],
    issue_ids: tuple[str, ...],
    *,
    max_total: int,
    max_per_issue: int,
) -> tuple[list[_Candidate], tuple[str, ...]]:
    """Select evidence in issue rounds so an early issue cannot consume context."""
    if not issue_ids:
        kept = candidates[:max_total]
        return kept, tuple(item.result.chunk_id for item in candidates[max_total:])

    selected: list[_Candidate] = []
    selected_ids: set[str] = set()
    selected_bases: set[tuple[str, str, str, str]] = set()
    issue_counts = {issue_id: 0 for issue_id in issue_ids}

    def issue_priority(candidate: _Candidate, issue_id: str) -> tuple:
        scores = candidate.result.metadata.get("issue_scores", {})
        ranks = candidate.result.metadata.get("issue_ranks", {})
        score = scores.get(issue_id, float("-inf"))
        rank = ranks.get(issue_id, candidate.original_rank)
        return (score, *candidate.priority, -rank)

    pools = {
        issue_id: sorted(
            (
                item
                for item in candidates
                if issue_id in item.result.metadata.get("issue_ids", ())
            ),
            key=lambda item: issue_priority(item, issue_id),
            reverse=True,
        )
        for issue_id in issue_ids
    }

    # Round one guarantees that every issue with a candidate gets a first chance.
    # Later rounds add supporting evidence without exceeding the per-issue cap.
    for _round in range(max_per_issue):
        made_progress = False
        for issue_id in issue_ids:
            if len(selected) >= max_total or issue_counts[issue_id] >= max_per_issue:
                continue
            available = [
                item
                for item in pools[issue_id]
                if item.result.chunk_id not in selected_ids
            ]
            candidate = next(
                (
                    item
                    for item in available
                    if _legal_basis_key(item.result) not in selected_bases
                ),
                None,
            )
            # In the mandatory first round, retaining one same-basis candidate
            # is safer than leaving an issue with no evidence at all.
            if candidate is None and _round == 0 and available:
                candidate = available[0]
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.result.chunk_id)
            selected_bases.add(_legal_basis_key(candidate.result))
            made_progress = True
            for covered_issue in candidate.result.metadata.get("issue_ids", ()):
                if covered_issue in issue_counts:
                    issue_counts[covered_issue] += 1
        if len(selected) >= max_total or not made_progress:
            break

    # Keep untagged/high-value candidates only after issue coverage has been served.
    for candidate in candidates:
        if len(selected) >= max_total:
            break
        known_tags = set(candidate.result.metadata.get("issue_ids", ())).intersection(
            issue_counts
        )
        if not known_tags and candidate.result.chunk_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.result.chunk_id)

    dropped = tuple(
        item.result.chunk_id
        for item in candidates
        if item.result.chunk_id not in selected_ids
    )
    return selected, dropped


def _group_key(result: RetrievedLegalChunk) -> tuple[str, str]:
    return result.citation.document_id, result.citation.article or ""


def _legal_basis_key(result: RetrievedLegalChunk) -> tuple[str, str, str, str]:
    citation = result.citation
    if not (citation.article or citation.clause or citation.point):
        return (citation.version_id or citation.document_id, result.chunk_id, "", "")
    return (
        citation.version_id or citation.document_id,
        citation.article or "",
        citation.clause or "",
        citation.point or "",
    )


def _group_header(result: RetrievedLegalChunk) -> str:
    citation = result.citation
    document = citation.document_number or citation.title
    article = f" — Điều {citation.article}" if citation.article else ""
    return f"## {document}{article}"


def _render_evidence(evidence_id: str, candidate: _Candidate) -> str:
    citation = candidate.result.citation
    location = " ".join(
        value
        for value in (
            f"Điều {citation.article}" if citation.article else None,
            f"Khoản {citation.clause}" if citation.clause else None,
            f"Điểm {citation.point}" if citation.point else None,
        )
        if value
    )
    fields = [
        f"[{evidence_id}]",
        f"Vai trò: {candidate.role}",
        f"Văn bản: {citation.document_number or citation.title}",
    ]
    issue_ids = candidate.result.metadata.get("issue_ids", ())
    if issue_ids:
        fields.append("Phục vụ vấn đề: " + ", ".join(issue_ids))
    if location:
        fields.append(f"Vị trí: {location}")
    if citation.version_id:
        fields.append(f"Version ID: {citation.version_id}")
        if citation.version_source_revision:
            fields.append(f"Dataset revision: {citation.version_source_revision}")
        if citation.source_url:
            fields.append(f"Nguồn: {citation.source_url}")
    fields.extend(
        (
            (
                f"Hiệu lực {_status_scope_label(citation.status_scope)} tại "
                f"{citation.as_of.isoformat()}: {_status_label(citation.status)}"
            ),
            f"Nội dung:\n{candidate.text}",
        )
    )
    return "\n".join(fields)


def _graph_evidence(index, edge, as_of, issue_ids=()) -> Evidence:
    source_number = edge.source_document_number or edge.source_title or edge.source_document_id
    target_number = edge.target_document_number or edge.target_title or edge.target_document_id
    relation = "REPLACED_BY" if (
        edge.relationship_type == "REPLACES" and edge.direction == "incoming"
    ) else edge.relationship_type
    relation_text = edge.citation_text or f"{source_number} {relation} {target_number}"
    details = [relation_text]
    if edge.source_status:
        details.append(f"Trạng thái văn bản nguồn: {edge.source_status}.")
    if edge.target_status:
        details.append(f"Trạng thái văn bản đích: {edge.target_status}.")
    if edge.effective_from or edge.effective_to:
        details.append(
            f"Thời gian áp dụng quan hệ: từ {edge.effective_from or '-'} "
            f"đến {edge.effective_to or '-'}."
        )
    text = " ".join(details)
    citation = LegalCitation(
        document_id=edge.source_document_id,
        title=edge.source_title or source_number,
        document_number=edge.source_document_number,
        article=None,
        clause=None,
        point=None,
        source_url=edge.source_url or "",
        as_of=as_of,
        status=edge.source_status or "UNKNOWN",
        status_scope="document",
    )
    return Evidence(
        evidence_id=f"G{index}",
        document_id=edge.source_document_id,
        chunk_id=f"graph:{edge.source_document_id}:{edge.target_document_id}:{relation}",
        text=text,
        citation=citation,
        role="graph_relationship",
        token_count=0,
        related_document_numbers=tuple(
            item
            for item in (
                edge.source_document_number,
                edge.target_document_number,
            )
            if item
        ),
        related_statuses=tuple(
            item for item in (edge.source_status, edge.target_status) if item
        ),
        issue_ids=issue_ids,
    )


def _render_graph_evidence(evidence: Evidence, edge) -> str:
    official = "official" if edge.is_official else "unverified-source"
    return "\n".join(
        (
            f"[{evidence.evidence_id}]",
            "Vai trò: graph_relationship",
            f"Loại quan hệ: {edge.relationship_type}",
            f"Nguồn: {evidence.citation.source_url}",
            f"Provenance: {official}",
            f"Nội dung:\n{evidence.text}",
        )
    )


def _append_graph_evidence(
    response,
    evidence,
    rendered_parts,
    *,
    token_counter,
    token_budget,
    max_graph_evidence,
    max_evidence_total,
    issue_ids,
) -> None:
    graph_index = 0
    for edge in response.graph_edges:
        if graph_index >= max_graph_evidence or len(evidence) >= max_evidence_total:
            break
        if not edge.is_official or not edge.source_url:
            continue
        graph_index += 1
        graph_evidence = _graph_evidence(
            graph_index, edge, response.as_of, issue_ids
        )
        block = _render_graph_evidence(graph_evidence, edge)
        proposed = "\n\n".join((*rendered_parts, block))
        if token_counter.count_tokens(proposed) > token_budget:
            break
        evidence.append(
            replace(graph_evidence, token_count=token_counter.count_tokens(block))
        )
        rendered_parts.append(block)


def _relationship_issue_ids(
    response: RetrievalResponse,
    query_parser: LegalQueryParser,
) -> tuple[str, ...]:
    related = []
    for issue in response.legal_issues:
        try:
            parsed = query_parser.parse(issue.search_query, as_of=response.as_of)
        except ValueError:
            continue
        if parsed.relationship_types:
            related.append(issue.issue_id)
    if related:
        return tuple(related)
    return tuple(item.issue_id for item in response.legal_issues)


def _status_label(status: str) -> str:
    labels = {
        "DRAFT": "Dự thảo",
        "NOT_YET_EFFECTIVE": "Chưa có hiệu lực",
        "EFFECTIVE": "Đang có hiệu lực",
        "PARTIALLY_EFFECTIVE": "Còn hiệu lực một phần",
        "SUSPENDED": "Đang bị đình chỉ",
        "EXPIRED": "Đã hết hiệu lực",
        "REPEALED": "Đã bị bãi bỏ",
        "UNKNOWN": "Chưa xác định",
    }
    return f"{labels.get(status.upper(), 'Chưa xác định')} ({status.upper()})"


def _status_scope_label(scope: str) -> str:
    return (
        "cấp Điều/Khoản/Điểm"
        if scope == "provision"
        else "cấp văn bản (chưa có dữ liệu hiệu lực riêng cho điều khoản)"
    )


def _strip_known_overlap(text: str, metadata: dict) -> str:
    value = text.strip()
    overlap = str(metadata.get("overlap_text") or "").strip()
    if overlap and value.startswith(overlap):
        value = value[len(overlap):].lstrip(" \n:;,-")
    return value


def _ancestor_excerpt(
    parent_text: str | None,
    leaf_text: str,
    *,
    token_counter: TokenCounter,
    max_tokens: int,
) -> str:
    parent = (parent_text or "").strip()
    if not parent or parent == leaf_text.strip() or max_tokens == 0:
        return ""
    if leaf_text.strip() in parent:
        parent = parent.replace(leaf_text.strip(), "", 1).strip(" \n:;,-")
        if not parent:
            return ""
    pieces = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(parent) if item.strip()]
    selected: list[str] = []
    for piece in pieces:
        proposed = " ".join((*selected, piece))
        if token_counter.count_tokens(proposed) > max_tokens:
            break
        selected.append(piece)
    return " ".join(selected)


def _texts_overlap(left: str, right: str) -> bool:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return False
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    if len(shorter) >= 40 and shorter in longer:
        return True
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.98


def _is_duplicate(left: _Candidate, right: _Candidate) -> bool:
    """Decide if two chunks are the same legal provision.

    - 0.98 SequenceMatcher ratio catches near-identical boilerplate.
    - Same article + same version_id ⇒ dedupe.
    - Same article + different version ⇒ keep the latest version.
    """
    ratio = SequenceMatcher(
        None,
        _normalize_text(left.dedup_text),
        _normalize_text(right.dedup_text),
    ).ratio()
    if ratio >= 0.98:
        return True
    left_citation = left.result.citation
    right_citation = right.result.citation
    if (
        left_citation.article
        and left_citation.article == right_citation.article
        and left_citation.clause == right_citation.clause
        and left_citation.point == right_citation.point
    ):
        if left_citation.version_id and right_citation.version_id:
            if left_citation.version_id == right_citation.version_id:
                return True
            # Same article across different versions ⇒ keep latest.
            left_from = left.result.metadata.get("effective_date") or ""
            right_from = right.result.metadata.get("effective_date") or ""
            if left_from and right_from:
                return left_from < right_from
            return True
    return _texts_overlap(left.dedup_text, right.dedup_text)


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def _normalize_identifier(value: str | None) -> str:
    return "".join((value or "").upper().split())
