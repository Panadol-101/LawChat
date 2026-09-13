from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .base import BaseChunkStrategy
from ..models import ChunkingContext
from ..utils import (
    clean_text,
    exact_token_count,
    iter_nodes,
    pack_units,
    render_article,
    render_clause,
    render_point,
)


@dataclass(frozen=True, slots=True)
class _LegalUnit:
    kind: str
    label: str
    text: str
    source: Mapping[str, Any]


class ArticleChunkStrategy(BaseChunkStrategy):
    """Pack complete Articles before decomposing along legal boundaries."""

    name = "article"

    def chunk(self, context: ChunkingContext) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        preamble = clean_text(context.parsed_document.get("preamble"))
        if preamble and context.config.index_preamble:
            chunks.extend(self._chunk_preamble(context, preamble))

        articles = list(
            iter_nodes(
                context.parsed_document.get("body") or [],
                target_type="article",
                excluded_ancestor_types={"appendix"},
            )
        )
        for ordinal, article in enumerate(articles, start=1):
            chunks.extend(
                self.chunk_article_node(
                    context=context,
                    article=article,
                    structural_parent_id=None,
                    structural_parent_type=None,
                    path_override=None,
                    ordinal=ordinal,
                )
            )
            context.consumed_node_ids.add(id(article))
        return chunks

    def chunk_article_node(
        self,
        *,
        context: ChunkingContext,
        article: Mapping[str, Any],
        structural_parent_id: str | None,
        structural_parent_type: str | None,
        path_override: Mapping[str, Any] | None,
        ordinal: int,
    ) -> list[dict[str, Any]]:
        article_no = clean_text(article.get("article")) or "unknown"
        article_title = clean_text(article.get("title"))
        path = dict(path_override or article.get("path") or {})
        article_text = render_article(article)
        article_id = context.factory.ids.make(
            f"article_{article_no}", line_start=article.get("line_start")
        )

        if self._retrieval_tokens(
            context,
            path=path,
            article=article_no,
            article_title=article_title,
            legal_text=article_text,
        ) <= context.config.max_indexable_tokens:
            return [
                self._make_chunk(
                    context,
                    chunk_id=article_id,
                    parent_chunk_id=structural_parent_id,
                    parent_type=structural_parent_type,
                    chunk_type="article",
                    text=article_text,
                    path=path,
                    article=article_no,
                    article_title=article_title,
                    ordinal=ordinal,
                )
            ]

        parent = self._make_chunk(
            context,
            chunk_id=article_id,
            parent_chunk_id=structural_parent_id,
            parent_type=structural_parent_type,
            chunk_type="article_parent",
            text=article_text,
            path=path,
            article=article_no,
            article_title=article_title,
            ordinal=ordinal,
            is_indexable=False,
        )

        clauses = list(article.get("clauses") or [])
        direct_points = list(article.get("points") or [])
        lead = clean_text(article.get("text"))
        if clauses:
            children = self._chunk_article_clauses(
                context=context,
                article_id=article_id,
                article_no=article_no,
                article_title=article_title,
                path=path,
                lead=lead,
                clauses=clauses,
            )
        elif direct_points:
            children = self._chunk_article_points(
                context=context,
                article_id=article_id,
                article_no=article_no,
                article_title=article_title,
                path=path,
                lead=lead,
                points=direct_points,
            )
        else:
            # The title is already an exact breadcrumb in retrieval_text and
            # remains in the Article parent.  Do not create a tiny title-only
            # leaf when the body itself requires token fallback splitting.
            body = lead
            children = self._fragment_prose(
                context=context,
                base_id=article_id,
                parent_chunk_id=article_id,
                parent_type="article",
                chunk_type="article_fragment",
                text=body or article_text,
                path=path,
                article=article_no,
                article_title=article_title,
            )
        return [parent, *children]

    def _chunk_article_clauses(
        self,
        *,
        context: ChunkingContext,
        article_id: str,
        article_no: str,
        article_title: str,
        path: Mapping[str, Any],
        lead: str,
        clauses: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        units: list[_LegalUnit] = []
        if lead:
            units.append(_LegalUnit("lead", "lead", lead, {}))
        units.extend(
            _LegalUnit(
                "clause",
                clean_text(clause.get("clause")) or "unknown",
                render_clause(clause),
                clause,
            )
            for clause in clauses
            if render_clause(clause)
        )
        groups = self._pack_units(
            units,
            token_count=lambda group: self._retrieval_tokens(
                context,
                path=path,
                article=article_no,
                article_title=article_title,
                clause=self._range_label(group, "clause"),
                legal_text=self._join_units(group),
            ),
            config=context.config,
        )
        output: list[dict[str, Any]] = []
        for ordinal, group in enumerate(groups, start=1):
            clause_units = [item for item in group if item.kind == "clause"]
            text = self._join_units(group)
            if not clause_units:
                output.extend(
                    self._fragment_prose(
                        context=context,
                        base_id=f"{article_id}::lead",
                        parent_chunk_id=article_id,
                        parent_type="article",
                        chunk_type="article_lead_fragment",
                        text=text,
                        path=path,
                        article=article_no,
                        article_title=article_title,
                    )
                )
                continue

            if len(group) == 1 and len(clause_units) == 1:
                clause = clause_units[0]
                if self._group_tokens(
                    context, path, article_no, article_title, group
                ) > context.config.max_indexable_tokens:
                    output.extend(
                        self._split_clause(
                            context=context,
                            article_id=article_id,
                            article_no=article_no,
                            article_title=article_title,
                            path=path,
                            clause=clause.source,
                        )
                    )
                    continue
                label = clause.label
                chunk_id = context.factory.ids.make(
                    f"article_{article_no}", f"clause_{label}"
                )
                chunk_type = "clause"
            else:
                label = self._range_label(group, "clause") or clause_units[0].label
                chunk_id = context.factory.ids.make(
                    f"article_{article_no}", f"clauses_{label}"
                )
                chunk_type = "clause_group"

            output.append(
                self._make_chunk(
                    context,
                    chunk_id=chunk_id,
                    parent_chunk_id=article_id,
                    parent_type="article",
                    chunk_type=chunk_type,
                    text=text,
                    path=path,
                    article=article_no,
                    article_title=article_title,
                    clause=label,
                    ordinal=ordinal,
                )
            )
        return output

    def _chunk_article_points(
        self,
        *,
        context: ChunkingContext,
        article_id: str,
        article_no: str,
        article_title: str,
        path: Mapping[str, Any],
        lead: str,
        points: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        units: list[_LegalUnit] = []
        if lead:
            units.append(_LegalUnit("lead", "lead", lead, {}))
        units.extend(
            _LegalUnit(
                "point",
                clean_text(point.get("point")) or "unknown",
                render_point(point),
                point,
            )
            for point in points
            if render_point(point)
        )
        return self._emit_point_groups(
            context=context,
            units=units,
            article_id=article_id,
            parent_chunk_id=article_id,
            parent_type="article",
            article_no=article_no,
            article_title=article_title,
            path=path,
            clause_no=None,
            retrieval_context="",
        )

    def _split_clause(
        self,
        *,
        context: ChunkingContext,
        article_id: str,
        article_no: str,
        article_title: str,
        path: Mapping[str, Any],
        clause: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        clause_no = clean_text(clause.get("clause")) or "unknown"
        clause_text = render_clause(clause)
        clause_id = context.factory.ids.make(
            f"article_{article_no}",
            f"clause_{clause_no}",
            line_start=clause.get("line_start"),
        )
        parent = self._make_chunk(
            context,
            chunk_id=clause_id,
            parent_chunk_id=article_id,
            parent_type="article",
            chunk_type="clause_parent",
            text=clause_text,
            path=path,
            article=article_no,
            article_title=article_title,
            clause=clause_no,
            is_indexable=False,
        )
        points = list(clause.get("points") or [])
        if not points:
            return [
                parent,
                *self._fragment_prose(
                    context=context,
                    base_id=clause_id,
                    parent_chunk_id=clause_id,
                    parent_type="clause",
                    chunk_type="clause_fragment",
                    text=clean_text(clause.get("text")) or clause_text,
                    path=path,
                    article=article_no,
                    article_title=article_title,
                    clause=clause_no,
                ),
            ]

        lead = clean_text(clause.get("text"))
        units = [
            _LegalUnit(
                "point",
                clean_text(point.get("point")) or "unknown",
                render_point(point),
                point,
            )
            for point in points
            if render_point(point)
        ]
        children = self._emit_point_groups(
            context=context,
            units=units,
            article_id=article_id,
            parent_chunk_id=clause_id,
            parent_type="clause",
            article_no=article_no,
            article_title=article_title,
            path=path,
            clause_no=clause_no,
            retrieval_context=lead,
        )
        return [parent, *children]

    def _emit_point_groups(
        self,
        *,
        context: ChunkingContext,
        units: Sequence[_LegalUnit],
        article_id: str,
        parent_chunk_id: str,
        parent_type: str,
        article_no: str,
        article_title: str,
        path: Mapping[str, Any],
        clause_no: str | None,
        retrieval_context: str,
    ) -> list[dict[str, Any]]:
        groups = self._pack_units(
            units,
            token_count=lambda group: self._retrieval_tokens(
                context,
                path=path,
                article=article_no,
                article_title=article_title,
                clause=clause_no,
                point=self._range_label(group, "point"),
                legal_text="\n".join(
                    item for item in [retrieval_context, self._join_units(group)] if item
                ),
            ),
            config=context.config,
        )
        output: list[dict[str, Any]] = []
        for ordinal, group in enumerate(groups, start=1):
            point_units = [item for item in group if item.kind == "point"]
            point_label = self._range_label(group, "point")
            text = self._join_units(group)
            retrieval_tokens = self._retrieval_tokens(
                context,
                path=path,
                article=article_no,
                article_title=article_title,
                clause=clause_no,
                point=point_label,
                legal_text="\n".join(
                    item for item in [retrieval_context, text] if item
                ),
            )
            if (
                len(point_units) == 1
                and retrieval_tokens > context.config.max_indexable_tokens
            ):
                point = point_units[0]
                id_parts = [f"article_{article_no}"]
                if clause_no:
                    id_parts.append(f"clause_{clause_no}")
                id_parts.append(f"point_{point.label}")
                point_base = context.factory.ids.make(
                    *id_parts, line_start=point.source.get("line_start")
                )
                output.append(
                    self._make_chunk(
                        context,
                        chunk_id=point_base,
                        parent_chunk_id=parent_chunk_id,
                        parent_type=parent_type,
                        chunk_type="point_parent",
                        text=point.text,
                        path=path,
                        article=article_no,
                        article_title=article_title,
                        clause=clause_no,
                        point=point.label,
                        is_indexable=False,
                    )
                )
                output.extend(
                    self._fragment_prose(
                        context=context,
                        base_id=point_base,
                        parent_chunk_id=point_base,
                        parent_type="point",
                        chunk_type="point_fragment",
                        text=clean_text(point.source.get("text")) or point.text,
                        path=path,
                        article=article_no,
                        article_title=article_title,
                        clause=clause_no,
                        point=point.label,
                        retrieval_context=retrieval_context,
                    )
                )
                continue

            if not point_units:
                output.extend(
                    self._fragment_prose(
                        context=context,
                        base_id=f"{article_id}::lead",
                        parent_chunk_id=parent_chunk_id,
                        parent_type=parent_type,
                        chunk_type="article_lead_fragment",
                        text=text,
                        path=path,
                        article=article_no,
                        article_title=article_title,
                    )
                )
                continue

            if len(point_units) == 1:
                chunk_type = "point"
                id_part = f"point_{point_units[0].label}"
            else:
                chunk_type = "point_group"
                id_part = f"points_{point_label}"
            id_parts = [f"article_{article_no}"]
            if clause_no:
                id_parts.append(f"clause_{clause_no}")
            id_parts.append(id_part)
            output.append(
                self._make_chunk(
                    context,
                    chunk_id=context.factory.ids.make(*id_parts),
                    parent_chunk_id=parent_chunk_id,
                    parent_type=parent_type,
                    chunk_type=chunk_type,
                    text=text,
                    path=path,
                    article=article_no,
                    article_title=article_title,
                    clause=clause_no,
                    point=point_label,
                    ordinal=ordinal,
                    retrieval_context=retrieval_context,
                )
            )
        return output

    def _fragment_prose(
        self,
        *,
        context: ChunkingContext,
        base_id: str,
        parent_chunk_id: str,
        parent_type: str,
        chunk_type: str,
        text: str,
        path: Mapping[str, Any],
        article: str,
        article_title: str,
        clause: str | None = None,
        point: str | None = None,
        retrieval_context: str = "",
    ) -> list[dict[str, Any]]:
        overhead_text = context.factory.build_retrieval_text(
            metadata=context.metadata,
            path=path,
            article=article,
            article_title=article_title,
            clause=clause,
            point=point,
            legal_text=retrieval_context,
        )
        budget = max(
            8,
            context.config.max_indexable_tokens - exact_token_count(overhead_text) - 4,
        )
        paragraphs = [clean_text(item) for item in text.splitlines() if clean_text(item)]
        blocks = pack_units(
            paragraphs or [text],
            target_tokens=min(context.config.effective_target_tokens, budget),
            max_tokens=budget,
            min_tail_tokens=min(context.config.min_tail_tokens, budget),
        )
        base_parts = base_id.split("::")[1:]
        output: list[dict[str, Any]] = []
        for ordinal, block in enumerate(blocks, start=1):
            output.append(
                self._make_chunk(
                    context,
                    chunk_id=context.factory.ids.make(
                        *base_parts, f"fragment_{ordinal}"
                    ),
                    parent_chunk_id=parent_chunk_id,
                    parent_type=parent_type,
                    chunk_type=chunk_type,
                    text=block,
                    path=path,
                    article=article,
                    article_title=article_title,
                    clause=clause,
                    point=point,
                    ordinal=ordinal,
                    retrieval_context=retrieval_context,
                )
            )
        return output

    def _chunk_preamble(
        self, context: ChunkingContext, preamble: str
    ) -> list[dict[str, Any]]:
        parent_id = context.factory.ids.make("preamble")
        budget = context.factory.available_legal_text_tokens(
            metadata=context.metadata,
            max_tokens=context.config.max_indexable_tokens,
        )
        blocks = pack_units(
            [clean_text(item) for item in preamble.splitlines() if clean_text(item)],
            target_tokens=min(context.config.effective_target_tokens, budget),
            max_tokens=budget,
            min_tail_tokens=min(context.config.min_tail_tokens, budget),
        )
        if len(blocks) == 1:
            return [
                self._make_chunk(
                    context,
                    chunk_id=parent_id,
                    parent_chunk_id=None,
                    parent_type=None,
                    chunk_type="preamble",
                    text=blocks[0],
                    path={},
                )
            ]
        parent = self._make_chunk(
            context,
            chunk_id=parent_id,
            parent_chunk_id=None,
            parent_type=None,
            chunk_type="preamble_parent",
            text=preamble,
            path={},
            is_indexable=False,
        )
        children = [
            self._make_chunk(
                context,
                chunk_id=context.factory.ids.make("preamble", f"fragment_{index}"),
                parent_chunk_id=parent_id,
                parent_type="preamble",
                chunk_type="preamble_fragment",
                text=block,
                path={},
                ordinal=index,
            )
            for index, block in enumerate(blocks, start=1)
        ]
        return [parent, *children]

    @staticmethod
    def _pack_units(
        units: Sequence[_LegalUnit],
        *,
        token_count: Callable[[Sequence[_LegalUnit]], int],
        config: Any,
    ) -> list[list[_LegalUnit]]:
        groups: list[list[_LegalUnit]] = []
        current: list[_LegalUnit] = []
        for unit in units:
            if not current:
                current = [unit]
                continue
            current_tokens = token_count(current)
            candidate = [*current, unit]
            candidate_tokens = token_count(candidate)
            if candidate_tokens > config.max_indexable_tokens:
                groups.append(current)
                current = [unit]
            elif current_tokens >= config.effective_target_min_tokens:
                groups.append(current)
                current = [unit]
            else:
                current = candidate
        if current:
            groups.append(current)
        if len(groups) >= 2 and token_count(groups[-1]) < config.min_tail_tokens:
            merged = [*groups[-2], *groups[-1]]
            if token_count(merged) <= config.max_indexable_tokens:
                groups[-2] = merged
                groups.pop()
        return groups

    @staticmethod
    def _join_units(units: Sequence[_LegalUnit]) -> str:
        return "\n".join(item.text for item in units if clean_text(item.text)).strip()

    @staticmethod
    def _range_label(units: Sequence[_LegalUnit], kind: str) -> str | None:
        labels = [item.label for item in units if item.kind == kind]
        if not labels:
            return None
        if len(labels) == 1:
            return labels[0]
        return f"{labels[0]}-{labels[-1]}"

    def _group_tokens(
        self,
        context: ChunkingContext,
        path: Mapping[str, Any],
        article_no: str,
        article_title: str,
        group: Sequence[_LegalUnit],
    ) -> int:
        return self._retrieval_tokens(
            context,
            path=path,
            article=article_no,
            article_title=article_title,
            clause=self._range_label(group, "clause"),
            legal_text=self._join_units(group),
        )

    @staticmethod
    def _retrieval_tokens(
        context: ChunkingContext,
        *,
        path: Mapping[str, Any],
        article: str | None = None,
        article_title: str = "",
        clause: str | None = None,
        point: str | None = None,
        legal_text: str,
    ) -> int:
        return exact_token_count(
            context.factory.build_retrieval_text(
                metadata=context.metadata,
                path=path,
                article=article,
                article_title=article_title,
                clause=clause,
                point=point,
                legal_text=legal_text,
            )
        )

    def _make_chunk(
        self,
        context: ChunkingContext,
        *,
        chunk_id: str,
        parent_chunk_id: str | None,
        parent_type: str | None,
        chunk_type: str,
        text: str,
        path: Mapping[str, Any],
        article: str | None = None,
        article_title: str = "",
        clause: str | None = None,
        point: str | None = None,
        ordinal: int = 0,
        is_indexable: bool = True,
        retrieval_context: str = "",
    ) -> dict[str, Any]:
        chunk = context.factory.make_chunk(
            chunk_id=chunk_id,
            parent_chunk_id=parent_chunk_id,
            parent_type=parent_type,
            chunk_type=chunk_type,
            strategy=self.name,
            is_indexable=is_indexable,
            structure_type=context.structure_type,
            text=text,
            metadata=context.metadata,
            path=path,
            article=article,
            article_title=article_title,
            clause=clause,
            point=point,
            ordinal=ordinal,
        )
        if is_indexable and retrieval_context:
            legal_text = "\n".join(
                item for item in [retrieval_context, clean_text(text)] if item
            )
            chunk["retrieval_text"] = context.factory.build_retrieval_text(
                metadata=context.metadata,
                path=path,
                article=article,
                article_title=article_title,
                clause=clause,
                point=point,
                legal_text=legal_text,
            )
            chunk["approx_token_count"] = exact_token_count(chunk["retrieval_text"])
        chunk["_article_title"] = article_title
        chunk["_retrieval_context"] = retrieval_context
        return chunk
