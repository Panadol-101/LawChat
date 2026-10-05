from __future__ import annotations

from typing import Any, Mapping

from .article import ArticleChunkStrategy
from .base import BaseChunkStrategy
from .semistructured import SemiStructuredChunkStrategy
from ..models import ChunkingContext
from ..utils import (
    exact_token_count,
    clean_text,
    iter_nodes,
    pack_units,
    path_with_node,
    render_generic_node,
    render_structural_heading,
    table_row_units,
)


class AppendixChunkStrategy(BaseChunkStrategy):
    """
    Phụ lục is always a hard boundary.

    Supported:
      - text/list appendices
      - heading-aware appendices
      - articles nested inside an appendix
      - future parser table nodes with headers/rows
    """

    name = "appendix"

    def __init__(self) -> None:
        self.article_strategy = ArticleChunkStrategy()
        self.semi_strategy = SemiStructuredChunkStrategy()

    def chunk(
        self,
        context: ChunkingContext,
    ) -> list[dict]:
        chunks: list[dict] = []

        appendices = list(
            iter_nodes(
                context.parsed_document.get("body") or [],
                target_type="appendix",
            )
        )

        for ordinal, appendix in enumerate(
            appendices,
            start=1,
        ):
            chunks.extend(
                self._chunk_appendix(
                    context=context,
                    appendix=appendix,
                    ordinal=ordinal,
                )
            )

            context.consumed_node_ids.add(
                id(appendix)
            )

        return chunks

    def _chunk_appendix(
        self,
        *,
        context: ChunkingContext,
        appendix: Mapping[str, Any],
        ordinal: int,
    ) -> list[dict]:
        label = clean_text(
            appendix.get("appendix")
        ) or str(ordinal)

        path = path_with_node(
            {},
            appendix,
        )

        heading = render_structural_heading(
            appendix
        )

        appendix_id = context.factory.ids.make(
            f"appendix_{label}",
            line_start=appendix.get("line_start"),
        )

        appendix_text = render_generic_node(
            appendix,
            recursive=True,
        )

        chunks = [
            context.factory.make_chunk(
                chunk_id=appendix_id,
                parent_chunk_id=None,
                parent_type=None,
                chunk_type="appendix_parent",
                strategy=self.name,
                is_indexable=False,
                structure_type=context.structure_type,
                text=appendix_text or heading,
                metadata=context.metadata,
                path=path,
                ordinal=ordinal,
            )
        ]

        child_ordinal = 0

        direct_nodes = list(
            appendix.get("children") or []
        )

        article_nodes: list[Mapping[str, Any]] = []
        table_nodes: list[Mapping[str, Any]] = []
        regular_nodes: list[Mapping[str, Any]] = []

        def collect(nodes: list[Mapping[str, Any]]) -> None:
            for node in nodes:
                node_type = clean_text(node.get("type"))

                # Nested appendices are processed independently by chunk().
                if node_type == "appendix":
                    continue
                if node_type == "article":
                    article_nodes.append(node)
                    continue
                if node_type == "table":
                    table_nodes.append(node)
                    continue
                if node_type in {"part", "chapter", "section"}:
                    structural_heading = render_structural_heading(node)
                    if structural_heading:
                        regular_nodes.append(
                            {"type": "heading", "text": structural_heading}
                        )
                    collect(list(node.get("children") or []))
                    continue

                regular_nodes.append(node)

        collect(direct_nodes)

        # 1) Articles anywhere inside an appendix retain legal semantics.

        for article in article_nodes:
            article_chunks = self.article_strategy.chunk_article_node(
                context=context,
                article=article,
                structural_parent_id=appendix_id,
                structural_parent_type="appendix",
                path_override={
                    **dict(article.get("path") or {}),
                    **path,
                },
                ordinal=child_ordinal + 1,
            )

            child_ordinal += 1
            chunks.extend(article_chunks)

        # 2) Tables at any hierarchy depth.

        for table_index, table in enumerate(
            table_nodes,
            start=1,
        ):
            units = table_row_units(
                table
            )
            header_line = " | ".join(
                clean_text(item)
                for item in table.get("headers") or []
                if clean_text(item)
            )

            base_budget = min(
                context.config.freeform_max_tokens,
                context.factory.available_legal_text_tokens(
                    metadata=context.metadata,
                    path=path,
                ),
            )
            heading_budget = max(
                1,
                base_budget - exact_token_count(heading),
            )
            header_tokens = exact_token_count(header_line)

            if header_line and header_tokens >= heading_budget // 2:
                header_context = f"Bảng {table_index} - tiêu đề và cấu trúc cột"
                header_content_budget = max(
                    1,
                    heading_budget - exact_token_count(header_context),
                )
                header_blocks = pack_units(
                    [header_line],
                    target_tokens=min(
                        context.config.freeform_target_tokens,
                        header_content_budget,
                    ),
                    max_tokens=header_content_budget,
                    min_tail_tokens=0,
                )
                for header_ordinal, header_block in enumerate(
                    header_blocks,
                    start=1,
                ):
                    child_ordinal += 1
                    header_id = context.factory.ids.make(
                        f"appendix_{label}",
                        f"table_{table_index}",
                        f"header_{header_ordinal}",
                    )
                    chunks.append(
                        context.factory.make_chunk(
                            chunk_id=header_id,
                            parent_chunk_id=appendix_id,
                            parent_type="appendix",
                            chunk_type="table_header_fragment",
                            strategy=self.name,
                            is_indexable=True,
                            structure_type=context.structure_type,
                            text="\n".join(
                                item
                                for item in [heading, header_context, header_block]
                                if item
                            ),
                            metadata=context.metadata,
                            path=path,
                            ordinal=child_ordinal,
                        )
                    )
                repeated_header = f"Bảng {table_index}"
            else:
                repeated_header = header_line

            # A normal header-only table still carries retrievable information.
            if not units:
                if header_line and repeated_header == header_line:
                    units = [header_line]
                    repeated_header = ""
                else:
                    continue

            content_budget = max(
                1,
                base_budget
                - exact_token_count(heading)
                - exact_token_count(repeated_header),
            )

            blocks = pack_units(
                units,
                target_tokens=min(context.config.freeform_target_tokens, content_budget),
                max_tokens=content_budget,
                min_tail_tokens=context.config.min_tail_tokens,
            )

            for block in blocks:
                child_ordinal += 1

                table_id = context.factory.ids.make(
                    f"appendix_{label}",
                    f"table_{table_index}",
                    f"block_{child_ordinal}",
                )

                chunks.append(
                    context.factory.make_chunk(
                        chunk_id=table_id,
                        parent_chunk_id=appendix_id,
                        parent_type="appendix",
                        chunk_type="table_block",
                        strategy=self.name,
                        is_indexable=True,
                        structure_type=context.structure_type,
                        text="\n".join(
                            item
                            for item in [heading, repeated_header, block]
                            if item
                        ),
                        metadata=context.metadata,
                        path=path,
                        ordinal=child_ordinal,
                    )
                )

        # 3) Regular text/list/headings at any hierarchy depth.

        if regular_nodes:
            groups = self.semi_strategy._group_by_heading(
                regular_nodes
            )

            for group_heading, group_nodes in groups:
                units = [
                    render_generic_node(
                        node,
                        recursive=False,
                    )
                    for node in group_nodes
                ]

                units = [
                    item
                    for item in units
                    if clean_text(item)
                ]

                repeated_context = "\n".join(
                    item for item in [heading, group_heading] if item
                )
                base_budget = min(
                    context.config.freeform_max_tokens,
                    context.factory.available_legal_text_tokens(
                        metadata=context.metadata,
                        path=path,
                    ),
                )
                content_budget = max(
                    1,
                    base_budget - exact_token_count(repeated_context),
                )
                blocks = pack_units(
                    units,
                    target_tokens=min(context.config.freeform_target_tokens, content_budget),
                    max_tokens=content_budget,
                    min_tail_tokens=context.config.min_tail_tokens,
                )

                for block in blocks:
                    child_ordinal += 1

                    text = "\n".join(
                        item
                        for item in [
                            heading,
                            group_heading,
                            block,
                        ]
                        if item
                    ).strip()

                    child_id = context.factory.ids.make(
                        f"appendix_{label}",
                        f"block_{child_ordinal}",
                    )

                    chunks.append(
                        context.factory.make_chunk(
                            chunk_id=child_id,
                            parent_chunk_id=appendix_id,
                            parent_type="appendix",
                            chunk_type="appendix_block",
                            strategy=self.name,
                            is_indexable=True,
                            structure_type=context.structure_type,
                            text=text,
                            metadata=context.metadata,
                            path=path,
                            ordinal=child_ordinal,
                        )
                    )

        return chunks
