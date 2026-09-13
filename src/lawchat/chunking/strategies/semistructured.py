from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseChunkStrategy
from ..models import ChunkingContext
from ..utils import (
    exact_token_count,
    clean_text,
    pack_node_blocks,
    pack_units,
    render_generic_node,
)


class SemiStructuredChunkStrategy(BaseChunkStrategy):
    """
    For Công văn / Thông báo / Quyết định free-layout / list-heavy documents.

    Headings are preserved as semantic context. Numbered list items outside
    Điều remain list items; they are never converted into Khoản.
    """

    name = "semi_structured"
    max_heading_context_tokens = 120

    def chunk(
        self,
        context: ChunkingContext,
    ) -> list[dict]:
        body = [
            node
            for node in context.parsed_document.get("body") or []
            if clean_text(node.get("type"))
            not in {
                "article",
                "appendix",
                "part",
                "chapter",
                "section",
            }
        ]

        chunks = self.chunk_nodes(
            context=context,
            nodes=body,
            preamble=clean_text(
                context.parsed_document.get("preamble")
            ),
            parent_key="document",
            parent_type="document",
        )
        context.consumed_node_ids.update(id(node) for node in body)
        return chunks

    def chunk_nodes(
        self,
        *,
        context: ChunkingContext,
        nodes: Sequence[Mapping[str, Any]],
        preamble: str = "",
        parent_key: str,
        parent_type: str,
        parent_chunk_id: str | None = None,
        path: Mapping[str, Any] | None = None,
        heading_prefix: str = "",
        chunk_type: str = "structured_block",
    ) -> list[dict]:
        path = path or {}

        groups = self._group_by_heading(
            nodes
        )

        all_rendered = []

        if preamble:
            all_rendered.append(preamble)

        for heading, group_nodes in groups:
            if heading:
                all_rendered.append(heading)

            all_rendered.extend(
                render_generic_node(
                    node,
                    recursive=False,
                )
                for node in group_nodes
            )

        all_rendered = [
            item
            for item in all_rendered
            if clean_text(item)
        ]

        if not all_rendered:
            return []

        parent_id = context.factory.ids.make(
            parent_key
        )

        parent_text = "\n".join(
            item
            for item in [
                heading_prefix,
                *all_rendered,
            ]
            if item
        ).strip()

        chunks = [
            context.factory.make_chunk(
                chunk_id=parent_id,
                parent_chunk_id=parent_chunk_id,
                parent_type=(
                    None
                    if parent_chunk_id is None
                    else parent_type
                ),
                chunk_type=f"{parent_type}_parent",
                strategy=self.name,
                is_indexable=False,
                structure_type=context.structure_type,
                text=parent_text,
                metadata=context.metadata,
                path=path,
            )
        ]

        # Containers that only contribute legal breadcrumbs (for example a
        # Chapter containing Articles and no direct prose) must be preserved
        # for hydration, but are not useful standalone retrieval vectors.
        if nodes and all(clean_text(node.get("type")) == "heading" for node in nodes):
            return chunks

        ordinal = 0

        base_budget = min(
            context.config.freeform_max_tokens,
            context.factory.available_legal_text_tokens(
                metadata=context.metadata,
                path=path,
            ),
        )

        if preamble:
            preamble_blocks = pack_units(
                preamble.splitlines(),
                target_tokens=min(context.config.freeform_target_tokens, base_budget),
                max_tokens=base_budget,
                min_tail_tokens=context.config.min_tail_tokens,
            )
            for preamble_block in preamble_blocks:
                ordinal += 1
                preamble_id = context.factory.ids.make(
                    parent_key,
                    f"preamble_{ordinal}",
                )
                chunks.append(
                    context.factory.make_chunk(
                        chunk_id=preamble_id,
                        parent_chunk_id=parent_id,
                        parent_type=parent_type,
                        chunk_type=chunk_type,
                        strategy=self.name,
                        is_indexable=True,
                        structure_type=context.structure_type,
                        text=preamble_block,
                        metadata=context.metadata,
                        path=path,
                        ordinal=ordinal,
                    )
                )

        for heading, group_nodes in groups:
            repeated_context = "\n".join(
                item for item in [heading_prefix, heading] if item
            )
            content_budget = max(
                1,
                base_budget - exact_token_count(repeated_context),
            )
            blocks = pack_node_blocks(
                group_nodes,
                target_tokens=min(context.config.freeform_target_tokens, content_budget),
                max_tokens=content_budget,
                min_tail_tokens=context.config.min_tail_tokens,
            )

            if not blocks:
                continue

            for block in blocks:
                ordinal += 1

                text = "\n".join(
                    item
                    for item in [
                        heading_prefix,
                        heading,
                        block,
                    ]
                    if item
                ).strip()

                child_id = context.factory.ids.make(
                    parent_key,
                    f"block_{ordinal}",
                )

                chunks.append(
                    context.factory.make_chunk(
                        chunk_id=child_id,
                        parent_chunk_id=parent_id,
                        parent_type=parent_type,
                        chunk_type=chunk_type,
                        strategy=self.name,
                        is_indexable=True,
                        structure_type=context.structure_type,
                        text=text,
                        metadata=context.metadata,
                        path=path,
                        ordinal=ordinal,
                    )
                )

        return chunks

    @staticmethod
    def _group_by_heading(
        nodes: Sequence[Mapping[str, Any]],
    ) -> list[
        tuple[
            str,
            list[Mapping[str, Any]],
        ]
    ]:
        groups: list[
            tuple[
                str,
                list[Mapping[str, Any]],
            ]
        ] = []

        current_heading = ""
        current_nodes: list[Mapping[str, Any]] = []

        def flush() -> None:
            nonlocal current_nodes

            if current_nodes:
                groups.append(
                    (
                        current_heading,
                        current_nodes,
                    )
                )
                current_nodes = []

        for node in nodes:
            node_type = clean_text(
                node.get("type")
            )

            if node_type == "heading":
                heading = clean_text(node.get("text"))
                if current_nodes:
                    flush()
                elif current_heading and heading:
                    combined = f"{current_heading}\n{heading}"
                    if (
                        exact_token_count(combined)
                        <= SemiStructuredChunkStrategy.max_heading_context_tokens
                    ):
                        current_heading = combined
                        continue
                    groups.append(
                        (
                            "",
                            [{"type": "paragraph", "text": current_heading}],
                        )
                    )
                current_heading = heading
                continue

            current_nodes.append(node)

        flush()

        # Preserve a final heading even when no paragraph follows it.
        if current_heading and not current_nodes and (
            not groups or groups[-1][0] != current_heading
        ):
            groups.append(
                (
                    "",
                    [
                        {
                            "type": "paragraph",
                            "text": current_heading,
                        }
                    ],
                )
            )

        return groups
