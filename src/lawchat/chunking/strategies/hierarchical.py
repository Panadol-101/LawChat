from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import BaseChunkStrategy
from ..models import ChunkingContext
from ..utils import (
    STRUCTURAL_TYPES,
    exact_token_count,
    clean_text,
    pack_node_blocks,
    pack_units,
    path_with_node,
    render_generic_node,
    render_structural_heading,
)


class HierarchicalBlockStrategy(BaseChunkStrategy):
    """
    For documents that have Phần/Chương/Mục but no Điều.

    The real hierarchy is preserved. Text/list blocks are attached to the
    deepest structural parent that actually contains them.
    """

    name = "hierarchical_block"

    def chunk(
        self,
        context: ChunkingContext,
    ) -> list[dict]:
        chunks: list[dict] = []

        preamble = clean_text(
            context.parsed_document.get("preamble")
        )

        top_nodes = [
            node
            for node in context.parsed_document.get("body") or []
            if clean_text(node.get("type")) != "appendix"
        ]

        # Top-level non-structural content gets a document parent.
        top_free = [
            node
            for node in top_nodes
            if clean_text(node.get("type"))
            not in {
                "part",
                "chapter",
                "section",
                "article",
            }
        ]

        if preamble or top_free:
            chunks.extend(
                self._chunk_document_level(
                    context=context,
                    preamble=preamble,
                    nodes=top_free,
                )
            )
            context.consumed_node_ids.update(id(node) for node in top_free)

        for node in top_nodes:
            if clean_text(node.get("type")) in {
                "part",
                "chapter",
                "section",
            }:
                chunks.extend(
                    self._chunk_container(
                        context=context,
                        node=node,
                        path={},
                        parent_chunk_id=None,
                        parent_type=None,
                        ordinal=1,
                    )
                )

        return chunks

    def _chunk_document_level(
        self,
        *,
        context: ChunkingContext,
        preamble: str,
        nodes: Sequence[Mapping[str, Any]],
    ) -> list[dict]:
        units = []

        if preamble:
            units.append(preamble)

        units.extend(
            render_generic_node(
                node,
                recursive=False,
            )
            for node in nodes
        )

        units = [
            item
            for item in units
            if clean_text(item)
        ]

        if not units:
            return []

        parent_text = "\n".join(units).strip()
        parent_id = context.factory.ids.make(
            "document_hierarchy_context"
        )

        chunks = [
            context.factory.make_chunk(
                chunk_id=parent_id,
                parent_chunk_id=None,
                parent_type=None,
                chunk_type="document_parent",
                strategy=self.name,
                is_indexable=False,
                structure_type=context.structure_type,
                text=parent_text,
                metadata=context.metadata,
            )
        ]

        blocks = pack_units(
            units,
            target_tokens=min(
                context.config.freeform_target_tokens,
                context.factory.available_legal_text_tokens(
                    metadata=context.metadata,
                ),
            ),
            max_tokens=min(
                context.config.freeform_max_tokens,
                context.factory.available_legal_text_tokens(
                    metadata=context.metadata,
                ),
            ),
            min_tail_tokens=context.config.min_tail_tokens,
        )

        for ordinal, block in enumerate(
            blocks,
            start=1,
        ):
            child_id = context.factory.ids.make(
                "document_hierarchy_context",
                f"block_{ordinal}",
            )

            chunks.append(
                context.factory.make_chunk(
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    parent_type="document",
                    chunk_type="hierarchical_block",
                    strategy=self.name,
                    is_indexable=True,
                    structure_type=context.structure_type,
                    text=block,
                    metadata=context.metadata,
                    ordinal=ordinal,
                )
            )

        return chunks

    def _chunk_container(
        self,
        *,
        context: ChunkingContext,
        node: Mapping[str, Any],
        path: Mapping[str, str],
        parent_chunk_id: str | None,
        parent_type: str | None,
        ordinal: int,
    ) -> list[dict]:
        node_type = clean_text(
            node.get("type")
        )

        current_path = path_with_node(
            path,
            node,
        )

        key_name = {
            "part": "part",
            "chapter": "chapter",
            "section": "section",
        }[node_type]

        label = clean_text(
            node.get(key_name)
        ) or "unknown"

        parent_id = context.factory.ids.make(
            f"{node_type}_{label}",
            line_start=node.get("line_start"),
        )

        direct_nodes = [
            child
            for child in node.get("children") or []
            if clean_text(child.get("type"))
            not in {
                "part",
                "chapter",
                "section",
                "article",
                "appendix",
            }
        ]

        direct_units = [
            render_generic_node(
                child,
                recursive=False,
            )
            for child in direct_nodes
        ]

        direct_units = [
            item
            for item in direct_units
            if clean_text(item)
        ]

        heading = render_structural_heading(
            node
        )

        parent_text = "\n".join(
            item
            for item in [
                heading,
                *direct_units,
            ]
            if item
        ).strip()

        chunks = [
            context.factory.make_chunk(
                chunk_id=parent_id,
                parent_chunk_id=parent_chunk_id,
                parent_type=parent_type,
                chunk_type=f"{node_type}_parent",
                strategy=self.name,
                is_indexable=False,
                structure_type=context.structure_type,
                text=parent_text or heading,
                metadata=context.metadata,
                path=current_path,
                ordinal=ordinal,
            )
        ]

        if direct_units:
            base_budget = min(
                context.config.freeform_max_tokens,
                context.factory.available_legal_text_tokens(
                    metadata=context.metadata,
                    path=current_path,
                ),
            )
            content_budget = max(
                1,
                base_budget - exact_token_count(heading),
            )
            blocks = pack_node_blocks(
                direct_nodes,
                target_tokens=min(context.config.freeform_target_tokens, content_budget),
                max_tokens=content_budget,
                min_tail_tokens=context.config.min_tail_tokens,
            )

            for block_ordinal, block in enumerate(
                blocks,
                start=1,
            ):
                text = "\n".join(
                    item
                    for item in [heading, block]
                    if item
                ).strip()

                child_id = context.factory.ids.make(
                    f"{node_type}_{label}",
                    f"block_{block_ordinal}",
                )

                chunks.append(
                    context.factory.make_chunk(
                        chunk_id=child_id,
                        parent_chunk_id=parent_id,
                        parent_type=node_type,
                        chunk_type="hierarchical_block",
                        strategy=self.name,
                        is_indexable=True,
                        structure_type=context.structure_type,
                        text=text,
                        metadata=context.metadata,
                        path=current_path,
                        ordinal=block_ordinal,
                    )
                )

        structural_children = [
            child
            for child in node.get("children") or []
            if clean_text(child.get("type"))
            in {
                "part",
                "chapter",
                "section",
            }
        ]

        for child_ordinal, child in enumerate(
            structural_children,
            start=1,
        ):
            chunks.extend(
                self._chunk_container(
                    context=context,
                    node=child,
                    path=current_path,
                    parent_chunk_id=parent_id,
                    parent_type=node_type,
                    ordinal=child_ordinal,
                )
            )

        context.consumed_node_ids.add(
            id(node)
        )

        return chunks
