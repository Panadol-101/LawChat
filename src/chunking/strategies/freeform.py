from __future__ import annotations

from .base import BaseChunkStrategy
from ..models import ChunkingContext
from ..utils import (
    clean_text,
    pack_units,
    render_generic_node,
)


class FreeFormChunkStrategy(BaseChunkStrategy):
    """
    Paragraph-aware adaptive chunking for documents without legal hierarchy.

    No fixed-size slicing is performed unless a single semantic unit itself
    exceeds the safety maximum.
    """

    name = "free_form"

    def chunk(
        self,
        context: ChunkingContext,
    ) -> list[dict]:
        units: list[str] = []

        preamble = clean_text(
            context.parsed_document.get("preamble")
        )

        if preamble:
            units.append(preamble)

        for node in context.parsed_document.get("body") or []:
            node_type = clean_text(
                node.get("type")
            )

            if node_type in {
                "article",
                "appendix",
                "part",
                "chapter",
                "section",
            }:
                continue

            rendered = render_generic_node(
                node,
                recursive=False,
            )

            if rendered:
                units.append(rendered)
            context.consumed_node_ids.add(id(node))

        if not units:
            return []

        parent_id = context.factory.ids.make(
            "document"
        )

        parent_text = "\n".join(
            units
        ).strip()

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
                "document",
                f"block_{ordinal}",
            )

            chunks.append(
                context.factory.make_chunk(
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    parent_type="document",
                    chunk_type="freeform_block",
                    strategy=self.name,
                    is_indexable=True,
                    structure_type=context.structure_type,
                    text=block,
                    metadata=context.metadata,
                    ordinal=ordinal,
                )
            )

        return chunks
