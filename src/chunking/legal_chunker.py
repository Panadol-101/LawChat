from __future__ import annotations

from typing import Any, Mapping

from .factory import ChunkFactory
from .models import (
    CHUNKER_VERSION as CURRENT_CHUNKER_VERSION,
    ChunkingConfig,
    ChunkingContext,
)
from .strategies import (
    AppendixChunkStrategy,
    ArticleChunkStrategy,
    FreeFormChunkStrategy,
    HierarchicalBlockStrategy,
    SemiStructuredChunkStrategy,
)
from .utils import (
    bounded_token_overlap_tail,
    clean_text,
    exact_token_count,
    exact_token_counts,
    has_node_type,
    render_article,
    render_generic_node,
    render_structural_heading,
    semantic_overlap_tail,
    split_exact_tokens,
)


class LegalChunker:
    """
    Router/composer for structure-aware Vietnamese legal chunking.

    It does NOT assume that every legal document has Điều.
    Multiple strategies may be applied to different subtrees of one document.
    """

    CHUNKER_VERSION = CURRENT_CHUNKER_VERSION

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()

        self.article_strategy = ArticleChunkStrategy()
        self.hierarchical_strategy = HierarchicalBlockStrategy()
        self.semi_strategy = SemiStructuredChunkStrategy()
        self.freeform_strategy = FreeFormChunkStrategy()
        self.appendix_strategy = AppendixChunkStrategy()

    def chunk_document(
        self,
        parsed_document: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        metadata = metadata or {}

        doc_id = clean_text(
            parsed_document.get("doc_id")
            or metadata.get("id")
        )

        if not doc_id:
            raise ValueError(
                "doc_id is required for chunking"
            )

        structure_type = clean_text(
            parsed_document.get("structure_type")
        ) or "unknown"

        if structure_type == "empty":
            return []

        body = parsed_document.get("body") or []

        factory = ChunkFactory(
            doc_id=doc_id,
            config=self.config,
        )

        context = ChunkingContext(
            doc_id=doc_id,
            structure_type=structure_type,
            parsed_document=parsed_document,
            metadata=metadata,
            config=self.config,
            factory=factory,
        )

        chunks: list[dict[str, Any]] = []

        has_non_appendix_articles = has_node_type(
            body,
            "article",
            exclude_under={"appendix"},
        )
        has_appendices = has_node_type(
            body,
            "appendix",
        )
        has_hierarchy = any(
            has_node_type(
                body,
                item,
                exclude_under={"appendix"},
            )
            for item in (
                "part",
                "chapter",
                "section",
            )
        )

        # One document may use multiple strategies.
        if has_non_appendix_articles:
            chunks.extend(
                self.article_strategy.chunk(
                    context
                )
            )

        if has_appendices:
            chunks.extend(
                self.appendix_strategy.chunk(
                    context
                )
            )

        if not has_non_appendix_articles:
            if has_hierarchy:
                chunks.extend(
                    self.hierarchical_strategy.chunk(
                        context
                    )
                )
            elif structure_type == "semi_structured":
                chunks.extend(
                    self.semi_strategy.chunk(
                        context
                    )
                )
            else:
                chunks.extend(
                    self.freeform_strategy.chunk(
                        context
                    )
                )

        # Article documents can still contain headings, paragraphs, lists or
        # tables outside Điều. Route those residual nodes instead of silently
        # dropping them merely because one article exists elsewhere.
        if has_non_appendix_articles:
            residual_nodes = self._collect_residual_nodes(
                body,
                context,
            )
            if residual_nodes:
                chunks.extend(
                    self.semi_strategy.chunk_nodes(
                        context=context,
                        nodes=residual_nodes,
                        parent_key="residual_content",
                        parent_type="document",
                        chunk_type="residual_block",
                    )
                )

        self._validate_top_level_coverage(body, context)

        chunks = self._apply_semantic_overlap(
            chunks,
            context,
        )

        chunks = self._enforce_indexable_token_limit(
            chunks,
            context,
        )

        for chunk in chunks:
            chunk.pop("_article_title", None)
            chunk.pop("_retrieval_context", None)

        self._validate_chunks(chunks)
        self._validate_content_coverage(parsed_document, chunks)

        return chunks

    @staticmethod
    def _validate_content_coverage(
        parsed_document: Mapping[str, Any],
        chunks: list[dict[str, Any]],
    ) -> None:
        """Ensure every parsed source unit survives in canonical chunk text."""
        haystacks = [clean_text(chunk.get("text")) for chunk in chunks]

        def covered(value: str) -> bool:
            value = clean_text(value)
            return not value or any(value in haystack for haystack in haystacks)

        preamble = clean_text(parsed_document.get("preamble"))
        if preamble and not covered(preamble):
            raise ValueError("Chunk content coverage failed for document preamble")

        def visit(nodes: list[Mapping[str, Any]]) -> None:
            for node in nodes:
                node_type = clean_text(node.get("type"))
                if node_type == "article":
                    source = render_article(node)
                elif node_type == "appendix":
                    source = render_generic_node(node, recursive=True)
                elif node_type in {"part", "chapter", "section"}:
                    # A structural container can be composed by multiple
                    # strategies.  For example, direct prose is routed to the
                    # semi-structured strategy while nested Articles are
                    # routed to the Article strategy.  Requiring the complete
                    # Section/Chapter text to remain contiguous in one chunk
                    # therefore produces false coverage failures.  Validate
                    # the heading here and validate every child independently
                    # in the recursive visit below.
                    source = render_structural_heading(node)
                else:
                    source = render_generic_node(node, recursive=False)
                if source and not covered(source):
                    raise ValueError(
                        "Chunk content coverage failed for parsed node: "
                        f"{node_type or 'unknown'}"
                    )
                if node_type != "appendix":
                    visit(list(node.get("children") or []))

        visit(list(parsed_document.get("body") or []))

    @staticmethod
    def _apply_semantic_overlap(
        chunks: list[dict[str, Any]],
        context: ChunkingContext,
    ) -> list[dict[str, Any]]:
        config = context.config
        if not config.enable_semantic_overlap:
            return chunks

        eligible_types = {
            "preamble_fragment",
            "article_fragment",
            "article_lead_fragment",
            "clause_fragment",
            "point_fragment",
            "freeform_block",
        }
        previous_by_group: dict[tuple[str, str], dict[str, Any]] = {}
        overlap_label = "[Ngữ cảnh tiếp nối]"
        label_tokens = exact_token_count(overlap_label)

        for chunk in chunks:
            chunk_type = clean_text(chunk.get("chunk_type"))
            parent_id = clean_text(chunk.get("parent_chunk_id"))
            if (
                not chunk.get("is_indexable")
                or chunk_type not in eligible_types
                or not parent_id
            ):
                continue

            group_key = (parent_id, chunk_type)
            previous = previous_by_group.get(group_key)
            previous_by_group[group_key] = chunk

            if previous is None or previous.get("doc_id") != chunk.get("doc_id"):
                continue

            previous_tokens = exact_token_count(
                clean_text(previous.get("text"))
            )
            if previous_tokens < config.overlap_min_source_tokens:
                continue

            available = (
                config.max_indexable_tokens
                - int(chunk.get("approx_token_count") or 0)
                - label_tokens
            )
            if available < config.min_overlap_tokens:
                continue

            target = max(
                config.min_overlap_tokens,
                round(previous_tokens * config.overlap_ratio),
            )
            max_semantic_overlap = max(
                config.min_overlap_tokens,
                round(previous_tokens * config.max_overlap_ratio),
            )
            available = min(available, max_semantic_overlap)
            target = min(target, available)
            overlap_text = semantic_overlap_tail(
                clean_text(previous.get("text")),
                target_tokens=target,
                max_tokens=available,
            )
            overlap_tokens = exact_token_count(overlap_text)
            if overlap_tokens < config.min_overlap_tokens:
                overlap_text = bounded_token_overlap_tail(
                    clean_text(previous.get("text")),
                    target_tokens=target,
                    max_tokens=available,
                )
                overlap_tokens = exact_token_count(overlap_text)
                if overlap_tokens < config.min_overlap_tokens:
                    continue

            path = {
                key: chunk.get(key)
                for key in ("part", "chapter", "section", "appendix")
                if chunk.get(key)
            }
            legal_text = "\n".join(
                item
                for item in [
                    overlap_label,
                    overlap_text,
                    clean_text(chunk.get("_retrieval_context")),
                    clean_text(chunk.get("text")),
                ]
                if item
            )
            retrieval_text = context.factory.build_retrieval_text(
                metadata=context.metadata,
                path=path,
                article=chunk.get("article"),
                article_title=clean_text(chunk.get("_article_title")),
                clause=chunk.get("clause"),
                point=chunk.get("point"),
                legal_text=legal_text,
            )
            final_tokens = exact_token_count(retrieval_text)
            if final_tokens > config.max_indexable_tokens:
                continue

            chunk["overlap_text"] = overlap_text
            chunk["overlap_from_chunk_id"] = previous["chunk_id"]
            chunk["overlap_token_count"] = overlap_tokens
            chunk["retrieval_text"] = retrieval_text
            chunk["approx_token_count"] = final_tokens

        return chunks

    @staticmethod
    def _validate_top_level_coverage(
        body: list[Mapping[str, Any]],
        context: ChunkingContext,
    ) -> None:
        missing = [
            clean_text(node.get("type")) or "unknown"
            for node in body
            if id(node) not in context.consumed_node_ids
        ]
        if missing:
            raise ValueError(
                "Unconsumed top-level parsed nodes: "
                + ", ".join(missing)
            )

    def _collect_residual_nodes(
        self,
        nodes: list[Mapping[str, Any]],
        context: ChunkingContext,
    ) -> list[Mapping[str, Any]]:
        def collect(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            residual: list[Mapping[str, Any]] = []
            for node in items:
                node_type = clean_text(node.get("type"))

                # These subtrees are owned by their dedicated strategies.
                if node_type in {"article", "appendix"}:
                    continue

                if node_type in {"part", "chapter", "section"}:
                    context.consumed_node_ids.add(id(node))
                    descendants = collect(list(node.get("children") or []))
                    heading = render_structural_heading(node)
                    if heading:
                        residual.append({"type": "heading", "text": heading})
                    residual.extend(descendants)
                    continue

                residual.append(node)
                context.consumed_node_ids.add(id(node))
            return residual

        return collect(nodes)

    def _enforce_indexable_token_limit(
        self,
        chunks: list[dict[str, Any]],
        context: ChunkingContext,
    ) -> list[dict[str, Any]]:
        limit = context.config.max_indexable_tokens
        indexable = [chunk for chunk in chunks if chunk.get("is_indexable")]
        counts = exact_token_counts(
            [clean_text(chunk.get("retrieval_text")) for chunk in indexable]
        )
        count_by_id = {
            str(chunk["chunk_id"]): count
            for chunk, count in zip(indexable, counts, strict=True)
        }
        output: list[dict[str, Any]] = []
        for chunk in chunks:
            if not chunk.get("is_indexable"):
                output.append(chunk)
                continue
            exact_count = count_by_id[str(chunk["chunk_id"])]
            chunk["approx_token_count"] = exact_count
            if chunk.get("overlap_text"):
                chunk["overlap_token_count"] = exact_token_count(
                    clean_text(chunk.get("overlap_text"))
                )
            if exact_count <= limit:
                output.append(chunk)
                continue
            output.extend(self._split_for_model(chunk, context))

        still_oversized = [
            chunk
            for chunk in output
            if chunk.get("is_indexable")
            and int(chunk.get("approx_token_count") or 0) > limit
        ]
        if still_oversized:
            details = "; ".join(
                f"{chunk['chunk_id']}:{chunk.get('approx_token_count')}"
                for chunk in still_oversized[:5]
            )
            raise ValueError("Model-token limit enforcement failed: " + details)
        return output

    @staticmethod
    def _split_for_model(
        chunk: dict[str, Any],
        context: ChunkingContext,
    ) -> list[dict[str, Any]]:
        limit = context.config.max_indexable_tokens
        path = {
            key: chunk.get(key)
            for key in ("part", "chapter", "section", "appendix")
            if chunk.get(key)
        }
        prefix = context.factory.build_retrieval_text(
            metadata=context.metadata,
            path=path,
            article=chunk.get("article"),
            article_title=clean_text(chunk.get("_article_title")),
            clause=chunk.get("clause"),
            point=chunk.get("point"),
            legal_text="",
        )
        # Both prefix and standalone piece counts include special tokens;
        # subtracting an extra small reserve keeps the combined encoding safe.
        legal_budget = max(8, limit - exact_token_count(prefix) - 4)
        legal_text = "\n".join(
            value
            for value in (
                "[Ngữ cảnh tiếp nối]" if chunk.get("overlap_text") else "",
                clean_text(chunk.get("overlap_text")),
                clean_text(chunk.get("_retrieval_context")),
                clean_text(chunk.get("text")),
            )
            if value
        )
        pieces = split_exact_tokens(legal_text, max_tokens=legal_budget)
        parent = dict(chunk)
        original_type = clean_text(chunk.get("chunk_type")) or "chunk"
        parent["chunk_type"] = f"{original_type}_model_parent"
        parent["is_indexable"] = False
        parent["retrieval_text"] = ""
        parent["approx_token_count"] = 0
        parent["overlap_text"] = None
        parent["overlap_from_chunk_id"] = None
        parent["overlap_token_count"] = 0

        output = [parent]
        for index, piece in enumerate(pieces, start=1):
            fragment = dict(chunk)
            fragment["chunk_id"] = (
                f"{chunk['chunk_id']}::fragment_{index}"
            )
            fragment["parent_chunk_id"] = chunk["chunk_id"]
            fragment["parent_type"] = original_type
            fragment["chunk_type"] = f"{original_type}_model_fragment"
            fragment["ordinal"] = index - 1
            fragment["text"] = piece
            fragment["overlap_text"] = None
            fragment["overlap_from_chunk_id"] = None
            fragment["overlap_token_count"] = 0
            fragment["retrieval_text"] = context.factory.build_retrieval_text(
                metadata=context.metadata,
                path=path,
                article=fragment.get("article"),
                article_title=clean_text(fragment.get("_article_title")),
                clause=fragment.get("clause"),
                point=fragment.get("point"),
                legal_text=piece,
            )
            fragment["approx_token_count"] = exact_token_count(
                fragment["retrieval_text"]
            )
            output.append(fragment)
        return output

    def _validate_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        ids = [
            str(chunk["chunk_id"])
            for chunk in chunks
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk_id generated "
                "within one document"
            )

        known = set(ids)

        for chunk in chunks:
            parent = chunk.get(
                "parent_chunk_id"
            )

            if (
                parent is not None
                and parent not in known
            ):
                raise ValueError(
                    "Orphan child chunk: "
                    f"{chunk['chunk_id']} -> {parent}"
                )

            if (
                chunk.get("is_indexable")
                and not clean_text(chunk.get("text"))
            ):
                raise ValueError(
                    "Indexable chunk has empty text: "
                    f"{chunk['chunk_id']}"
                )

            if int(chunk.get("ordinal") or 0) < 0:
                raise ValueError(
                    "Chunk ordinal must be non-negative: "
                    f"{chunk['chunk_id']}"
                )

            if (
                chunk.get("is_indexable")
                and not clean_text(
                    chunk.get("retrieval_text")
                )
            ):
                raise ValueError(
                    "Indexable chunk has empty retrieval_text: "
                    f"{chunk['chunk_id']}"
                )

            if (
                chunk.get("is_indexable")
                and int(chunk.get("approx_token_count") or 0)
                > self.config.max_indexable_tokens
            ):
                raise ValueError(
                    "Indexable chunk exceeds max_indexable_tokens: "
                    f"{chunk['chunk_id']}"
                )
