from __future__ import annotations

import re
from typing import Any, Mapping

from .models import CHUNKER_VERSION, ChunkingConfig
from .utils import (
    clean_text,
    exact_token_count,
    metadata_value,
    nullable_text,
)


class ChunkIdBuilder:
    """
    Stable deterministic IDs.

    Numbering is preferred (article/clause/point). If malformed source data
    contains duplicate numbers, line_start or a stable duplicate suffix is
    used to prevent collisions without random UUIDs.
    """

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self._seen: dict[str, int] = {}

    def make(
        self,
        *parts: str,
        line_start: Any = None,
    ) -> str:
        safe = [
            self._safe_part(part)
            for part in parts
            if clean_text(part)
        ]

        base = "::".join(
            [self._safe_part(self.doc_id), *safe]
        )

        count = self._seen.get(base, 0)

        if count == 0:
            self._seen[base] = 1
            return base

        line = clean_text(line_start)

        if line:
            candidate = f"{base}::line_{self._safe_part(line)}"
            if candidate not in self._seen:
                self._seen[candidate] = 1
                self._seen[base] = count + 1
                return candidate

        self._seen[base] = count + 1
        return f"{base}::dup_{count + 1}"

    @staticmethod
    def _safe_part(value: Any) -> str:
        text = clean_text(value)

        text = re.sub(
            r"[^0-9A-Za-zÀ-ỹ_.:-]+",
            "_",
            text,
        )

        return text.strip("_")


class ChunkFactory:
    def __init__(
        self,
        *,
        doc_id: str,
        config: ChunkingConfig,
    ) -> None:
        self.doc_id = doc_id
        self.config = config
        self.ids = ChunkIdBuilder(doc_id)

    def build_retrieval_text(
        self,
        *,
        metadata: Mapping[str, Any],
        path: Mapping[str, Any],
        article: str | None,
        article_title: str | None = None,
        clause: str | None,
        point: str | None,
        legal_text: str,
    ) -> str:
        lines: list[str] = []

        title = metadata_value(
            metadata,
            "title",
            "document_title",
        )
        number = metadata_value(
            metadata,
            "so_ky_hieu",
            "document_number",
        )
        document_type = metadata_value(
            metadata,
            "loai_van_ban",
            "document_type",
        )
        authority = metadata_value(
            metadata,
            "co_quan_ban_hanh",
            "authority",
        )

        if title:
            lines.append(f"Văn bản: {title}")

        if number:
            lines.append(f"Số hiệu: {number}")

        if document_type:
            lines.append(f"Loại văn bản: {document_type}")

        if authority:
            lines.append(f"Cơ quan ban hành: {authority}")

        if self.config.include_temporal_metadata_in_retrieval_text:
            effective = metadata_value(
                metadata,
                "ngay_co_hieu_luc",
                "effective_date",
            )
            expiry = metadata_value(
                metadata,
                "ngay_het_hieu_luc",
                "expiry_date",
            )
            status = metadata_value(
                metadata,
                "tinh_trang_hieu_luc",
                "legal_status",
            )

            if effective:
                lines.append(
                    f"Ngày có hiệu lực: {effective}"
                )

            if expiry:
                lines.append(
                    f"Ngày hết hiệu lực: {expiry}"
                )

            if status:
                lines.append(
                    f"Tình trạng hiệu lực: {status}"
                )

        if path.get("part"):
            lines.append(f"Phần {path['part']}")

        if path.get("chapter"):
            lines.append(f"Chương {path['chapter']}")

        if path.get("section"):
            lines.append(f"Mục {path['section']}")

        if path.get("appendix"):
            lines.append(f"Phụ lục {path['appendix']}")

        if article:
            article_line = f"Điều {article}"
            if clean_text(article_title):
                article_line += f": {clean_text(article_title)}"
            lines.append(article_line)

        if clause:
            lines.append(f"Khoản {clause}")

        if point:
            lines.append(f"Điểm {point}")

        if legal_text:
            lines.append(legal_text)

        return "\n".join(lines).strip()

    def available_legal_text_tokens(
        self,
        *,
        metadata: Mapping[str, Any],
        path: Mapping[str, Any] | None = None,
        article: str | None = None,
        article_title: str | None = None,
        clause: str | None = None,
        point: str | None = None,
        max_tokens: int | None = None,
    ) -> int:
        """Token budget left after retrieval metadata/structure is added."""
        path = path or {}
        limit = max_tokens or self.config.base_indexable_tokens
        overhead = exact_token_count(
            self.build_retrieval_text(
                metadata=metadata,
                path=path,
                article=article,
                article_title=article_title,
                clause=clause,
                point=point,
                legal_text="",
            )
        )
        return max(1, limit - overhead)

    def make_chunk(
        self,
        *,
        chunk_id: str,
        parent_chunk_id: str | None,
        parent_type: str | None,
        chunk_type: str,
        strategy: str,
        is_indexable: bool,
        structure_type: str,
        text: str,
        metadata: Mapping[str, Any],
        path: Mapping[str, Any] | None = None,
        article: str | None = None,
        article_title: str | None = None,
        clause: str | None = None,
        point: str | None = None,
        ordinal: int = 0,
    ) -> dict[str, Any]:
        path = path or {}

        retrieval_text = (
            self.build_retrieval_text(
                metadata=metadata,
                path=path,
                article=article,
                article_title=article_title,
                clause=clause,
                point=point,
                legal_text=text,
            )
            if is_indexable
            else ""
        )

        return {
            "chunk_id": chunk_id,
            "doc_id": self.doc_id,
            "parent_chunk_id": parent_chunk_id,
            "parent_type": parent_type,
            "chunk_type": chunk_type,
            "strategy": strategy,
            "is_indexable": bool(is_indexable),
            "structure_type": structure_type,
            "part": nullable_text(path.get("part")),
            "chapter": nullable_text(path.get("chapter")),
            "section": nullable_text(path.get("section")),
            "appendix": nullable_text(path.get("appendix")),
            "article": nullable_text(article),
            "clause": nullable_text(clause),
            "point": nullable_text(point),
            "ordinal": int(ordinal),
            "text": clean_text(text),
            "retrieval_text": retrieval_text,
            "approx_token_count": exact_token_count(
                retrieval_text
            ),
            "overlap_text": None,
            "overlap_from_chunk_id": None,
            "overlap_token_count": 0,
            "document_title": metadata_value(
                metadata,
                "title",
                "document_title",
            ),
            "document_number": metadata_value(
                metadata,
                "so_ky_hieu",
                "document_number",
            ),
            "document_type": metadata_value(
                metadata,
                "loai_van_ban",
                "document_type",
            ),
            "authority": metadata_value(
                metadata,
                "co_quan_ban_hanh",
                "authority",
            ),
            "issued_date": metadata_value(
                metadata,
                "ngay_ban_hanh",
                "issued_date",
            ),
            "effective_date": metadata_value(
                metadata,
                "ngay_co_hieu_luc",
                "effective_date",
            ),
            "expiry_date": metadata_value(
                metadata,
                "ngay_het_hieu_luc",
                "expiry_date",
            ),
            "legal_status": metadata_value(
                metadata,
                "tinh_trang_hieu_luc",
                "legal_status",
            ),
            "legal_field": metadata_value(
                metadata,
                "linh_vuc",
                "legal_field",
            ),
            "chunker_version": CHUNKER_VERSION,
        }
