from __future__ import annotations

from html import escape
import os
import re
import unicodedata
from urllib.parse import urlparse

import streamlit as st


STATUS_LABELS = {
    "queue.completed": "Đã tiếp nhận yêu cầu",
    "decomposition.started": "Đang phân tích vấn đề pháp lý…",
    "decomposition.completed": "Đã phân tích câu hỏi",
    "retrieval.started": "Đang tra cứu văn bản pháp luật…",
    "retrieval.completed": "Đã tìm thấy nguồn liên quan",
    "context.started": "Đang xây dựng căn cứ trả lời…",
    "context.completed": "Đã chuẩn bị căn cứ",
    "generation.started": "Đang soạn câu trả lời…",
    "verification.started": "Đang kiểm tra căn cứ và trích dẫn…",
    "verification.completed": "Đã kiểm tra căn cứ và trích dẫn",
}

PUBLIC_API_BASE_URL = os.getenv(
    "LAWCHAT_PUBLIC_API_BASE_URL", "http://localhost:8000"
).rstrip("/")

_TRAILING_EVIDENCE_MARKERS = re.compile(
    r"(?:[ \t]*\[(?:E|G)\d+\])+(?=[ \t]*(?:\n|$))",
    re.MULTILINE,
)


def render_citation(citation: dict) -> None:
    title = citation.get("title") or "Văn bản pháp luật"
    number = citation.get("document_number") or ""
    provision = " · ".join(
        str(value) for value in (
            citation.get("article"), citation.get("clause"), citation.get("point")
        ) if value
    )
    status = citation.get("status") or "UNKNOWN"
    as_of = citation.get("as_of") or ""
    st.markdown(f"**{escape(str(title))}**")
    detail = " · ".join(value for value in (str(number), provision) if value)
    if detail:
        st.caption(detail)
    st.caption(f"Trạng thái: {status} tại {as_of}")
    document_id = str(citation.get("document_id") or "")
    if document_id:
        st.link_button(
            "Mở toàn văn",
            f"{PUBLIC_API_BASE_URL}/api/v1/documents/{document_id}/open-source",
        )
    else:
        source_url = str(citation.get("source_url") or "")
        if _safe_http_url(source_url):
            st.link_button("Mở nguồn", source_url)


def deduplicate_citations(citations: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen_document_numbers: set[str] = set()
    for citation in citations:
        document_number = _document_number_key(citation.get("document_number"))
        if document_number:
            if document_number in seen_document_numbers:
                continue
            seen_document_numbers.add(document_number)
        unique.append(citation)
    return unique


def render_citations(citations: list[dict]) -> None:
    unique = deduplicate_citations(citations)
    if not unique:
        return
    with st.expander(f"📚 {len(unique)} nguồn được trích dẫn"):
        for index, citation in enumerate(unique):
            render_citation(citation)
            if index < len(unique) - 1:
                st.divider()


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        content = message.get("content") or ""
        if role == "assistant":
            content = answer_for_display(content)
        st.markdown(content)
        render_citations(message.get("citations") or [])


def answer_for_display(value: object) -> str:
    return _TRAILING_EVIDENCE_MARKERS.sub("", str(value or "")).rstrip()


def _document_number_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = normalized.translate(str.maketrans("‐‑‒–—−", "------"))
    return re.sub(r"\s+", "", normalized)


def _safe_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
