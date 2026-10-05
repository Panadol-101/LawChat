from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from preprocessing.html_cleaner import (
    TABLE_END_MARKER,
    TABLE_HEADERS_MARKER,
    TABLE_ROW_MARKER,
    TABLE_START_MARKER,
)


@dataclass(frozen=True, slots=True)
class LineToken:
    line_no: int
    raw: str
    kind: str
    label: str | None = None
    title: str | None = None


class LegalStructureParser:
    """
    Structure-preserving parser for Vietnamese legal documents.

    Design goals:
    1. Never require every document to contain Điều/Khoản/Điểm.
    2. Parse legal structure conservatively:
         Điều -> Khoản -> Điểm
       then place Điều into:
         Mục -> Chương -> Phần
    3. Preserve text that cannot be safely classified.
    4. Never silently discard free-form legal documents, appendices or forms.
    5. Return a uniform root object suitable for later RAG chunking.

    The parser does not decide whether a document is legally valid/effective.
    That belongs to the metadata/temporal layer.
    """

    PARSER_VERSION = "2.0.0"

    # ------------------------------------------------------------------
    # Structural headings
    # ------------------------------------------------------------------

    _PART_RE = re.compile(
        r"^\s*PHẦN\s+"
        r"(?P<label>(?:THỨ\s+\S+)|[IVXLCDM]+|\d+[A-ZĐ]?)"
        r"(?=\s|[.:\-–—]|$)"
        r"\s*(?:[.:\-–—]\s*)?"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )

    _CHAPTER_RE = re.compile(
        r"^\s*CHƯƠNG\s+"
        r"(?P<label>[IVXLCDM]+|\d+[A-ZĐ]?)"
        r"(?=\s|[.:\-–—]|$)"
        r"\s*(?:[.:\-–—]\s*)?"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )

    _SECTION_RE = re.compile(
        r"^\s*MỤC\s+"
        r"(?P<label>[IVXLCDM]+|\d+[A-ZĐ]?)"
        r"(?=\s|[.:\-–—]|$)"
        r"\s*(?:[.:\-–—]\s*)?"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )

    _ROMAN_SECTION_RE = re.compile(
        r"^\s*(?P<label>[IVXLCDM]+)[.)]\s+(?P<title>.+)$",
        re.IGNORECASE,
    )

    _ARTICLE_RE = re.compile(
        r"^\s*ĐIỀU\s+"
        r"(?P<label>\d+[A-ZĐ]?)"
        r"(?=\s|[.:\-–—]|$)"
        r"\s*(?:[.:\-–—]\s*)?"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )

    # Khoản: 1. ..., 2) ..., 1a. ...
    _CLAUSE_RE = re.compile(
        r"^\s*(?P<label>\d{1,3}[a-zđ]?)"
        r"(?P<sep>[.)])"
        r"\s*(?P<text>.*)$",
        re.IGNORECASE,
    )

    # Điểm: a) ..., b. ..., đ) ..., a1) ...
    _POINT_RE = re.compile(
        r"^\s*(?P<label>[a-zđ](?:\d+)?)"
        r"(?P<sep>[.)])"
        r"\s*(?P<text>.*)$",
        re.IGNORECASE,
    )

    _BULLET_RE = re.compile(
        r"^\s*[-–—•▪▫◦‣⁃∙]\s*(?P<text>.*)$"
    )

    _APPENDIX_RE = re.compile(
        r"^\s*(?P<label>PHỤ\s+LỤC(?:\s+[IVXLCDM\d]+)?|MẪU\s+SỐ\s+\S+)"
        r"(?=\s|[.:\-–—]|$)"
        r"\s*(?:[.:\-–—]\s*)?"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )

    _SIGNATURE_RE = re.compile(
        r"^\s*(?:KT\.|TM\.|TL\.|TUQ\.)\s+.+$|"
        r"^\s*NƠI\s+NHẬN\s*:?\s*$",
        re.IGNORECASE,
    )

    _POST_SIGNATURE_APPENDIX_RE = re.compile(
        r"^\s*(?:DANH\s+MỤC|BẢNG|BIỂU)(?:\s|$)",
        re.IGNORECASE,
    )

    _KNOWN_FREE_HEADINGS = re.compile(
        r"^\s*(?:"
        r"QUYẾT\s+ĐỊNH|"
        r"NGHỊ\s+QUYẾT|"
        r"THÔNG\s+BÁO|"
        r"CHỈ\s+THỊ|"
        r"CÔNG\s+VĂN|"
        r"KẾ\s+HOẠCH|"
        r"BÁO\s+CÁO|"
        r"TỜ\s+TRÌNH|"
        r"NƠI\s+NHẬN|"
        r"KÍNH\s+GỬI"
        r")\s*:?\s*$",
        re.IGNORECASE,
    )

    _ARTICLE_TITLE_BLOCKERS = re.compile(
        r"^\s*(?:"
        r"Căn\s+cứ|"
        r"Theo\s+đề\s+nghị|"
        r"Xét\s+đề\s+nghị|"
        r"Kính\s+gửi|"
        r"Nơi\s+nhận|"
        r"Thực\s+hiện|"
        r"Quyết\s+định\s*:|"
        r"Điều\s+\d+|"
        r"\d{1,3}[a-zđ]?[.)]|"
        r"[a-zđ](?:\d+)?[.)]"
        r")",
        re.IGNORECASE,
    )

    _STRUCTURAL_KINDS = {
        "part",
        "chapter",
        "section",
        "article",
        "appendix",
        "table_start",
        "signature",
    }

    def __init__(
        self,
        *,
        infer_split_titles: bool = True,
        max_inferred_title_chars: int = 180,
        max_inferred_title_words: int = 18,
    ) -> None:
        self.infer_split_titles = infer_split_titles
        self.max_inferred_title_chars = max_inferred_title_chars
        self.max_inferred_title_words = max_inferred_title_words

    # ==================================================================
    # Public API
    # ==================================================================

    def parse_document(
        self,
        content_clean: str | None,
        *,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Parse one cleaned legal document into a structured tree.

        The root always has the same shape:
        {
          "doc_id": ...,
          "parser_version": ...,
          "structure_type": ...,
          "preamble": ...,
          "body": [...],
          "stats": {...},
          "warnings": [...]
        }
        """
        text = (content_clean or "").strip()

        if not text:
            return self._empty_document(doc_id)

        tokens = self._tokenize(text)
        warnings: list[str] = []

        if not tokens:
            return self._empty_document(doc_id)

        has_any_structural_marker = any(
            token.kind in self._STRUCTURAL_KINDS
            for token in tokens
        )

        # Documents without legal hierarchy are not errors.
        # Preserve them as free/semi-structured blocks.
        if not has_any_structural_marker:
            body = self._build_free_form_blocks(tokens)
            stats = self._collect_stats(body)
            structure_type = self._classify_structure(stats, body)

            return {
                "doc_id": None if doc_id is None else str(doc_id),
                "parser_version": self.PARSER_VERSION,
                "structure_type": structure_type,
                "preamble": "",
                "body": body,
                "stats": stats,
                "warnings": warnings,
            }

        first_structural_idx = next(
            i
            for i, token in enumerate(tokens)
            if token.kind in self._STRUCTURAL_KINDS
        )

        preamble_tokens = tokens[:first_structural_idx]
        preamble = self._join_raw_lines(preamble_tokens)

        body_tokens = tokens[first_structural_idx:]
        body = self._build_hierarchy(body_tokens, warnings)

        stats = self._collect_stats(body)
        structure_type = self._classify_structure(stats, body)

        self._append_duplicate_number_warnings(body, warnings)

        return {
            "doc_id": None if doc_id is None else str(doc_id),
            "parser_version": self.PARSER_VERSION,
            "structure_type": structure_type,
            "preamble": preamble,
            "body": body,
            "stats": stats,
            "warnings": warnings,
        }

    def parse_json(
        self,
        content_clean: str | None,
        *,
        doc_id: str | None = None,
        pretty: bool = False,
    ) -> str:
        parsed = self.parse_document(
            content_clean,
            doc_id=doc_id,
        )

        if pretty:
            return json.dumps(
                parsed,
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # ==================================================================
    # Tokenization
    # ==================================================================

    def _tokenize(self, text: str) -> list[LineToken]:
        tokens: list[LineToken] = []

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()

            if not line:
                continue

            tokens.append(self._classify_line(line_no, line))

        return tokens

    def _classify_line(self, line_no: int, line: str) -> LineToken:
        if line == TABLE_START_MARKER:
            return LineToken(line_no=line_no, raw=line, kind="table_start")
        if line.startswith(TABLE_HEADERS_MARKER):
            return LineToken(line_no=line_no, raw=line, kind="table_headers")
        if line.startswith(TABLE_ROW_MARKER):
            return LineToken(line_no=line_no, raw=line, kind="table_row")
        if line == TABLE_END_MARKER:
            return LineToken(line_no=line_no, raw=line, kind="table_end")

        match = self._PART_RE.match(line)
        if match:
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="part",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        match = self._CHAPTER_RE.match(line)
        if match:
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="chapter",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        match = self._SECTION_RE.match(line)
        if match:
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="section",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        match = self._ROMAN_SECTION_RE.match(line)
        if match and self._is_free_heading(match.group("title")):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="section",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        match = self._ARTICLE_RE.match(line)
        if match:
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="article",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        match = self._APPENDIX_RE.match(line)
        if match:
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="appendix",
                label=self._clean_label(match.group("label")),
                title=self._clean_title(match.group("title")),
            )

        if self._SIGNATURE_RE.match(line):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="signature",
            )

        if self._CLAUSE_RE.match(line):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="numbered_item",
            )

        if self._POINT_RE.match(line):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="lettered_item",
            )

        if self._BULLET_RE.match(line):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="bullet",
            )

        if self._is_free_heading(line):
            return LineToken(
                line_no=line_no,
                raw=line,
                kind="free_heading",
            )

        return LineToken(
            line_no=line_no,
            raw=line,
            kind="text",
        )

    # ==================================================================
    # Hierarchy construction
    # ==================================================================

    def _build_hierarchy(
        self,
        tokens: list[LineToken],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        body: list[dict[str, Any]] = []

        current_appendix: dict[str, Any] | None = None
        current_part: dict[str, Any] | None = None
        current_chapter: dict[str, Any] | None = None
        current_section: dict[str, Any] | None = None
        after_signature = False

        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token.kind == "signature":
                current_appendix = None
                current_part = None
                current_chapter = None
                current_section = None
                after_signature = True
                body.append(
                    {
                        "type": "heading",
                        "text": token.raw,
                        "line_start": token.line_no,
                    }
                )
                i += 1
                continue

            if (
                after_signature
                and token.kind == "free_heading"
                and self._POST_SIGNATURE_APPENDIX_RE.match(token.raw)
            ):
                inferred = LineToken(
                    line_no=token.line_no,
                    raw=token.raw,
                    kind="appendix",
                    label=token.raw,
                    title="",
                )
                node, consumed = self._make_container_node(
                    inferred,
                    tokens,
                    i,
                    node_type="appendix",
                )
                body.append(node)
                current_appendix = node
                after_signature = False
                i += consumed
                continue

            if token.kind == "table_start":
                node, consumed = self._parse_table(tokens, i, warnings)
                target_children = self._deepest_children(
                    body=body,
                    current_appendix=current_appendix,
                    current_part=current_part,
                    current_chapter=current_chapter,
                    current_section=current_section,
                )
                target_children.append(node)
                i += consumed
                continue

            if token.kind == "appendix":
                node, consumed = self._make_container_node(
                    token,
                    tokens,
                    i,
                    node_type="appendix",
                )

                body.append(node)

                current_appendix = node
                current_part = None
                current_chapter = None
                current_section = None
                after_signature = False

                i += consumed
                continue

            if token.kind == "part":
                node, consumed = self._make_container_node(
                    token,
                    tokens,
                    i,
                    node_type="part",
                )

                parent_children = (
                    current_appendix["children"]
                    if current_appendix is not None
                    else body
                )
                parent_children.append(node)

                current_part = node
                current_chapter = None
                current_section = None

                i += consumed
                continue

            if token.kind == "chapter":
                node, consumed = self._make_container_node(
                    token,
                    tokens,
                    i,
                    node_type="chapter",
                )

                if current_part is not None:
                    current_part["children"].append(node)
                elif current_appendix is not None:
                    current_appendix["children"].append(node)
                else:
                    body.append(node)

                current_chapter = node
                current_section = None

                i += consumed
                continue

            if token.kind == "section":
                node, consumed = self._make_container_node(
                    token,
                    tokens,
                    i,
                    node_type="section",
                )

                if current_chapter is not None:
                    current_chapter["children"].append(node)
                elif current_part is not None:
                    current_part["children"].append(node)
                elif current_appendix is not None:
                    current_appendix["children"].append(node)
                else:
                    body.append(node)

                current_section = node

                i += consumed
                continue

            if token.kind == "article":
                article_tokens: list[LineToken] = []
                j = i + 1

                while j < len(tokens):
                    if tokens[j].kind in self._STRUCTURAL_KINDS:
                        break

                    article_tokens.append(tokens[j])
                    j += 1

                article = self._parse_article(
                    heading=token,
                    body_tokens=article_tokens,
                    warnings=warnings,
                )

                article["path"] = self._current_path(
                    current_appendix=current_appendix,
                    current_part=current_part,
                    current_chapter=current_chapter,
                    current_section=current_section,
                )

                if current_section is not None:
                    current_section["children"].append(article)
                elif current_chapter is not None:
                    current_chapter["children"].append(article)
                elif current_part is not None:
                    current_part["children"].append(article)
                elif current_appendix is not None:
                    current_appendix["children"].append(article)
                else:
                    body.append(article)

                i = j
                continue

            # Text outside Điều is preserved instead of discarded.
            target_children = self._deepest_children(
                body=body,
                current_appendix=current_appendix,
                current_part=current_part,
                current_chapter=current_chapter,
                current_section=current_section,
            )

            target_children.append(
                self._free_block_from_token(token)
            )
            i += 1

        return body

    def _parse_table(
        self,
        tokens: list[LineToken],
        index: int,
        warnings: list[str],
    ) -> tuple[dict[str, Any], int]:
        headers: list[str] = []
        rows: list[list[str]] = []
        i = index + 1
        found_end = False

        while i < len(tokens):
            token = tokens[i]
            if token.kind == "table_end":
                found_end = True
                i += 1
                break

            marker = None
            if token.kind == "table_headers":
                marker = TABLE_HEADERS_MARKER
            elif token.kind == "table_row":
                marker = TABLE_ROW_MARKER
            else:
                warnings.append(f"unexpected_table_line:{token.line_no}")
                i += 1
                continue

            try:
                payload = json.loads(token.raw[len(marker):].strip())
                values = [self._clean_text(str(value)) for value in payload]
                if token.kind == "table_headers":
                    headers = values
                else:
                    rows.append(values)
            except (json.JSONDecodeError, TypeError):
                warnings.append(f"invalid_table_row:{token.line_no}")
            i += 1

        if not found_end:
            warnings.append(f"unterminated_table:{tokens[index].line_no}")

        line_end = tokens[i - 1].line_no if i > index else tokens[index].line_no
        return (
            {
                "type": "table",
                "headers": headers,
                "rows": rows,
                "line_start": tokens[index].line_no,
                "line_end": line_end,
            },
            i - index,
        )

    def _make_container_node(
        self,
        token: LineToken,
        tokens: list[LineToken],
        index: int,
        *,
        node_type: str,
    ) -> tuple[dict[str, Any], int]:
        title = token.title or ""
        title_inferred = False
        consumed = 1

        if (
            not title
            and self.infer_split_titles
            and index + 1 < len(tokens)
        ):
            candidate = tokens[index + 1]

            if self._looks_like_split_title(
                candidate,
                strict=False,
            ):
                title = candidate.raw
                title_inferred = True
                consumed = 2

        key = {
            "part": "part",
            "chapter": "chapter",
            "section": "section",
            "appendix": "appendix",
        }[node_type]

        return (
            {
                "type": node_type,
                key: token.label or token.raw,
                "title": title,
                "title_inferred": title_inferred,
                "raw_heading": token.raw,
                "line_start": token.line_no,
                "children": [],
            },
            consumed,
        )

    # ==================================================================
    # Điều -> Khoản -> Điểm
    # ==================================================================

    def _parse_article(
        self,
        *,
        heading: LineToken,
        body_tokens: list[LineToken],
        warnings: list[str],
    ) -> dict[str, Any]:
        working_tokens = list(body_tokens)

        title = heading.title or ""
        title_inferred = False

        if (
            not title
            and self.infer_split_titles
            and working_tokens
            and self._looks_like_split_title(
                working_tokens[0],
                strict=True,
            )
        ):
            title = working_tokens[0].raw
            title_inferred = True
            working_tokens = working_tokens[1:]

        article_text_parts: list[str] = []
        clauses: list[dict[str, Any]] = []
        direct_points: list[dict[str, Any]] = []

        current_clause: dict[str, Any] | None = None
        current_point: dict[str, Any] | None = None

        for token in working_tokens:
            line = token.raw

            clause_match = self._CLAUSE_RE.match(line)

            if clause_match:
                current_point = None

                current_clause = {
                    "clause": self._clean_label(
                        clause_match.group("label")
                    ),
                    "text": self._clean_text(
                        clause_match.group("text")
                    ),
                    "points": [],
                    "line_start": token.line_no,
                }

                clauses.append(current_clause)
                continue

            point_match = self._POINT_RE.match(line)

            if point_match:
                point = {
                    "point": self._clean_label(
                        point_match.group("label")
                    ),
                    "text": self._clean_text(
                        point_match.group("text")
                    ),
                    "line_start": token.line_no,
                }

                if current_clause is not None:
                    current_clause["points"].append(point)
                else:
                    direct_points.append(point)

                current_point = point
                continue

            # Everything else is continuation text.
            if current_point is not None:
                current_point["text"] = self._append_text(
                    current_point["text"],
                    line,
                )
            elif current_clause is not None:
                current_clause["text"] = self._append_text(
                    current_clause["text"],
                    line,
                )
            else:
                article_text_parts.append(line)

        article_text = "\n".join(article_text_parts).strip()

        self._validate_article_children(
            article_number=heading.label or "",
            clauses=clauses,
            direct_points=direct_points,
            warnings=warnings,
        )

        if (
            not article_text
            and not clauses
            and not direct_points
        ):
            warnings.append(
                f"article_without_body:{heading.label}"
            )

        line_end = (
            body_tokens[-1].line_no
            if body_tokens
            else heading.line_no
        )

        return {
            "type": "article",
            "article": heading.label or "",
            "title": title,
            "title_inferred": title_inferred,
            "raw_heading": heading.raw,
            "text": article_text,
            "clauses": clauses,
            "points": direct_points,
            "line_start": heading.line_no,
            "line_end": line_end,
        }

    def _validate_article_children(
        self,
        *,
        article_number: str,
        clauses: list[dict[str, Any]],
        direct_points: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        clause_numbers = [
            str(clause["clause"]).casefold()
            for clause in clauses
        ]

        if len(clause_numbers) != len(set(clause_numbers)):
            warnings.append(
                f"duplicate_clause_number:article={article_number}"
            )

        for clause in clauses:
            point_numbers = [
                str(point["point"]).casefold()
                for point in clause["points"]
            ]

            if len(point_numbers) != len(set(point_numbers)):
                warnings.append(
                    "duplicate_point_number:"
                    f"article={article_number},"
                    f"clause={clause['clause']}"
                )

        direct_point_numbers = [
            str(point["point"]).casefold()
            for point in direct_points
        ]

        if (
            len(direct_point_numbers)
            != len(set(direct_point_numbers))
        ):
            warnings.append(
                f"duplicate_direct_point_number:article={article_number}"
            )

    # ==================================================================
    # Free/semi-structured documents
    # ==================================================================

    def _build_free_form_blocks(
        self,
        tokens: list[LineToken],
    ) -> list[dict[str, Any]]:
        return [
            self._free_block_from_token(token)
            for token in tokens
        ]

    def _free_block_from_token(
        self,
        token: LineToken,
    ) -> dict[str, Any]:
        line = token.raw

        clause_match = self._CLAUSE_RE.match(line)
        if clause_match:
            return {
                "type": "list_item",
                "marker": clause_match.group("label")
                + clause_match.group("sep"),
                "text": self._clean_text(
                    clause_match.group("text")
                ),
                "line_start": token.line_no,
            }

        point_match = self._POINT_RE.match(line)
        if point_match:
            return {
                "type": "list_item",
                "marker": point_match.group("label")
                + point_match.group("sep"),
                "text": self._clean_text(
                    point_match.group("text")
                ),
                "line_start": token.line_no,
            }

        bullet_match = self._BULLET_RE.match(line)
        if bullet_match:
            return {
                "type": "list_item",
                "marker": "-",
                "text": self._clean_text(
                    bullet_match.group("text")
                ),
                "line_start": token.line_no,
            }

        if token.kind == "free_heading":
            return {
                "type": "heading",
                "text": line,
                "line_start": token.line_no,
            }

        return {
            "type": "paragraph",
            "text": line,
            "line_start": token.line_no,
        }

    # ==================================================================
    # Heuristics
    # ==================================================================

    def _looks_like_split_title(
        self,
        token: LineToken,
        *,
        strict: bool,
    ) -> bool:
        if token.kind in self._STRUCTURAL_KINDS:
            return False

        line = token.raw.strip()

        if not line:
            return False

        if self._CLAUSE_RE.match(line):
            return False

        if self._POINT_RE.match(line):
            return False

        if self._BULLET_RE.match(line):
            return False

        if len(line) > self.max_inferred_title_chars:
            return False

        words = line.split()

        if len(words) > self.max_inferred_title_words:
            return False

        if self._ARTICLE_TITLE_BLOCKERS.match(line):
            return False

        # A legal heading/title is normally not a complete sentence.
        if line.endswith((".", ";", "?", "!")):
            return False

        if strict:
            # Article title inference is intentionally conservative.
            # Too much punctuation usually indicates prose rather than a title.
            if "," in line or ";" in line:
                return False

            if len(words) > 14:
                return False

        return True

    def _is_free_heading(self, line: str) -> bool:
        if self._KNOWN_FREE_HEADINGS.match(line):
            return True

        # Uppercase short lines are often document headings.
        letters = [char for char in line if char.isalpha()]

        if not letters:
            return False

        if len(line) > 140:
            return False

        uppercase_letters = sum(
            1
            for char in letters
            if char.isupper()
        )

        uppercase_ratio = uppercase_letters / len(letters)

        return uppercase_ratio >= 0.90

    # ==================================================================
    # Paths / statistics / validation
    # ==================================================================

    def _current_path(
        self,
        *,
        current_appendix: dict[str, Any] | None,
        current_part: dict[str, Any] | None,
        current_chapter: dict[str, Any] | None,
        current_section: dict[str, Any] | None,
    ) -> dict[str, str]:
        path: dict[str, str] = {}

        if current_appendix is not None:
            path["appendix"] = str(
                current_appendix.get("appendix", "")
            )

        if current_part is not None:
            path["part"] = str(
                current_part.get("part", "")
            )

        if current_chapter is not None:
            path["chapter"] = str(
                current_chapter.get("chapter", "")
            )

        if current_section is not None:
            path["section"] = str(
                current_section.get("section", "")
            )

        return path

    def _deepest_children(
        self,
        *,
        body: list[dict[str, Any]],
        current_appendix: dict[str, Any] | None,
        current_part: dict[str, Any] | None,
        current_chapter: dict[str, Any] | None,
        current_section: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if current_section is not None:
            return current_section["children"]

        if current_chapter is not None:
            return current_chapter["children"]

        if current_part is not None:
            return current_part["children"]

        if current_appendix is not None:
            return current_appendix["children"]

        return body

    def _collect_stats(
        self,
        body: list[dict[str, Any]],
    ) -> dict[str, int]:
        stats = {
            "parts": 0,
            "chapters": 0,
            "sections": 0,
            "articles": 0,
            "clauses": 0,
            "points": 0,
            "appendices": 0,
            "headings": 0,
            "paragraphs": 0,
            "list_items": 0,
            "tables": 0,
        }

        def visit(node: dict[str, Any]) -> None:
            node_type = node.get("type")

            if node_type == "part":
                stats["parts"] += 1
            elif node_type == "chapter":
                stats["chapters"] += 1
            elif node_type == "section":
                stats["sections"] += 1
            elif node_type == "appendix":
                stats["appendices"] += 1
            elif node_type == "article":
                stats["articles"] += 1

                clauses = node.get("clauses", [])
                stats["clauses"] += len(clauses)

                stats["points"] += len(
                    node.get("points", [])
                )

                for clause in clauses:
                    stats["points"] += len(
                        clause.get("points", [])
                    )

            elif node_type == "heading":
                stats["headings"] += 1
            elif node_type == "paragraph":
                stats["paragraphs"] += 1
            elif node_type == "list_item":
                stats["list_items"] += 1
            elif node_type == "table":
                stats["tables"] += 1

            for child in node.get("children", []):
                visit(child)

        for root_node in body:
            visit(root_node)

        return stats

    def _classify_structure(
        self,
        stats: dict[str, int],
        body: list[dict[str, Any]],
    ) -> str:
        if stats["articles"] > 0:
            if (
                stats["parts"] > 0
                or stats["chapters"] > 0
                or stats["sections"] > 0
            ):
                if stats["appendices"] > 0:
                    return "mixed_hierarchical"
                return "hierarchical_article_based"

            if stats["appendices"] > 0:
                return "mixed_article_appendix"

            return "article_based"

        if stats["appendices"] > 0:
            return "appendix_or_form"

        if (
            stats["parts"] > 0
            or stats["chapters"] > 0
            or stats["sections"] > 0
        ):
            return "hierarchical_no_articles"

        if (
            stats["headings"] > 0
            or stats["list_items"] > 0
            or stats["tables"] > 0
        ):
            return "semi_structured"

        return "free_form"

    def _append_duplicate_number_warnings(
        self,
        body: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        seen: dict[tuple[tuple[tuple[str, str], ...], str], int] = {}

        def visit(
            node: dict[str, Any],
            path: tuple[tuple[str, str], ...],
        ) -> None:
            node_type = node.get("type")

            if node_type in {
                "part",
                "chapter",
                "section",
                "appendix",
            }:
                key_name = {
                    "part": "part",
                    "chapter": "chapter",
                    "section": "section",
                    "appendix": "appendix",
                }[node_type]

                new_path = path + (
                    (
                        node_type,
                        str(node.get(key_name, "")),
                    ),
                )
            else:
                new_path = path

            if node_type == "article":
                article_number = str(
                    node.get("article", "")
                ).casefold()

                key = (path, article_number)
                seen[key] = seen.get(key, 0) + 1

            for child in node.get("children", []):
                visit(child, new_path)

        for root in body:
            visit(root, ())

        for (path, article_number), count in seen.items():
            if count > 1:
                warnings.append(
                    "duplicate_article_number:"
                    f"article={article_number},"
                    f"count={count},"
                    f"path={dict(path)}"
                )

    # ==================================================================
    # Small helpers
    # ==================================================================

    def _empty_document(
        self,
        doc_id: str | None,
    ) -> dict[str, Any]:
        return {
            "doc_id": None if doc_id is None else str(doc_id),
            "parser_version": self.PARSER_VERSION,
            "structure_type": "empty",
            "preamble": "",
            "body": [],
            "stats": {
                "parts": 0,
                "chapters": 0,
                "sections": 0,
                "articles": 0,
                "clauses": 0,
                "points": 0,
                "appendices": 0,
                "headings": 0,
                "paragraphs": 0,
                "list_items": 0,
                "tables": 0,
            },
            "warnings": [],
        }

    @staticmethod
    def _clean_label(value: str | None) -> str:
        return (value or "").strip()

    @staticmethod
    def _clean_title(value: str | None) -> str:
        return (value or "").strip(" \t\r\n.:-–—")

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return (value or "").strip()

    @staticmethod
    def _append_text(existing: str, extra: str) -> str:
        existing = existing.strip()
        extra = extra.strip()

        if not existing:
            return extra

        if not extra:
            return existing

        return f"{existing}\n{extra}"

    @staticmethod
    def _join_raw_lines(tokens: Iterable[LineToken]) -> str:
        return "\n".join(
            token.raw
            for token in tokens
        ).strip()
