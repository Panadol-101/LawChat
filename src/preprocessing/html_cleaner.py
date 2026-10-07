from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Iterable

from lxml import etree
from lxml import html as lxml_html


CLEANER_VERSION = "2.2.0"


# ---------------------------------------------------------------------------
# Elements that never contain useful legal-document content.
# ---------------------------------------------------------------------------

REMOVE_TAGS = {
    "head",
    "title",
    "meta",
    "link",
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
}


# HTML block elements where boundaries should normally become newlines.
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "legend",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}


# Characters that frequently appear as visual bullets.
BULLET_CHARS = {
    "•",
    "●",
    "▪",
    "▫",
    "◦",
    "‣",
    "⁃",
    "∙",
}


_ZERO_WIDTH_RE = re.compile(
    r"[\u200B\u200C\u200D\u2060\uFEFF]"
)

_HORIZONTAL_SPACE_RE = re.compile(
    r"[ \t\f\v]+"
)

_SPACE_BEFORE_PUNCT_RE = re.compile(
    r"\s+([,.;:!?])"
)

_MULTI_BLANK_RE = re.compile(
    r"\n[ \t]*\n(?:[ \t]*\n)+"
)

_LEADING_NUMBER_DOT_RE = re.compile(
    r"^\s*(\d+)\s*\.\s*"
)

_LEADING_LETTER_PAREN_RE = re.compile(
    r"^\s*([a-zđ])\s*\)\s*",
    flags=re.IGNORECASE,
)

_LEADING_NUMBER_PAREN_RE = re.compile(
    r"^\s*(\d+)\s*\)\s*"
)

_LEADING_BULLET_RE = re.compile(
    r"^\s*([•●▪▫◦‣⁃∙])\s*"
)


# Conservative garbage selectors.
#
# Do not add selectors such as ".header" or ".footer" blindly:
# some legal documents may use these classes for actual document content.
GARBAGE_XPATHS = (
    "//head",
    "//title",
    "//script",
    "//style",
    "//noscript",
    "//iframe",
    "//svg",
    "//canvas",
    "//object",
    "//embed",
)


TABLE_START_MARKER = "[[LAWCHAT_TABLE_START]]"
TABLE_HEADERS_MARKER = "[[LAWCHAT_TABLE_HEADERS]]"
TABLE_ROW_MARKER = "[[LAWCHAT_TABLE_ROW]]"
TABLE_END_MARKER = "[[LAWCHAT_TABLE_END]]"

_FILLER_RUN_RE = re.compile(r"(?:[._…]\s*){12,}")
FILLER_PLACEHOLDER = "[THÔNG TIN ĐỂ TRỐNG]"


def normalize_unicode(text: str) -> str:
    """
    Normalize Vietnamese Unicode while preserving characters.

    NFC is important because Vietnamese characters can otherwise have
    multiple binary representations.
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)

    # Normalize common non-breaking spaces.
    text = (
        text.replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2007", " ")
    )

    return text


def remove_unwanted_nodes(root: etree._Element) -> None:
    """
    Remove only nodes that are extremely unlikely to represent legal text.

    Cleaning is intentionally conservative.
    """
    for xpath in GARBAGE_XPATHS:
        for node in root.xpath(xpath):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def _append_text(parts: list[str], value: str | None) -> None:
    if not value:
        return

    parts.append(value)


def _node_to_text(
    node: etree._Element,
    parts: list[str],
) -> None:
    """
    Recursively convert HTML into text while retaining block boundaries.

    Inline formatting such as:
        <strong>, <b>, <em>, <span>
    does not create arbitrary line breaks.

    Block elements such as:
        <p>, <div>, <tr>, headings
    produce boundaries.
    """

    tag = node.tag.lower() if isinstance(node.tag, str) else ""

    if tag in REMOVE_TAGS:
        return

    if tag == "table":
        layout_cell = _layout_cell(node)
        if layout_cell is None:
            _append_text(parts, _table_to_text(node))
        else:
            parts.append("\n")
            _node_to_text(layout_cell, parts)
            parts.append("\n")
        return

    # Explicit line break.
    if tag == "br":
        parts.append("\n")

    # Add node text.
    _append_text(parts, node.text)

    for child in node:
        child_tag = (
            child.tag.lower()
            if isinstance(child.tag, str)
            else ""
        )

        _node_to_text(child, parts)

        if child_tag in BLOCK_TAGS:
            parts.append("\n")

        _append_text(parts, child.tail)



_LAYOUT_TABLE_MIN_BLOCKS = 10


def _layout_cell(table: etree._Element) -> etree._Element | None:
    """The only cell of a table used as a page wrapper, else None.

    Some sources put a whole law inside one <td>; serializing it as a table
    row collapses every paragraph onto one line and hides "Điều N." headings.
    """
    cells = [
        cell
        for cell in table.iter("td", "th")
        if next(
            (a for a in cell.iterancestors() if isinstance(a.tag, str) and a.tag.lower() == "table"),
            None,
        ) is table
    ]
    if len(cells) != 1:
        return None
    blocks = sum(
        1
        for element in cells[0].iter("p", "div")
        if element is not cells[0]
    )
    return cells[0] if blocks >= _LAYOUT_TABLE_MIN_BLOCKS else None


def _table_to_text(table: etree._Element) -> str:
    """Serialize an HTML table without losing row/cell boundaries."""
    headers: list[str] = []
    rows: list[list[str]] = []

    def belongs_to_current_table(element: etree._Element) -> bool:
        return next(
            (
                ancestor
                for ancestor in element.iterancestors()
                if isinstance(ancestor.tag, str)
                and ancestor.tag.lower() == "table"
            ),
            None,
        ) is table

    def cell_text_without_nested_tables(cell: etree._Element) -> str:
        parts: list[str] = []

        def visit(node: etree._Element) -> None:
            if node is not cell and isinstance(node.tag, str) and node.tag.lower() == "table":
                return
            if node.text:
                parts.append(node.text)
            for child in node:
                visit(child)
                if child.tail:
                    parts.append(child.tail)

        visit(cell)
        return re.sub(
            r"\s+",
            " ",
            normalize_unicode("".join(parts)),
        ).strip()

    for tr in table.iter("tr"):
        if not belongs_to_current_table(tr):
            continue
        cells = [
            child
            for child in tr
            if isinstance(child.tag, str)
            and child.tag.lower() in {"th", "td"}
        ]
        if not cells:
            continue

        values = [
            cell_text_without_nested_tables(cell)
            for cell in cells
        ]

        if not headers and any(cell.tag.lower() == "th" for cell in cells):
            headers = values
        else:
            rows.append(values)

    lines: list[str] = []
    if headers or any(any(value for value in row) for row in rows):
        lines.append(TABLE_START_MARKER)
        if headers:
            lines.append(
                f"{TABLE_HEADERS_MARKER} "
                + json.dumps(headers, ensure_ascii=False)
            )
        lines.extend(
            f"{TABLE_ROW_MARKER} "
            + json.dumps(row, ensure_ascii=False)
            for row in rows
            if any(row)
        )
        lines.append(TABLE_END_MARKER)

    nested_tables = [
        nested
        for nested in table.iter("table")
        if nested is not table and belongs_to_current_table(nested)
    ]
    lines.extend(
        serialized
        for nested in nested_tables
        if (serialized := _table_to_text(nested))
    )
    return "\n".join(lines)


def html_to_raw_text(content_html: str | None) -> str:
    """
    Convert HTML to plain text while preserving legal-document structure.
    """
    if content_html is None:
        return ""

    content_html = content_html.strip()

    if not content_html:
        return ""

    try:
        root = lxml_html.fromstring(content_html)
    except (etree.ParserError, ValueError):
        # Broken HTML is common in crawled datasets.
        # Try wrapping it in a container before giving up.
        try:
            root = lxml_html.fromstring(
                f"<div>{content_html}</div>"
            )
        except Exception:
            # Last fallback: strip tags conservatively.
            return re.sub(r"<[^>]+>", " ", content_html)

    remove_unwanted_nodes(root)

    # A full HTML document may contain crawler/page metadata in <head>.
    # Only the body is legal-document content. Fragments without <body>
    # continue to use their parsed root.
    bodies = root.xpath("//body")
    content_root = bodies[0] if bodies else root

    parts: list[str] = []
    _node_to_text(content_root, parts)

    return "".join(parts)


def normalize_legal_marker(line: str) -> str:
    """
    Normalize only obvious leading legal/list markers.

    Examples:
        "1 . Nội dung" -> "1. Nội dung"
        "a ) Nội dung" -> "a) Nội dung"
        "• Nội dung"   -> "- Nội dung"

    This function deliberately does NOT attempt to identify whether the
    line is actually a Khoản or Điểm.
    """

    line = line.strip()

    if not line:
        return ""

    match = _LEADING_NUMBER_DOT_RE.match(line)
    if match:
        marker = match.group(1)
        rest = line[match.end():].strip()

        if rest:
            return f"{marker}. {rest}"

        return f"{marker}."

    match = _LEADING_LETTER_PAREN_RE.match(line)
    if match:
        marker = match.group(1)
        rest = line[match.end():].strip()

        if rest:
            return f"{marker}) {rest}"

        return f"{marker})"

    match = _LEADING_NUMBER_PAREN_RE.match(line)
    if match:
        marker = match.group(1)
        rest = line[match.end():].strip()

        if rest:
            return f"{marker}) {rest}"

        return f"{marker})"

    match = _LEADING_BULLET_RE.match(line)
    if match:
        rest = line[match.end():].strip()

        if rest:
            return f"- {rest}"

        return "-"

    return line


def normalize_line(line: str) -> str:
    """
    Normalize whitespace within a line without destroying legal markers.
    """
    line = normalize_unicode(line)

    line = _HORIZONTAL_SPACE_RE.sub(" ", line)

    # Long dotted/underscored runs are blank fields in legal forms, not useful
    # lexical content. Keep their meaning with one bounded placeholder.
    line = _FILLER_RUN_RE.sub(FILLER_PLACEHOLDER, line)

    # Remove whitespace before punctuation:
    # "Điều 5 ." -> "Điều 5."
    line = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)

    line = line.strip()

    if not line:
        return ""

    line = normalize_legal_marker(line)

    return line


def normalize_text(text: str) -> str:
    """
    Normalize the full legal document.
    """

    text = normalize_unicode(text)

    # Normalize line-ending variants.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    normalized_lines: list[str] = []

    for raw_line in text.split("\n"):
        line = normalize_line(raw_line)

        if line:
            normalized_lines.append(line)
        else:
            # Preserve paragraph separation, but never accumulate many
            # blank lines.
            if (
                normalized_lines
                and normalized_lines[-1] != ""
            ):
                normalized_lines.append("")

    text = "\n".join(normalized_lines)

    # Collapse 3+ newlines into exactly two.
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


def clean_html(content_html: str | None) -> str:
    """
    Main public API.

    HTML -> structure-preserving clean text.
    """
    if not content_html:
        return ""

    raw_text = html_to_raw_text(content_html)
    return normalize_text(raw_text)
