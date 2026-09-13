from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+")
_OVERLAP_BOUNDARY_RE = re.compile(r"(?<=[.!?;:,])\s+")


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def count_many_tokens(self, texts: Sequence[str]) -> list[int]: ...

    def split_text(self, text: str, *, max_tokens: int) -> list[str]: ...

    def tail_text(self, text: str, *, max_tokens: int) -> str: ...


_MODEL_TOKEN_COUNTER: TokenCounter | None = None


def configure_token_counter(counter: TokenCounter | None) -> None:
    """Use the embedding model tokenizer for all chunk budgets."""
    global _MODEL_TOKEN_COUNTER
    _MODEL_TOKEN_COUNTER = counter

STRUCTURAL_TYPES = {
    "part",
    "chapter",
    "section",
    "article",
    "appendix",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def nullable_text(value: Any) -> str | None:
    text = clean_text(value)
    return text or None


def metadata_value(
    metadata: Mapping[str, Any],
    *keys: str,
) -> str | None:
    for key in keys:
        value = metadata.get(key)

        if value is None:
            continue

        if isinstance(value, (dt.date, dt.datetime)):
            return value.isoformat()

        if isinstance(value, (list, tuple, set)):
            text = "; ".join(
                clean_text(item)
                for item in value
                if clean_text(item)
            )
            if text:
                return text
            continue

        text = clean_text(value)
        if text:
            return text

    return None


def approximate_token_count(text: str) -> int:
    """
    Cheap deterministic approximation for preprocessing QA.

    This is NOT a replacement for the tokenizer of the embedding model.
    Recompute exact counts later when the embedding model is chosen.
    """
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def exact_token_count(text: str) -> int:
    if not text:
        return 0
    if _MODEL_TOKEN_COUNTER is None:
        return approximate_token_count(text)
    return _MODEL_TOKEN_COUNTER.count_tokens(text)


def exact_token_counts(texts: Sequence[str]) -> list[int]:
    if _MODEL_TOKEN_COUNTER is None:
        return [approximate_token_count(text) for text in texts]
    return _MODEL_TOKEN_COUNTER.count_many_tokens(texts)


def split_exact_tokens(text: str, *, max_tokens: int) -> list[str]:
    if _MODEL_TOKEN_COUNTER is None:
        return hard_word_fallback(text, max_tokens=max_tokens)
    return _MODEL_TOKEN_COUNTER.split_text(text, max_tokens=max_tokens)


def split_sentence_aware(
    text: str,
    *,
    max_tokens: int,
) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    if exact_token_count(text) <= max_tokens:
        return [text]

    sentences = [
        item.strip()
        for item in _SENTENCE_BOUNDARY_RE.split(text)
        if item.strip()
    ]

    # Abnormally long sentence / table cell / badly normalized text.
    if len(sentences) <= 1:
        return split_exact_tokens(text, max_tokens=max_tokens)

    output: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence_tokens = exact_token_count(sentence)

        if sentence_tokens > max_tokens:
            if buffer:
                output.append(" ".join(buffer).strip())
                buffer = []
                buffer_tokens = 0

            output.extend(
                split_exact_tokens(
                    sentence,
                    max_tokens=max_tokens,
                )
            )
            continue

        if (
            buffer
            and buffer_tokens + sentence_tokens > max_tokens
        ):
            output.append(" ".join(buffer).strip())
            buffer = []
            buffer_tokens = 0

        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        output.append(" ".join(buffer).strip())

    return output


def hard_word_fallback(
    text: str,
    *,
    max_tokens: int,
) -> list[str]:
    """
    Final fallback only. It is intentionally isolated so chunks created by
    this path can be audited by their fragment chunk_type.
    """
    words = text.split()

    if not words:
        return []

    output: list[str] = []
    buffer: list[str] = []

    for word in words:
        # text.split() can return one enormous punctuation/URL-like token.
        # Split that token using the same units as approximate_token_count so
        # the fallback itself can never return an oversized fragment.
        if approximate_token_count(word) > max_tokens:
            if buffer:
                output.append(" ".join(buffer).strip())
                buffer = []

            token_units = _TOKEN_RE.findall(word)
            output.extend(
                "".join(token_units[index:index + max_tokens])
                for index in range(0, len(token_units), max_tokens)
            )
            continue

        candidate = " ".join([*buffer, word])

        if (
            buffer
            and approximate_token_count(candidate) > max_tokens
        ):
            output.append(" ".join(buffer).strip())
            buffer = [word]
        else:
            buffer.append(word)

    if buffer:
        output.append(" ".join(buffer).strip())

    return output


def semantic_overlap_tail(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
) -> str:
    """Return a contiguous tail made only from complete lines/sentences."""
    if not text or target_tokens <= 0 or max_tokens <= 0:
        return ""

    units: list[str] = []
    for line in text.splitlines():
        line = clean_text(line)
        if not line:
            continue
        sentences = [
            item.strip()
            for item in _OVERLAP_BOUNDARY_RE.split(line)
            if item.strip()
        ]
        units.extend(sentences or [line])

    selected: list[str] = []
    selected_tokens = 0
    for unit in reversed(units):
        unit_tokens = exact_token_count(unit)
        if unit_tokens > max_tokens:
            break
        if selected and selected_tokens + unit_tokens > max_tokens:
            break
        selected.append(unit)
        selected_tokens += unit_tokens
        if selected_tokens >= target_tokens:
            break

    return "\n".join(reversed(selected)).strip()


def bounded_token_overlap_tail(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
) -> str:
    """Fallback tail for overlap only; canonical chunk text is untouched."""
    words = text.split()
    selected: list[str] = []

    for word in reversed(words):
        candidate = " ".join([word, *selected])
        candidate_tokens = exact_token_count(candidate)
        if candidate_tokens > max_tokens:
            break
        selected.insert(0, word)
        if candidate_tokens >= target_tokens:
            break

    result = " ".join(selected).strip()
    if exact_token_count(result) >= target_tokens:
        return result

    # Exact fallback using tokenizer spans across the whole source tail.
    # This affects overlap context only; canonical text is never modified.
    matches = list(_TOKEN_RE.finditer(text))
    take = min(target_tokens, max_tokens, len(matches))
    if take >= target_tokens:
        return text[matches[-take].start():].strip()
    return ""


def pack_units(
    units: Sequence[str],
    *,
    target_tokens: int,
    max_tokens: int,
    min_tail_tokens: int,
) -> list[str]:
    """
    Merge consecutive semantic units without crossing max_tokens.

    Each input unit is already a meaningful boundary: paragraph, list item,
    heading group, table row group, etc. Oversized units are split only after
    preserving this boundary as long as possible.
    """
    normalized = [
        clean_text(unit)
        for unit in units
        if clean_text(unit)
    ]

    if not normalized:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append("\n".join(buffer).strip())
            buffer = []
            buffer_tokens = 0

    for unit in normalized:
        unit_tokens = exact_token_count(unit)

        if unit_tokens > max_tokens:
            flush()
            chunks.extend(
                split_sentence_aware(
                    unit,
                    max_tokens=max_tokens,
                )
            )
            continue

        if (
            buffer
            and buffer_tokens + unit_tokens > max_tokens
        ):
            flush()

        buffer.append(unit)
        buffer_tokens += unit_tokens

        if buffer_tokens >= target_tokens:
            flush()

    flush()

    # Avoid a tiny tail when it can safely be merged into the preceding block.
    if (
        len(chunks) >= 2
        and exact_token_count(chunks[-1]) < min_tail_tokens
    ):
        merged = f"{chunks[-2]}\n{chunks[-1]}".strip()

        if exact_token_count(merged) <= max_tokens:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def iter_nodes(
    nodes: Iterable[Mapping[str, Any]],
    *,
    target_type: str | None = None,
    excluded_ancestor_types: set[str] | None = None,
    ancestors: tuple[str, ...] = (),
) -> Iterable[Mapping[str, Any]]:
    excluded = excluded_ancestor_types or set()

    for node in nodes:
        node_type = clean_text(node.get("type"))

        if not (set(ancestors) & excluded):
            if target_type is None or node_type == target_type:
                yield node

        yield from iter_nodes(
            node.get("children") or [],
            target_type=target_type,
            excluded_ancestor_types=excluded,
            ancestors=(*ancestors, node_type),
        )


def has_node_type(
    nodes: Iterable[Mapping[str, Any]],
    node_type: str,
    *,
    exclude_under: set[str] | None = None,
) -> bool:
    return any(
        True
        for _ in iter_nodes(
            nodes,
            target_type=node_type,
            excluded_ancestor_types=exclude_under,
        )
    )


def render_point(point: Mapping[str, Any]) -> str:
    number = clean_text(point.get("point"))
    text = clean_text(point.get("text"))

    if number:
        return f"{number}) {text}".strip()

    return text


def render_clause(
    clause: Mapping[str, Any],
    *,
    include_points: bool = True,
) -> str:
    number = clean_text(clause.get("clause"))
    text = clean_text(clause.get("text"))

    head = (
        f"{number}. {text}".strip()
        if number
        else text
    )

    parts = [head] if head else []

    if include_points:
        for point in clause.get("points") or []:
            rendered = render_point(point)
            if rendered:
                parts.append(rendered)

    return "\n".join(parts).strip()


def render_article(
    article: Mapping[str, Any],
) -> str:
    number = clean_text(article.get("article"))
    title = clean_text(article.get("title"))

    if number and title:
        heading = f"Điều {number}. {title}"
    elif number:
        heading = f"Điều {number}"
    else:
        heading = title

    parts = [heading] if heading else []

    article_text = clean_text(article.get("text"))
    if article_text:
        parts.append(article_text)

    for clause in article.get("clauses") or []:
        rendered = render_clause(clause)
        if rendered:
            parts.append(rendered)

    for point in article.get("points") or []:
        rendered = render_point(point)
        if rendered:
            parts.append(rendered)

    return "\n".join(parts).strip()


def structural_label(
    node: Mapping[str, Any],
) -> tuple[str, str]:
    node_type = clean_text(node.get("type"))

    key_map = {
        "part": ("Phần", "part"),
        "chapter": ("Chương", "chapter"),
        "section": ("Mục", "section"),
        "appendix": ("Phụ lục", "appendix"),
    }

    human, key = key_map.get(
        node_type,
        ("", node_type),
    )

    value = clean_text(node.get(key))

    # Current LegalStructureParser stores appendix labels such as
    # "PHỤ LỤC I", not only "I". Avoid producing "Phụ lục PHỤ LỤC I".
    if node_type == "appendix":
        match = re.match(
            r"^PHỤ\s+LỤC(?:\s+(.*))?$",
            value,
            flags=re.IGNORECASE,
        )

        if match:
            suffix = clean_text(
                match.group(1)
            )

            if suffix:
                value = suffix
            else:
                # The label itself already fully names the appendix.
                human = ""
                value = "PHỤ LỤC"

    return human, value


def render_structural_heading(
    node: Mapping[str, Any],
) -> str:
    human, label = structural_label(node)
    title = clean_text(node.get("title"))

    return " ".join(
        item
        for item in [human, label, title]
        if item
    ).strip()


def render_generic_node(
    node: Mapping[str, Any],
    *,
    recursive: bool = True,
) -> str:
    node_type = clean_text(node.get("type"))

    if node_type == "article":
        return render_article(node)

    if node_type in {
        "part",
        "chapter",
        "section",
        "appendix",
    }:
        parts = [render_structural_heading(node)]

        if recursive:
            for child in node.get("children") or []:
                text = render_generic_node(child)
                if text:
                    parts.append(text)

        return "\n".join(
            item
            for item in parts
            if item
        ).strip()

    if node_type in {"paragraph", "heading"}:
        return clean_text(node.get("text"))

    if node_type == "list_item":
        marker = clean_text(node.get("marker"))
        text = clean_text(node.get("text"))
        return f"{marker} {text}".strip()

    if node_type == "table":
        return render_table(node)

    return clean_text(
        node.get("text")
        or node.get("raw_heading")
    )


def render_document_text(
    parsed_document: Mapping[str, Any],
    *,
    exclude_top_level_types: set[str] | None = None,
) -> str:
    exclude = exclude_top_level_types or set()
    parts: list[str] = []

    preamble = clean_text(parsed_document.get("preamble"))
    if preamble:
        parts.append(preamble)

    for node in parsed_document.get("body") or []:
        if clean_text(node.get("type")) in exclude:
            continue

        rendered = render_generic_node(node)
        if rendered:
            parts.append(rendered)

    return "\n".join(parts).strip()


def path_with_node(
    path: Mapping[str, str],
    node: Mapping[str, Any],
) -> dict[str, str]:
    output = dict(path)
    node_type = clean_text(node.get("type"))

    if node_type == "part":
        value = clean_text(node.get("part"))
        if value:
            output["part"] = value

    elif node_type == "chapter":
        value = clean_text(node.get("chapter"))
        if value:
            output["chapter"] = value

    elif node_type == "section":
        value = clean_text(node.get("section"))
        if value:
            output["section"] = value

    elif node_type == "appendix":
        value = clean_text(node.get("appendix"))

        match = re.match(
            r"^PHỤ\s+LỤC(?:\s+(.*))?$",
            value,
            flags=re.IGNORECASE,
        )

        if match:
            value = clean_text(
                match.group(1)
            ) or "PHỤ LỤC"

        if value:
            output["appendix"] = value

    return output


def render_table(node: Mapping[str, Any]) -> str:
    headers = [
        clean_text(item)
        for item in node.get("headers") or []
    ]

    rows = node.get("rows") or []

    lines: list[str] = []

    if headers:
        lines.append(" | ".join(headers))

    for row in rows:
        if isinstance(row, Mapping):
            values = [
                clean_text(value)
                for value in row.values()
            ]
        else:
            values = [
                clean_text(value)
                for value in row
            ]

        lines.append(" | ".join(values))

    return "\n".join(
        line
        for line in lines
        if line
    ).strip()


def table_row_units(
    node: Mapping[str, Any],
) -> list[str]:
    units: list[str] = []

    for row in node.get("rows") or []:
        if isinstance(row, Mapping):
            values = [
                clean_text(value)
                for value in row.values()
            ]
        else:
            values = [
                clean_text(value)
                for value in row
            ]

        row_line = " | ".join(values)

        units.append(row_line)

    return [
        unit
        for unit in units
        if unit
    ]


def pack_node_blocks(
    nodes: Sequence[Mapping[str, Any]],
    *,
    target_tokens: int,
    max_tokens: int,
    min_tail_tokens: int,
) -> list[str]:
    """Pack regular nodes and keep every table row attached to its header."""
    blocks: list[str] = []
    pending: list[str] = []

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            blocks.extend(
                pack_units(
                    pending,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    min_tail_tokens=min_tail_tokens,
                )
            )
            pending = []

    for node in nodes:
        if clean_text(node.get("type")) != "table":
            rendered = render_generic_node(node, recursive=False)
            if rendered:
                pending.append(rendered)
            continue

        flush_pending()
        header = " | ".join(
            clean_text(item)
            for item in node.get("headers") or []
            if clean_text(item)
        )
        rows = table_row_units(node)

        if not rows:
            if header:
                blocks.extend(
                    split_sentence_aware(header, max_tokens=max_tokens)
                )
            continue

        header_tokens = exact_token_count(header)
        if header and header_tokens >= max_tokens // 2:
            header_context = "Tiêu đề và cấu trúc cột của bảng"
            header_budget = max(
                1,
                max_tokens - exact_token_count(header_context),
            )
            blocks.extend(
                "\n".join(
                    item for item in [header_context, fragment] if item
                )
                for fragment in split_sentence_aware(
                    header,
                    max_tokens=header_budget,
                )
            )
            header = "Bảng dữ liệu"
            header_tokens = exact_token_count(header)

        row_budget = max(1, max_tokens - header_tokens)
        row_target = max(1, min(target_tokens, row_budget))
        table_blocks = pack_units(
            rows,
            target_tokens=row_target,
            max_tokens=row_budget,
            min_tail_tokens=0,
        )
        blocks.extend(
            "\n".join(item for item in [header, block] if item)
            for block in table_blocks
        )

    flush_pending()
    return blocks
