from __future__ import annotations


def provision_key(
    article: str | None,
    clause: str | None = None,
    point: str | None = None,
) -> str:
    normalized_article = _normalize(article)
    if not normalized_article:
        raise ValueError("article is required for a provision status")
    return (
        f"article:{normalized_article}/clause:{_normalize(clause)}"
        f"/point:{_normalize(point)}"
    )


def _normalize(value: str | None) -> str:
    return " ".join((value or "").strip().casefold().split())
