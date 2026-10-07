"""User counts for the admin Users page.

REJECTED accounts are kept for audit/history but are not system users:
Users = ACTIVE + PENDING.
"""
from __future__ import annotations

from typing import Any


def group_users(users: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split USER accounts by status; ADMIN accounts are left out."""
    groups: dict[str, list[dict[str, Any]]] = {
        "ACTIVE": [],
        "PENDING": [],
        "REJECTED": [],
    }
    for item in users:
        if item.get("role") != "USER":
            continue
        groups.setdefault(item.get("status") or "ACTIVE", []).append(item)
    return groups


def user_metrics(users: list[dict[str, Any]]) -> dict[str, int]:
    groups = group_users(users)
    active = len(groups["ACTIVE"])
    pending = len(groups["PENDING"])
    return {
        "users": active + pending,
        "active": active,
        "pending": pending,
        "rejected": len(groups["REJECTED"]),
    }
