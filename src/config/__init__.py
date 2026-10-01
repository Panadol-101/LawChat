"""Configuration utilities for LawChat."""

from __future__ import annotations

from .feature_flags import (
    FeatureFlag,
    FeatureFlags,
    get_feature_flags,
    load_feature_flags,
)

__all__ = [
    "FeatureFlag",
    "FeatureFlags",
    "get_feature_flags",
    "load_feature_flags",
]