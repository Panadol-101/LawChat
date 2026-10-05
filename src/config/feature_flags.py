"""Feature flags for safe rollout of audit fixes.

Each flag controls one of the 29 fixes from the audit plan. Flags are
opt-in: defaults are conservative (safer behaviour, smaller scope) and
engineers can flip them on after staging verification.

Flags use environment variables of the form ``LAWCHAT_FLAG_<UPPER>``.
Bool parsing accepts ``1|true|yes|on`` (case-insensitive).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class FeatureFlag(str, Enum):
    """Catalogue of every rollout flag.

    The string value is the canonical environment-variable suffix.
    """

    FAIL_CLOSED_PARTIAL = "fail_closed_partial"
    AMENDMENT_LOOKUP_ENABLED = "amendment_lookup_enabled"
    BM25_VI_TOKENIZER = "bm25_vi_tokenizer"
    FIX_UNALIASED_SQL = "fix_unaliased_sql"
    FIX_COLUMN_ALIAS = "fix_column_alias"
    SEMANTIC_REWRITER_V2 = "semantic_rewriter_v2"
    MULTI_HOP_GRAPH = "multi_hop_graph"
    RERANK_TOP_N = "rerank_top_n"
    CONVERSATION_CONTEXT_ENABLED = "conversation_context_enabled"
    SCORE_NORMALIZATION_ENABLED = "score_normalization_enabled"
    CITES_RELATIONSHIPS_ENABLED = "cites_relationships_enabled"
    PROVISION_UNKNOWN_EXCLUDE = "profile_unknown_exclude"
    STRICT_PARTIAL_REFUSAL = "strict_partial_refusal"
    FAST_AMENDMENT_FALLBACK = "fast_amendment_fallback"
    TRANSITIVE_GRAPH_DEPTH = "transitive_graph_depth"


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env var. Empty/unset => default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class FeatureFlags:
    """Resolved feature flags snapshot.

    Reading from a single snapshot (rather than calling ``_env_flag`` ad-hoc)
    makes tests deterministic and lets us override flags in one place.
    """

    # E1+E2: PARTIALLY_EFFECTIVE documents without resolved provision
    # status must be REFUSED, not warned. Default ON because the
    # alternative produces legally wrong answers.
    fail_closed_partial: bool
    strict_partial_refusal: bool

    # E9: AmendmentEvent lookup in graph expansion.
    amendment_lookup_enabled: bool
    fast_amendment_fallback: bool

    # Bug BM25 thiếu tokenizer tiếng Việt.
    bm25_vi_tokenizer: bool

    # Bug SQL unaliased model class (queries.py:261-267).
    fix_unaliased_sql: bool

    # Bug column alias doc_status vs reason (queries.py:374).
    fix_column_alias: bool

    # W1: real SemanticQueryRewriter implementation.
    semantic_rewriter_v2: bool

    # W6: reranker top-N candidate limit override.
    rerank_top_n: bool

    # W4: always-on conversation context (not just vague markers).
    conversation_context_enabled: bool

    # W12: min-max score normalization before fusion.
    score_normalization_enabled: bool

    # G5: route CITES/IMPLEMENTS/RELATED_TO roles explicitly.
    cites_relationships_enabled: bool

    # E10: treat UNKNOWN provision status as exclusion rather than fallback.
    profile_unknown_exclude: bool

    # E7/E8: multi-hop graph traversal.
    multi_hop_graph: bool
    transitive_graph_depth: int


def _env(name: FeatureFlag, default: bool) -> bool:
    env_name = f"LAWCHAT_FLAG_{name.value.upper()}"
    return _env_flag(env_name, default)


def _envi(name: FeatureFlag, default: int) -> int:
    env_name = f"LAWCHAT_FLAG_{name.value.upper()}"
    return _env_int(env_name, default)


def load_feature_flags() -> FeatureFlags:
    """Resolve current feature flag values from the environment."""
    return FeatureFlags(
        fail_closed_partial=_env(FeatureFlag.FAIL_CLOSED_PARTIAL, False),
        strict_partial_refusal=_env(FeatureFlag.STRICT_PARTIAL_REFUSAL, False),
        amendment_lookup_enabled=_env(FeatureFlag.AMENDMENT_LOOKUP_ENABLED, False),
        fast_amendment_fallback=_env(FeatureFlag.FAST_AMENDMENT_FALLBACK, True),
        bm25_vi_tokenizer=_env(FeatureFlag.BM25_VI_TOKENIZER, False),
        fix_unaliased_sql=_env(FeatureFlag.FIX_UNALIASED_SQL, True),
        fix_column_alias=_env(FeatureFlag.FIX_COLUMN_ALIAS, True),
        semantic_rewriter_v2=_env(FeatureFlag.SEMANTIC_REWRITER_V2, False),
        rerank_top_n=_env(FeatureFlag.RERANK_TOP_N, True),
        conversation_context_enabled=_env(FeatureFlag.CONVERSATION_CONTEXT_ENABLED, False),
        score_normalization_enabled=_env(FeatureFlag.SCORE_NORMALIZATION_ENABLED, False),
        cites_relationships_enabled=_env(FeatureFlag.CITES_RELATIONSHIPS_ENABLED, True),
        profile_unknown_exclude=_env(FeatureFlag.PROVISION_UNKNOWN_EXCLUDE, False),
        multi_hop_graph=_env(FeatureFlag.MULTI_HOP_GRAPH, False),
        transitive_graph_depth=_envi(FeatureFlag.TRANSITIVE_GRAPH_DEPTH, 2),
    )


def get_feature_flags() -> FeatureFlags:
    """Module-level accessor used by callers.

    Kept separate from :func:`load_feature_flags` so tests can monkey-patch
    a constant instance.
    """
    return load_feature_flags()


__all__ = [
    "FeatureFlag",
    "FeatureFlags",
    "load_feature_flags",
    "get_feature_flags",
]