from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .models import HydratedLegalChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# Colloquial phrases mapped to the statutory term used in the corpus, plus an
# optional context pattern that must also match. Expansions are appended to
# the query, never substituted, so the user's own wording is still searched.
LEGAL_SYNONYMS: tuple[tuple[str, str, str | None], ...] = (
    (r"\bngh[ỉi]\s+vi[ệe]c\b(?!\s+(?:riêng|không\s+lương))", "chấm dứt hợp đồng lao động", None),
    (r"\bngh[ỉi]\s+ph[ée]p\b", "nghỉ hằng năm", None),
    (r"\bth[ôo]i\s+vi[ệe]c\b", "trợ cấp thôi việc", None),
    (
        r"\b[đd]ơn\s+ph[ưu]ơng\b",
        "đơn phương chấm dứt hợp đồng lao động",
        r"\b(?:hợp\s+đồng\s+lao\s+động|người\s+lao\s+động|người\s+sử\s+dụng\s+lao\s+động|"
        r"công\s+ty|doanh\s+nghiệp|nghỉ\s+việc|làm\s+việc)\b",
    ),
    (r"\bsa\s+th[ảa]i\b", "kỷ luật lao động sa thải", None),
    (r"\bkh[ởo]i\s+ki[ệe]n\b", "khởi kiện vụ án", None),
)


class SemanticQueryRewriter:
    """Lightweight, deterministic legal-jargon expansion.

    Appends canonical statutory terms for known colloquial phrases. Audit fix
    W1 (Phase 3) adds :meth:`rewrite_variants` which returns a list of
    paraphrased queries the caller can route through hybrid retrieval and
    merge via RRF.
    """

    def rewrite(self, query: str) -> str:
        folded = query.casefold()
        expansions: list[str] = []
        for pattern, expansion, context in LEGAL_SYNONYMS:
            if not re.search(pattern, query, flags=re.IGNORECASE):
                continue
            if context is not None and not re.search(context, query, flags=re.IGNORECASE):
                continue
            if expansion.casefold() in folded or expansion in expansions:
                continue
            expansions.append(expansion)
        expansions = [
            item for item in expansions
            if not any(item != other and item in other for other in expansions)
        ]
        if not expansions:
            return query
        return f"{query} ({'; '.join(expansions)})"

    def rewrite_variants(
        self,
        query: str,
        *,
        llm_client: Any | None = None,
        n_variants: int = 2,
    ) -> tuple[str, ...]:
        """Return ``[original, *paraphrases]``.

        Paraphrases are produced by the LLM when one is provided and
        :data:`FeatureFlags.semantic_rewriter_v2` is on. Falls back to the
        single deterministic rewrite otherwise.
        """
        flags = _flags_or_default()
        rewritten = self.rewrite(query)
        if not flags.semantic_rewriter_v2 or llm_client is None or n_variants <= 0:
            return (rewritten,)
        try:
            variants = llm_client.generate_variants(
                rewritten, n=n_variants, locale="vi"
            )
        except Exception:
            # Never fail the request because of paraphrase generation.
            import logging

            logging.getLogger(__name__).warning(
                "semantic rewriter v2 LLM failed; returning deterministic rewrite"
            )
            return (rewritten,)
        # Always include the deterministic rewrite first so downstream RRF
        # is anchored to the canonical phrasing.
        return tuple(dict.fromkeys((rewritten, *variants)))


def _flags_or_default():
    try:
        try:
            from config import get_feature_flags
        except (ImportError, ValueError):
            from ..config import get_feature_flags  # type: ignore[import-not-found]

        return get_feature_flags()
    except Exception:
        from dataclasses import replace

        try:
            from config import load_feature_flags
        except (ImportError, ValueError):
            from ..config import load_feature_flags

        return replace(load_feature_flags(), semantic_rewriter_v2=False)


_rewriter = SemanticQueryRewriter()


@dataclass(frozen=True, slots=True)
class RerankerSettings:
    enabled: bool = False
    model_name: str = DEFAULT_RERANKER_MODEL
    # Audit fix W6 (Phase 3): raised the legacy 12-candidate rerank window
    # to 20. With the default retrieval pool (50 candidates) the previous
    # limit dropped ~38 candidates before they ever reached the cross-
    # encoder, leaving them invisible to the final ranking step.
    candidate_limit: int = 20
    batch_size: int = 32
    max_length: int = 1536
    cache_dir: str = "data/.cache/huggingface"
    device: str = "cpu"
    local_files_only: bool = False
    use_fp16: bool = False
    # Upper bound for one cross-encoder call; on expiry the RRF order is used.
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("reranker candidate_limit must be > 0")
        if self.batch_size <= 0:
            raise ValueError("reranker batch_size must be > 0")
        if self.max_length <= 2:
            raise ValueError("reranker max_length must be > 2")
        if self.timeout_seconds <= 0:
            raise ValueError("reranker timeout_seconds must be > 0")

    @classmethod
    def from_env(cls) -> "RerankerSettings":
        device = os.getenv("RERANKER_DEVICE", "cpu")
        use_fp16 = (
            os.getenv("RERANKER_USE_FP16", "true" if device != "cpu" else "false")
            .casefold()
            in {"1", "true", "yes", "on"}
        )
        return cls(
            enabled=os.getenv("RERANKER_ENABLED", "false").casefold()
            in {"1", "true", "yes", "on"},
            model_name=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            candidate_limit=int(os.getenv("RERANKER_CANDIDATES", "20")),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "32")),
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", "1536")),
            cache_dir=os.getenv(
                "RERANKER_CACHE_PATH", "data/.cache/huggingface"
            ),
            device=device,
            use_fp16=use_fp16,
            timeout_seconds=float(os.getenv("RERANKER_TIMEOUT_SECONDS", "60")),
            local_files_only=(
                os.getenv("RERANKER_LOCAL_FILES_ONLY", "false").casefold()
                in {"1", "true", "yes", "on"}
                or os.getenv("HF_HUB_OFFLINE", "false").casefold()
                in {"1", "true", "yes", "on"}
            ),
        )


@dataclass(frozen=True, slots=True)
class RerankResult:
    point_id: str
    score: float
    rank: int


class LegalReranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: Sequence[HydratedLegalChunk],
    ) -> list[RerankResult]: ...


class NoOpReranker:
    def rerank(
        self,
        query: str,
        chunks: Sequence[HydratedLegalChunk],
    ) -> list[RerankResult]:
        return [
            RerankResult(chunk.point_id, float(-rank), rank)
            for rank, chunk in enumerate(chunks, start=1)
        ]


class SentenceTransformerCrossEncoderReranker:
    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self.settings = settings or RerankerSettings.from_env()
        from sentence_transformers import CrossEncoder

        kwargs = {
            "device": self.settings.device,
            "cache_folder": self.settings.cache_dir,
            "local_files_only": self.settings.local_files_only,
            "max_length": self.settings.max_length,
        }
        self.model = CrossEncoder(self.settings.model_name, **kwargs)
        if self.settings.use_fp16 and self.settings.device != "cpu":
            try:
                import torch
                self.model.model.half()
            except Exception:
                # fp16 is an optimization; stay on fp32 if not supported.
                pass

    def rerank(
        self,
        query: str,
        chunks: Sequence[HydratedLegalChunk],
    ) -> list[RerankResult]:
        if not chunks:
            return []
        texts = [chunk.retrieval_text or chunk.text for chunk in chunks]
        scores = list(
            self.model.predict(
                [(query, text) for text in texts],
                batch_size=self.settings.batch_size,
                show_progress_bar=False,
            )
        )
        if len(scores) != len(chunks):
            raise RuntimeError("reranker returned a different number of scores")
        ordered = sorted(
            zip(chunks, scores),
            key=lambda item: (-float(item[1]), item[0].point_id),
        )
        return [
            RerankResult(chunk.point_id, float(score), rank)
            for rank, (chunk, score) in enumerate(ordered, start=1)
        ]
