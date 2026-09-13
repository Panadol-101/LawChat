from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

from .models import HydratedLegalChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True, slots=True)
class RerankerSettings:
    enabled: bool = False
    model_name: str = DEFAULT_RERANKER_MODEL
    candidate_limit: int = 12
    batch_size: int = 1
    max_length: int = 1536
    cache_dir: str = "data/.cache/huggingface"
    device: str = "cpu"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0:
            raise ValueError("reranker candidate_limit must be > 0")
        if self.batch_size <= 0:
            raise ValueError("reranker batch_size must be > 0")
        if self.max_length <= 2:
            raise ValueError("reranker max_length must be > 2")

    @classmethod
    def from_env(cls) -> "RerankerSettings":
        return cls(
            enabled=os.getenv("RERANKER_ENABLED", "false").casefold()
            in {"1", "true", "yes", "on"},
            model_name=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            candidate_limit=int(os.getenv("RERANKER_CANDIDATES", "12")),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "1")),
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", "1536")),
            cache_dir=os.getenv(
                "RERANKER_CACHE_PATH", "data/.cache/huggingface"
            ),
            device=os.getenv("RERANKER_DEVICE", "cpu"),
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

        self.model = CrossEncoder(
            self.settings.model_name,
            device=self.settings.device,
            cache_folder=self.settings.cache_dir,
            local_files_only=self.settings.local_files_only,
            max_length=self.settings.max_length,
        )

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
