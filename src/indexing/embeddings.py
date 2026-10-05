from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_EMBEDDING_DIMENSION = 1024


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """BGE-M3 dense embeddings through SentenceTransformers/PyTorch.

    The same class is used for batch document indexing on CUDA/ROCm and for
    query-time CPU inference after a Qdrant snapshot is restored locally.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    revision: str | None = DEFAULT_EMBEDDING_REVISION
    cache_dir: str = "data/.cache/huggingface"
    batch_size: int = 16
    device: str = "auto"
    max_length: int = 1200
    expected_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    local_files_only: bool = False
    cache_size: int = 1024
    _model: Any | None = field(default=None, init=False, repr=False)
    _resolved_device: str | None = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _cache_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_length <= 2:
            raise ValueError("max_length must be > 2")
        if self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be > 0")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        self.cache_dir = os.path.abspath(self.cache_dir)

    @classmethod
    def from_env(cls, **overrides: Any) -> "SentenceTransformerEmbedder":
        model_name = str(
            overrides.pop(
                "model_name",
                os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            )
        )
        default_revision = (
            DEFAULT_EMBEDDING_REVISION
            if model_name == DEFAULT_EMBEDDING_MODEL
            else None
        )
        values: dict[str, Any] = {
            "model_name": model_name,
            "revision": os.getenv("EMBEDDING_MODEL_REVISION", default_revision),
            "cache_dir": os.getenv(
                "EMBEDDING_CACHE_PATH", "data/.cache/huggingface"
            ),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            "device": os.getenv("EMBEDDING_DEVICE", "auto"),
            "max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "1200")),
            "expected_dimension": int(
                os.getenv("EMBEDDING_DIMENSION", "1024")
            ),
            "local_files_only": _env_flag("EMBEDDING_LOCAL_FILES_ONLY")
            or _env_flag("HF_HUB_OFFLINE"),
        }
        values.update(overrides)
        return cls(**values)

    @property
    def dimension(self) -> int:
        return self.expected_dimension

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()
        return self._resolved_device

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=True)

    def _encode(self, texts: Sequence[str], *, query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        cached_indices: dict[int, np.ndarray] = {}
        missing: list[str] = []
        missing_indices: list[int] = []
        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            hit = self._cache.get(key)
            if hit is not None:
                cached_indices[idx] = hit
            else:
                missing.append(text)
                missing_indices.append(idx)

        if missing:
            self._ensure_model()
            method_name = "encode_query" if query else "encode_document"
            method = getattr(self._model, method_name, None)
            if not callable(method):
                method = self._model.encode
            vectors = method(
                list(missing),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            computed = np.asarray(vectors, dtype=np.float32)
            if computed.shape != (len(missing), self.dimension):
                raise ValueError(
                    f"Unexpected embedding shape {computed.shape}; "
                    f"expected ({len(missing)}, {self.dimension})"
                )
            for original_idx, text, vec in zip(missing_indices, missing, computed):
                cached_indices[original_idx] = vec
                self._store_cache(self._cache_key(text), vec)

        ordered = np.stack(
            [cached_indices[idx] for idx in range(len(texts))],
            axis=0,
        )
        if not np.isfinite(ordered).all():
            raise ValueError("Embedding contains NaN or infinity")
        norms = np.linalg.norm(ordered, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding contains a zero vector")
        return ordered / norms

    def invalidate_model(self, *, model_name: str | None = None, revision: str | None = None) -> None:
        """Drop in-process embeddings; called when model or revision changes."""
        if (model_name is not None and model_name != self.model_name) or (
            revision is not None and revision != self.revision
        ):
            return
        self._cache.clear()
        self._cache_keys.clear()

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model_name}:{self.revision or 'none'}:{digest}"

    def _store_cache(self, key: str, vector: np.ndarray) -> None:
        if self.cache_size <= 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache_keys.remove(key)
            self._cache_keys.append(key)
            self._cache[key] = vector
            return
        self._cache[key] = vector
        self._cache_keys.append(key)
        if len(self._cache_keys) > self.cache_size:
            oldest = self._cache_keys.pop(0)
            self._cache.pop(oldest, None)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.model_name,
            device=self.resolved_device,
            cache_folder=self.cache_dir,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        model.max_seq_length = self.max_length
        dimension_getter = getattr(model, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = model.get_sentence_embedding_dimension
        actual_dimension = dimension_getter()
        if actual_dimension is None or int(actual_dimension) != self.dimension:
            raise ValueError(
                f"Model {self.model_name} has dimension {actual_dimension}, "
                f"expected {self.dimension}"
            )
        model.eval()
        self._model = model

    def _resolve_device(self) -> str:
        requested = self.device.casefold()
        if requested not in {"auto", "cpu", "cuda", "mps"} and not requested.startswith(
            ("cuda:", "xpu:")
        ):
            raise ValueError(
                "EMBEDDING_DEVICE must be auto, cpu, cuda, cuda:N, mps or xpu:N"
            )
        if requested == "cpu":
            return "cpu"

        import torch

        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"EMBEDDING_DEVICE={self.device!r}, but PyTorch cannot access CUDA/ROCm"
            )
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError(
                    "EMBEDDING_DEVICE='mps', but PyTorch MPS is unavailable"
                )
        return self.device
