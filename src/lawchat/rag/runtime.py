from __future__ import annotations

import os
import asyncio
import json
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter

from lawchat.chunking import HuggingFaceTokenizerCounter
from lawchat.indexing import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION
from lawchat.retrieval import LegalQueryParser

from .context_builder import RAGContextBuilder
from .generator import (
    OpenAICompatibleLegalAnswerGenerator,
    OpenAICompatibleSettings,
)
from .service import GroundedRAGService
from .semantic import OpenAICompatibleClaimJudge, SemanticReviewCache, SemanticVerifier
from .issues import OpenAICompatibleIssueDecomposer
from .context_models import TokenBudget


@dataclass(frozen=True, slots=True)
class RAGRuntime:
    """Process-scoped generation dependencies reused across HTTP requests."""

    context_builder: RAGContextBuilder
    generation_service: GroundedRAGService
    execution_gate: RAGExecutionGate = field(
        default_factory=lambda: RAGExecutionGate(
            max_concurrency=1,
            max_queue=4,
            queue_timeout_seconds=10,
            request_timeout_seconds=180,
        )
    )
    llm_health: LLMHealthClient | None = None
    issue_decomposer: OpenAICompatibleIssueDecomposer | None = None


class RAGQueueFullError(RuntimeError):
    pass


class RAGExecutionGate:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue: int,
        queue_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> None:
        if min(max_concurrency, max_queue + 1) <= 0:
            raise ValueError("RAG concurrency settings must be positive")
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_queue = max_queue
        self.queue_timeout_seconds = queue_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._waiting = 0
        self._waiting_lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self):
        wait_started = perf_counter()
        async with self._waiting_lock:
            if self.semaphore.locked() and self._waiting >= self.max_queue:
                raise RAGQueueFullError("RAG request queue is full")
            self._waiting += 1
        try:
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.queue_timeout_seconds,
                )
            except TimeoutError as exc:
                raise RAGQueueFullError("RAG request queue timed out") from exc
        finally:
            async with self._waiting_lock:
                self._waiting -= 1
        try:
            yield perf_counter() - wait_started
        finally:
            self.semaphore.release()


@dataclass(frozen=True, slots=True)
class LLMHealthClient:
    settings: OpenAICompatibleSettings

    def check(self) -> dict[str, str]:
        endpoint = self.settings.base_url.rstrip("/") + "/models"
        request = urllib.request.Request(endpoint, headers=self._headers())
        with urllib.request.urlopen(request, timeout=min(5, self.settings.timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = {
            item.get("id")
            for item in payload.get("data", ())
            if isinstance(item, dict)
        }
        if self.settings.model not in model_ids:
            raise RuntimeError("configured LLM model is not available")
        return {"status": "ready", "model": self.settings.model}

    def warmup(self) -> None:
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "reasoning_effort": "none",
            "max_tokens": 8,
            "temperature": 0,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._headers()},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            response.read()

    def _headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.settings.api_key}"}
            if self.settings.api_key
            else {}
        )


def create_rag_runtime(*, enforce_configured_cutoff: bool = True) -> RAGRuntime:
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    default_revision = (
        DEFAULT_EMBEDDING_REVISION
        if model_name == DEFAULT_EMBEDDING_MODEL
        else None
    )
    tokenizer = HuggingFaceTokenizerCounter(
        model_name=model_name,
        revision=os.getenv("EMBEDDING_MODEL_REVISION", default_revision),
        cache_dir=os.getenv(
            "EMBEDDING_CACHE_PATH",
            "data/.cache/huggingface",
        ),
        local_files_only=_env_flag("EMBEDDING_LOCAL_FILES_ONLY")
        or _env_flag("HF_HUB_OFFLINE"),
    )
    context_builder = RAGContextBuilder(
        tokenizer,
        budget=TokenBudget(
            model_context_window=_positive_int(
                "LAWCHAT_LLM_CONTEXT_WINDOW",
                8192,
            ),
            system_prompt_tokens=_non_negative_int(
                "LAWCHAT_LLM_SYSTEM_PROMPT_TOKENS",
                1500,
            ),
            answer_reserve_tokens=_non_negative_int(
                "LAWCHAT_LLM_ANSWER_RESERVE_TOKENS",
                1500,
            ),
        ),
        max_evidence_total=_positive_int(
            "LAWCHAT_CONTEXT_MAX_EVIDENCE_TOTAL", 12
        ),
        max_evidence_per_issue=_positive_int(
            "LAWCHAT_CONTEXT_MAX_EVIDENCE_PER_ISSUE", 3
        ),
        max_graph_evidence=_non_negative_int(
            "LAWCHAT_CONTEXT_MAX_GRAPH_EVIDENCE", 2
        ),
        query_parser=LegalQueryParser(
            enforce_configured_cutoff=enforce_configured_cutoff
        ),
    )
    llm_settings = OpenAICompatibleSettings.from_env()
    generation_service = create_generation_service(llm_settings)
    return RAGRuntime(
        context_builder=context_builder,
        generation_service=generation_service,
        execution_gate=RAGExecutionGate(
            max_concurrency=_positive_int("LAWCHAT_RAG_MAX_CONCURRENCY", 1),
            max_queue=_non_negative_int("LAWCHAT_RAG_MAX_QUEUE", 4),
            queue_timeout_seconds=_positive_float(
                "LAWCHAT_RAG_QUEUE_TIMEOUT_SECONDS", 10
            ),
            request_timeout_seconds=_positive_float(
                "LAWCHAT_RAG_REQUEST_TIMEOUT_SECONDS", 180
            ),
        ),
        llm_health=LLMHealthClient(llm_settings),
        issue_decomposer=OpenAICompatibleIssueDecomposer(
            replace(
                llm_settings,
                max_tokens=_positive_int("LAWCHAT_ISSUE_MAX_TOKENS", 1024),
                timeout_seconds=_positive_float("LAWCHAT_ISSUE_TIMEOUT_SECONDS", 60),
            )
        ),
    )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").casefold() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def create_generation_service(settings: OpenAICompatibleSettings | None = None) -> GroundedRAGService:
    settings = settings or OpenAICompatibleSettings.from_env()
    mode = os.getenv("LAWCHAT_SEMANTIC_MODE", "off").lower()
    judge_settings = replace(settings,
        model=os.getenv("LAWCHAT_SEMANTIC_MODEL") or settings.model,
        timeout_seconds=_positive_float("LAWCHAT_SEMANTIC_TIMEOUT_SECONDS", 60),
        max_tokens=_positive_int("LAWCHAT_SEMANTIC_MAX_TOKENS", 3072))
    shadow_model = os.getenv("LAWCHAT_SEMANTIC_SHADOW_MODEL", "").strip()
    cache_entries = _non_negative_int("LAWCHAT_SEMANTIC_CACHE_MAX_ENTRIES", 256)
    cache_ttl = _positive_float("LAWCHAT_SEMANTIC_CACHE_TTL_SECONDS", 300)
    return GroundedRAGService(
        OpenAICompatibleLegalAnswerGenerator(settings),
        semantic_verifier=(
            SemanticVerifier(
                OpenAICompatibleClaimJudge(judge_settings),
                cache=SemanticReviewCache(
                    max_entries=cache_entries,
                    ttl_seconds=cache_ttl,
                ),
            )
            if mode != "off"
            else None
        ),
        shadow_semantic_verifier=(
            SemanticVerifier(
                OpenAICompatibleClaimJudge(replace(judge_settings, model=shadow_model)),
                cache=SemanticReviewCache(cache_entries, cache_ttl),
            )
            if mode != "off" and shadow_model
            else None
        ),
        semantic_mode=mode,
    )
