import asyncio
import io
import json
import urllib.error

import pytest

from rag import (
    GenerationTimeoutError,
    InvalidStructuredResponseError,
    LLMHealthClient,
    OpenAICompatibleSettings,
    ProviderUnavailableError,
    RAGExecutionGate,
    RAGQueueFullError,
)
from rag.generator import _post_json


def test_execution_gate_rejects_when_concurrency_and_queue_are_full():
    async def scenario():
        gate = RAGExecutionGate(
            max_concurrency=1,
            max_queue=0,
            queue_timeout_seconds=0.1,
            request_timeout_seconds=1,
        )
        async with gate.slot():
            with pytest.raises(RAGQueueFullError):
                async with gate.slot():
                    pass

    asyncio.run(scenario())


def test_execution_gate_reports_queue_wait_and_releases_cancelled_holder():
    async def scenario():
        gate = RAGExecutionGate(
            max_concurrency=1,
            max_queue=1,
            queue_timeout_seconds=1,
            request_timeout_seconds=1,
        )
        entered = asyncio.Event()

        async def holder():
            async with gate.slot() as waited:
                assert waited >= 0
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with gate.slot() as waited:
            assert waited >= 0

    asyncio.run(scenario())


def test_provider_transport_classifies_timeout_unavailable_and_invalid_json(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("private detail")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    with pytest.raises(GenerationTimeoutError):
        _post_json("http://llm", {}, {}, 1)

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("private detail")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(ProviderUnavailableError):
        _post_json("http://llm", {}, {}, 1)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(InvalidStructuredResponseError):
        _post_json("http://llm", {}, {}, 1)


@pytest.mark.parametrize("provider_model_id", ["local-model", "models/local-model"])
def test_llm_health_checks_configured_model(monkeypatch, provider_model_id):
    payload = json.dumps({"data": [{"id": provider_model_id}]}).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    health = LLMHealthClient(
        OpenAICompatibleSettings(
            base_url="http://ollama/v1",
            api_key="ollama",
            model="local-model",
        )
    )

    assert health.check() == {"status": "ready", "model": "local-model"}
