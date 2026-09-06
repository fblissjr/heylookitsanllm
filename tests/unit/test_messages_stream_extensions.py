# tests/unit/test_messages_stream_extensions.py
"""The /v1/messages heylook extension namespace (Phase 3b).

Messages has no timing of its own; consumers that ported off the OpenAI route
(removed in v1.79.66) must not lose it (spec §4's extension rule): streaming
message_stop.performance carries the shared timing names (peak_memory_gb,
kv_cache_bytes, queue_wait_ms, draft_acceptance -- the heylook_saved.timing
vocabulary), None fields skipped.

The heylook_logprobs half of this namespace was removed in v1.79.74 with the
token explorer, its only consumer.
"""

import json as std_json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm.config import AppConfig
from heylook_llm.messages_api import messages_router
from heylook_llm.providers.base import BaseProvider, GenerationChunk

TEST_MODELS = {
    "models": [
        {
            "id": "fake-model",
            "provider": "mlx",
            "enabled": True,
            "config": {"model_path": "/fake/model", "vision": False},
        },
    ],
    "default_model": "fake-model",
    "max_loaded_models": 1,
}


class FakeTokenizer:
    def decode(self, ids):
        return "".join(f"tok{i}" for i in ids)


class FakeProvider(BaseProvider):
    """Yields canned chunks."""

    provider_name = "mlx"

    def __init__(self, chunks):
        super().__init__("fake-model", {"model_path": "/fake/model", "vision": False}, False)
        self.chunks = chunks
        self.last_request = None

    def load_model(self):
        pass

    def template_info(self):
        return None  # pass-through parser

    def get_tokenizer(self):
        return FakeTokenizer()

    def create_chat_completion(self, request, abort_event=None):
        self.last_request = request

        def gen():
            yield from self.chunks
        return gen()


class FakeRouter:
    def __init__(self, provider):
        self.app_config = AppConfig(**TEST_MODELS)
        self.provider = provider

    def get_provider(self, model_id):
        return self.provider


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = std_json.loads(line[len("data: "):])
        if ev:
            events.append((ev, data))
    return events


def make_app(chunks):
    provider = FakeProvider(chunks)
    app = FastAPI()
    app.include_router(messages_router)
    app.state.router_instance = FakeRouter(provider)
    return app, provider


def token_chunks():
    return [
        GenerationChunk(text="Hello", token=0),
        GenerationChunk(text=" world", token=1,
                        finish_reason="stop", prompt_tokens=3, generation_tokens=2,
                        peak_memory=1.5, kv_cache_bytes=4096, queue_wait_ms=2.0),
    ]


@pytest_asyncio.fixture
async def client_factory():
    clients = []

    async def make(chunks):
        app, provider = make_app(chunks)
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        clients.append(client)
        return client, provider

    yield make
    for c in clients:
        await c.aclose()


@pytest.mark.unit
class TestMessageStopTiming:
    @pytest.mark.asyncio
    async def test_timing_rides_performance(self, client_factory):
        client, _ = await client_factory(token_chunks())
        res = await client.post("/v1/messages", json={
            "model": "fake-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
        stop = next(d for ev, d in sse_events(res.text) if ev == "message_stop")
        perf = stop["performance"]
        assert perf["peak_memory_gb"] == pytest.approx(1.5)
        assert perf["kv_cache_bytes"] == 4096
        assert perf["queue_wait_ms"] == pytest.approx(2.0)
        # absent telemetry is SKIPPED, never null (no spec decode ran here)
        assert "draft_acceptance" not in perf

    @pytest.mark.asyncio
    async def test_absent_telemetry_is_omitted(self, client_factory):
        chunks = [GenerationChunk(text="hi", token=0, finish_reason="stop")]
        client, _ = await client_factory(chunks)
        res = await client.post("/v1/messages", json={
            "model": "fake-model", "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
        stop = next(d for ev, d in sse_events(res.text) if ev == "message_stop")
        perf = stop["performance"]
        assert "peak_memory_gb" not in perf and "kv_cache_bytes" not in perf
        # The span the stream can always measure is still there. A measured
        # zero queue wait is NOT absent -- that is the v1.79.58 rule, and this
        # run really did wait zero.
        assert "generation_duration_ms" in perf
        # An unmeasured queue wait is ABSENT, not a published 0.0 (v1.79.59 --
        # .58 had this backwards on a premise that measurement refuted).
        assert "queue_wait_ms" not in perf


@pytest.mark.unit
class TestExtensionSamplerFields:
    @pytest.mark.asyncio
    async def test_sampler_and_vision_tokens_reach_the_provider(self, client_factory):
        client, provider = await client_factory(token_chunks())
        res = await client.post("/v1/messages", json={
            "model": "fake-model", "stream": True,
            "vision_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert res.status_code == 200
        assert provider.last_request is not None
        assert provider.last_request.vision_tokens == 256
