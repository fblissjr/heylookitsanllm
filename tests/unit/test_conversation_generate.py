# tests/unit/test_conversation_generate.py
"""Contract for POST/DELETE /v1/conversations/{id}/generate.

Runs the generate router on a minimal FastAPI app with an in-memory store and
a fake router/provider -- no server, no models. Pins the Phase 1 invariants
(plan_chat_orchestration.md):

- the store IS the request (system prompt + params + rows reach the provider)
- append persists the user turn, then the assistant turn; regenerate and
  continue commit their truncation ONLY together with the produced row
- the final heylook_saved event carries the authoritative stored rows
- an empty generation leaves the thread untouched (no truncation)
- one active generation per conversation (409), DELETE aborts / 404s

NOT covered here (needs a live server / real concurrency): persistence on a
mid-stream client disconnect. The mechanism (detached task in the stream
generator's finally) is exercised only by code review + the e2e suite once
the v3 cutover lands.
"""

import json as std_json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db
from heylook_llm.config import AppConfig
from heylook_llm.conversation_api import conversation_router
from heylook_llm.conversation_generate_api import _ACTIVE, generate_router
from heylook_llm.providers.abort import AbortEvent
from heylook_llm.providers.base import BaseProvider, GenerationChunk


TEST_MODELS = {
    "models": [{
        "id": "fake-model",
        "provider": "mlx",
        "enabled": True,
        "config": {"model_path": "/fake/model", "vision": False},
    }],
    "default_model": "fake-model",
    "max_loaded_models": 1,
}


class FakeProvider(BaseProvider):
    """Yields canned chunks; records the request it was driven with."""

    provider_name = "mlx"

    def __init__(self, model_id="fake-model", chunks=("Hello", " world")):
        super().__init__(model_id, {"model_path": "/fake/model", "vision": False}, False)
        self.chunks = chunks
        self.last_request = None

    def load_model(self):
        pass

    def template_info(self):
        return None  # pass-through parser

    def create_chat_completion(self, request, abort_event=None):
        self.last_request = request

        def gen():
            for i, text in enumerate(self.chunks):
                yield GenerationChunk(text=text, token=i)
        return gen()


class FakeRouter:
    def __init__(self, provider: FakeProvider):
        self.app_config = AppConfig(**TEST_MODELS)
        self.provider = provider

    def get_provider(self, model_id):
        from heylook_llm.router import ModelNotFound
        if not self.app_config.get_model_config(model_id):
            raise ModelNotFound(f"Model '{model_id}' not found or not enabled")
        return self.provider


def sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse '(event, data)' pairs out of an SSE body."""
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


def saved_event(text: str) -> dict:
    for ev, data in sse_events(text):
        if ev == "heylook_saved":
            return data
    raise AssertionError(f"no heylook_saved event in stream:\n{text}")


@pytest_asyncio.fixture
async def ctx():
    provider = FakeProvider()
    app = FastAPI()
    app.include_router(conversation_router)
    app.include_router(generate_router)
    app.state.db = await db.get_connection(path=":memory:")
    app.state.router_instance = FakeRouter(provider)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app.state.db, provider
    _ACTIVE.clear()  # a failed test must not leak a claim into the next one
    await app.state.db.close()


async def make_conv(store, *messages, system_prompt=None, params=None):
    conv = await db.create_conversation(
        store, title="t", model_id="fake-model",
        system_prompt=system_prompt, params=params or {})
    rows = []
    for role, content in messages:
        rows.append(await db.append_message(store, conv["id"], role=role, content=content))
    return conv, rows


@pytest.mark.unit
class TestAppend:
    @pytest.mark.asyncio
    async def test_append_persists_user_and_assistant(self, ctx):
        client, store, provider = ctx
        conv, _ = await make_conv(store)
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "hi there"})
        assert res.status_code == 200
        data = saved_event(res.text)
        assert data["end_reason"] == "complete"
        roles = [(m["role"], m["content"]) for m in data["messages"]]
        assert roles == [("user", "hi there"), ("assistant", "Hello world")]
        # the store agrees (authoritative rows are STORED rows)
        stored = await db.get_conversation(store, conv["id"])
        assert stored is not None
        assert [(m["role"], m["content"]) for m in stored["messages"]] == roles
        assert [m["position"] for m in stored["messages"]] == [0, 1]

    @pytest.mark.asyncio
    async def test_store_is_the_request(self, ctx):
        client, store, provider = ctx
        conv, _ = await make_conv(
            store, ("user", "earlier turn"),
            system_prompt="be terse", params={"temperature": 0.4, "max_tokens": 64})
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 200
        req = provider.last_request
        assert req is not None
        assert req.messages[0].role == "system" and req.messages[0].content == "be terse"
        assert req.messages[1].content == "earlier turn"
        assert req.temperature == 0.4 and req.max_tokens == 64
        assert req.model == "fake-model"

    @pytest.mark.asyncio
    async def test_append_empty_conversation_400(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_conversation_404(self, ctx):
        client, _, _ = ctx
        res = await client.post("/v1/conversations/nope/generate",
                                json={"mode": "append", "user_content": "x"})
        assert res.status_code == 404


@pytest.mark.unit
class TestRegenerate:
    @pytest.mark.asyncio
    async def test_regenerate_replaces_tail_atomically(self, ctx):
        client, store, provider = ctx
        conv, rows = await make_conv(
            store, ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "regenerate", "message_id": rows[1]["id"]})
        assert res.status_code == 200
        stored = await db.get_conversation(store, conv["id"])
        assert stored is not None
        got = [(m["role"], m["content"], m["position"]) for m in stored["messages"]]
        # q1 survives; a1, q2, a2 replaced by the fresh assistant turn at position 1
        assert got == [("user", "q1", 0), ("assistant", "Hello world", 1)]
        # the prompt saw only what precedes the anchor
        assert [m.content for m in provider.last_request.messages] == ["q1"]

    @pytest.mark.asyncio
    async def test_regenerate_first_message_400(self, ctx):
        client, store, _ = ctx
        conv, rows = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "regenerate", "message_id": rows[0]["id"]})
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_anchor_404(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "regenerate", "message_id": "missing"})
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_generation_leaves_thread_untouched(self, ctx):
        client, store, provider = ctx
        provider.chunks = ()  # model produces nothing
        conv, rows = await make_conv(
            store, ("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "regenerate", "message_id": rows[1]["id"]})
        assert res.status_code == 200
        assert saved_event(res.text)["messages"] == []
        stored = await db.get_conversation(store, conv["id"])
        assert stored is not None
        # the commit-together rule: no output, no truncation
        assert [(m["role"], m["content"]) for m in stored["messages"]] == [
            ("user", "q1"), ("assistant", "a1"), ("user", "q2")]


@pytest.mark.unit
class TestContinue:
    @pytest.mark.asyncio
    async def test_continue_merges_onto_anchor(self, ctx):
        client, store, _ = ctx
        conv, rows = await make_conv(
            store, ("user", "q1"), ("assistant", "partial"), ("user", "stale tail"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "continue", "message_id": rows[1]["id"]})
        assert res.status_code == 200
        data = saved_event(res.text)
        assert [(m["role"], m["content"]) for m in data["messages"]] == [
            ("assistant", "partialHello world")]
        stored = await db.get_conversation(store, conv["id"])
        assert stored is not None
        # anchor absorbed the continuation; the stale tail is gone; NO new row
        assert [(m["role"], m["content"], m["id"]) for m in stored["messages"]] == [
            ("user", "q1", rows[0]["id"]),
            ("assistant", "partialHello world", rows[1]["id"])]

    @pytest.mark.asyncio
    async def test_continue_requires_message_id(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "continue"})
        assert res.status_code == 400


@pytest.mark.unit
class TestArbitration:
    @pytest.mark.asyncio
    async def test_second_generate_409(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        _ACTIVE[conv["id"]] = AbortEvent()  # simulate an in-flight generation
        try:
            res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                    json={"mode": "append", "user_content": "x"})
            assert res.status_code == 409
            assert res.json()["error"]["code"] == "generation_in_progress"
        finally:
            _ACTIVE.pop(conv["id"], None)

    @pytest.mark.asyncio
    async def test_delete_aborts_active(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        ev = AbortEvent()
        _ACTIVE[conv["id"]] = ev
        try:
            res = await client.delete(f"/v1/conversations/{conv['id']}/generate")
            assert res.status_code == 200
            assert ev.is_set()
        finally:
            _ACTIVE.pop(conv["id"], None)

    @pytest.mark.asyncio
    async def test_delete_without_active_404(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        res = await client.delete(f"/v1/conversations/{conv['id']}/generate")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_claim_released_after_stream(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "x"})
        assert res.status_code == 200
        assert conv["id"] not in _ACTIVE

    @pytest.mark.asyncio
    async def test_claim_released_on_early_failure(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "regenerate", "message_id": "missing"})
        assert res.status_code == 404
        assert conv["id"] not in _ACTIVE


@pytest.mark.unit
class TestWireShape:
    @pytest.mark.asyncio
    async def test_messages_grammar_events_present(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "hi"})
        names = [ev for ev, _ in sse_events(res.text)]
        for expected in ("message_start", "content_block_start", "content_block_delta",
                         "content_block_stop", "message_delta", "message_stop", "heylook_saved"):
            assert expected in names, f"missing {expected} in {names}"
        # extension event is LAST -- the client's final state assignment
        assert names[-1] == "heylook_saved"

    @pytest.mark.asyncio
    async def test_model_override_reaches_provider_and_unknown_400(self, ctx):
        client, store, provider = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "overrides": {"model": "nope"}})
        assert res.status_code == 400
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "overrides": {"model": "fake-model",
                                                                       "temperature": 0.9}})
        assert res.status_code == 200
        assert provider.last_request.temperature == 0.9
