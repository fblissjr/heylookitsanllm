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

import asyncio
import time

import pytest

from heylook_llm.providers.common.generation_gate import ModelBusyError
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db
from heylook_llm.config import AppConfig
from heylook_llm.conversation_api import conversation_router
from heylook_llm.conversation_generate_api import (
    _ACTIVE, _Run, _pump, _subscribe, generate_router)
from heylook_llm.providers.abort import AbortEvent
from heylook_llm.providers.base import BaseProvider, GenerationChunk
from helpers.sse import sse_events, streamed_text


TEST_MODELS = {
    "models": [
        {
            "id": "fake-model",
            "provider": "mlx",
            "enabled": True,
            "config": {"model_path": "/fake/model", "vision": False},
        },
        {
            # thinking + vision caps, for the cap-gating tests
            "id": "fake-capable",
            "provider": "mlx",
            "enabled": True,
            "config": {"model_path": "/fake/capable", "vision": True,
                       "enable_thinking": True},
        },
    ],
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
        # Seconds per chunk. Default 0 keeps every existing test instant; the
        # detached-run tests need a generation that is still going when they
        # look at it. This is a SYNC generator driven on an executor thread,
        # so a plain sleep is the right blocking call.
        self.chunk_delay = 0.0

    def load_model(self):
        pass

    def template_info(self):
        return None  # pass-through parser

    def create_chat_completion(self, request, abort_event=None):
        self.last_request = request

        def gen():
            for i, text in enumerate(self.chunks):
                if self.chunk_delay:
                    time.sleep(self.chunk_delay)
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
        _ACTIVE[conv["id"]] = _Run(AbortEvent())  # simulate an in-flight generation
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
        _ACTIVE[conv["id"]] = _Run(ev)
        try:
            res = await client.delete(f"/v1/conversations/{conv['id']}/generate")
            assert res.status_code == 200
            assert ev.is_set()
        finally:
            _ACTIVE.pop(conv["id"], None)

    @pytest.mark.asyncio
    async def test_the_conversation_list_names_an_active_run(self, ctx):
        """A run nobody is subscribed to must still be visible and stoppable.

        Without this the disconnect fix trades a truncation bug for a runaway
        one: a generation that survives the client walking away, with no
        surface that admits it is running.
        """
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        ev = AbortEvent()
        _ACTIVE[conv["id"]] = _Run(ev)
        try:
            listed = (await client.get("/v1/conversations")).json()["conversations"]
            assert any(c["id"] == conv["id"] and c["generating"] for c in listed), \
                "an active run is invisible in the conversation list"
            body = (await client.get(f"/v1/conversations/{conv['id']}")).json()
            assert body["generating"] is True
            assert (await client.delete(
                f"/v1/conversations/{conv['id']}/generate")).status_code == 200
            assert ev.is_set(), "Stop did not reach a run with no subscriber"
        finally:
            _ACTIVE.pop(conv["id"], None)

    @pytest.mark.asyncio
    async def test_idle_conversation_is_not_marked_generating(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store)
        listed = (await client.get("/v1/conversations")).json()["conversations"]
        assert all(c["generating"] is False for c in listed)

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


@pytest.mark.unit
class TestCapGating:
    """The server-side twin of v3's old samplerParams(caps) filter
    (_CAP_GATED): cap-gated keys stored on a conversation -- or sent as
    overrides -- must not reach a provider whose model lacks the cap.
    Added 2026-08-13: the e2e suite pins only that the WIRE is sampler-free;
    nothing pinned the gate itself (review finding)."""

    @pytest.mark.asyncio
    async def test_cap_gated_params_dropped_for_incapable_model(self, ctx):
        client, store, provider = ctx
        # fake-model: vision False, no enable_thinking -> no thinking/vision caps
        conv = await db.create_conversation(
            store, title="t", model_id="fake-model",
            params={"enable_thinking": True, "vision_tokens": 512, "temperature": 0.5})
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "hi"})
        assert res.status_code == 200
        req = provider.last_request
        assert req is not None
        assert req.enable_thinking is None, "enable_thinking leaked to a non-thinking model"
        assert req.vision_tokens is None, "vision_tokens leaked to a non-vision model"
        assert req.temperature == 0.5  # ungated keys still flow

    @pytest.mark.asyncio
    async def test_cap_gated_params_pass_for_capable_model(self, ctx):
        client, store, provider = ctx
        conv = await db.create_conversation(
            store, title="t", model_id="fake-capable",
            params={"enable_thinking": True, "vision_tokens": 512})
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "hi"})
        assert res.status_code == 200
        req = provider.last_request
        assert req is not None
        assert req.enable_thinking is True
        assert req.vision_tokens == 512

    @pytest.mark.asyncio
    async def test_overrides_are_cap_gated_too(self, ctx):
        client, store, provider = ctx
        conv = await db.create_conversation(store, title="t", model_id="fake-model")
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "hi",
                                      "overrides": {"enable_thinking": True, "temperature": 0.7}})
        assert res.status_code == 200
        req = provider.last_request
        assert req is not None
        assert req.enable_thinking is None, "an override bypassed the cap gate"
        assert req.temperature == 0.7


@pytest.mark.unit
class TestMessageAndMediaEndpoints:
    """The v7 HTTP surface: single-message DELETE and the media serve route."""

    IMAGE = {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": "aGV5bG9vaw=="}}

    @pytest.mark.asyncio
    async def test_delete_one_message_leaves_neighbors(self, ctx):
        client, store, _ = ctx
        conv, rows = await make_conv(store, ("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        res = await client.delete(f"/v1/conversations/{conv['id']}/messages/{rows[1]['id']}")
        assert res.status_code == 200
        got = await db.get_conversation(store, conv["id"])
        assert [m["content"] for m in got["messages"]] == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_delete_missing_message_404s(self, ctx):
        client, store, _ = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.delete(f"/v1/conversations/{conv['id']}/messages/ghost")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_409s_while_generating(self, ctx):
        client, store, _ = ctx
        conv, rows = await make_conv(store, ("user", "q1"))
        _ACTIVE[conv["id"]] = _Run(AbortEvent())
        try:
            res = await client.delete(f"/v1/conversations/{conv['id']}/messages/{rows[0]['id']}")
            assert res.status_code == 409
        finally:
            _ACTIVE.pop(conv["id"], None)

    @pytest.mark.asyncio
    async def test_media_endpoint_serves_immutable_bytes(self, ctx):
        client, store, _ = ctx
        conv = await db.create_conversation(store, title="t", model_id="fake-model")
        row = await db.append_message(store, conv["id"], role="user", content=[self.IMAGE])
        url = row["content_blocks"][0]["source"]["url"]
        res = await client.get(url)
        assert res.status_code == 200
        assert res.content == b"heylook"
        assert res.headers["content-type"].startswith("image/png")
        # private (conversation media stays out of shared caches), immutable
        # (content-addressed), nosniff (the type is caller-derived)
        assert "immutable" in res.headers["cache-control"]
        assert "private" in res.headers["cache-control"]
        assert res.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.asyncio
    async def test_off_family_media_type_degrades_to_octet_stream(self, ctx):
        # A block claiming text/html would otherwise be SERVED as same-origin
        # HTML -- stored XSS. The write boundary constrains the type to the
        # block's own family; off-family claims store as octet-stream.
        client, store, _ = ctx
        conv = await db.create_conversation(store, title="t", model_id="fake-model")
        evil = {"type": "image",
                "source": {"type": "base64", "media_type": "text/html",
                           "data": "PHNjcmlwdD5ib29tPC9zY3JpcHQ+"}}
        row = await db.append_message(store, conv["id"], role="user", content=[evil])
        res = await client.get(row["content_blocks"][0]["source"]["url"])
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("application/octet-stream")

    @pytest.mark.asyncio
    async def test_media_endpoint_404s_across_conversations(self, ctx):
        client, store, _ = ctx
        conv = await db.create_conversation(store, title="t", model_id="fake-model")
        other = await db.create_conversation(store, title="o", model_id="fake-model")
        row = await db.append_message(store, conv["id"], role="user", content=[self.IMAGE])
        media_id = row["content_blocks"][0]["source"]["media_id"]
        res = await client.get(f"/v1/conversations/{other['id']}/media/{media_id}")
        assert res.status_code == 404


@pytest.mark.unit
class TestMediaWire:
    """Schema v7: stored rows carry blob-backed url sources; the wire build
    must resolve them back to inline bytes (providers cannot fetch our own
    relative URLs)."""

    IMAGE = {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": "aGV5bG9vaw=="}}

    @pytest.mark.asyncio
    async def test_blob_backed_image_reaches_the_wire_as_data_url(self, ctx):
        client, store, provider = ctx
        conv = await db.create_conversation(store, title="t", model_id="fake-capable")
        row = await db.append_message(
            store, conv["id"], role="user",
            content=[self.IMAGE, {"type": "text", "text": "what is this?"}])
        # precondition: the store externalized -- no base64 in the row
        assert row["content_blocks"][0]["source"]["type"] == "url"

        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 200
        req = provider.last_request
        assert req is not None
        parts = req.messages[-1].content
        image_parts = [p for p in parts if getattr(p, "type", None) == "image_url"]
        assert len(image_parts) == 1
        # the original bytes, re-inlined from the blob
        assert image_parts[0].image_url.url == "data:image/png;base64,aGV5bG9vaw=="

    @pytest.mark.asyncio
    async def test_cap_dropped_media_is_counted_not_fetched(self, ctx):
        client, store, provider = ctx
        # conversation on the NON-vision default model
        conv = await db.create_conversation(store, title="t", model_id="fake-model")
        await db.append_message(
            store, conv["id"], role="user",
            content=[self.IMAGE, {"type": "text", "text": "see this?"}])
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 200
        assert saved_event(res.text)["dropped_media"] == {"images": 1, "audio": 0}
        req = provider.last_request
        assert req is not None
        assert req.messages[-1].content == "see this?"  # text-only wire shape


@pytest.mark.unit
class TestClaimLeaks:
    @pytest.mark.asyncio
    async def test_busy_503_releases_the_claim(self, ctx):
        """The MODEL_BUSY 503 RETURNS (no raise), skipping the
        except-BaseException release -- the explicit pop on that path is
        what this pins (review finding 2026-08-13: leak = permanent 409)."""
        client, store, provider = ctx
        conv, _ = await make_conv(store, ("user", "q1"))

        def busy():
            raise ModelBusyError("MODEL_BUSY")
        provider.check_capacity = busy
        try:
            res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                    json={"mode": "append"})
            assert res.status_code == 503
            assert conv["id"] not in _ACTIVE, "the 503 path leaked the _ACTIVE claim"
            # and the conversation is immediately usable again
            del provider.check_capacity
            res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                    json={"mode": "append"})
            assert res.status_code == 200
        finally:
            if "check_capacity" in provider.__dict__:
                del provider.check_capacity


    @pytest.mark.asyncio
    async def test_a_finished_run_cannot_pop_a_NEWER_claim(self, ctx):
        """The stream's finally must release only ITS OWN claim.

        `_ACTIVE` is installed with an identity-guarded releaser and the code
        says so where the claim is made ("every release is identity-guarded so
        none can pop a NEWER generation's claim"), but the stream generator's
        own finally did a bare `_ACTIVE.pop(conv_id, None)` -- and that is the
        release most able to break the invariant, because it runs on a detached
        task that outlives the response by the whole length of an abandoned
        run.

        The consequence of popping someone else's claim is not a leak, it is
        the opposite and worse: run B is left with no entry, so `DELETE
        .../generate` 404s (B is unstoppable), `is_generating` reports idle so
        the composer offers Send, and a third POST passes the 409 gate and
        writes into the same conversation through the positional commit.

        Staged by having the PROVIDER swap the claim mid-generation, which is
        the real shape: something released A (the 60s watchdog, or response
        cleanup racing `started`) and B claimed the conversation while A was
        still running.
        """
        client, store, provider = ctx
        conv, _ = await make_conv(store, ("user", "q1"))
        newer = _Run(AbortEvent())

        def swap_mid_run(request, abort_event=None):
            def gen():
                yield GenerationChunk(text="Hello", token=0)
                _ACTIVE[conv["id"]] = newer   # the claim changed hands
                yield GenerationChunk(text=" world", token=1)
            return gen()
        provider.create_chat_completion = swap_mid_run
        try:
            res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                    json={"mode": "append"})
            assert res.status_code == 200
            assert _ACTIVE.get(conv["id"]) is newer, (
                "the finished run popped a NEWER generation's claim -- that run "
                "is now unstoppable and the conversation reports idle while it "
                "is still writing")
        finally:
            del provider.create_chat_completion
            _ACTIVE.pop(conv["id"], None)


@pytest.mark.unit
class TestShowSpecialTokens:
    """v3's "Show special tokens" display pref on the wire (DESIGN.md §6).

    The server strips a model's DECLARED specials before the text is streamed
    and before it is persisted, so "show them" has to be asked for per request.
    Both halves matter: a reply is stored exactly as it was parsed, so the
    streamed deltas and the saved row must agree -- one parser built two ways
    would put markers on screen that the row does not have (or the reverse).
    """

    SPECIAL = "<|im_end|>"

    @pytest_asyncio.fixture
    async def ctx_specials(self):
        """Same app as `ctx`, but the provider DECLARES a special and emits it
        -- the fake in `ctx` returns template_info() None (pass-through), where
        nothing is ever stripped and this pref could not be observed."""
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

        class DeclaringProvider(FakeProvider):
            def __init__(self):
                super().__init__(chunks=("Hello", f" world{TestShowSpecialTokens.SPECIAL}"))

            def template_info(self):
                return ModelTemplateInfo(
                    chat_template="",
                    special_tokens=frozenset([TestShowSpecialTokens.SPECIAL]),
                    template_source="jinja",
                )

        provider = DeclaringProvider()
        app = FastAPI()
        app.include_router(conversation_router)
        app.include_router(generate_router)
        app.state.db = await db.get_connection(path=":memory:")
        app.state.router_instance = FakeRouter(provider)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app.state.db, provider
        _ACTIVE.clear()
        await app.state.db.close()

    @pytest.mark.asyncio
    async def test_asking_keeps_them_on_screen_and_in_the_row(self, ctx_specials):
        client, store, _ = ctx_specials
        conv, _ = await make_conv(store, ("user", "q"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "show_special_tokens": True})
        assert res.status_code == 200
        assert self.SPECIAL in streamed_text(res.text)
        saved = saved_event(res.text)["messages"][-1]
        assert saved["content"] == f"Hello world{self.SPECIAL}"

    @pytest.mark.asyncio
    async def test_default_strips_both(self, ctx_specials):
        """Opt-IN: a body that says nothing gets today's behavior, streamed
        and stored."""
        client, store, _ = ctx_specials
        conv, _ = await make_conv(store, ("user", "q"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 200
        assert self.SPECIAL not in streamed_text(res.text)
        assert saved_event(res.text)["messages"][-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_it_never_reaches_the_model(self, ctx_specials):
        """It is a display pref: it must not land in the request the provider
        is driven with (params is the sampler bag -- CLAUDE.md)."""
        client, store, provider = ctx_specials
        conv, _ = await make_conv(store, ("user", "q"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "show_special_tokens": True})
        assert res.status_code == 200
        dumped = provider.last_request.model_dump()
        assert "show_special_tokens" not in dumped
        assert not any("special" in k for k in dumped)

    @pytest.mark.asyncio
    async def test_kept_specials_do_not_come_back_as_prompt(self, ctx_specials):
        """The store IS the request, so a row recorded WITH specials is
        replayed into the next turn's prompt -- and a fast tokenizer encodes a
        declared special's string as the real control token, putting a turn
        boundary inside prior assistant content. Display state must not reach
        the model: the replay strips (code review finding, 2026-08-23)."""
        client, store, provider = ctx_specials
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "show_special_tokens": True})
        assert res.status_code == 200
        stored = saved_event(res.text)["messages"][-1]["content"]
        assert self.SPECIAL in stored, "precondition: the row must carry a special"

        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "user_content": "q2",
                                      "show_special_tokens": True})
        assert res.status_code == 200
        replayed = [m for m in provider.last_request.messages if m.role == "assistant"]
        assert replayed, "the second turn replayed no assistant history"
        assert all(self.SPECIAL not in (m.content or "") for m in replayed), \
            f"the model was fed its own control token back: {[m.content for m in replayed]}"

    @pytest.mark.asyncio
    async def test_continue_prefill_never_ends_in_a_control_token(self, ctx_specials):
        """Worst case of the same bug: in continue mode the anchor row rides as
        the FINAL message with continue_final_message=True, so an unstripped
        row would make the prefill end on a turn boundary."""
        client, store, provider = ctx_specials
        conv, _ = await make_conv(store, ("user", "q1"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append", "show_special_tokens": True})
        assert res.status_code == 200
        anchor = saved_event(res.text)["messages"][-1]
        assert self.SPECIAL in anchor["content"], "precondition: anchor carries a special"

        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "continue", "message_id": anchor["id"],
                                      "show_special_tokens": True})
        assert res.status_code == 200
        req = provider.last_request
        assert req.continue_final_message is True
        assert self.SPECIAL not in (req.messages[-1].content or "")

    @pytest.mark.asyncio
    async def test_user_text_is_left_alone(self, ctx_specials):
        """The strip is scoped to what the SERVER recorded unstripped. What a
        user typed keeps the semantics every other API surface gives it."""
        client, store, provider = ctx_specials
        conv, _ = await make_conv(store, ("user", f"keep {self.SPECIAL} mine"))
        res = await client.post(f"/v1/conversations/{conv['id']}/generate",
                                json={"mode": "append"})
        assert res.status_code == 200
        user_msgs = [m for m in provider.last_request.messages if m.role == "user"]
        assert any(self.SPECIAL in (m.content or "") for m in user_msgs)


@pytest.mark.unit
class TestRunOutlivesResponse:
    """The generation is the SERVER's, not the HTTP response's.

    These drive the mechanism directly rather than through the test client.
    That is not a shortcut -- httpx's ASGITransport runs the app to completion
    before it yields the first line, so a test that opens a stream, breaks out
    and calls it a disconnect never disconnects anything: the run has already
    finished. The first version of these tests did exactly that and passed
    while proving nothing.
    """

    @pytest.mark.asyncio
    async def test_pump_finishes_after_its_subscriber_leaves(self):
        run = _Run(AbortEvent())
        produced = []

        async def saga():
            for i in range(10):
                await asyncio.sleep(0.005)
                produced.append(i)
                yield f"event: tick\ndata: {i}\n\n"

        task = asyncio.create_task(_pump("c" * 8, run, saga()))
        sub = _subscribe(run)
        seen = []
        async for event_str in sub:
            seen.append(event_str)
            break
        await sub.aclose()  # the client went away

        await asyncio.wait_for(task, timeout=5)
        assert produced == list(range(10)), \
            f"the run stopped when its subscriber left (got {produced})"
        assert len(seen) == 1
        assert run.detached

    @pytest.mark.asyncio
    async def test_detached_run_does_not_grow_a_queue(self):
        """A run nobody drains must not accumulate its own output."""
        run = _Run(AbortEvent())

        async def saga():
            for i in range(500):
                yield f"event: tick\ndata: {i}\n\n"

        run.detached = True  # as if the subscriber had already gone
        await asyncio.wait_for(_pump("c" * 8, run, saga()), timeout=5)
        # Only the end-of-stream sentinel.
        assert run.queue.qsize() == 1
        assert run.queue.get_nowait() is None


@pytest.mark.unit
class TestDisconnectPolicy:
    """`abort_on_disconnect` is the whole behaviour switch; pin it both ways."""

    @staticmethod
    def _request(disconnected: bool):
        class _Req:
            async def is_disconnected(self):
                return disconnected
        return _Req()

    @pytest.mark.asyncio
    async def test_stateless_wires_still_abort_on_disconnect(self):
        # /v1/messages and /v1/chat/completions persist nothing, so a client
        # that leaves means the work has nowhere to go. This must not change.
        from heylook_llm.streaming_utils import async_generator_with_abort
        ev = AbortEvent()
        produced = []

        def gen():
            for i in range(50):
                time.sleep(0.005)
                produced.append(i)
                yield GenerationChunk(text=str(i), token=i)

        out = []
        async for chunk in async_generator_with_abort(
                gen(), self._request(True), ev):
            out.append(chunk)
        assert ev.is_set(), "a disconnect no longer aborts the stateless wire"
        assert len(produced) < 50, "generation ran to completion despite the abort"

    @pytest.mark.asyncio
    async def test_conversation_saga_keeps_going_when_the_client_leaves(self):
        from heylook_llm.streaming_utils import async_generator_with_abort
        ev = AbortEvent()
        produced = []

        def gen():
            for i in range(20):
                produced.append(i)
                yield GenerationChunk(text=str(i), token=i)

        out = []
        async for chunk in async_generator_with_abort(
                gen(), self._request(True), ev, abort_on_disconnect=False):
            out.append(chunk)
        assert not ev.is_set(), "a disconnect aborted a server-persisted run"
        assert produced == list(range(20)), \
            f"the run was truncated by the disconnect (got {len(produced)} chunks)"
        assert len(out) == 20
