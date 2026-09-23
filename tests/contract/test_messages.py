# tests/contract/test_messages.py
#
# Contract tests for POST /v1/messages (Anthropic Messages-inspired API).

import json

import pytest

from helpers.sse import streamed_text


class TestMessagesNonStreaming:
    """Tests for POST /v1/messages (non-streaming)."""

    def test_valid_request_returns_200(self, client):
        """A valid messages request returns 200 with content blocks."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        assert resp.status_code == 200

        data = resp.json()
        assert data["role"] == "assistant"
        assert isinstance(data["content"], list)
        assert len(data["content"]) >= 1

    def test_response_has_text_block(self, client):
        """Response content includes at least one text block."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        text_blocks = [b for b in data["content"] if b["type"] == "text"]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0]["text"]) > 0

    def test_response_has_usage(self, client):
        """Response includes usage with input/output tokens."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        assert "usage" in data
        assert "output_tokens" in data["usage"]

    def test_response_has_model_and_id(self, client):
        """Response includes model and id fields."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
        })
        data = resp.json()
        assert "id" in data
        assert data["model"] == "test-mlx-model"

    def test_content_blocks_with_typed_input(self, client):
        """Content blocks as input (not just string) are accepted."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello typed"}],
            }],
            "max_tokens": 128,
        })
        assert resp.status_code == 200

    def test_system_prompt_as_top_level_param(self, client):
        """System prompt is a top-level parameter, not in messages."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 128,
        })
        assert resp.status_code == 200

    def test_missing_messages_returns_422(self, client):
        """Request without messages returns 422."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "max_tokens": 128,
        })
        assert resp.status_code == 422


class TestMessagesStreaming:
    """Tests for POST /v1/messages with stream=true."""

    def test_streaming_returns_sse_events(self, client):
        """stream=true returns SSE with structured event types."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        # Extract event type lines
        event_lines = [l for l in body.split("\n") if l.startswith("event: ")]
        event_types = [l.split("event: ", 1)[1] for l in event_lines]

        # Must start with message_start and end with message_stop
        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"

    def test_streaming_has_content_block_events(self, client):
        """Streaming includes content_block_start, content_block_delta, content_block_stop."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        body = resp.text
        event_lines = [l for l in body.split("\n") if l.startswith("event: ")]
        event_types = [l.split("event: ", 1)[1] for l in event_lines]

        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types

    def test_streaming_data_lines_are_valid_json(self, client):
        """Each data: line in SSE is valid JSON."""
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 128,
            "stream": True,
        })
        body = resp.text
        for line in body.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                parsed = json.loads(payload)
                assert "type" in parsed


class TestDeclaredSpecialsAreStrippedOnThisWire:
    """The strip on `/v1/messages`, pinned at the ROUTE.

    It is unconditional since v2.0.38 removed `show_special_tokens`, and that is
    exactly why this class had to survive the removal rather than go with it:
    delete the flag's tests and the SURVIVING behaviour is unpinned, so dropping
    `strip_specials` from messages_api's two `select_reasoning_parser` calls --
    or `StripSpecials` ceasing to compose -- would leak raw control tokens into
    both the streamed text and the non-streaming body with every contract test
    green.

    The fixture is the load-bearing half. conftest's FakeProvider returns
    ``template_info() -> None``, i.e. a pass-through parser with NO declared
    specials, so on that path nothing is ever stripped and the strip is
    UNOBSERVABLE -- an assertion against it would pass whether or not the
    handler strips anything. These swap in a provider that declares and emits
    one.
    """

    SPECIAL = "<|im_end|>"

    @pytest.fixture
    def declaring_model(self, mock_router):
        """Point one model id at a provider that declares + emits a special.

        The router fixture is session-scoped, so the swap is undone after the
        test or every later test would inherit this provider."""
        from helpers.mlx_mock import FakeChunk
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

        special = TestDeclaredSpecialsAreStrippedOnThisWire.SPECIAL
        model_id = "test-mlx-model"
        # Subclass the fixture's OWN provider class rather than importing it:
        # the contract conftest is not importable by name (bare `conftest`
        # resolves to tests/conftest.py), and inheriting from whatever the
        # router hands out keeps this in step with that fake.
        base = type(mock_router.get_provider(model_id))

        class DeclaringProvider(base):
            def template_info(self):
                return ModelTemplateInfo(
                    chat_template="",
                    special_tokens=frozenset([special]),
                    template_source="jinja",
                )

            def create_chat_completion(self, request, abort_event=None):
                yield FakeChunk("Hello", token_id=1)
                yield FakeChunk(f" world{special}", token_id=2)

        previous = mock_router.providers.get(model_id)
        mock_router.providers[model_id] = DeclaringProvider(model_id)
        try:
            yield model_id
        finally:
            if previous is None:
                mock_router.providers.pop(model_id, None)
            else:
                mock_router.providers[model_id] = previous

    def _body(self, model, **extra):
        return {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128, **extra}

    def test_streaming_strips(self, client, declaring_model):
        resp = client.post("/v1/messages", json=self._body(declaring_model, stream=True))
        assert resp.status_code == 200
        text = streamed_text(resp.text)
        assert self.SPECIAL not in text, f"a declared special reached the stream: {text!r}"
        assert "Hello world" in text, "the strip ate the surrounding text too"

    def test_non_streaming_strips(self, client, declaring_model):
        resp = client.post("/v1/messages", json=self._body(declaring_model))
        assert resp.status_code == 200
        text = "".join(b.get("text", "") for b in resp.json()["content"]
                       if b["type"] == "text")
        assert self.SPECIAL not in text, f"a declared special reached the body: {text!r}"
        assert "Hello world" in text, "the strip ate the surrounding text too"


class TestMidThoughtResumeIsFiledAsThinking:
    """A trailing assistant message with thinking and NO content resumes INSIDE
    the thinking block, so the model's first token continues the reasoning and
    the parser must start in thinking state. The generate route always armed
    it; this one did not until v2.0.52, and the resumed trace came back as a
    TEXT block with the closing marker visible (found live on Qwen3.5).

    Needs a marker-template provider for the same reason the class above needs
    a declaring one: FakeProvider's ``template_info() -> None`` selects the
    pass-through parser, where the start state is unobservable. Through the
    ROUTE, because the defect was the route not forwarding a request field."""

    @pytest.fixture
    def marker_model(self, mock_router):
        from helpers.mlx_mock import FakeChunk
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

        model_id = "test-mlx-model"
        base = type(mock_router.get_provider(model_id))

        class MarkerProvider(base):
            def template_info(self):
                return ModelTemplateInfo(
                    chat_template="",
                    special_tokens=frozenset(),
                    template_source="jinja",
                    has_thinking_markers=True,
                )

            def create_chat_completion(self, request, abort_event=None):
                yield FakeChunk(" rest of the thought", token_id=1)
                yield FakeChunk("</think>\n\nThe answer.", token_id=2)

        previous = mock_router.providers.get(model_id)
        mock_router.providers[model_id] = MarkerProvider(model_id)
        try:
            yield model_id
        finally:
            if previous is None:
                mock_router.providers.pop(model_id, None)
            else:
                mock_router.providers[model_id] = previous

    def test_the_resumed_trace_is_a_thinking_block(self, client, marker_model):
        resp = client.post("/v1/messages", json={
            "model": marker_model, "max_tokens": 64, "thinking": True,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "The start of the thought, then the"}]},
            ]})
        assert resp.status_code == 200
        blocks = {b["type"]: (b.get("thinking") or b.get("text")) for b in resp.json()["content"]}
        assert blocks.get("thinking", "").strip() == "rest of the thought"
        assert blocks.get("text", "").strip() == "The answer."


class TestNonStreamingPerformance:
    """What a NON-STREAMING client can actually read off `performance`.

    The Messages wire returns this object unconditionally (there is no
    `include_performance` on it since v1.79.49), so its contents are a
    contract, not an option. Three of the six declared PerformanceInfo fields
    used to arrive null here while the streaming half of the same wire filled
    them, which a consuming client hit before any test did.
    """

    def _perf(self, client):
        resp = client.post("/v1/messages", json={
            "model": "test-mlx-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        })
        assert resp.status_code == 200
        return resp.json()["performance"]

    def test_performance_is_unconditional(self, client):
        assert self._perf(client) is not None

    def test_peak_memory_reaches_the_non_streaming_response(self, client):
        """The fake's last chunk carries peak_memory=1.25; this asserts it
        survives ChunkTelemetry -> the builder -> PerformanceInfo. This was
        the field the non-streaming builder dropped."""
        assert self._perf(client)["peak_memory_gb"] == 1.25

    def test_rates_and_duration_survive_to_the_response(self, client):
        """Asserts VALUES, not non-nullness.

        The `is not None` version of this could not fail: `converters` fills
        both rates with `.get(key, 0)` and `PerformanceInfo` declares them as
        required floats, so a dropped rate arrives as `0.0` and passes. The
        fake's first chunk carries `prompt_tps=42.5` precisely so that a
        dropped rate is distinguishable from the fake's own zero.
        """
        perf = self._perf(client)
        assert perf["prompt_tps"] == 42.5
        # The fake reports a PREFILL rate and no DECODE rate, which is what
        # makes this pair worth asserting together: the measured one arrives
        # verbatim, and the unmeasured one is ABSENT rather than synthesized.
        # Until v1.79.58 this line read `generation_tps > 0` and passed only
        # because `headline_tps` invented a figure from wall-clock that the
        # engine never produced -- a derived stand-in and a measurement
        # sharing one field name, indistinguishable to a client.
        # NON-STREAMING SPELLS UNMEASURED AS EXPLICIT null, streaming spells
        # it as an absent key -- the response goes through PerformanceInfo,
        # which materialises every declared field. Same meaning, two
        # spellings, and both are "the engine did not report this".
        assert perf["generation_tps"] is None, (
            "a decode rate appeared for a run whose engine never reported one "
            "-- headline_tps used to invent one from wall-clock here"
        )
        assert perf["request_duration_ms"] is not None
        assert "total_duration_ms" not in perf, "retired in v1.79.58"

    def test_the_three_telemetry_keys_reach_this_path_too(self, client):
        """Declared on PerformanceInfo and built here as of v1.79.54.

        They rode `message_stop` from the start and were absent here by
        OMISSION -- and the model did not declare them, so they could not have
        arrived even if the builder had set them. The fake reports no KV bytes
        or queue wait, so this asserts the KEYS exist on the model rather than
        values the fake cannot produce; `test_peak_memory_reaches_the_non_
        streaming_response` is the one that follows a value end to end.
        """
        perf = self._perf(client)
        for key in ("queue_wait_ms", "cache", "speculative"):
            assert key in perf, key

    def test_phase_durations_are_measured_non_streaming_too(self, client):
        """Until v2.0.64 both were pinned ABSENT here on the claim that only
        the stream translator could time them. This mode now clocks the
        chunks as they land through a second parser instance. The fake emits
        plain text, so the content span is filled and the thinking span --
        which nothing opened -- stays null: absent means "not seen", never
        "not measured"."""
        perf = self._perf(client)
        assert isinstance(perf["content_duration_ms"], int) and perf["content_duration_ms"] >= 0
        assert perf["thinking_duration_ms"] is None


def test_retired_request_fields_are_refused_not_ignored(client):
    """A removed field must be REFUSED at the wire, not silently dropped.

    Route-level on purpose, and that is the whole lesson. v1.79.74 put this
    guard on the internal ChatRequest and tested it by constructing a
    ChatRequest directly -- green, while the real answer to a client sending
    `logprobs` on /v1/messages was a normal 200 with the key dropped, because
    nothing binds ChatRequest as a request body and pydantic's default extra
    policy is *ignore*. A model-level test passes whether or not any route
    binds the model it tests; only this shape can tell the difference.
    """
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    # `sampler` joined this list in v2.0.30: named sampler bundles were
    # removed, so a client still sending one must be told rather than have
    # it silently ignored -- the same failure `preset` is here for.
    # `show_special_tokens` joined in v2.0.38: it was a per-BROWSER display pref
    # that decided what the conversation store PERSISTED, so a client still
    # asking to KEEP them must be told rather than quietly get stripped text.
    for field, value in (("logprobs", True), ("top_logprobs", 5),
                         ("preset", "x"), ("sampler", "balanced"),
                         ("show_special_tokens", True)):
        r = client.post("/v1/messages", json={**body, field: value})
        assert r.status_code == 422, f"{field} was accepted: {r.status_code}"
        assert field.split("_")[-1] in r.text or field in r.text, \
            f"the {field} refusal does not name the field: {r.text[:200]}"

    # ...but `show_special_tokens: false` asked for exactly what the server now
    # always does, so refusing it would break the one client shape that needed
    # no change. The guard is on the VALUE, not the key's presence.
    r = client.post("/v1/messages", json={**body, "show_special_tokens": False})
    assert r.status_code != 422, \
        "show_special_tokens=false was refused, but it requests current behaviour"


def test_internal_spellings_are_refused_not_silently_dropped(client):
    """A WRONG SPELLING of a live field must 422, not be dropped.

    Distinct from the retired-field guard above: nothing here was removed.
    The capability exists under another name, so the silent drop is worse --
    the client asked for a control that is present, got a normal 200, and
    received the cascade default. There is no error and no hint anywhere.

    Route-level for the same reason the sibling test is, and this one had a
    second reason to be: it was reported by a client author (2026-09-20) who
    only discovered `enable_thinking` is not a wire field by reading the live
    schema, and whose own note was that this is the one failure a test cannot
    easily catch, BECAUSE THE REQUEST SUCCEEDS. A green integration suite is
    exactly what a client in this state sees.

    Each spelling is a habit with an origin: `max_new_tokens` is transformers'
    and is what Qwen's own reference runners use, `enable_thinking` is
    heylook's internal name (models.toml, provider configs,
    chat_template_kwargs), `system_prompt` is what the preset store calls it.
    """
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    for wrong, right, value in (("enable_thinking", "thinking", True),
                                ("max_new_tokens", "max_tokens", 256),
                                ("system_prompt", "system", "be brief")):
        r = client.post("/v1/messages", json={**body, wrong: value})
        assert r.status_code == 422, f"{wrong} was accepted: {r.status_code}"
        # The refusal has to NAME THE RIGHT SPELLING. Telling a client its
        # field is invalid without saying what to send instead leaves it
        # exactly as stuck as the silent drop did.
        assert right in r.text, \
            f"the {wrong} refusal does not name `{right}`: {r.text[:200]}"

    # The correct spellings must still be accepted -- a guard that refuses the
    # real field name would be a far worse bug than the one it fixes.
    r = client.post("/v1/messages", json={
        **body, "thinking": True, "system": "be brief"})
    assert r.status_code != 422, f"correct spellings were refused: {r.text[:200]}"


class TestUsageIsAnthropicShaped:
    """Plan W5 (owner decision): `input_tokens` is what the request
    PROCESSED and `cache_read_input_tokens` what it reused, on both modes of
    /v1/messages, taken from the provider's CacheReport through one helper
    (perf_collector.usage_counts)."""

    def _serve_a_cache_hit(self, mock_router, monkeypatch):
        from heylook_llm.providers.base import CacheReport, GenerationChunk

        provider = mock_router.get_provider("test-mlx-model")
        report = CacheReport(prompt_tokens=100, cached_tokens=60, outcome="reused")

        def generate(request, abort_event=None):
            yield GenerationChunk(text="Hi", token=1, prompt_tokens=100,
                                  generation_tokens=1, cache=report)
            yield GenerationChunk(text="!", token=2, prompt_tokens=100,
                                  generation_tokens=2, finish_reason="stop")

        monkeypatch.setattr(provider, "create_chat_completion", generate)

    def test_both_modes_report_processed_and_reused(self, client, mock_router, monkeypatch):
        self._serve_a_cache_hit(mock_router, monkeypatch)
        body = {"model": "test-mlx-model", "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}]}

        plain = client.post("/v1/messages", json=body).json()
        assert plain["usage"]["input_tokens"] == 40
        assert plain["usage"]["cache_read_input_tokens"] == 60
        assert plain["performance"]["cache"]["processed_tokens"] == 40
        assert plain["performance"]["cache"]["outcome"] == "reused"

        with client.stream("POST", "/v1/messages", json={**body, "stream": True}) as resp:
            lines = [line for line in resp.iter_lines()]
        deltas = [json.loads(line[len("data: "):]) for line in lines
                  if line.startswith("data: ") and '"message_delta"' in line]
        assert deltas, "no message_delta on the stream"
        usage = deltas[-1]["usage"]
        assert (usage["input_tokens"], usage["cache_read_input_tokens"]) == (40, 60)
