# tests/unit/test_abort_stop_reason.py
"""A cancelled run must not claim the model finished (v1.79.46).

This rule has now been broken on THREE routes in three consecutive releases,
each time the same way: a path becomes cancellable, the generator simply ends
early, and the end-state falls through to its default -- `end_turn` on the
Messages wire, `"stop"` on the OpenAI one -- both of which positively assert
the model reached its own end. v1.79.40 fixed `conversation_generate_api`,
v1.79.44 fixed `/v1/messages`, and .44 introduced it on
`/v1/chat/completions` in the very commit that made that path cancellable.

Why THIS file exists rather than another meta-test. v1.79.40's response was
`TestStopReasonHasOneMapper`, which checks HOW `stop_reason` is written -- and
it was green through both later regressions, because a wrong VALUE written the
right way satisfies it, and because the non-streaming overrides set
`finish_reason` and are outside its regex entirely. Deleting either override
today leaves that whole suite green. These are behavioural: they drive an
aborted generation and read what a client would receive.

The oracle is deliberately "not the value that means completed", not an exact
string -- the honest value differs per wire (`max_tokens` vs `length`) and the
claim is about what the client can conclude, not about a spelling.
"""

import logging
from types import SimpleNamespace

import pytest

from heylook_llm.providers.abort import AbortEvent

# non_stream_response reads router.log_level for its debug dump; nothing else
# on the router is touched on this path.
_ROUTER = SimpleNamespace(log_level=logging.INFO)


def _finish_reason(result):
    """non_stream_response returns a ChatCompletionResponse OR a dict depending
    on the request (the route branches on `isinstance(result, dict)`), so read
    through both rather than pinning today's shape."""
    choices = result["choices"] if isinstance(result, dict) else result.choices
    choice = choices[0]
    return (choice["finish_reason"] if isinstance(choice, dict)
            else choice.finish_reason)


class _Chunk:
    """Minimal provider chunk. Real generation stops mid-stream on abort, so
    the last chunk carries NO finish_reason -- that absence is the whole
    mechanism, and a fixture that supplied one would test nothing."""

    def __init__(self, text, finish_reason=None):
        self.text = text
        self.finish_reason = finish_reason
        self.thinking = None
        self.token = None
        self.logprobs = None


def _aborted_generator(abort_event, n=3):
    """Yields a few chunks, then stops the way an aborted decode loop does:
    `generation_core` just breaks, so the stream simply ends."""
    def gen():
        for i in range(n):
            yield _Chunk(f"tok{i} ")
        abort_event.set()   # cancelled between tokens
    return gen()


@pytest.mark.unit
class TestMessagesWireReportsCancellationHonestly:
    @pytest.mark.asyncio
    async def test_non_streaming_abort_does_not_say_end_turn(self):
        from heylook_llm.messages_api import _non_stream_messages
        from heylook_llm.schema.messages import MessageCreateRequest

        abort = AbortEvent()
        req = MessageCreateRequest.model_validate(
            {"model": "m", "max_tokens": 64,
             "messages": [{"role": "user", "content": "hi"}]})
        resp = await _non_stream_messages(
            _aborted_generator(abort), req, "rid", 0.0, abort_event=abort)
        assert resp.stop_reason != "end_turn", (
            "a cancelled run reported the model's own end")
        assert resp.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_a_completed_run_still_says_end_turn(self):
        """The override must not fire on every request -- that would make the
        honest value meaningless by making it universal."""
        from heylook_llm.messages_api import _non_stream_messages
        from heylook_llm.schema.messages import MessageCreateRequest

        abort = AbortEvent()  # never set
        req = MessageCreateRequest.model_validate(
            {"model": "m", "max_tokens": 64,
             "messages": [{"role": "user", "content": "hi"}]})

        def gen():
            yield _Chunk("done")

        resp = await _non_stream_messages(
            gen(), req, "rid", 0.0, abort_event=abort)
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_an_engine_reported_reason_outranks_the_abort_override(self):
        """The guard is on the DEFAULT, not unconditional: if a DELETE lands
        between the last token and the response build, a real `length` from the
        engine is the more specific truth and keeps priority. The non-streaming
        half overwrote unconditionally until v1.79.46 while its streaming
        sibling guarded -- two halves of one rule, quietly disagreeing."""
        from heylook_llm.messages_api import _non_stream_messages
        from heylook_llm.schema.messages import MessageCreateRequest

        abort = AbortEvent()
        req = MessageCreateRequest.model_validate(
            {"model": "m", "max_tokens": 64,
             "messages": [{"role": "user", "content": "hi"}]})

        def gen():
            yield _Chunk("a", finish_reason="stop_sequence")
            abort.set()

        resp = await _non_stream_messages(
            gen(), req, "rid", 0.0, abort_event=abort)
        assert resp.stop_reason == "stop_sequence"


@pytest.mark.unit
class TestOpenAIWireReportsCancellationHonestly:
    """`/v1/chat/completions` became cancellable in v1.79.44 and kept
    `finish_reason: "stop"` for a cancelled run -- verified live against a
    running server before this test was written."""

    @pytest.mark.asyncio
    async def test_non_streaming_abort_does_not_say_stop(self):
        from heylook_llm.api import non_stream_response
        from heylook_llm.config import ChatRequest

        abort = AbortEvent()
        req = ChatRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        result = await non_stream_response(
            _aborted_generator(abort), req, _ROUTER, "rid", 0.0, abort_event=abort)
        finish = _finish_reason(result)
        assert finish != "stop", "a cancelled run claimed the model finished"
        assert finish == "length"

    @pytest.mark.asyncio
    async def test_a_completed_run_still_says_stop(self):
        from heylook_llm.api import non_stream_response
        from heylook_llm.config import ChatRequest

        abort = AbortEvent()  # never set
        req = ChatRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]})

        def gen():
            yield _Chunk("done")

        result = await non_stream_response(
            gen(), req, _ROUTER, "rid", 0.0, abort_event=abort)
        assert _finish_reason(result) == "stop"
