# tests/unit/test_messages_api.py
"""
Unit tests for the /v1/messages endpoint converters and streaming translator.

Tests cover:
1. MessageCreateRequest -> ChatRequest conversion (via converters)
2. OpenAI response dict -> MessageResponse conversion
3. StreamingEventTranslator event sequencing
"""
import pathlib
from typing import get_args

import pytest
from pydantic import ValidationError

from heylook_llm.schema.responses import StopReason
from heylook_llm.schema.content_blocks import (
    ImageBlock,
    TextBlock,
    ThinkingBlock,
)
from heylook_llm.schema.converters import (
    from_openai_response_dict,
    to_chat_request,
)
from heylook_llm.schema.messages import Message, MessageCreateRequest


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToRequestConversion:
    def test_simple_text_message(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )
        chat_req = to_chat_request(req)
        assert chat_req.model == "test"
        assert len(chat_req.messages) == 1
        assert chat_req.messages[0].role == "user"
        assert chat_req.messages[0].content == "hello"

    def test_with_system_prompt(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            system="You are helpful.",
        )
        chat_req = to_chat_request(req)
        # System prompt becomes the first message
        assert len(chat_req.messages) == 2
        assert chat_req.messages[0].role == "system"
        assert chat_req.messages[0].content == "You are helpful."

    def test_replayed_thinking_blocks_become_message_thinking(self):
        # v1.79.63: an Anthropic-shaped client replays the assistant's
        # thinking blocks in history (its own tool loop requires it); this
        # used to 422 on the first one. They become ChatMessage.thinking.
        from heylook_llm.schema.content_blocks import ThinkingBlock, TextBlock
        req = MessageCreateRequest(
            model="test",
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content=[
                    ThinkingBlock(thinking="so far"), TextBlock(text="answer")]),
                Message(role="user", content="next"),
            ],
        )
        chat_req = to_chat_request(req)
        assistant = chat_req.messages[1]
        assert assistant.thinking == "so far"
        assert [p.type for p in assistant.content] == ["text"]
        # And from raw JSON, the way a client sends it (spelled `thinking`).
        raw = MessageCreateRequest.model_validate({
            "model": "test",
            "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": [
                             {"type": "thinking", "thinking": "partial"}]}],
        })
        last = to_chat_request(raw).messages[-1]
        assert last.thinking == "partial" and last.content == []

    def test_with_thinking_enabled(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="think")],
            thinking=True,
        )
        chat_req = to_chat_request(req)
        assert chat_req.enable_thinking is True

    def test_sampler_params_forwarded(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            seed=42,
        )
        chat_req = to_chat_request(req)
        assert chat_req.temperature == 0.7
        assert chat_req.top_p == 0.9
        assert chat_req.max_tokens == 512
        assert chat_req.seed == 42

    def test_show_special_tokens_does_not_reach_the_model(self):
        """Display pref, not a sampler: it selects the response PARSER and must
        not survive the conversion into the request the provider is driven with
        (DESIGN.md §6 -- "never changes what is sent to the model")."""
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            show_special_tokens=True,
        )
        assert req.show_special_tokens is True
        dumped = to_chat_request(req).model_dump()
        assert not any("special" in k for k in dumped)

    def test_show_special_tokens_defaults_to_off(self):
        """Opt-IN: every existing client omits it and keeps the strip."""
        req = MessageCreateRequest(
            model="test", messages=[Message(role="user", content="hi")])
        assert req.show_special_tokens is False

    def test_stream_flag_forwarded(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )
        chat_req = to_chat_request(req)
        assert chat_req.stream is True

    def test_with_image_blocks(self):
        req = MessageCreateRequest(
            model="test",
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="describe"),
                        ImageBlock(
                            source_type="base64",
                            media_type="image/png",
                            data="iVBORw0KGgo...",
                        ),
                    ],
                )
            ],
        )
        chat_req = to_chat_request(req)
        # Should have one message with list content
        assert len(chat_req.messages) == 1
        content = chat_req.messages[0].content
        assert isinstance(content, list)
        assert len(content) == 2

    def test_logprobs_forwarded(self):
        req = MessageCreateRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            logprobs=True,
            top_logprobs=10,
        )
        chat_req = to_chat_request(req)
        assert chat_req.logprobs is True
        assert chat_req.top_logprobs == 10


@pytest.mark.unit
class TestFromResponseConversion:
    def test_simple_text_response(self):
        d = {
            "model": "test-model",
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "index": 0, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        resp = from_openai_response_dict(d)
        assert resp.model == "test-model"
        # Anthropic vocabulary, not the provider's OpenAI finish_reason.
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 2
        # Content blocks should have a TextBlock
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "hello"

    def test_response_with_thinking(self):
        d = {
            "model": "qwen3",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "thinking": "2+2=4, 4*10+2=42",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
        }
        resp = from_openai_response_dict(d)
        # Should have ThinkingBlock + TextBlock
        assert len(resp.content) == 2
        assert isinstance(resp.content[0], ThinkingBlock)
        assert resp.content[0].text == "2+2=4, 4*10+2=42"
        assert isinstance(resp.content[1], TextBlock)
        assert resp.content[1].text == "42"

    def test_response_length_stop_reason(self):
        d = {
            "model": "m",
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 50},
        }
        resp = from_openai_response_dict(d)
        assert resp.stop_reason == "max_tokens"

    def test_response_with_performance(self):
        d = {
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "performance": {
                "prompt_tps": 100.0,
                "generation_tps": 50.0,
                "total_duration_ms": 500,
            },
        }
        resp = from_openai_response_dict(d)
        assert resp.performance is not None
        assert resp.performance.generation_tps == 50.0

    def test_metadata_passthrough(self):
        d = {
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }
        resp = from_openai_response_dict(d, metadata={"session": "abc"})
        assert resp.metadata == {"session": "abc"}

    def test_empty_response(self):
        d = {
            "model": "m",
            "choices": [],
            "usage": {},
        }
        resp = from_openai_response_dict(d)
        assert resp.content == []
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# StreamingEventTranslator tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStreamingEventTranslator:
    @pytest.fixture
    def translator(self):
        from heylook_llm.messages_api import StreamingEventTranslator
        return StreamingEventTranslator("msg_test123", "test-model")

    def test_message_start_event(self, translator):
        event = translator.message_start_event()
        assert "event: message_start" in event
        # orjson produces compact JSON (no spaces after colons)
        assert '"type":"message_start"' in event or '"type": "message_start"' in event
        assert "test-model" in event

    def test_text_only_event_sequence(self, translator):
        """Text-only generation should produce: block_start(text), delta(s), block_stop."""
        events = translator.process_chunk("Hello world")
        events += translator.flush()

        event_text = "\n".join(events)
        assert "content_block_start" in event_text
        assert "text_delta" in event_text
        assert "content_block_stop" in event_text

    def test_thinking_then_text_sequence(self, translator):
        """Thinking + content should produce two blocks."""
        events = []
        events += translator.process_chunk("<think>")
        events += translator.process_chunk("reasoning")
        events += translator.process_chunk("</think>")
        events += translator.process_chunk("answer")
        events += translator.flush()

        # Count SSE event lines (not data payload occurrences)
        event_lines = [line for line in "\n".join(events).split("\n") if line.startswith("event: content_block_start")]
        assert len(event_lines) == 2
        event_text = "\n".join(events)
        assert "thinking_delta" in event_text
        assert "text_delta" in event_text

    def test_message_delta_event(self, translator):
        # Simulate some generation
        translator.process_chunk("hello")
        translator.content_tokens = 5

        event = translator.message_delta_event()
        assert "message_delta" in event
        assert '"stop_reason"' in event

    def test_message_stop_event(self, translator):
        event = translator.message_stop_event()
        assert "message_stop" in event
        # The translator's clock starts with the stream, so the span it can
        # always report is the GENERATION one. `total_duration_ms` was retired
        # in v1.79.58 for meaning request-arrival in the other mode.
        assert "generation_duration_ms" in event

    def test_block_index_increments(self, translator):
        """Each new block should get an incremented index."""
        import re

        # Force two blocks by simulating thinking then text
        events = []
        events += translator.process_chunk("<think>")
        events += translator.process_chunk("think")
        events += translator.process_chunk("</think>")
        events += translator.process_chunk("text")
        events += translator.flush()

        # Check that we see index 0 and index 1
        # orjson may produce "index":0 (no space) or "index": 0
        indices_seen = set()
        for e in events:
            if '"index"' in e:
                match = re.search(r'"index":\s*(\d+)', e)
                if match:
                    indices_seen.add(int(match.group(1)))

        assert 0 in indices_seen
        assert 1 in indices_seen

    def test_empty_chunks_ignored(self, translator):
        events = translator.process_chunk("")
        assert events == []

    def test_token_counting(self, translator):
        translator.process_chunk("<think>")
        translator.process_chunk("a")
        translator.process_chunk("b")
        translator.process_chunk("</think>")
        translator.process_chunk("c")
        translator.flush()

        # "a" and "b" are thinking tokens, "c" is content
        # The text parser may buffer, so check total >= expected
        assert translator.thinking_tokens >= 1
        assert translator.content_tokens >= 1


# ---------------------------------------------------------------------------
# Anthropic wire conformance (v1.79.39)
#
# /v1/messages advertises itself as Messages-shaped, and three payloads were
# not: the image block was flat where Anthropic nests it under `source`, the
# thinking block and its delta named the field `text` where Anthropic names
# it `thinking`, and `stop_reason` carried the provider's OpenAI
# finish_reason vocabulary ("stop"/"length") straight onto the wire.
# ---------------------------------------------------------------------------


class TestAnthropicImageBlockShape:
    """Both spellings are accepted and normalize to one internal shape."""

    ANTHROPIC = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
    }
    HEYLOOK = {
        "type": "image",
        "source_type": "base64",
        "media_type": "image/jpeg",
        "data": "AAAA",
    }

    def _req(self, block):
        return MessageCreateRequest(
            model="m", messages=[{"role": "user", "content": [block]}]
        )

    def test_anthropic_nested_source_is_accepted(self):
        # This was a 422 before v1.79.39 -- a correctly-formed Messages
        # request rejected by an endpoint claiming to speak Messages.
        block = self._req(self.ANTHROPIC).messages[0].content[0]
        assert block.source_type == "base64"
        assert block.media_type == "image/jpeg"
        assert block.data == "AAAA"

    def test_flat_spelling_still_accepted(self):
        block = self._req(self.HEYLOOK).messages[0].content[0]
        assert (block.source_type, block.data) == ("base64", "AAAA")

    def test_both_produce_identical_wire_parts(self):
        # The point of normalizing in the schema: nothing downstream needs to
        # know which spelling arrived.
        assert (
            to_chat_request(self._req(self.ANTHROPIC)).messages[0].content
            == to_chat_request(self._req(self.HEYLOOK)).messages[0].content
        )

    def test_nested_url_source(self):
        block = self._req({
            "type": "image", "source": {"type": "url", "url": "https://e/x.png"},
        }).messages[0].content[0]
        assert (block.source_type, block.url) == ("url", "https://e/x.png")

    def test_nested_audio_source(self):
        block = MessageCreateRequest(model="m", messages=[{"role": "user", "content": [
            {"type": "audio",
             "source": {"type": "base64", "media_type": "audio/wav", "data": "UklG"}},
        ]}]).messages[0].content[0]
        assert (block.source_type, block.media_type, block.data) == (
            "base64", "audio/wav", "UklG",
        )

    # A generated client serializes absent optionals as explicit nulls rather
    # than omitting them -- ordinary behaviour for the Java/Go/TS generators
    # and for pydantic's own model_dump() without exclude_none. Since
    # `source_type` is Optional in the published schema, that client sends
    # nulls BESIDE the nested object, which is the exact spelling
    # docs/api_integration.md tells integrators to prefer. Both the gate and
    # the assignment below it tested key PRESENCE, so the nulls survived and
    # the block 422'd. Two tests: one per mechanism, because fixing only the
    # gate leaves this passing at the discriminator and failing at the data.
    NULLED = {
        "type": "image",
        "source_type": None, "media_type": None, "data": None, "url": None,
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
    }

    def test_explicit_nulls_beside_a_nested_source_still_flatten(self):
        block = self._req(self.NULLED).messages[0].content[0]
        assert block.source_type == "base64", "the presence gate short-circuited"
        assert (block.media_type, block.data) == ("image/jpeg", "AAAA"), (
            "setdefault left the explicit nulls in place"
        )

    def test_nulled_nested_url_source_flattens(self):
        block = self._req({
            "type": "image", "source_type": None, "url": None, "data": None,
            "source": {"type": "url", "url": "https://e/x.png"},
        }).messages[0].content[0]
        assert (block.source_type, block.url) == ("url", "https://e/x.png")

    # "Absent" has THREE spellings and an earlier version of this test pinned
    # one of them, while reading as a guarantee about the class. The other two
    # were accepted and then silently dropped in conversion -- a 200, the text
    # parts intact, and a confident answer about an image the model never saw.
    # Parametrized so adding a spelling is one line rather than a new test that
    # nobody writes.
    @pytest.mark.parametrize("name,block", [
        ("no source in any spelling",
         {"type": "image", "source_type": None}),
        ("flat discriminator, no payload",
         {"type": "image", "source_type": "base64"}),
        ("flat url discriminator, no url",
         {"type": "image", "source_type": "url"}),
        ("nested type, no data",
         {"type": "image", "source_type": None,
          "source": {"type": "base64", "media_type": "image/jpeg"}}),
        ("nested type, no data, no nulls",
         {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg"}}),
        ("nested url type, no url",
         {"type": "image", "source": {"type": "url"}}),
    ])
    def test_a_block_that_would_be_dropped_is_rejected_instead(self, name, block):
        # converters.py requires `source_type == "base64" and data` else `url`,
        # and otherwise `continue`s -- so a block that validates without a
        # payload does not reach the model and nothing says so. A 422 naming
        # the missing field is strictly better on a vision request.
        with pytest.raises(ValidationError):
            self._req(block)

    def test_a_flat_discriminator_does_not_suppress_the_nested_payload(self):
        # The whole-block early return was the same presence-vs-value mistake
        # one level up: a client that sets `source_type` and leaves the payload
        # to `source` had the type resolved and the image dropped. Filling is
        # PER FIELD, which is what the spec §4 contract promises.
        block = self._req({
            "type": "image", "source_type": "base64", "data": None,
            "media_type": None,
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
        }).messages[0].content[0]
        assert (block.media_type, block.data) == ("image/jpeg", "AAAA")

    def test_flat_values_still_win_over_a_nested_source(self):
        # The one case the nested object is ignored wholesale: the two
        # spellings DISAGREE about the kind of source. Mixing a base64 payload
        # into a block declaring itself a url would build something the caller
        # never described, so the flat spelling wins and nothing is merged.
        block = self._req({
            "type": "image", "source_type": "url", "url": "https://flat/a.png",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "ZZ"},
        }).messages[0].content[0]
        assert (block.source_type, block.url) == ("url", "https://flat/a.png")
        assert block.data is None

    def test_every_accepted_spelling_actually_reaches_the_wire(self):
        # The oracle the other tests lack: validation succeeding is not the
        # property anyone cares about -- the image ARRIVING is. Each spelling
        # must produce an image part, not just a valid block.
        spellings = [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": "AAAA"}},
            {"type": "image", "source_type": None, "media_type": None, "data": None,
             "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
            {"type": "image", "source_type": "base64",
             "media_type": "image/jpeg", "data": "AAAA"},
            {"type": "image", "source_type": None,
             "source": {"type": "url", "url": "https://e/x.png"}},
        ]
        for block in spellings:
            parts = to_chat_request(self._req(block)).messages[0].content
            kinds = [getattr(p, "type", None) for p in parts]
            assert "image_url" in kinds, f"{block} validated but never reached the wire"


class TestThinkingBlockCarriesBothFields:
    """Anthropic names it `thinking`; v3 reads `text`. Both are populated."""

    def test_constructed_from_text(self):
        b = ThinkingBlock(text="abc")
        assert b.thinking == "abc" and b.text == "abc"

    def test_constructed_from_thinking(self):
        b = ThinkingBlock(thinking="xyz")
        assert b.text == "xyz" and b.thinking == "xyz"

    def test_response_conversion_populates_both(self):
        resp = from_openai_response_dict({
            "model": "m",
            "choices": [{"finish_reason": "stop",
                         "message": {"content": "hi", "thinking": "hmm"}}],
            "usage": {},
        })
        block = resp.content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.thinking == "hmm" and block.text == "hmm"


class TestStopReasonVocabulary:
    """The provider speaks OpenAI; the Messages wire must not."""

    @pytest.mark.parametrize("finish_reason,expected", [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        (None, "end_turn"),
        # An unrecognised provider value must not reach the wire verbatim --
        # that is the whole defect being fixed.
        ("something_new", "end_turn"),
        # Already-Anthropic values pass through unchanged.
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
    ])
    def test_mapping(self, finish_reason, expected):
        from heylook_llm.schema.converters import to_stop_reason
        assert to_stop_reason(finish_reason) == expected

    def test_no_openai_vocabulary_survives(self):
        from heylook_llm.schema.converters import STOP_REASON_FROM_FINISH_REASON
        assert "stop" not in STOP_REASON_FROM_FINISH_REASON.values()
        assert "length" not in STOP_REASON_FROM_FINISH_REASON.values()


class TestStreamingConformance:
    """The SSE payloads a Messages client actually reads."""

    def _events(self, text):
        from heylook_llm.messages_api import StreamingEventTranslator
        t = StreamingEventTranslator("msg_1", "m")
        out = t.process_chunk(text) + t.flush()
        return "".join(out)

    def test_thinking_delta_carries_thinking_field(self):
        blob = self._events("<think>reasoning</think>answer")
        assert '"thinking_delta"' in blob
        # Anthropic's field name must be present, and v3's must survive.
        assert '"thinking": "reasoning"' in blob or '"thinking":"reasoning"' in blob
        assert '"text": "reasoning"' in blob or '"text":"reasoning"' in blob

    def test_text_delta_unchanged(self):
        blob = self._events("plain answer")
        assert '"text_delta"' in blob
        assert '"thinking_delta"' not in blob

    def test_message_delta_uses_anthropic_stop_reason(self):
        from heylook_llm.messages_api import StreamingEventTranslator
        t = StreamingEventTranslator("msg_1", "m")
        # Default with no provider signal is a natural stop.
        assert '"end_turn"' in t.message_delta_event()


# Repo-root relative, not CWD relative: this file is
# tests/unit/test_messages_api.py, so parents[2] is the repo root. Resolving
# against the process CWD made the class below raise FileNotFoundError when
# pytest ran from anywhere else.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _messages_grammar_routes():
    """Every module that drives StreamingEventTranslator's stop_reason.

    DERIVED, not hand-listed. A literal two-entry list is the hand-copied
    constant CLAUDE.md calls "a defect with a delay": the premise of the class
    below is "there are two routes on this grammar", so a third surface growing
    a Messages stream would go unwatched by the very test written to stop
    routes diverging. Driving the translator IS the membership rule.
    """
    src = _REPO_ROOT / "src" / "heylook_llm"
    found = []
    for p in sorted(src.rglob("*.py")):
        text = p.read_text()
        if "StreamingEventTranslator" in text and ".stop_reason" in text:
            found.append(str(p.relative_to(_REPO_ROOT)))
    assert len(found) >= 2, (
        f"expected at least the two known Messages-grammar routes, found "
        f"{found} -- if StreamingEventTranslator was renamed, fix this rule "
        "rather than pinning a list"
    )
    return found


MESSAGES_GRAMMAR_ROUTES = _messages_grammar_routes()


class TestStopReasonHasOneMapper:
    """Both Messages-grammar routes must rename the provider's vocabulary.

    This is the defect that actually shipped: /v1/messages and
    /v1/conversations/{id}/generate share StreamingEventTranslator, so the
    block deltas agree by construction -- but each assigned
    `translator.stop_reason` from the provider itself, and fixing one in
    v1.79.39 left the other emitting "length" for one more commit. Per-path
    behavioural tests cannot see a disagreement between paths; this asserts
    the shared mapper is the only writer.
    """

    def _source(self, path):
        return (_REPO_ROOT / path).read_text()

    def _assignments(self, path):
        import re
        # Strip comments and docstring-ish lines first: the raw regex matched
        # prose mentioning the field and would have failed on a comment.
        lines = [ln for ln in self._source(path).splitlines()
                 if not ln.lstrip().startswith("#")]
        # Also match ANNOTATED assignment. messages_api.py:87 already spells
        # it `self.stop_reason: str = "end_turn"`, so respelling a bad
        # default that way evaded the check entirely -- the pattern was
        # not hypothetical, it was one line away from live.
        # `=(?!=)` -- a COMPARISON is not a write. Without the lookahead,
        # `if translator.stop_reason == "end_turn":` was read as an assignment
        # of `= "end_turn"):`, which matches neither the literal branch nor
        # the mapper branch and failed. Guarding a write by first reading the
        # field is the natural way to express "only override the default",
        # so the check has to tell reads from writes or it forbids the shape
        # rather than the defect.
        return re.findall(r"\.stop_reason(?:\s*:\s*[^=\n]+)?\s*=(?!=)\s*(.+)",
                          "\n".join(lines))

    @pytest.mark.parametrize("path", MESSAGES_GRAMMAR_ROUTES)
    def test_every_stop_reason_write_goes_through_the_mapper(self, path):
        writes = self._assignments(path)
        assert writes, f"{path} no longer writes stop_reason -- update this test"
        for rhs in writes:
            rhs = rhs.strip()
            # A literal from the shared vocabulary is fine (an explicit
            # end-state, e.g. an abort); a bare provider value is not.
            #
            # The literal must be IN that vocabulary, not merely be a literal.
            # Exempting every quoted string waved through the exact bug this
            # class exists to catch: `stop_reason = "length"` is OpenAI's
            # finish_reason vocabulary reaching the Messages wire verbatim, and
            # it passed -- it starts with a quote. Checking membership costs
            # nothing and closes the literal-shaped path back to the defect.
            if rhs.startswith('"') or rhs.startswith("'"):
                literal = rhs[1:].split(rhs[0])[0]
                assert literal in get_args(StopReason), (
                    f"{path} assigns stop_reason = {literal!r}, which is not in "
                    f"the Messages vocabulary {get_args(StopReason)} -- a "
                    "provider finish_reason spelled as a literal is still a "
                    "provider finish_reason on the wire"
                )
                continue
            assert "to_stop_reason(" in rhs, (
                f"{path} assigns stop_reason from {rhs!r} without "
                "to_stop_reason() -- a provider finish_reason would reach the "
                "Messages wire verbatim"
            )

    @pytest.mark.parametrize("path", MESSAGES_GRAMMAR_ROUTES)
    def test_route_imports_the_shared_mapper(self, path):
        # Assert the IMPORT, not the mere presence of the name: the call site
        # guarantees the substring, so a substring check passed even with the
        # import deleted -- which is a NameError at runtime. A comment
        # mentioning the name satisfied it too.
        src = self._source(path)
        assert "from heylook_llm.schema.converters import" in src
        assert any(
            "to_stop_reason" in ln
            for ln in src.splitlines()
            if ln.startswith(("from ", "    to_stop_reason", "import "))
        ), f"{path} calls to_stop_reason without importing it"
