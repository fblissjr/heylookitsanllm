# tests/unit/test_messages_api.py
"""
Unit tests for the /v1/messages endpoint converters and streaming translator.

Tests cover:
1. MessageCreateRequest -> ChatRequest conversion (via converters)
2. OpenAI response dict -> MessageResponse conversion
3. StreamingEventTranslator event sequencing
"""
import pytest

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
        assert resp.stop_reason == "stop"
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
        assert resp.stop_reason == "length"

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
        assert resp.stop_reason == "stop"


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
        assert "total_duration_ms" in event

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
