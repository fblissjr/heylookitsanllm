# tests/unit/test_per_request_parser.py
"""Phase 1 item 4: reasoning parser must be per-request, not per-provider.

The parser was built once at model load and stored on the provider; every
request called reset() and streamed through the SHARED instance. Two
interleaved streams on the same model therefore corrupted each other's
buffer state, and request B's reset() clobbered request A mid-flight (an
aborted stream's leftover buffer was only cleared when the NEXT request
happened to reset it). Fix: each request instantiates its own parser via
select_reasoning_parser(provider.template_info()); the once-per-load rationale
(Mistral's ~1000-token strip-regex compile) is preserved by caching the
compiled pattern, which is stateless and safely shared -- only the buffers
must be per-request.

Driven through the /v1/messages handlers, the inference wire (the OpenAI route
this was first pinned on went in v1.79.66).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from heylook_llm.perf_collector import PerfCollector
from heylook_llm.reasoning_parser import _compile_strip_pattern, select_reasoning_parser
from heylook_llm.schema.messages import MessageCreateRequest

from _fake_chunk import fake_chunk as _chunk


def _poisoned_shared_parser():
    """Stand-in for the provider's load-time parser with abandoned state."""
    shared = MagicMock()
    shared.process_chunk.return_value = []
    shared.flush.return_value = []
    return shared


def _provider(template_info=None):
    provider = MagicMock()
    provider.template_info.return_value = template_info
    provider._reasoning_parser = _poisoned_shared_parser()
    return provider


def _msg_request():
    return MessageCreateRequest(model="m", messages=[{"role": "user", "content": "x"}])


class TestStreamingUsesPerRequestParser:
    def _run(self, provider):
        from heylook_llm.messages_api import _stream_messages

        def gen():
            yield _chunk("hello")

        async def drain():
            return [
                part
                async for part in _stream_messages(
                    gen(), _msg_request(), "req-parser-1",
                    http_request=None, provider=provider,
                )
            ]

        with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
            return asyncio.run(drain())

    def test_shared_provider_parser_never_touched(self):
        provider = _provider()
        out = self._run(provider)
        # The request must run on its own parser instance: the shared one is
        # neither reset (clobbers a concurrent request's buffer) nor fed.
        assert provider._reasoning_parser.process_chunk.call_count == 0
        assert provider._reasoning_parser.reset.call_count == 0
        # And the stream still delivers the text via the fresh parser.
        assert any("hello" in part for part in out)


class TestNonStreamingUsesPerRequestParser:
    def test_shared_provider_parser_never_touched(self):
        from heylook_llm.messages_api import _non_stream_messages

        provider = _provider()

        def gen():
            yield _chunk("hello")

        with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
            response = asyncio.run(_non_stream_messages(
                gen(), _msg_request(), "req-parser-2",
                request_start_time=time.time(), provider=provider,
            ))

        assert provider._reasoning_parser.process_chunk.call_count == 0
        assert provider._reasoning_parser.reset.call_count == 0
        text = "".join(b.text for b in response.content if b.type == "text")
        assert text == "hello"


class TestStripPatternCache:
    def test_same_token_set_reuses_compiled_pattern(self):
        # The once-per-load rationale for the shared parser was regex-compile
        # cost. Per-request parsers stay cheap because the compiled pattern
        # (stateless) is cached and shared across instances.
        tokens = frozenset({"<|endoftext|>", "<|im_end|>"})
        assert _compile_strip_pattern(tokens) is _compile_strip_pattern(frozenset(tokens))

    def test_parser_instances_are_distinct(self):
        info = SimpleNamespace(
            special_tokens=frozenset({"<|endoftext|>"}),
            has_harmony_structure=True,
            has_thinking_markers=False,
        )
        a = select_reasoning_parser(info)
        b = select_reasoning_parser(info)
        assert a is not b  # buffers must never be shared across requests
