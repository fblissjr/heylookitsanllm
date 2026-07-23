# tests/unit/test_finish_reason.py
"""finish_reason must distinguish a natural stop from a truncated one.

Both non-streaming paths (chat/completions and messages) hardcoded
``"stop"``, so a response cut off by ``max_tokens`` was indistinguishable
from one the model chose to end -- clients could not tell a complete answer
from a truncated one, and tests/eval had to infer it by comparing
completion_tokens against max_tokens. mlx-lm reports the real reason on the
chunk; the streaming path already forwarded it.
"""

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

from heylook_llm.config import ChatRequest
from heylook_llm.perf_collector import ChunkTelemetry, PerfCollector

from _fake_chunk import fake_chunk as _chunk


def _provider():
    provider = MagicMock()
    provider._template_info = None
    return provider


def _run_non_stream(chunks):
    from heylook_llm.api import non_stream_response

    router = MagicMock()
    router.log_level = logging.INFO

    def gen():
        yield from chunks

    with patch("heylook_llm.api.get_perf_collector", return_value=PerfCollector()):
        return asyncio.run(non_stream_response(
            gen(),
            ChatRequest(model="m", messages=[{"role": "user", "content": "x"}]),
            router, "req-finish-1",
            request_start_time=time.time(), provider=_provider(),
        ))


class TestChunkTelemetryCarriesFinishReason:
    """The scrape lives in absorb() -- one place, per CLAUDE.md, so the four
    consume loops cannot drift apart."""

    def test_absorbs_finish_reason(self):
        t = ChunkTelemetry()
        t.absorb(_chunk("hi", finish_reason=None))
        assert t.finish_reason is None
        t.absorb(_chunk("", finish_reason="length"))
        assert t.finish_reason == "length"

    def test_later_none_does_not_clear_a_seen_reason(self):
        # the reason arrives on the FINAL chunk; a trailing empty chunk
        # without one must not erase it
        t = ChunkTelemetry()
        t.absorb(_chunk("", finish_reason="length"))
        t.absorb(_chunk("", finish_reason=None))
        assert t.finish_reason == "length"


class TestNonStreamingFinishReason:
    def test_budget_exhausted_reports_length(self):
        response = _run_non_stream([
            _chunk("a lot of "),
            _chunk("text", finish_reason="length"),
        ])
        assert response.choices[0]["finish_reason"] == "length"

    def test_natural_stop_reports_stop(self):
        response = _run_non_stream([
            _chunk("done", finish_reason="stop"),
        ])
        assert response.choices[0]["finish_reason"] == "stop"

    def test_missing_reason_defaults_to_stop(self):
        # mlx-lm may report nothing; "stop" stays the safe default
        response = _run_non_stream([_chunk("hello")])
        assert response.choices[0]["finish_reason"] == "stop"
