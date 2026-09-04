# tests/unit/test_finish_reason.py
"""The stop reason must distinguish a natural stop from a truncated one.

The non-streaming paths hardcoded the engine's ``"stop"``, so a response cut
off by ``max_tokens`` was indistinguishable from one the model chose to end --
clients could not tell a complete answer from a truncated one, and tests/eval
had to infer it by comparing completion_tokens against max_tokens. mlx-lm
reports the real reason on the chunk; the streaming path already forwarded it.

Pinned on the /v1/messages non-streaming handler, where the engine's
finish_reason is renamed to Anthropic's stop_reason at the boundary.
"""

import asyncio
import time
from unittest.mock import patch

from heylook_llm.perf_collector import ChunkTelemetry, PerfCollector
from heylook_llm.schema.messages import MessageCreateRequest

from _fake_chunk import fake_chunk as _chunk


def _run_non_stream(chunks):
    from heylook_llm.messages_api import _non_stream_messages

    def gen():
        yield from chunks

    with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
        return asyncio.run(_non_stream_messages(
            gen(),
            MessageCreateRequest(model="m", messages=[{"role": "user", "content": "x"}]),
            "req-finish-1",
            request_start_time=time.time(),
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


class TestNonStreamingStopReason:
    def test_budget_exhausted_reports_max_tokens(self):
        response = _run_non_stream([
            _chunk("a lot of "),
            _chunk("text", finish_reason="length"),
        ])
        assert response.stop_reason == "max_tokens"

    def test_natural_stop_reports_end_turn(self):
        response = _run_non_stream([
            _chunk("done", finish_reason="stop"),
        ])
        assert response.stop_reason == "end_turn"

    def test_missing_reason_defaults_to_end_turn(self):
        # mlx-lm may report nothing; the model's own end stays the default
        # (an ABORTED run is the exception, pinned in test_abort_stop_reason)
        response = _run_non_stream([_chunk("hello")])
        assert response.stop_reason == "end_turn"
