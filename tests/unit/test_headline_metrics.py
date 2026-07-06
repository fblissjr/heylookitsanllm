# tests/unit/test_headline_metrics.py
"""Phase 1 item 2: honest headline metrics.

The 2026-07-06 measurement audit found the recorded perf numbers
untrustworthy:
- mlx-lm's native per-chunk prompt_tps/generation_tps (computed tightly
  around real prefill/decode) were never read anywhere in src/,
- headline tok/s and TTFT silently included FIFO queue-wait,
- /v1/messages non-streaming prompt_tps divided prompt tokens by
  whole-request elapsed time,
- hourly trends averaged failed requests in at 0.0 tok/s.

These tests pin the fixed behavior: native engine numbers are the recorded
generation numbers, queue-wait lives ONLY in its own field, and trend
averages are success-only.
"""

import asyncio
import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

from heylook_llm.perf_collector import (
    PerfCollector,
    RequestEvent,
    headline_tps,
)

from _fake_chunk import fake_chunk as _chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_event_kwargs(**overrides) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        timestamp=time.time(),
        model="m",
        success=True,
        total_ms=1000.0,
        queue_ms=10.0,
        model_load_ms=0.0,
        image_processing_ms=0.0,
        token_generation_ms=950.0,
        first_token_ms=50.0,
        prompt_tokens=20,
        completion_tokens=100,
        tokens_per_second=100.0,
        had_images=False,
        was_streaming=True,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# headline_tps helper
# ---------------------------------------------------------------------------

class TestHeadlineTps:
    def test_native_engine_number_wins(self):
        # Wall-clock says 1 tok/s; the engine measured 88.2 around the decode
        # loop. The engine is right.
        assert headline_tps(88.2, tokens=100, elapsed_s=100.0, queue_wait_ms=0.0) == 88.2

    def test_fallback_excludes_queue_wait(self):
        # 100 tokens over 10s elapsed, but 5s of that was FIFO queue wait:
        # honest rate is 20 tok/s, not 10.
        assert headline_tps(0.0, tokens=100, elapsed_s=10.0, queue_wait_ms=5000.0) == 20.0

    def test_zero_when_no_data(self):
        assert headline_tps(0.0, tokens=0, elapsed_s=1.0, queue_wait_ms=0.0) == 0.0

    def test_zero_when_queue_wait_swallows_elapsed(self):
        # Clock skew / rounding: queue wait >= elapsed must not produce a
        # negative or infinite rate.
        assert headline_tps(0.0, tokens=10, elapsed_s=1.0, queue_wait_ms=2000.0) == 0.0


# ---------------------------------------------------------------------------
# RequestEvent carries native prompt_tps
# ---------------------------------------------------------------------------

class TestRequestEventPromptTps:
    def test_field_exists_with_backcompat_default(self):
        e = RequestEvent(**_base_event_kwargs())
        assert e.prompt_tps == 0.0

    def test_field_records_value(self):
        e = RequestEvent(**_base_event_kwargs(), prompt_tps=456.7)
        assert e.prompt_tps == 456.7


# ---------------------------------------------------------------------------
# Trends aggregate success-only
# ---------------------------------------------------------------------------

class TestTrendsSuccessOnly:
    def test_failed_events_excluded_from_averages(self):
        c = PerfCollector()
        now = (time.time() // 3600) * 3600 + 1800  # mid-hour, no boundary flake
        c.record_request(RequestEvent(**_base_event_kwargs(
            timestamp=now, total_ms=400.0, tokens_per_second=100.0, success=True)))
        c.record_request(RequestEvent(**_base_event_kwargs(
            timestamp=now, total_ms=5.0, tokens_per_second=0.0, success=False)))

        trends = c.build_profile("1h")["trends"]
        assert len(trends) == 1
        # Averages must reflect the one successful request only; the failure
        # still counts in requests/errors.
        assert trends[0]["tokens_per_second"] == 100.0
        assert trends[0]["response_time_ms"] == 400.0
        assert trends[0]["requests"] == 2
        assert trends[0]["errors"] == 1

    def test_all_failed_hour_reports_zero(self):
        c = PerfCollector()
        now = (time.time() // 3600) * 3600 + 1800
        c.record_request(RequestEvent(**_base_event_kwargs(
            timestamp=now, tokens_per_second=0.0, success=False)))

        trends = c.build_profile("1h")["trends"]
        assert trends[0]["tokens_per_second"] == 0.0
        assert trends[0]["response_time_ms"] == 0.0
        assert trends[0]["errors"] == 1


# ---------------------------------------------------------------------------
# OpenAI streaming path records native numbers
# ---------------------------------------------------------------------------

class TestOpenAIStreamingRecordsNativeTps:
    def _run(self, chunks):
        from heylook_llm.api import stream_response_generator_async
        from heylook_llm.config import ChatRequest

        chat_request = ChatRequest(
            model="test-model",
            messages=[{"role": "user", "content": "x"}],
        )
        router = MagicMock()
        router.log_level = logging.INFO
        perf_ctx = {
            "request_start_time": time.time(),
            "provider_get_ms": 5.0,
            "image_resize_ms": 0.0,
            "had_images": False,
        }
        collector = PerfCollector()

        def gen():
            yield from chunks

        async def drain():
            out = []
            async for part in stream_response_generator_async(
                gen(), chat_request, router, "req-test-123",
                http_request=None, provider=None, perf_ctx=perf_ctx,
            ):
                out.append(part)
            return out

        with patch("heylook_llm.api.get_perf_collector", return_value=collector):
            asyncio.run(drain())
        assert len(collector._events) == 1
        return collector._events[0]

    def test_recorded_tps_is_native_generation_tps(self):
        event = self._run([_chunk(generation_tps=87.6, prompt_tps=123.4)])
        assert event.tokens_per_second == 87.6
        assert event.prompt_tps == 123.4

    def test_ttft_excludes_queue_wait(self):
        # Queue wait far exceeds actual wall time -> honest TTFT clamps to 0
        # instead of reporting queue pressure as model latency.
        event = self._run([_chunk(queue_wait_ms=1_000_000.0)])
        assert event.first_token_ms == 0.0
        assert event.queue_wait_ms == 1_000_000.0


# ---------------------------------------------------------------------------
# Messages non-streaming path: formula fix + native numbers
# ---------------------------------------------------------------------------

class TestMessagesNonStreamNativeTps:
    def _run(self, chunks):
        from heylook_llm.messages_api import _non_stream_messages
        from heylook_llm.schema.messages import MessageCreateRequest

        msg_request = MessageCreateRequest(
            model="test-model",
            messages=[{"role": "user", "content": "x"}],
        )
        perf_ctx = {"provider_get_ms": 5.0, "had_images": False}
        collector = PerfCollector()

        def gen():
            yield from chunks

        with patch("heylook_llm.messages_api.get_perf_collector", return_value=collector):
            response = asyncio.run(_non_stream_messages(
                gen(), msg_request, "req-test-456",
                request_start_time=time.time() - 10.0,  # 10s elapsed
                perf_ctx=perf_ctx,
            ))
        assert len(collector._events) == 1
        return response, collector._events[0]

    def test_performance_prompt_tps_is_native_not_elapsed_division(self):
        # Old bug: prompt_tps = prompt_tokens / whole-request elapsed
        # (10 tokens / 10s = 1.0). Native measurement is 123.4.
        response, event = self._run([_chunk(prompt_tokens=10, prompt_tps=123.4)])
        assert response.performance is not None
        assert response.performance.prompt_tps == 123.4
        assert event.prompt_tps == 123.4

    def test_recorded_tps_is_native_generation_tps(self):
        response, event = self._run([_chunk(generation_tps=87.6)])
        assert event.tokens_per_second == 87.6
        assert response.performance is not None
        assert response.performance.generation_tps == 87.6


# ---------------------------------------------------------------------------
# Messages streaming path records native numbers
# ---------------------------------------------------------------------------

class TestMessagesStreamingRecordsNativeTps:
    def _run(self, chunks):
        from heylook_llm.messages_api import _stream_messages
        from heylook_llm.schema.messages import MessageCreateRequest

        msg_request = MessageCreateRequest(
            model="test-model",
            messages=[{"role": "user", "content": "x"}],
        )
        perf_ctx = {
            "request_start_time": time.time(),
            "provider_get_ms": 5.0,
            "had_images": False,
        }
        collector = PerfCollector()

        def gen():
            yield from chunks

        async def drain():
            out = []
            async for part in _stream_messages(
                gen(), msg_request, "req-test-789",
                http_request=None, provider=None, perf_ctx=perf_ctx,
                abort_event=None,
            ):
                out.append(part)
            return out

        with patch("heylook_llm.messages_api.get_perf_collector", return_value=collector):
            asyncio.run(drain())
        assert len(collector._events) == 1
        return collector._events[0]

    def test_recorded_tps_is_native_generation_tps(self):
        event = self._run([_chunk(generation_tps=87.6, prompt_tps=123.4)])
        assert event.tokens_per_second == 87.6
        assert event.prompt_tps == 123.4

    def test_ttft_excludes_queue_wait(self):
        event = self._run([_chunk(queue_wait_ms=1_000_000.0)])
        assert event.first_token_ms == 0.0
        assert event.queue_wait_ms == 1_000_000.0
