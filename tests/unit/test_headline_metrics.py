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
averages are success-only. Both /v1/messages modes are driven; the OpenAI
streaming path that was pinned beside them went in v1.79.66.
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

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
        model_load_ms=0.0,
        token_generation_ms=950.0,
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

# native rate wins; the wall-clock fallback excludes FIFO queue wait; never
# negative or infinite.
# - native_wins: wall-clock says 1 tok/s; the engine measured 88.2 around the
#   decode loop. The engine is right.
# - fallback_excludes_queue_wait: 100 tokens over 10s elapsed, 5s of it queue
#   wait: the honest rate is 20 tok/s, not 10.
# - queue_wait_swallows_elapsed: clock skew / rounding (queue wait >= elapsed)
#   must not produce a negative or infinite rate.
@pytest.mark.parametrize("native, tokens, elapsed_s, queue_wait_ms, expected", [
    (88.2, 100, 100.0, 0.0, 88.2),
    (0.0, 100, 10.0, 5000.0, 20.0),
    (0.0, 0, 1.0, 0.0, 0.0),
    (0.0, 10, 1.0, 2000.0, 0.0),
], ids=["native_wins", "fallback_excludes_queue_wait", "zero_when_no_data",
        "zero_when_queue_wait_swallows_elapsed"])
def test_headline_tps(native, tokens, elapsed_s, queue_wait_ms, expected):
    assert headline_tps(native, tokens=tokens, elapsed_s=elapsed_s,
                        queue_wait_ms=queue_wait_ms) == expected


# ---------------------------------------------------------------------------
# Trends aggregate success-only
# ---------------------------------------------------------------------------

# Averages reflect successful requests only; a failure still counts in
# requests/errors. An all-failed hour reports zero averages.
@pytest.mark.parametrize("events, expected", [
    ([dict(total_ms=400.0, tokens_per_second=100.0, success=True),
      dict(total_ms=5.0, tokens_per_second=0.0, success=False)],
     {"tokens_per_second": 100.0, "response_time_ms": 400.0, "requests": 2, "errors": 1}),
    ([dict(tokens_per_second=0.0, success=False)],
     {"tokens_per_second": 0.0, "response_time_ms": 0.0, "errors": 1}),
], ids=["failed_excluded_from_averages", "all_failed_hour_reports_zero"])
def test_trends_average_success_only(events, expected):
    c = PerfCollector()
    now = (time.time() // 3600) * 3600 + 1800  # mid-hour, no boundary flake
    for overrides in events:
        c.record_request(RequestEvent(**_base_event_kwargs(timestamp=now, **overrides)))

    trends = c.build_profile("1h")["trends"]
    assert len(trends) == 1
    assert {k: trends[0][k] for k in expected} == expected


# ---------------------------------------------------------------------------
# Both /v1/messages modes record the engine's native numbers
# ---------------------------------------------------------------------------

def _msg_request():
    from heylook_llm.schema.messages import MessageCreateRequest
    return MessageCreateRequest(model="test-model", messages=[{"role": "user", "content": "x"}])


def _run_non_stream(chunks):
    from heylook_llm.messages_api import _non_stream_messages

    perf_ctx = {"provider_get_ms": 5.0, "had_images": False}
    collector = PerfCollector()

    def gen():
        yield from chunks

    with patch("heylook_llm.messages_api.get_perf_collector", return_value=collector):
        response = asyncio.run(_non_stream_messages(
            gen(), _msg_request(), "req-test-456",
            request_start_time=time.time() - 10.0,  # 10s elapsed
            perf_ctx=perf_ctx,
        ))
    assert len(collector._events) == 1
    return response, collector._events[0]


def _run_stream(chunks):
    from heylook_llm.messages_api import _stream_messages

    perf_ctx = {
        "request_start_time": time.time(),
        "provider_get_ms": 5.0,
        "had_images": False,
    }
    collector = PerfCollector()

    def gen():
        yield from chunks

    async def drain():
        return [part async for part in _stream_messages(
            gen(), _msg_request(), "req-test-789",
            http_request=None, provider=None, perf_ctx=perf_ctx,
            abort_event=None,
        )]

    with patch("heylook_llm.messages_api.get_perf_collector", return_value=collector):
        asyncio.run(drain())
    assert len(collector._events) == 1
    return None, collector._events[0]


# The recorded event (and the non-stream response's `performance`) carry the
# chunk's own prompt_tps/generation_tps. Old bug on the non-stream path:
# prompt_tps = prompt_tokens / whole-request elapsed (10 tokens / 10s = 1.0).
@pytest.mark.parametrize("run, chunk, event_expected, perf_expected", [
    (_run_non_stream, dict(prompt_tokens=10, prompt_tps=123.4),
     {"prompt_tps": 123.4}, {"prompt_tps": 123.4}),
    (_run_non_stream, dict(generation_tps=87.6),
     {"tokens_per_second": 87.6}, {"generation_tps": 87.6}),
    (_run_stream, dict(generation_tps=87.6, prompt_tps=123.4),
     {"tokens_per_second": 87.6, "prompt_tps": 123.4}, None),
], ids=["non_stream_prompt_tps_native_not_elapsed_division",
        "non_stream_generation_tps_native", "stream_records_native_rates"])
def test_messages_record_native_rates(run, chunk, event_expected, perf_expected):
    response, event = run([_chunk(**chunk)])
    assert {k: getattr(event, k) for k in event_expected} == event_expected
    if perf_expected is not None:
        assert response.performance is not None
        assert {k: getattr(response.performance, k) for k in perf_expected} == perf_expected
