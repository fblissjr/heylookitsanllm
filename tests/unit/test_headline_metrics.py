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

import time
from typing import Any

import pytest

from heylook_llm.perf_collector import (
    PerfCollector,
    RequestEvent,
    headline_tps,
)


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
# Both /v1/messages modes report and record the engine's native numbers
# ---------------------------------------------------------------------------

def _app_serving(chunk):
    """/v1/messages and the perf profile, over a provider that yields one
    engine chunk. The app's own routes, a stand-in only for the model."""
    from fastapi import FastAPI

    from heylook_llm.config import AppConfig
    from heylook_llm.messages_api import messages_router
    from heylook_llm.monitoring_api import monitoring_router
    from heylook_llm.providers.base import BaseProvider

    class Provider(BaseProvider):
        provider_name = "mlx"

        def load_model(self):
            pass

        def template_info(self):
            return None  # pass-through parser

        def create_chat_completion(self, request, abort_event=None):
            yield chunk

    provider = Provider("m", {"model_path": "/fake/m", "vision": False}, False)

    class Router:
        app_config = AppConfig(models=[{"id": "m", "provider": "mlx", "enabled": True,
                                        "config": {"model_path": "/fake/m", "vision": False}}])

        def get_provider(self, model_id):
            return provider

    app = FastAPI()
    app.include_router(messages_router)
    app.include_router(monitoring_router)
    app.state.router_instance = Router()
    return app


def _stop_performance(text):
    import json

    for block in text.split("\n\n"):
        if "event: message_stop" in block:
            data = next(line for line in block.split("\n") if line.startswith("data: "))
            return json.loads(data[len("data: "):])["performance"]
    raise AssertionError("no message_stop event")


# What the client is told (the response's `performance`, message_stop's on a
# stream) and the rate the perf page trends are the chunk's own
# prompt_tps/generation_tps. Old bug on the non-stream path: prompt_tps =
# prompt tokens / whole-request elapsed. A wall-clock rate over this
# near-instant request would be far from either number.
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_messages_report_and_record_native_rates(stream, monkeypatch):
    from fastapi.testclient import TestClient

    from heylook_llm import perf_collector
    from heylook_llm.providers.base import GenerationChunk

    monkeypatch.setattr(perf_collector, "_collector", PerfCollector())  # this test's events only
    chunk = GenerationChunk(text="hi", token=0, finish_reason="stop", prompt_tokens=10,
                            generation_tokens=1, prompt_tps=123.4, generation_tps=87.6)
    client = TestClient(_app_serving(chunk))
    res = client.post("/v1/messages", json={
        "model": "m", "stream": stream, "max_tokens": 16,
        "messages": [{"role": "user", "content": "x"}]})
    assert res.status_code == 200, res.text
    perf = _stop_performance(res.text) if stream else res.json()["performance"]
    assert (perf["prompt_tps"], perf["generation_tps"]) == (123.4, 87.6)

    (trend,) = client.get("/v1/performance/profile/1h").json()["trends"]
    assert trend["tokens_per_second"] == 87.6
