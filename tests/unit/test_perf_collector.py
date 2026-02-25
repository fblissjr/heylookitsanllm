# tests/unit/test_perf_collector.py
"""Unit tests for PerfCollector aggregation logic."""

import time

import pytest

from heylook_llm.perf_collector import (
    PerfCollector,
    RequestEvent,
    ResourceSnapshot,
    _parse_time_range,
    get_perf_collector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    timestamp: float | None = None,
    model: str = "test-model",
    success: bool = True,
    total_ms: float = 1000.0,
    queue_ms: float = 10.0,
    model_load_ms: float = 0.0,
    image_processing_ms: float = 0.0,
    token_generation_ms: float = 950.0,
    first_token_ms: float = 50.0,
    prompt_tokens: int = 20,
    completion_tokens: int = 100,
    tokens_per_second: float = 100.0,
    had_images: bool = False,
    was_streaming: bool = True,
) -> RequestEvent:
    return RequestEvent(
        timestamp=timestamp or time.time(),
        model=model,
        success=success,
        total_ms=total_ms,
        queue_ms=queue_ms,
        model_load_ms=model_load_ms,
        image_processing_ms=image_processing_ms,
        token_generation_ms=token_generation_ms,
        first_token_ms=first_token_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=tokens_per_second,
        had_images=had_images,
        was_streaming=was_streaming,
    )


def _make_snapshot(
    timestamp: float | None = None,
    memory_gb: float = 8.0,
    gpu_percent: float = 0.0,
    tokens_per_second: float = 50.0,
    requests: int = 5,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=timestamp or time.time(),
        memory_gb=memory_gb,
        gpu_percent=gpu_percent,
        tokens_per_second=tokens_per_second,
        requests=requests,
    )


# ---------------------------------------------------------------------------
# Tests: _parse_time_range
# ---------------------------------------------------------------------------

class TestParseTimeRange:
    def test_valid_ranges(self):
        assert _parse_time_range("1h") == 3600
        assert _parse_time_range("6h") == 6 * 3600
        assert _parse_time_range("24h") == 24 * 3600
        assert _parse_time_range("7d") == 7 * 24 * 3600

    def test_unknown_range_defaults_to_1h(self):
        assert _parse_time_range("invalid") == 3600
        assert _parse_time_range("") == 3600


# ---------------------------------------------------------------------------
# Tests: PerfCollector recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_request(self):
        c = PerfCollector(max_events=100)
        e = _make_event()
        c.record_request(e)
        assert len(c._events) == 1

    def test_record_resource_snapshot(self):
        c = PerfCollector(max_snapshots=100)
        s = _make_snapshot()
        c.record_resource_snapshot(s)
        assert len(c._resource_snapshots) == 1

    def test_ring_buffer_eviction(self):
        c = PerfCollector(max_events=3)
        for i in range(5):
            c.record_request(_make_event(total_ms=float(i)))
        assert len(c._events) == 3
        # Oldest events should be evicted
        assert c._events[0].total_ms == 2.0


# ---------------------------------------------------------------------------
# Tests: build_profile -- empty state
# ---------------------------------------------------------------------------

class TestBuildProfileEmpty:
    def test_empty_returns_valid_structure(self):
        c = PerfCollector()
        profile = c.build_profile("1h")

        assert profile["time_range"] == "1h"
        assert len(profile["timing_breakdown"]) == 5
        assert profile["resource_timeline"] == []
        assert profile["bottlenecks"] == []
        assert profile["trends"] == []

    def test_empty_timing_breakdown_has_all_operations(self):
        c = PerfCollector()
        profile = c.build_profile("1h")
        ops = {item["operation"] for item in profile["timing_breakdown"]}
        assert ops == {"queue", "model_load", "image_processing", "token_generation", "other"}


# ---------------------------------------------------------------------------
# Tests: build_profile -- timing_breakdown
# ---------------------------------------------------------------------------

class TestTimingBreakdown:
    def test_averages_across_events(self):
        c = PerfCollector()
        c.record_request(_make_event(queue_ms=10, token_generation_ms=90, total_ms=100))
        c.record_request(_make_event(queue_ms=20, token_generation_ms=80, total_ms=100))

        profile = c.build_profile("1h")
        breakdown = {item["operation"]: item for item in profile["timing_breakdown"]}

        assert breakdown["queue"]["avg_time_ms"] == 15.0
        assert breakdown["token_generation"]["avg_time_ms"] == 85.0
        assert breakdown["queue"]["count"] == 2

    def test_percentages_sum_to_approximately_one(self):
        c = PerfCollector()
        c.record_request(_make_event(queue_ms=10, model_load_ms=0, image_processing_ms=0, token_generation_ms=90, total_ms=100))

        profile = c.build_profile("1h")
        total_pct = sum(item["percentage"] for item in profile["timing_breakdown"])
        assert 0.99 <= total_pct <= 1.01


# ---------------------------------------------------------------------------
# Tests: build_profile -- bottlenecks
# ---------------------------------------------------------------------------

class TestBottlenecks:
    def test_per_model_breakdown(self):
        c = PerfCollector()
        c.record_request(_make_event(model="model-a", total_ms=200))
        c.record_request(_make_event(model="model-a", total_ms=300))
        c.record_request(_make_event(model="model-b", total_ms=100))

        profile = c.build_profile("1h")
        bottlenecks = profile["bottlenecks"]

        assert len(bottlenecks) == 2
        # Sorted by avg_total_ms descending
        assert bottlenecks[0]["model"] == "model-a"
        assert bottlenecks[0]["avg_total_ms"] == 250.0
        assert bottlenecks[0]["request_count"] == 2
        assert bottlenecks[1]["model"] == "model-b"
        assert bottlenecks[1]["request_count"] == 1

    def test_breakdown_fields_present(self):
        c = PerfCollector()
        c.record_request(_make_event(model="m", first_token_ms=50))

        profile = c.build_profile("1h")
        breakdown = profile["bottlenecks"][0]["breakdown"]
        assert "queue" in breakdown
        assert "model_load" in breakdown
        assert "image_processing" in breakdown
        assert "token_generation" in breakdown
        assert "first_token" in breakdown


# ---------------------------------------------------------------------------
# Tests: build_profile -- trends
# ---------------------------------------------------------------------------

class TestTrends:
    def test_single_hour_bucket(self):
        c = PerfCollector()
        now = time.time()
        c.record_request(_make_event(timestamp=now, total_ms=500, tokens_per_second=80))
        c.record_request(_make_event(timestamp=now + 60, total_ms=600, tokens_per_second=120))

        profile = c.build_profile("1h")
        trends = profile["trends"]

        assert len(trends) == 1
        assert trends[0]["requests"] == 2
        assert trends[0]["response_time_ms"] == 550.0  # (500 + 600) / 2
        assert trends[0]["tokens_per_second"] == 100.0

    def test_error_counting(self):
        c = PerfCollector()
        now = time.time()
        c.record_request(_make_event(timestamp=now, success=True))
        c.record_request(_make_event(timestamp=now, success=False))
        c.record_request(_make_event(timestamp=now, success=False))

        profile = c.build_profile("1h")
        assert profile["trends"][0]["errors"] == 2

    def test_change_between_hours(self):
        c = PerfCollector()
        now = time.time()
        # Two distinct hours
        hour1 = now - 7200  # 2 hours ago
        hour2 = now - 3600  # 1 hour ago
        c.record_request(_make_event(timestamp=hour1, total_ms=100, tokens_per_second=50))
        c.record_request(_make_event(timestamp=hour2, total_ms=200, tokens_per_second=100))

        profile = c.build_profile("24h")
        trends = profile["trends"]

        if len(trends) == 2:
            # Second hour should show change relative to first
            assert trends[1]["response_time_change"] == 1.0  # doubled
            assert trends[1]["tps_change"] == 1.0  # doubled


# ---------------------------------------------------------------------------
# Tests: build_profile -- time range filtering
# ---------------------------------------------------------------------------

class TestTimeRangeFiltering:
    def test_old_events_excluded(self):
        c = PerfCollector()
        old = time.time() - 7200  # 2 hours ago
        recent = time.time()
        c.record_request(_make_event(timestamp=old, model="old"))
        c.record_request(_make_event(timestamp=recent, model="recent"))

        profile = c.build_profile("1h")
        assert len(profile["bottlenecks"]) == 1
        assert profile["bottlenecks"][0]["model"] == "recent"

    def test_snapshots_filtered_by_time_range(self):
        c = PerfCollector()
        old = time.time() - 7200
        recent = time.time()
        c.record_resource_snapshot(_make_snapshot(timestamp=old))
        c.record_resource_snapshot(_make_snapshot(timestamp=recent))

        profile = c.build_profile("1h")
        assert len(profile["resource_timeline"]) == 1


# ---------------------------------------------------------------------------
# Tests: resource_timeline format
# ---------------------------------------------------------------------------

class TestResourceTimeline:
    def test_snapshot_format(self):
        c = PerfCollector()
        c.record_resource_snapshot(_make_snapshot(memory_gb=12.5, tokens_per_second=75.3, requests=10))

        profile = c.build_profile("1h")
        timeline = profile["resource_timeline"]
        assert len(timeline) == 1
        point = timeline[0]
        assert "timestamp" in point
        assert point["memory_gb"] == 12.5
        assert point["gpu_percent"] == 0.0
        assert point["tokens_per_second"] == 75.3
        assert point["requests"] == 10


# ---------------------------------------------------------------------------
# Tests: singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_perf_collector_returns_same_instance(self):
        a = get_perf_collector()
        b = get_perf_collector()
        assert a is b
