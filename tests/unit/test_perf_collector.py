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
    model_load_ms: float = 0.0,
    token_generation_ms: float = 950.0,
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
        model_load_ms=model_load_ms,
        token_generation_ms=token_generation_ms,
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
    # range string -> seconds; anything unknown (incl. empty) is 1h
    @pytest.mark.parametrize(
        "expected",
        [
            {"1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600, "7d": 7 * 24 * 3600},
            {"invalid": 3600, "": 3600},
        ],
        ids=["valid_ranges", "unknown_range_defaults_to_1h"],
    )
    def test_parse_time_range(self, expected):
        for text, seconds in expected.items():
            assert _parse_time_range(text) == seconds, text


# ---------------------------------------------------------------------------
# Tests: PerfCollector recording
# ---------------------------------------------------------------------------

class TestRecording:
    # Recording lands in a bounded ring buffer. Each row: collector bounds,
    # the recorder, how many items it gets (item i from make(i)), the buffer
    # it lands in, its length after, and the oldest surviving total_ms
    # (None = not checked).
    @pytest.mark.parametrize(
        "bounds, recorder, make, n, buffer, expected_len, oldest_total_ms",
        [
            ({"max_events": 100}, "record_request",
             lambda i: _make_event(), 1, "_events", 1, None),
            ({"max_snapshots": 100}, "record_resource_snapshot",
             lambda i: _make_snapshot(), 1, "_resource_snapshots", 1, None),
            # past the bound the oldest events are evicted (0 and 1 go)
            ({"max_events": 3}, "record_request",
             lambda i: _make_event(total_ms=float(i)), 5, "_events", 3, 2.0),
        ],
        ids=["record_request", "record_resource_snapshot", "ring_buffer_eviction"],
    )
    def test_recording(self, bounds, recorder, make, n, buffer, expected_len,
                       oldest_total_ms):
        c = PerfCollector(**bounds)
        for i in range(n):
            getattr(c, recorder)(make(i))
        items = getattr(c, buffer)
        assert len(items) == expected_len
        if oldest_total_ms is not None:
            assert items[0].total_ms == oldest_total_ms


# ---------------------------------------------------------------------------
# Tests: build_profile -- empty state
# ---------------------------------------------------------------------------

class TestBuildProfileEmpty:
    # An empty collector still returns a valid profile. Each row projects the
    # profile and names what the projection must equal.
    @pytest.mark.parametrize(
        "project, expected",
        [
            # every section present and empty; timing_breakdown keeps its 3 rows
            (lambda p: (p["time_range"], len(p["timing_breakdown"]),
                        p["resource_timeline"], p["bottlenecks"], p["trends"]),
             ("1h", 3, [], [], [])),
            # the 3 rows are exactly the three operations
            (lambda p: {item["operation"] for item in p["timing_breakdown"]},
             {"model_load", "token_generation", "other"}),
        ],
        ids=["empty_returns_valid_structure", "empty_timing_breakdown_has_all_operations"],
    )
    def test_empty_profile(self, project, expected):
        assert project(PerfCollector().build_profile("1h")) == expected


# ---------------------------------------------------------------------------
# Tests: build_profile -- timing_breakdown
# ---------------------------------------------------------------------------

class TestTimingBreakdown:
    def test_averages_across_events(self):
        c = PerfCollector()
        c.record_request(_make_event(model_load_ms=10, token_generation_ms=90, total_ms=100))
        c.record_request(_make_event(model_load_ms=20, token_generation_ms=80, total_ms=100))

        profile = c.build_profile("1h")
        breakdown = {item["operation"]: item for item in profile["timing_breakdown"]}

        assert breakdown["model_load"]["avg_time_ms"] == 15.0
        assert breakdown["token_generation"]["avg_time_ms"] == 85.0
        assert breakdown["model_load"]["count"] == 2

    def test_percentages_sum_to_approximately_one(self):
        c = PerfCollector()
        c.record_request(_make_event(model_load_ms=10, token_generation_ms=90, total_ms=100))

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
        c.record_request(_make_event(model="m"))

        profile = c.build_profile("1h")
        breakdown = profile["bottlenecks"][0]["breakdown"]
        # Exactly the measured rows: a key here is a promise something fills it.
        assert set(breakdown) == {"queue_wait", "model_load", "token_generation"}


# ---------------------------------------------------------------------------
# Tests: build_profile -- trends
# ---------------------------------------------------------------------------

class TestTrends:
    def test_single_hour_bucket(self):
        c = PerfCollector()
        # Anchor to the middle of the current hour: with live time.time() and
        # a +60s offset, the two events straddled an hour boundary whenever
        # the suite ran in the last minute of an hour -- a real observed
        # 1-in-60 flake (2026-07-06).
        now = (time.time() // 3600) * 3600 + 1800
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

        # Unconditional: an `if len(trends) == 2` guard here let any other
        # bucketing pass having asserted nothing.
        assert len(trends) == 2, trends
        # Second hour should show change relative to first
        assert trends[1]["response_time_change"] == 1.0  # doubled
        assert trends[1]["tps_change"] == 1.0  # doubled


# ---------------------------------------------------------------------------
# Tests: build_profile -- time range filtering
# ---------------------------------------------------------------------------

class TestTimeRangeFiltering:
    # Items older than the window (2h old vs a 1h range) are excluded, for
    # request events and resource snapshots alike. Each row: the recorder,
    # make(timestamp, label), the profile projection, and what it must equal.
    @pytest.mark.parametrize(
        "recorder, make, project, expected",
        [
            ("record_request",
             lambda ts, label: _make_event(timestamp=ts, model=label),
             lambda p: [b["model"] for b in p["bottlenecks"]],
             ["recent"]),
            ("record_resource_snapshot",
             lambda ts, label: _make_snapshot(timestamp=ts),
             lambda p: len(p["resource_timeline"]),
             1),
        ],
        ids=["old_events_excluded", "snapshots_filtered_by_time_range"],
    )
    def test_window_excludes_old_items(self, recorder, make, project, expected):
        c = PerfCollector()
        old = time.time() - 7200  # 2 hours ago
        recent = time.time()
        getattr(c, recorder)(make(old, "old"))
        getattr(c, recorder)(make(recent, "recent"))

        assert project(c.build_profile("1h")) == expected


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
