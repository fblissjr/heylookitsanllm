# src/heylook_llm/perf_collector.py
"""In-memory performance metrics collector.

Stores completed request events in a bounded ring buffer and serves
aggregated profiles to the frontend Performance applet via
GET /v1/performance/profile/{time_range}.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(slots=True)
class RequestEvent:
    """One completed request's timing breakdown (~200 bytes)."""

    timestamp: float  # time.time() at completion
    model: str
    success: bool
    total_ms: float
    queue_ms: float  # time in get_provider() call
    model_load_ms: float  # 0 if cache hit, else ~= queue_ms
    image_processing_ms: float  # 0 if text-only
    token_generation_ms: float  # generation loop time
    first_token_ms: float  # TTFT (streaming only, else 0)
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    had_images: bool
    was_streaming: bool
    # Time blocked in the FIFO generation queue waiting for an in-flight
    # generation to finish (distinct from queue_ms, which is get_provider /
    # model-load time). Defaulted for back-compat with older event records.
    queue_wait_ms: float = 0.0
    # mlx-lm's own prefill rate, measured tightly around the prefill loop
    # (chunk.prompt_tps). Defaulted for back-compat with older event records.
    prompt_tps: float = 0.0


@dataclass(slots=True)
class ChunkTelemetry:
    """Accumulates mlx-lm per-chunk telemetry in one place.

    Chunks are non-slotted GenerationResponse attr-bags; every API consume
    loop (api.py / messages_api.py, streaming and non-streaming) needs the
    same scrape. One ``absorb(chunk)`` per chunk replaces four hand-copied
    getattr blocks -- a new telemetry field is added HERE once, not at four
    call sites that would otherwise drift apart.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    peak_memory_gb: float = 0.0  # monotonic max across chunks
    kv_cache_bytes: int = 0  # snapshot tagged on the first chunk
    queue_wait_ms: float = 0.0  # FIFO generation-queue wait
    prompt_tps: float = 0.0  # mlx-lm's own prefill rate
    generation_tps: float = 0.0  # mlx-lm's own decode rate
    finish_reason: Optional[str] = None  # "stop" | "length" | None (mlx-lm's)

    def absorb(self, chunk) -> None:
        self.prompt_tokens = getattr(chunk, "prompt_tokens", self.prompt_tokens)
        self.completion_tokens = getattr(chunk, "generation_tokens", self.completion_tokens)
        self.cached_tokens = getattr(chunk, "cached_tokens", self.cached_tokens)
        self.peak_memory_gb = max(self.peak_memory_gb, getattr(chunk, "peak_memory", 0.0))
        self.kv_cache_bytes = getattr(chunk, "kv_cache_bytes", self.kv_cache_bytes)
        self.queue_wait_ms = getattr(chunk, "queue_wait_ms", self.queue_wait_ms)
        self.prompt_tps = getattr(chunk, "prompt_tps", self.prompt_tps)
        self.generation_tps = getattr(chunk, "generation_tps", self.generation_tps)
        # arrives on the FINAL chunk only -- a later chunk without one must
        # not erase it, so this latches rather than overwrites
        self.finish_reason = getattr(chunk, "finish_reason", None) or self.finish_reason


def net_ttft_ms(raw_ttft_ms: float, queue_wait_ms: float) -> float:
    """TTFT with FIFO queue wait excluded (clamped at 0).

    Admission pressure is not model latency; it stays visible in the
    separate queue_wait_ms field.
    """
    return max(0.0, raw_ttft_ms - queue_wait_ms)


def headline_tps(
    native_tps: float,
    tokens: int,
    elapsed_s: float,
    queue_wait_ms: float = 0.0,
) -> float:
    """Headline generation tok/s: prefer the engine's own measurement.

    mlx-lm computes generation_tps tightly around the decode loop -- wall-clock
    division at the API layer bakes in prefill, FIFO queue wait, and SSE
    delivery overhead (the 2026-07-06 measurement audit). When no native
    number is available, the fallback at least excludes queue wait so
    admission pressure can't masquerade as model slowness.
    """
    if native_tps > 0:
        return native_tps
    gen_s = elapsed_s - queue_wait_ms / 1000.0
    return tokens / gen_s if gen_s > 0 and tokens > 0 else 0.0


@dataclass(slots=True)
class ResourceSnapshot:
    """Periodic system resource sample (~48 bytes)."""

    timestamp: float
    memory_gb: float
    gpu_percent: float  # 0.0 on Apple Silicon (no discrete GPU)
    tokens_per_second: float  # rolling avg from recent events
    requests: int  # requests in this interval


# ---------------------------------------------------------------------------
# Time-range helpers
# ---------------------------------------------------------------------------

_TIME_RANGE_SECONDS = {
    "1h": 3600,
    "6h": 6 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
}


def _parse_time_range(time_range: str) -> int:
    """Return seconds for a time_range string, default 3600."""
    return _TIME_RANGE_SECONDS.get(time_range, 3600)


# ---------------------------------------------------------------------------
# PerfCollector
# ---------------------------------------------------------------------------

class PerfCollector:
    """Thread-safe ring-buffer collector with profile aggregation.

    Memory budget:
      - _events: 10 000 RequestEvent  ~2 MB
      - _resource_snapshots: 10 080 ResourceSnapshot  ~500 KB
    """

    def __init__(
        self,
        max_events: int = 10_000,
        max_snapshots: int = 10_080,
    ):
        self._events: deque[RequestEvent] = deque(maxlen=max_events)
        self._resource_snapshots: deque[ResourceSnapshot] = deque(maxlen=max_snapshots)
        self._lock = threading.Lock()

    # -- Recording ----------------------------------------------------------

    def record_request(self, event: RequestEvent) -> None:
        """Append a completed-request event. O(1)."""
        with self._lock:
            self._events.append(event)

    def record_resource_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Append a periodic resource snapshot. O(1)."""
        with self._lock:
            self._resource_snapshots.append(snapshot)

    # -- Aggregation --------------------------------------------------------

    def build_profile(self, time_range: str) -> dict:
        """Build a frontend-compatible performance profile.

        Returns the PerformanceProfile dict shape consumed by the v3
        performance page (apps/heylook-frontend-v3/js/pages/perf.js).
        """
        window_seconds = _parse_time_range(time_range)
        cutoff = time.time() - window_seconds

        with self._lock:
            events = [e for e in self._events if e.timestamp >= cutoff]
            snapshots = [s for s in self._resource_snapshots if s.timestamp >= cutoff]

        return {
            "time_range": time_range,
            "timing_breakdown": self._timing_breakdown(events),
            "resource_timeline": self._resource_timeline(snapshots),
            "bottlenecks": self._bottlenecks(events),
            "trends": self._trends(events),
        }

    # -- Private aggregation helpers ----------------------------------------

    @staticmethod
    def _timing_breakdown(events: list[RequestEvent]) -> list[dict]:
        """Per-operation average times + percentage of total."""
        if not events:
            ops = ["queue", "model_load", "image_processing", "token_generation", "other"]
            return [{"operation": op, "avg_time_ms": 0, "count": 0, "percentage": 0.0} for op in ops]

        n = len(events)
        sums = {
            "queue": sum(e.queue_ms for e in events),
            "model_load": sum(e.model_load_ms for e in events),
            "image_processing": sum(e.image_processing_ms for e in events),
            "token_generation": sum(e.token_generation_ms for e in events),
        }
        total_accounted = sum(sums.values())
        total_all = sum(e.total_ms for e in events)
        sums["other"] = max(0.0, total_all - total_accounted)

        grand_total = total_all if total_all > 0 else 1.0  # avoid division by zero
        return [
            {
                "operation": op,
                "avg_time_ms": round(ms / n, 1),
                "count": n,
                "percentage": round(ms / grand_total, 4),
            }
            for op, ms in sums.items()
        ]

    @staticmethod
    def _resource_timeline(snapshots: list[ResourceSnapshot]) -> list[dict]:
        """Convert snapshots to frontend ResourceTimepoint format."""
        return [
            {
                "timestamp": datetime.fromtimestamp(s.timestamp, tz=timezone.utc).isoformat(),
                "memory_gb": round(s.memory_gb, 2),
                "gpu_percent": round(s.gpu_percent, 1),
                "tokens_per_second": round(s.tokens_per_second, 1),
                "requests": s.requests,
            }
            for s in snapshots
        ]

    @staticmethod
    def _bottlenecks(events: list[RequestEvent]) -> list[dict]:
        """Per-model breakdown sorted by avg_total_ms descending."""
        if not events:
            return []

        by_model: dict[str, list[RequestEvent]] = {}
        for e in events:
            by_model.setdefault(e.model, []).append(e)

        result = []
        for model_id, model_events in by_model.items():
            n = len(model_events)
            result.append({
                "model": model_id,
                "avg_total_ms": round(sum(e.total_ms for e in model_events) / n, 1),
                "breakdown": {
                    "queue": round(sum(e.queue_ms for e in model_events) / n, 1),
                    "queue_wait": round(sum(e.queue_wait_ms for e in model_events) / n, 1),
                    "model_load": round(sum(e.model_load_ms for e in model_events) / n, 1),
                    "image_processing": round(sum(e.image_processing_ms for e in model_events) / n, 1),
                    "token_generation": round(sum(e.token_generation_ms for e in model_events) / n, 1),
                    "first_token": round(sum(e.first_token_ms for e in model_events) / n, 1),
                },
                "request_count": n,
            })

        result.sort(key=lambda b: b["avg_total_ms"], reverse=True)
        return result

    @staticmethod
    def _trends(events: list[RequestEvent]) -> list[dict]:
        """Per-hour buckets with response time, TPS, request/error counts, and deltas.

        Averages are success-only: failed/503 events carry 0.0 tok/s and
        near-zero total_ms, so mixing them in silently drags the trend lines
        toward zero. They still count in requests/errors.
        """
        if not events:
            return []

        # Bucket by hour
        buckets: dict[str, list[RequestEvent]] = {}
        for e in events:
            hour_key = datetime.fromtimestamp(
                e.timestamp, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:00:00+00:00")
            buckets.setdefault(hour_key, []).append(e)

        sorted_hours = sorted(buckets.keys())
        result = []
        prev_rt: Optional[float] = None
        prev_tps: Optional[float] = None

        for hour in sorted_hours:
            hour_events = buckets[hour]
            n = len(hour_events)
            successes = [e for e in hour_events if e.success]
            n_ok = len(successes)
            avg_rt = sum(e.total_ms for e in successes) / n_ok if n_ok else 0.0
            avg_tps = sum(e.tokens_per_second for e in successes) / n_ok if n_ok else 0.0
            errors = n - n_ok

            rt_change = 0.0
            if prev_rt is not None and prev_rt > 0:
                rt_change = round((avg_rt - prev_rt) / prev_rt, 4)

            tps_change = 0.0
            if prev_tps is not None and prev_tps > 0:
                tps_change = round((avg_tps - prev_tps) / prev_tps, 4)

            result.append({
                "hour": hour,
                "response_time_ms": round(avg_rt, 1),
                "tokens_per_second": round(avg_tps, 1),
                "requests": n,
                "errors": errors,
                "response_time_change": rt_change,
                "tps_change": tps_change,
            })

            prev_rt = avg_rt
            prev_tps = avg_tps

        return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: Optional[PerfCollector] = None
_collector_lock = threading.Lock()


def get_perf_collector() -> PerfCollector:
    """Get or create the module-level PerfCollector singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = PerfCollector()
    return _collector
