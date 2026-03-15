#!/usr/bin/env python3
"""Shared utilities for text and VLM benchmark scripts.

Provides composite scoring, baseline management, timing helpers,
hardware info, and grep-friendly output formatting.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import orjson


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "bench"
TEXT_DIR = DATA_DIR / "text"
VLM_DIR = DATA_DIR / "vlm"


def ensure_dirs():
    """Create data directories if they don't exist."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    VLM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict:
    """Collect chip and memory info for result metadata."""
    info = {"chip": "unknown", "memory_gb": 0}
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except Exception:
        pass

    # Try to get Apple Silicon chip name
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.chip_model"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["chip"] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["memory_gb"] = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def sync_barrier():
    """Force GPU synchronization via mx.eval on a trivial operation."""
    mx.eval(mx.zeros(1))


@dataclass
class TimingContext:
    """Context manager for timed sections with mx.eval barriers."""

    elapsed_ms: float = 0.0
    _start: float = 0.0

    def __enter__(self):
        sync_barrier()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync_barrier()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Weights: 40% gen_tps, 25% TTFT, 20% prefill_tps, 15% memory
SCORE_WEIGHTS = {
    "gen_tps": 0.40,
    "ttft": 0.25,
    "prefill_tps": 0.20,
    "memory": 0.15,
}

# Hard constraints
MAX_REGRESSION_PCT = 0.30  # 30% regression in any single metric = auto-fail
MAX_TOKEN_DEVIATION_PCT = 0.20  # 20% token count deviation = correctness guard


def compute_composite_score(
    gen_tps: float,
    ttft_ms: float,
    prefill_tps: float,
    memory_gb: float,
    baseline: dict,
) -> float:
    """Compute composite score relative to baseline.

    Baseline = 1.0. >1.0 = improvement.

    For TTFT and memory, lower is better so we invert the ratio.
    """
    b = baseline
    if b["gen_tps"] <= 0 or b["ttft_ms"] <= 0 or b["prefill_tps"] <= 0 or b["memory_gb"] <= 0:
        return 1.0

    score = (
        SCORE_WEIGHTS["gen_tps"] * (gen_tps / b["gen_tps"])
        + SCORE_WEIGHTS["ttft"] * (b["ttft_ms"] / ttft_ms)
        + SCORE_WEIGHTS["prefill_tps"] * (prefill_tps / b["prefill_tps"])
        + SCORE_WEIGHTS["memory"] * (b["memory_gb"] / memory_gb)
    )
    return score


def check_hard_constraints(
    gen_tps: float,
    ttft_ms: float,
    prefill_tps: float,
    memory_gb: float,
    avg_completion_tokens: float,
    baseline: dict,
) -> list[str]:
    """Check hard constraints against baseline. Returns list of violations."""
    violations = []
    b = baseline

    # Metric regression checks (>30% regression = fail)
    if b["gen_tps"] > 0 and gen_tps < b["gen_tps"] * (1 - MAX_REGRESSION_PCT):
        violations.append(
            f"gen_tps regressed {(1 - gen_tps / b['gen_tps']) * 100:.1f}% "
            f"({gen_tps:.1f} vs baseline {b['gen_tps']:.1f})"
        )

    if b["ttft_ms"] > 0 and ttft_ms > b["ttft_ms"] * (1 + MAX_REGRESSION_PCT):
        violations.append(
            f"ttft regressed {(ttft_ms / b['ttft_ms'] - 1) * 100:.1f}% "
            f"({ttft_ms:.1f}ms vs baseline {b['ttft_ms']:.1f}ms)"
        )

    if b["prefill_tps"] > 0 and prefill_tps < b["prefill_tps"] * (1 - MAX_REGRESSION_PCT):
        violations.append(
            f"prefill_tps regressed {(1 - prefill_tps / b['prefill_tps']) * 100:.1f}% "
            f"({prefill_tps:.1f} vs baseline {b['prefill_tps']:.1f})"
        )

    if b["memory_gb"] > 0 and memory_gb > b["memory_gb"] * (1 + MAX_REGRESSION_PCT):
        violations.append(
            f"memory regressed {(memory_gb / b['memory_gb'] - 1) * 100:.1f}% "
            f"({memory_gb:.2f}GB vs baseline {b['memory_gb']:.2f}GB)"
        )

    # Token count deviation (correctness guard)
    if b.get("avg_completion_tokens", 0) > 0:
        deviation = abs(avg_completion_tokens - b["avg_completion_tokens"]) / b["avg_completion_tokens"]
        if deviation > MAX_TOKEN_DEVIATION_PCT:
            violations.append(
                f"token count deviated {deviation * 100:.1f}% "
                f"({avg_completion_tokens:.0f} vs baseline {b['avg_completion_tokens']:.0f})"
            )

    return violations


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def load_baseline(bench_dir: Path) -> dict | None:
    """Load baseline.json from bench directory. Returns None if not found."""
    path = bench_dir / "baseline.json"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_baseline(bench_dir: Path, data: dict):
    """Save baseline.json to bench directory."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / "baseline.json"
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def save_run(bench_dir: Path, data: dict, timestamp: str):
    """Save a run result as run_<timestamp>.json."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"run_{timestamp}.json"
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


# ---------------------------------------------------------------------------
# Grep-friendly output
# ---------------------------------------------------------------------------

def print_results(
    composite_score: float,
    avg_gen_tps: float,
    avg_ttft_ms: float,
    avg_prefill_tps: float,
    peak_memory_gb: float,
    runs: int,
    model: str,
    bench: str,
    extra_lines: dict | None = None,
):
    """Print grep-friendly results block to stdout."""
    print("---")
    print(f"composite_score:  {composite_score:.4f}")
    print(f"avg_gen_tps:      {avg_gen_tps:.1f}")
    print(f"avg_ttft_ms:      {avg_ttft_ms:.1f}")
    print(f"avg_prefill_tps:  {avg_prefill_tps:.1f}")
    if extra_lines:
        for key, value in extra_lines.items():
            print(f"{key}:  {value}")
    print(f"peak_memory_gb:   {peak_memory_gb:.1f}")
    print(f"runs:             {runs}")
    print(f"model:            {model}")
    print(f"bench:            {bench}")


# ---------------------------------------------------------------------------
# Result data construction
# ---------------------------------------------------------------------------

def build_result_data(
    bench: str,
    model: str,
    timestamp: str,
    composite_score: float,
    metrics: dict,
    per_prompt: list[dict],
    hardware: dict | None = None,
) -> dict:
    """Build the JSON result dict for saving."""
    if hardware is None:
        hardware = get_hardware_info()
    return {
        "timestamp": timestamp,
        "bench": bench,
        "model": model,
        "hardware": hardware,
        "composite_score": composite_score,
        "metrics": metrics,
        "per_prompt": per_prompt,
    }


def baseline_metrics_from_result(result: dict) -> dict:
    """Extract baseline comparison metrics from a result dict."""
    m = result["metrics"]
    avg_tokens = 0
    if result.get("per_prompt"):
        avg_tokens = sum(p.get("completion_tokens", 0) for p in result["per_prompt"]) / len(result["per_prompt"])
    return {
        "gen_tps": m["avg_gen_tps"],
        "ttft_ms": m["avg_ttft_ms"],
        "prefill_tps": m["avg_prefill_tps"],
        "memory_gb": m["peak_memory_gb"],
        "avg_completion_tokens": avg_tokens,
    }
