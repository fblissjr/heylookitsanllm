#!/usr/bin/env python3
"""Shared utilities for text and VLM benchmark scripts.

Provides composite scoring, baseline management, timing helpers,
hardware info, grep-friendly output formatting, output fingerprinting,
config loading, and per-cycle structured logging.
"""

import hashlib
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import orjson


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OPTLOOP_ROOT = SCRIPT_DIR.parent                # apps/optloop/
REPO_ROOT = OPTLOOP_ROOT.parent.parent          # repo root (for test running)
DATA_DIR = OPTLOOP_ROOT / "data"
TEXT_DIR = DATA_DIR / "text"
VLM_DIR = DATA_DIR / "vlm"
CYCLES_DIR = DATA_DIR / "cycles"
RESULTS_TSV = OPTLOOP_ROOT / "results.tsv"
CONFIG_PATH = OPTLOOP_ROOT / "bench_config.toml"
MODELS_TOML = REPO_ROOT / "models.toml"


def load_models_toml() -> list[dict]:
    """Load models from the project root models.toml. Returns list of model entries."""
    if not MODELS_TOML.exists():
        return []
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    return data.get("models", [])


def resolve_model_from_toml(model_id: str) -> str | None:
    """Look up a model ID in models.toml and return its local model_path.

    Matches on the 'id' field. Also tries stripping an org/ prefix
    (e.g. 'mlx-community/foo' -> 'foo') as a fallback.
    """
    models = load_models_toml()
    # Direct match
    for m in models:
        if m.get("id") == model_id:
            return m.get("config", {}).get("model_path")
    # Fallback: strip org prefix
    if "/" in model_id:
        short_id = model_id.split("/", 1)[1]
        for m in models:
            if m.get("id") == short_id:
                return m.get("config", {}).get("model_path")
    return None


def ensure_dirs():
    """Create data directories if they don't exist."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """Load bench_config.toml. Falls back to empty dict if not found."""
    path = config_path or CONFIG_PATH
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_scoring_weights(config: dict) -> dict:
    """Extract scoring weights from config, falling back to defaults."""
    scoring = config.get("scoring", {})
    return {
        "gen_tps": scoring.get("gen_tps_weight", 0.40),
        "ttft": scoring.get("ttft_weight", 0.25),
        "prefill_tps": scoring.get("prefill_tps_weight", 0.20),
        "memory": scoring.get("memory_weight", 0.15),
    }


def get_bench_params(config: dict) -> dict:
    """Extract bench parameters from config, falling back to defaults."""
    bench = config.get("bench", {})
    return {
        "runs": bench.get("runs", 3),
        "warmup": bench.get("warmup", 1),
        "max_tokens": bench.get("max_tokens", 256),
        "seed": bench.get("seed", 42),
        "timeout_seconds": bench.get("timeout_seconds", 600),
    }


def get_constraints(config: dict) -> dict:
    """Extract constraint thresholds from config, falling back to defaults."""
    constraints = config.get("constraints", {})
    return {
        "max_single_metric_regression": constraints.get("max_single_metric_regression", 0.30),
        "max_token_deviation": constraints.get("max_token_deviation", 0.20),
        "per_prompt_regression_check": constraints.get("per_prompt_regression_check", True),
        "require_fingerprint_match": constraints.get("require_fingerprint_match", True),
        "require_tests_pass": constraints.get("require_tests_pass", True),
        "max_suspicion_gain": constraints.get("max_suspicion_gain", 0.30),
        "variance_max_cv": constraints.get("variance_max_cv", 0.15),
    }


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
    import mlx.core as mx
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

    def __exit__(self, _t: object, _v: object, _tb: object) -> None:
        sync_barrier()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ---------------------------------------------------------------------------
# Output fingerprinting
# ---------------------------------------------------------------------------

def fingerprint_output(token_ids: list[int]) -> str:
    """SHA-256 hash of token ID sequence. Deterministic for greedy decode."""
    token_bytes = b",".join(str(t).encode() for t in token_ids)
    return hashlib.sha256(token_bytes).hexdigest()[:16]


def check_fingerprints(per_prompt: list[dict], baseline: dict) -> list[str]:
    """Compare output fingerprints against baseline. Returns violations."""
    violations = []
    baseline_prompts = {p["name"]: p for p in baseline.get("per_prompt", [])}
    for prompt in per_prompt:
        name = prompt["name"]
        bp = baseline_prompts.get(name)
        if bp and bp.get("fingerprint") and prompt.get("fingerprint"):
            if prompt["fingerprint"] != bp["fingerprint"]:
                violations.append(
                    f"{name}: output fingerprint mismatch "
                    f"(got {prompt['fingerprint']}, baseline {bp['fingerprint']})"
                )
    return violations


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Default weights (overridden by config)
DEFAULT_SCORE_WEIGHTS = {
    "gen_tps": 0.40,
    "ttft": 0.25,
    "prefill_tps": 0.20,
    "memory": 0.15,
}


def compute_composite_score(
    gen_tps: float,
    ttft_ms: float,
    prefill_tps: float,
    memory_gb: float,
    baseline: dict,
    weights: dict | None = None,
) -> float:
    """Compute composite score relative to baseline.

    Baseline = 1.0. >1.0 = improvement.

    For TTFT and memory, lower is better so we invert the ratio.
    """
    w = weights or DEFAULT_SCORE_WEIGHTS
    total = sum(w.values())
    if abs(total - 1.0) > 0.01:
        print(f"WARNING: scoring weights sum to {total:.3f}, expected 1.0", file=sys.stderr)
    b = baseline
    if b["gen_tps"] <= 0 or b["ttft_ms"] <= 0 or b["prefill_tps"] <= 0 or b["memory_gb"] <= 0:
        return 1.0

    score = (
        w["gen_tps"] * (gen_tps / b["gen_tps"])
        + w["ttft"] * (b["ttft_ms"] / ttft_ms)
        + w["prefill_tps"] * (prefill_tps / b["prefill_tps"])
        + w["memory"] * (b["memory_gb"] / memory_gb)
    )
    return score


def check_hard_constraints(
    gen_tps: float,
    ttft_ms: float,
    prefill_tps: float,
    memory_gb: float,
    avg_completion_tokens: float,
    baseline: dict,
    constraints: dict | None = None,
) -> list[str]:
    """Check hard constraints against baseline. Returns list of violations."""
    c = constraints or {}
    max_regression = c.get("max_single_metric_regression", 0.30)
    max_token_dev = c.get("max_token_deviation", 0.20)

    violations = []
    b = baseline

    # Metric regression checks
    if b["gen_tps"] > 0 and gen_tps < b["gen_tps"] * (1 - max_regression):
        violations.append(
            f"gen_tps regressed {(1 - gen_tps / b['gen_tps']) * 100:.1f}% "
            f"({gen_tps:.1f} vs baseline {b['gen_tps']:.1f})"
        )

    if b["ttft_ms"] > 0 and ttft_ms > b["ttft_ms"] * (1 + max_regression):
        violations.append(
            f"ttft regressed {(ttft_ms / b['ttft_ms'] - 1) * 100:.1f}% "
            f"({ttft_ms:.1f}ms vs baseline {b['ttft_ms']:.1f}ms)"
        )

    if b["prefill_tps"] > 0 and prefill_tps < b["prefill_tps"] * (1 - max_regression):
        violations.append(
            f"prefill_tps regressed {(1 - prefill_tps / b['prefill_tps']) * 100:.1f}% "
            f"({prefill_tps:.1f} vs baseline {b['prefill_tps']:.1f})"
        )

    if b["memory_gb"] > 0 and memory_gb > b["memory_gb"] * (1 + max_regression):
        violations.append(
            f"memory regressed {(memory_gb / b['memory_gb'] - 1) * 100:.1f}% "
            f"({memory_gb:.2f}GB vs baseline {b['memory_gb']:.2f}GB)"
        )

    # Token count deviation (correctness guard)
    if b.get("avg_completion_tokens", 0) > 0:
        deviation = abs(avg_completion_tokens - b["avg_completion_tokens"]) / b["avg_completion_tokens"]
        if deviation > max_token_dev:
            violations.append(
                f"token count deviated {deviation * 100:.1f}% "
                f"({avg_completion_tokens:.0f} vs baseline {b['avg_completion_tokens']:.0f})"
            )

    return violations


def check_per_prompt_constraints(
    per_prompt: list[dict],
    baseline: dict,
    constraints: dict | None = None,
) -> list[str]:
    """Check per-prompt regression constraints. Returns violations."""
    c = constraints or {}
    max_regression = c.get("max_single_metric_regression", 0.30)

    if not c.get("per_prompt_regression_check", True):
        return []

    violations = []
    baseline_prompts = {p["name"]: p for p in baseline.get("per_prompt", [])}

    for prompt in per_prompt:
        name = prompt["name"]
        bp = baseline_prompts.get(name)
        if not bp:
            continue

        if bp.get("gen_tps", 0) > 0 and prompt.get("gen_tps", 0) < bp["gen_tps"] * (1 - max_regression):
            violations.append(
                f"{name}: gen_tps regressed {(1 - prompt['gen_tps'] / bp['gen_tps']) * 100:.1f}%"
            )

        if bp.get("ttft_ms", 0) > 0 and prompt.get("ttft_ms", 0) > bp["ttft_ms"] * (1 + max_regression):
            violations.append(
                f"{name}: ttft regressed {(prompt['ttft_ms'] / bp['ttft_ms'] - 1) * 100:.1f}%"
            )

    return violations


def check_suspicion(
    composite_score: float,
    constraints: dict | None = None,
) -> list[str]:
    """Check for suspiciously large gains. Returns warnings (not auto-reject)."""
    c = constraints or {}
    max_gain = c.get("max_suspicion_gain", 0.30)

    warnings = []
    gain = composite_score - 1.0
    if gain > max_gain:
        warnings.append(
            f"Suspiciously large gain: {gain * 100:.1f}% (threshold {max_gain * 100:.0f}%)"
        )
    return warnings


def check_variance(
    run_results: list[list[dict]],
    constraints: dict | None = None,
) -> list[str]:
    """Check if variance across runs is too high. Returns warnings."""
    c = constraints or {}
    max_cv = c.get("variance_max_cv", 0.15)

    warnings = []
    if not run_results or len(run_results) < 2:
        return warnings

    prompt_names = {r["name"] for run in run_results for r in run}
    for name in prompt_names:
        values = [r["gen_tps"] for run in run_results for r in run if r["name"] == name]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if mean <= 0:
            continue
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        stddev = variance ** 0.5
        cv = stddev / mean
        if cv > max_cv:
            warnings.append(
                f"{name}: gen_tps CV={cv:.2f} exceeds threshold {max_cv:.2f} "
                f"(mean={mean:.1f}, stddev={stddev:.1f})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def load_baseline(bench_dir: Path) -> dict | None:
    """Load baseline.json from bench directory. Returns None if not found or corrupt."""
    path = bench_dir / "baseline.json"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except (ValueError, OSError) as exc:
        print(f"WARNING: corrupt baseline file {path}: {exc}", file=sys.stderr)
        return None


def save_baseline(bench_dir: Path, data: dict):
    """Save baseline.json to bench directory (atomic write via tmp+rename)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / "baseline.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.rename(path)


def save_run(bench_dir: Path, data: dict, timestamp: str):
    """Save a run result as run_<timestamp>.json (atomic write via tmp+rename)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / f"run_{timestamp}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.rename(path)


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
    fingerprint_match: bool | None = None,
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
    if fingerprint_match is not None:
        print(f"fingerprint_match: {str(fingerprint_match).lower()}")


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


# ---------------------------------------------------------------------------
# Per-cycle structured logging
# ---------------------------------------------------------------------------

def next_cycle_id() -> int:
    """Get next cycle ID from existing cycle files."""
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CYCLES_DIR.glob("cycle_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem  # cycle_0007
    return int(last.split("_")[1]) + 1


def save_cycle(cycle_id: int, data: dict):
    """Save cycle_NNNN.json to cycles directory (atomic write via tmp+rename)."""
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    path = CYCLES_DIR / f"cycle_{cycle_id:04d}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    tmp.rename(path)


def load_cycles() -> list[dict]:
    """Load all cycle JSON files, sorted by cycle_id."""
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    cycles = []
    for path in sorted(CYCLES_DIR.glob("cycle_*.json")):
        try:
            with open(path, "rb") as f:
                cycles.append(orjson.loads(f.read()))
        except (ValueError, OSError) as exc:
            print(f"WARNING: skipping corrupt cycle file {path.name}: {exc}", file=sys.stderr)
    return cycles


def build_cycle_data(
    cycle_id: int,
    git_info: dict,
    optimizer_info: dict,
    config_snapshot: dict,
    text_results: dict | None,
    vlm_results: dict | None,
    verification: dict,
    cumulative: dict,
    decision: str,
    decision_reason: str,
) -> dict:
    """Build the per-cycle JSON structure."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}
    if text_results:
        results["text"] = text_results
    if vlm_results:
        results["vlm"] = vlm_results

    return {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "git": git_info,
        "optimizer": optimizer_info,
        "config_snapshot": config_snapshot,
        "results": results,
        "verification": verification,
        "cumulative": cumulative,
        "decision": decision,
        "decision_reason": decision_reason,
    }
