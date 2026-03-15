#!/usr/bin/env python3
"""Benchmark results analysis tool.

Reads results.tsv and per-run JSON files to produce summary tables,
per-metric breakdowns, and progress charts.

Usage:
    uv run scripts/bench_analysis.py
    uv run scripts/bench_analysis.py --no-chart
"""

import argparse
import sys
from pathlib import Path

import orjson

from bench_common import DATA_DIR, REPO_ROOT, TEXT_DIR, VLM_DIR


RESULTS_TSV = REPO_ROOT / "results.tsv"


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

def load_results_tsv() -> list[dict]:
    """Load results.tsv into list of dicts."""
    if not RESULTS_TSV.exists():
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            rows.append(row)
    return rows


def load_json_runs(bench_dir: Path) -> list[dict]:
    """Load all run_*.json files from a bench directory."""
    if not bench_dir.exists():
        return []
    runs = []
    for path in sorted(bench_dir.glob("run_*.json")):
        with open(path, "rb") as f:
            runs.append(orjson.loads(f.read()))
    return runs


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]):
    """Print experiment summary from results.tsv."""
    if not rows:
        print("No results found in results.tsv")
        return

    kept = [r for r in rows if r.get("status") == "keep"]
    discarded = [r for r in rows if r.get("status") == "discard"]
    crashed = [r for r in rows if r.get("status") == "crash"]
    baseline = [r for r in rows if r.get("status") == "baseline"]

    print("=== Experiment Summary ===")
    print(f"  Total:     {len(rows)}")
    print(f"  Baseline:  {len(baseline)}")
    print(f"  Kept:      {len(kept)}")
    print(f"  Discarded: {len(discarded)}")
    print(f"  Crashed:   {len(crashed)}")
    print()

    if kept:
        print("=== Kept Experiments ===")
        header = f"{'Commit':<10} {'Text':>8} {'VLM':>8} {'TxtTPS':>8} {'VlmTPS':>8} {'Description'}"
        print(header)
        print("-" * len(header))
        for r in kept:
            print(f"{r.get('commit', '?'):<10} "
                  f"{r.get('text_score', '?'):>8} "
                  f"{r.get('vlm_score', '?'):>8} "
                  f"{r.get('text_tps', '?'):>8} "
                  f"{r.get('vlm_tps', '?'):>8} "
                  f"{r.get('description', '')}")
        print()


# ---------------------------------------------------------------------------
# Per-metric breakdown from JSON runs
# ---------------------------------------------------------------------------

def print_metric_breakdown(bench_type: str, runs: list[dict]):
    """Print per-metric breakdown for a bench type."""
    if not runs:
        print(f"No {bench_type} runs found.")
        return

    print(f"=== {bench_type.upper()} Metric History ===")
    header = f"{'Timestamp':<22} {'Score':>8} {'GenTPS':>8} {'TTFT':>10} {'Prefill':>10} {'Memory':>8}"
    print(header)
    print("-" * len(header))

    for run in runs:
        m = run.get("metrics", {})
        print(f"{run.get('timestamp', '?'):<22} "
              f"{run.get('composite_score', 0):>8.4f} "
              f"{m.get('avg_gen_tps', 0):>8.1f} "
              f"{m.get('avg_ttft_ms', 0):>10.1f} "
              f"{m.get('avg_prefill_tps', 0):>10.1f} "
              f"{m.get('peak_memory_gb', 0):>8.1f}")
    print()


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------

def print_rankings(rows: list[dict]):
    """Rank kept experiments by combined score delta."""
    kept = [r for r in rows if r.get("status") == "keep" and r.get("status") != "baseline"]

    if not kept:
        print("No kept experiments to rank.")
        return

    # Compute combined delta
    ranked = []
    for r in kept:
        try:
            text_score = float(r.get("text_score", 1.0))
            vlm_score = float(r.get("vlm_score", 1.0))
            combined_delta = (text_score - 1.0) + (vlm_score - 1.0)
            ranked.append((combined_delta, r))
        except (ValueError, TypeError):
            continue

    ranked.sort(key=lambda x: x[0], reverse=True)

    print("=== Top Improvements (by combined delta) ===")
    header = f"{'#':>3} {'Delta':>8} {'Text':>8} {'VLM':>8} {'Description'}"
    print(header)
    print("-" * len(header))
    for i, (delta, r) in enumerate(ranked[:10], 1):
        print(f"{i:>3} {delta:>+8.4f} "
              f"{r.get('text_score', '?'):>8} "
              f"{r.get('vlm_score', '?'):>8} "
              f"{r.get('description', '')}")
    print()


# ---------------------------------------------------------------------------
# Progress chart
# ---------------------------------------------------------------------------

def generate_chart(rows: list[dict]):
    """Generate progress.png chart with dual-axis scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping chart generation.", file=sys.stderr)
        return

    kept = [r for r in rows if r.get("status") in ("keep", "baseline")]
    if len(kept) < 2:
        print("Not enough data points for chart.", file=sys.stderr)
        return

    indices = list(range(len(kept)))
    text_scores = []
    vlm_scores = []

    for r in kept:
        try:
            text_scores.append(float(r.get("text_score", 1.0)))
        except (ValueError, TypeError):
            text_scores.append(1.0)
        try:
            vlm_scores.append(float(r.get("vlm_score", 1.0)))
        except (ValueError, TypeError):
            vlm_scores.append(1.0)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Experiment #")
    ax1.set_ylabel("Text Score", color="tab:blue")
    ax1.plot(indices, text_scores, "o-", color="tab:blue", label="Text")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("VLM Score", color="tab:orange")
    ax2.plot(indices, vlm_scores, "s-", color="tab:orange", label="VLM")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Optimization Progress")
    fig.tight_layout()

    chart_path = DATA_DIR / "progress.png"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark results analysis")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    # TSV summary
    rows = load_results_tsv()
    print_summary(rows)
    print_rankings(rows)

    # Per-bench JSON breakdown
    text_runs = load_json_runs(TEXT_DIR)
    vlm_runs = load_json_runs(VLM_DIR)

    print_metric_breakdown("text", text_runs)
    print_metric_breakdown("vlm", vlm_runs)

    # Chart
    if not args.no_chart and rows:
        generate_chart(rows)


if __name__ == "__main__":
    main()
