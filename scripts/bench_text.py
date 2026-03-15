#!/usr/bin/env python3
"""Text-only benchmark for mlx-lm inference path.

Loads a model directly via mlx-lm (no HTTP server) and measures generation
performance with fixed prompts. Produces grep-friendly stdout and detailed
JSON results.

Usage:
    uv run scripts/bench_text.py
    uv run scripts/bench_text.py --model-path <path>
    uv run scripts/bench_text.py --reset-baseline
    uv run scripts/bench_text.py --runs 5
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load as lm_load
from mlx_lm.generate import stream_generate as lm_stream_generate
from mlx_lm.sample_utils import make_sampler

from bench_common import (
    TEXT_DIR,
    baseline_metrics_from_result,
    build_result_data,
    check_hard_constraints,
    compute_composite_score,
    ensure_dirs,
    get_hardware_info,
    load_baseline,
    print_results,
    save_baseline,
    save_run,
    sync_barrier,
)


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google_gemma-3-27b-it-mlx-bf16"


def resolve_model_path(model_path: str | None) -> str:
    """Resolve model path -- use provided path or default HF model ID."""
    if model_path:
        return model_path
    # Try HF cache first via hub
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(f"mlx-community/{DEFAULT_MODEL}")
    except Exception:
        return f"mlx-community/{DEFAULT_MODEL}"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "name": "short",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
    },
    {
        "name": "medium",
        "messages": [
            {"role": "user", "content": "Explain how hash tables work in 3-4 sentences."},
        ],
    },
    {
        "name": "long",
        "messages": [
            {"role": "system", "content": (
                "You are a senior software engineer with deep expertise in systems programming, "
                "distributed systems, and performance optimization. Provide detailed, technically "
                "accurate responses with concrete examples."
            )},
            {"role": "user", "content": (
                "I'm designing a write-ahead log (WAL) for an embedded database that needs to "
                "handle 100k writes per second on consumer hardware. The log needs to support "
                "crash recovery, group commit for batching fsync calls, and efficient truncation "
                "of committed entries. What data structure should I use for the in-memory "
                "representation, how should I handle the on-disk format for crash safety, and "
                "what are the key trade-offs between throughput and durability? Please discuss "
                "the pros and cons of at least two approaches."
            )},
        ],
    },
]

MAX_TOKENS = 256
SEED = 42


# ---------------------------------------------------------------------------
# Single prompt benchmark
# ---------------------------------------------------------------------------

def bench_prompt(
    model,
    tokenizer,
    prompt: dict,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run a single prompt and measure performance metrics.

    Returns dict with: gen_tps, ttft_ms, prefill_tps, completion_tokens, prompt_tokens
    """
    messages = prompt["messages"]

    # Tokenize via chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(formatted)
    num_prompt_tokens = len(prompt_tokens)

    # Build sampler
    mx.random.seed(SEED)
    sampler = make_sampler(temp=0.0)  # greedy for reproducibility

    # Track timing
    completion_tokens = 0
    ttft_ms = 0.0
    gen_start = 0.0

    # Sync before starting
    sync_barrier()
    start = time.perf_counter()

    for _ in lm_stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        sampler=sampler,
        max_tokens=max_tokens,
    ):
        if completion_tokens == 0:
            # First token received -- measure TTFT and prefill
            sync_barrier()
            now = time.perf_counter()
            ttft_ms = (now - start) * 1000
            gen_start = now

        completion_tokens += 1

    # Final sync
    sync_barrier()
    end = time.perf_counter()

    # Compute metrics
    if completion_tokens > 1 and gen_start > 0:
        gen_time_s = end - gen_start
        gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0
    else:
        gen_tps = 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s if prefill_time_s > 0 else 0.0

    return {
        "name": prompt["name"],
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": num_prompt_tokens,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path: str,
    runs: int = 3,
    warmup: int = 1,
    reset_baseline: bool = False,
) -> dict:
    """Run the full text benchmark suite."""
    ensure_dirs()

    # Load model
    print(f"Loading model: {model_path}", file=sys.stderr)
    loaded = lm_load(model_path)
    model, tokenizer = loaded[0], loaded[1]
    model_name = Path(model_path).name if "/" not in model_path or Path(model_path).exists() else model_path.split("/")[-1]
    print(f"Model loaded: {model_name}", file=sys.stderr)

    # Reset peak memory tracking
    mx.reset_peak_memory()

    hardware = get_hardware_info()
    all_prompt_results = []

    for prompt in PROMPTS:
        # Warmup
        for w in range(warmup):
            print(f"  warmup {w + 1}/{warmup}: {prompt['name']}...", end="\r", file=sys.stderr)
            bench_prompt(model, tokenizer, prompt)
            mx.clear_cache()

        # Measured runs
        prompt_runs = []
        for r in range(runs):
            print(f"  run {r + 1}/{runs}: {prompt['name']}...   ", end="\r", file=sys.stderr)
            result = bench_prompt(model, tokenizer, prompt)
            prompt_runs.append(result)
            mx.clear_cache()

        # Average across runs
        avg_result = {
            "name": prompt["name"],
            "gen_tps": sum(r["gen_tps"] for r in prompt_runs) / len(prompt_runs),
            "ttft_ms": sum(r["ttft_ms"] for r in prompt_runs) / len(prompt_runs),
            "prefill_tps": sum(r["prefill_tps"] for r in prompt_runs) / len(prompt_runs),
            "completion_tokens": sum(r["completion_tokens"] for r in prompt_runs) / len(prompt_runs),
            "prompt_tokens": prompt_runs[0]["prompt_tokens"],
        }
        all_prompt_results.append(avg_result)
        print(f"  {prompt['name']}: gen={avg_result['gen_tps']:.1f} tps, "
              f"ttft={avg_result['ttft_ms']:.1f}ms, "
              f"prefill={avg_result['prefill_tps']:.1f} tps", file=sys.stderr)

    # Aggregate metrics
    avg_gen_tps = sum(r["gen_tps"] for r in all_prompt_results) / len(all_prompt_results)
    avg_ttft_ms = sum(r["ttft_ms"] for r in all_prompt_results) / len(all_prompt_results)
    avg_prefill_tps = sum(r["prefill_tps"] for r in all_prompt_results) / len(all_prompt_results)
    peak_memory_gb = mx.get_peak_memory() / (1024 ** 3)
    total_runs = len(PROMPTS) * runs

    # Build result
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    metrics = {
        "avg_gen_tps": round(avg_gen_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "avg_prefill_tps": round(avg_prefill_tps, 1),
        "peak_memory_gb": round(peak_memory_gb, 1),
    }

    # Load or create baseline
    baseline_data = load_baseline(TEXT_DIR)
    if baseline_data is None or reset_baseline:
        # This IS the baseline
        result_data = build_result_data(
            bench="text", model=model_name, timestamp=timestamp,
            composite_score=1.0, metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        save_baseline(TEXT_DIR, result_data)
        save_run(TEXT_DIR, result_data, timestamp.replace(":", ""))
        composite_score = 1.0
        print("\nBaseline established.", file=sys.stderr)
    else:
        baseline_metrics = baseline_metrics_from_result(baseline_data)
        composite_score = compute_composite_score(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb, baseline_metrics,
        )
        avg_completion = sum(r["completion_tokens"] for r in all_prompt_results) / len(all_prompt_results)
        violations = check_hard_constraints(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            avg_completion, baseline_metrics,
        )
        if violations:
            print(f"\nHard constraint violations:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)

        result_data = build_result_data(
            bench="text", model=model_name, timestamp=timestamp,
            composite_score=round(composite_score, 4), metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        save_run(TEXT_DIR, result_data, timestamp.replace(":", ""))

    # Print grep-friendly output
    print("", file=sys.stderr)
    print_results(
        composite_score=composite_score,
        avg_gen_tps=avg_gen_tps,
        avg_ttft_ms=avg_ttft_ms,
        avg_prefill_tps=avg_prefill_tps,
        peak_memory_gb=peak_memory_gb,
        runs=total_runs,
        model=model_name,
        bench="text",
    )

    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text-only benchmark (mlx-lm path)")
    parser.add_argument("--model-path", default=None, help="Model path or HF repo ID")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per prompt")
    parser.add_argument("--reset-baseline", action="store_true", help="Re-establish baseline")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    run_benchmark(
        model_path=model_path,
        runs=args.runs,
        warmup=args.warmup,
        reset_baseline=args.reset_baseline,
    )


if __name__ == "__main__":
    main()
