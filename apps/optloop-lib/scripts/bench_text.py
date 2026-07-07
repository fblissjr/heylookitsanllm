#!/usr/bin/env python3
"""Text-only benchmark for mlx-lm inference path.

Loads a model directly via mlx-lm (no HTTP server) and measures generation
performance with fixed prompts. Produces grep-friendly stdout and detailed
JSON results. Includes output fingerprinting for correctness verification.

Usage:
    uv run apps/optloop-lib/scripts/bench_text.py
    uv run apps/optloop-lib/scripts/bench_text.py --model-path <path>
    uv run apps/optloop-lib/scripts/bench_text.py --reset-baseline
    uv run apps/optloop-lib/scripts/bench_text.py --runs 5
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
    check_fingerprints,
    check_hard_constraints,
    check_per_prompt_constraints,
    check_suspicion,
    check_variance,
    compute_composite_score,
    ensure_dirs,
    fingerprint_output,
    get_bench_params,
    get_constraints,
    get_hardware_info,
    get_scoring_weights,
    load_baseline,
    load_config,
    print_results,
    resolve_model_from_toml,
    save_baseline,
    save_run,
    sync_barrier,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# A genuinely long-context document (~2.5-3k prompt tokens) so at least one
# workload exercises real prefill scaling + a large KV cache, not just decode.
# Fixed + coherent so the greedy fingerprint is stable. Added BEFORE the first
# baseline (plan Phase 5 item 1) -- adding it later would invalidate baselines.
LONG_CONTEXT_DOC = """\
Design note: a columnar analytics engine for append-mostly event data.

Section 1 -- Storage layout. Events arrive as flat records with a fixed set of
typed columns plus a sparse map of user-defined attributes. The engine stores
each column in its own segment file, compressed independently, so a scan that
touches three of forty columns reads only those three segments. Segments are
immutable once sealed; a background writer accumulates rows in an in-memory row
group until it reaches a target of one million rows or five minutes elapse,
whichever comes first, then encodes and seals the group. Sealing is the only
point at which encoding decisions are made, which keeps the hot write path free
of per-row branching.

Section 2 -- Encodings. Integer columns use frame-of-reference plus bit-packing:
the minimum value in a row group is subtracted from every value, and the residuals
are packed into the smallest number of bits that covers the observed range. Low
cardinality string columns use dictionary encoding, where each distinct value is
assigned a small integer code and the codes are themselves bit-packed. Timestamps,
which are usually monotonic within a group, use delta-of-delta encoding so that a
steady arrival rate compresses to nearly nothing. The encoder picks per-segment
between these schemes by sampling the first few thousand values and estimating the
compressed size of each candidate, choosing the smallest.

Section 3 -- The zone map. Every sealed segment carries a small header recording
the minimum and maximum value, the null count, and the distinct-count estimate for
its column. A query planner consults these zone maps before reading any segment
data: a predicate of the form price greater than one thousand can skip every
segment whose recorded maximum is at or below one thousand. Zone maps are cheap to
build during sealing and cheap to keep resident in memory, and they are the primary
mechanism by which the engine avoids I/O on selective queries. For high cardinality
columns the engine also builds a per-segment bloom filter to accelerate equality
predicates that zone maps cannot prune.

Section 4 -- Late materialization. The execution engine defers reading a column's
values until a row has survived every predicate that can be evaluated on cheaper
columns. A filter on an indexed timestamp runs first and produces a selection
vector of surviving row positions; only then does the engine gather the wide,
expensive columns at exactly those positions. This ordering matters because gather
cost is proportional to surviving rows, not scanned rows, so a query that keeps one
row in ten thousand pays almost nothing for its wide projections. The planner
orders predicates by estimated selectivity divided by estimated evaluation cost.

Section 5 -- Vectorized execution. Operators process batches of a few thousand
values at a time rather than one row at a time. Each operator consumes a batch,
produces a batch, and passes it to the next operator, which keeps intermediate
results in cache and amortizes per-call overhead across the whole batch. Selection
is represented as a dense vector of surviving positions that flows between
operators, so a filter never has to physically compact the columns it filters. The
batch size is tuned so that a working set of all active operator buffers fits in
the last-level cache of a single core.

Section 6 -- Concurrency and durability. Writes append to a durable log before the
in-memory row group acknowledges them, so a crash loses at most the unsealed group,
which recovery rebuilds by replaying the log tail. Readers see a consistent
snapshot defined by the set of sealed segments visible at query start; a segment
sealed during a long-running scan is simply invisible to that scan. Compaction
merges many small sealed segments into fewer large ones to bound the number of
files a scan must open, and it runs opportunistically when write pressure is low.
Compaction never blocks readers because it produces new segments and atomically
swaps them in, leaving the old segments for in-flight scans to finish with.

Section 7 -- Trade-offs. Column-at-a-time storage penalizes point lookups that
want a whole row, since it must gather from every segment, but the engine is not
built for point lookups; it is built for aggregations over large row counts. The
one-million-row group target trades recovery granularity against encoding
efficiency: larger groups compress better and prune better but lose more work on a
crash and delay the visibility of freshly written data. The sampling-based encoder
can occasionally mispredict the best scheme when a group's tail differs sharply
from its head, which the engine accepts in exchange for keeping sealing fast.
"""

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
    # -- Multi-turn prompts (test context carry and growing KV cache) -----------
    {
        "name": "multi_turn_short",
        "messages": [
            {"role": "user", "content": "What are the three laws of thermodynamics?"},
            {"role": "assistant", "content": (
                "The three laws of thermodynamics are: (1) Energy cannot be created or "
                "destroyed, only transformed from one form to another (conservation of energy). "
                "(2) The total entropy of an isolated system always increases over time; heat "
                "flows spontaneously from hot to cold, never the reverse. (3) As temperature "
                "approaches absolute zero, the entropy of a perfect crystal approaches zero."
            )},
            {"role": "user", "content": (
                "How does the second law relate to the concept of heat death of the universe?"
            )},
        ],
    },
    {
        "name": "multi_turn_long",
        "messages": [
            {"role": "system", "content": (
                "You are a database architect with expertise in distributed storage systems. "
                "Give precise, implementation-oriented answers."
            )},
            {"role": "user", "content": (
                "What is the difference between B-trees and LSM-trees for on-disk storage?"
            )},
            {"role": "assistant", "content": (
                "B-trees store data in a balanced tree of fixed-size pages, typically 4-16 KB. "
                "Reads are fast because locating a key requires O(log N) page reads. Writes "
                "require random I/O to update pages in place, plus WAL writes for crash safety. "
                "LSM-trees buffer writes in an in-memory memtable, then flush sorted runs to "
                "disk. Reads may need to check multiple levels and merge results. Writes are "
                "sequential and fast, but background compaction is required to bound read "
                "amplification. B-trees favor read-heavy workloads; LSM-trees favor write-heavy "
                "workloads."
            )},
            {"role": "user", "content": (
                "Given that trade-off, how do modern systems like RocksDB tune compaction "
                "strategies to reduce read amplification while maintaining write throughput?"
            )},
            {"role": "assistant", "content": (
                "RocksDB offers several compaction strategies. Level compaction (default) "
                "organizes data into levels where each level is ~10x larger than the previous. "
                "When a level fills, its SST files are merged into the next level. This bounds "
                "read amplification to the number of levels but causes significant write "
                "amplification. Universal compaction reduces write amplification by allowing "
                "more sorted runs before triggering compaction, at the cost of higher read "
                "amplification. FIFO compaction simply drops the oldest data and is used for "
                "time-series workloads. Subcompactions parallelize work within a single "
                "compaction job. Bloom filters on each SST file reduce point-read I/O."
            )},
            {"role": "user", "content": (
                "How would you design a compaction scheduler that adapts its strategy based on "
                "real-time workload characteristics, switching between level and universal "
                "compaction depending on the current read/write ratio?"
            )},
        ],
    },
    # -- Long-context prompt (large prefill + KV cache; the only workload that
    #    stresses prompt processing at scale) -----------------------------------
    {
        "name": "long_context",
        "messages": [
            {"role": "user", "content": (
                LONG_CONTEXT_DOC
                + "\n\nUsing only the design note above, answer concisely: (1) which two "
                "mechanisms let a selective query avoid reading segment data, and how do they "
                "differ; (2) why does late materialization make the cost of wide projections "
                "depend on surviving rows rather than scanned rows; (3) name one concrete "
                "trade-off the one-million-row group target imposes."
            )},
        ],
    },
]


def resolve_model_path(model_path: str | None, config: dict) -> str:
    """Resolve model path from CLI arg, bench_config.toml, or models.toml.

    Priority: CLI --model-path > bench_config.toml model ID looked up in the
    server's models.toml (local path, no download) > HF download fallback.
    """
    if model_path:
        return model_path
    text_config = config.get("bench", {}).get("text", {})
    model_id = text_config.get("model", "mlx-community/google_gemma-3-27b-it-mlx-bf16")
    local_path = resolve_model_from_toml(model_id)
    if local_path:
        return local_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id)
    except Exception:
        return model_id


def resolve_draft_path(draft_path: str | None, config: dict, use_config_draft: bool) -> str | None:
    """Resolve the speculative-decoding draft model, or None.

    Priority: CLI --draft-model-path > (if --spec-decode) [bench.text].draft_model
    looked up in models.toml, else HF download > None (no speculative decoding).
    """
    if draft_path:
        return draft_path
    if not use_config_draft:
        return None
    draft_id = config.get("bench", {}).get("text", {}).get("draft_model")
    if not draft_id:
        return None
    local_path = resolve_model_from_toml(draft_id)
    if local_path:
        return local_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(draft_id)
    except Exception:
        return draft_id


# ---------------------------------------------------------------------------
# Single prompt benchmark
# ---------------------------------------------------------------------------

def bench_prompt(
    model,
    tokenizer,
    prompt: dict,
    max_tokens: int = 256,
    seed: int = 42,
    draft_model=None,
) -> dict:
    """Run a single prompt and measure performance metrics.

    Returns dict with: gen_tps, ttft_ms, prefill_tps, completion_tokens,
    prompt_tokens, token_ids, fingerprint

    When draft_model is provided, stream_generate uses speculative decoding.
    It is LOSSLESS (the target verifies every draft token), so the greedy
    fingerprint must match the non-speculative baseline -- that identity is the
    correctness guard for the speedup.
    """
    messages = prompt["messages"]

    # Tokenize via chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(formatted)
    num_prompt_tokens = len(prompt_tokens)

    # Build sampler
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0)  # greedy for reproducibility

    # Track timing and tokens
    completion_tokens = 0
    token_ids = []
    ttft_ms = 0.0
    gen_start = 0.0

    # Sync before starting
    sync_barrier()
    start = time.perf_counter()

    for response in lm_stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        sampler=sampler,
        max_tokens=max_tokens,
        draft_model=draft_model,
    ):
        if completion_tokens == 0:
            # First token received -- measure TTFT and prefill
            sync_barrier()
            now = time.perf_counter()
            ttft_ms = (now - start) * 1000
            gen_start = now

        # Collect token ID for fingerprinting
        if hasattr(response, "token"):
            token_ids.append(int(response.token))

        completion_tokens += 1

    # Final sync
    sync_barrier()
    end = time.perf_counter()

    if completion_tokens == 0:
        raise RuntimeError(f"No tokens generated for prompt '{prompt['name']}'")

    # Compute metrics
    if completion_tokens > 1 and gen_start > 0:
        gen_time_s = end - gen_start
        gen_tps = (completion_tokens - 1) / gen_time_s if gen_time_s > 0 else 0.0
    else:
        gen_tps = 0.0

    prefill_time_s = ttft_ms / 1000 if ttft_ms > 0 else 0.001
    prefill_tps = num_prompt_tokens / prefill_time_s if prefill_time_s > 0 else 0.0

    # Compute fingerprint
    fp = fingerprint_output(token_ids) if token_ids else ""

    return {
        "name": prompt["name"],
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": num_prompt_tokens,
        "token_ids": token_ids,
        "fingerprint": fp,
    }


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    model_path: str,
    runs: int = 3,
    warmup: int = 1,
    max_tokens: int = 256,
    seed: int = 42,
    reset_baseline: bool = False,
    scoring_weights: dict | None = None,
    constraints: dict | None = None,
    draft_model_path: str | None = None,
) -> dict:
    """Run the full text benchmark suite.

    draft_model_path enables speculative decoding (compared against the
    non-speculative baseline). Establish the baseline WITHOUT a draft, then run
    WITH one -- gen_tps should rise and the fingerprint must still match.
    """
    ensure_dirs()

    # Load model
    print(f"Loading model: {model_path}", file=sys.stderr)
    loaded = lm_load(model_path)
    model, tokenizer = loaded[0], loaded[1]
    model_name = Path(model_path).name if "/" not in model_path or Path(model_path).exists() else model_path.split("/")[-1]
    print(f"Model loaded: {model_name}", file=sys.stderr)

    # Optional draft model for speculative decoding (must share the tokenizer).
    draft_model = None
    draft_name = None
    if draft_model_path:
        print(f"Loading draft model: {draft_model_path}", file=sys.stderr)
        draft_model = lm_load(draft_model_path)[0]
        draft_name = Path(draft_model_path).name if "/" not in draft_model_path or Path(draft_model_path).exists() else draft_model_path.split("/")[-1]
        print(f"Draft loaded: {draft_name} (speculative decoding ON)", file=sys.stderr)

    # Reset peak memory tracking
    mx.reset_peak_memory()

    hardware = get_hardware_info()
    all_prompt_results = []
    all_run_results = []  # for variance checking

    for prompt in PROMPTS:
        # Warmup
        for w in range(warmup):
            print(f"  warmup {w + 1}/{warmup}: {prompt['name']}...", end="\r", file=sys.stderr)
            bench_prompt(model, tokenizer, prompt, max_tokens=max_tokens, seed=seed, draft_model=draft_model)
            mx.clear_cache()

        # Measured runs
        prompt_runs = []
        for r in range(runs):
            print(f"  run {r + 1}/{runs}: {prompt['name']}...   ", end="\r", file=sys.stderr)
            result = bench_prompt(model, tokenizer, prompt, max_tokens=max_tokens, seed=seed, draft_model=draft_model)
            prompt_runs.append(result)
            mx.clear_cache()

        # Store per-run results for variance checking
        all_run_results.append(prompt_runs)

        # Average across runs (use last run's fingerprint -- should be identical for greedy)
        avg_result = {
            "name": prompt["name"],
            "gen_tps": sum(r["gen_tps"] for r in prompt_runs) / len(prompt_runs),
            "ttft_ms": sum(r["ttft_ms"] for r in prompt_runs) / len(prompt_runs),
            "prefill_tps": sum(r["prefill_tps"] for r in prompt_runs) / len(prompt_runs),
            "completion_tokens": sum(r["completion_tokens"] for r in prompt_runs) / len(prompt_runs),
            "prompt_tokens": prompt_runs[0]["prompt_tokens"],
            "fingerprint": prompt_runs[-1]["fingerprint"],
        }
        all_prompt_results.append(avg_result)
        print(f"  {prompt['name']}: gen={avg_result['gen_tps']:.1f} tps, "
              f"ttft={avg_result['ttft_ms']:.1f}ms, "
              f"prefill={avg_result['prefill_tps']:.1f} tps, "
              f"fp={avg_result['fingerprint']}", file=sys.stderr)

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
    fingerprint_match = True
    all_violations = []
    suspicion_warnings = []
    variance_warnings = []

    if baseline_data is None or reset_baseline:
        # This IS the baseline
        result_data = build_result_data(
            bench="text", model=model_name, timestamp=timestamp,
            composite_score=1.0, metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        result_data["draft_model"] = draft_name
        save_baseline(TEXT_DIR, result_data)
        save_run(TEXT_DIR, result_data, timestamp.replace(":", ""))
        composite_score = 1.0
        print("\nBaseline established.", file=sys.stderr)
    else:
        baseline_metrics = baseline_metrics_from_result(baseline_data)
        composite_score = compute_composite_score(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            baseline_metrics, weights=scoring_weights,
        )
        avg_completion = sum(r["completion_tokens"] for r in all_prompt_results) / len(all_prompt_results)

        # Hard constraints
        violations = check_hard_constraints(
            avg_gen_tps, avg_ttft_ms, avg_prefill_tps, peak_memory_gb,
            avg_completion, baseline_metrics, constraints=constraints,
        )
        all_violations.extend(violations)

        # Per-prompt constraints
        per_prompt_violations = check_per_prompt_constraints(
            all_prompt_results, baseline_data, constraints=constraints,
        )
        all_violations.extend(per_prompt_violations)

        # Fingerprint check
        fp_violations = check_fingerprints(all_prompt_results, baseline_data)
        if fp_violations:
            fingerprint_match = False
            all_violations.extend(fp_violations)

        # Suspicion check
        suspicion_warnings = check_suspicion(composite_score, constraints=constraints)

        # Variance check
        transposed_runs = []
        for run_idx in range(runs):
            run_slice = [per_prompt[run_idx] for per_prompt in all_run_results]
            transposed_runs.append(run_slice)
        variance_warnings = check_variance(transposed_runs, constraints=constraints)

        if all_violations:
            print(f"\nHard constraint violations:", file=sys.stderr)
            for v in all_violations:
                print(f"  - {v}", file=sys.stderr)

        if suspicion_warnings:
            print(f"\nSuspicion warnings:", file=sys.stderr)
            for w in suspicion_warnings:
                print(f"  - {w}", file=sys.stderr)

        if variance_warnings:
            print(f"\nVariance warnings:", file=sys.stderr)
            for w in variance_warnings:
                print(f"  - {w}", file=sys.stderr)

        result_data = build_result_data(
            bench="text", model=model_name, timestamp=timestamp,
            composite_score=round(composite_score, 4), metrics=metrics,
            per_prompt=all_prompt_results, hardware=hardware,
        )
        # Add verification metadata to result
        result_data["draft_model"] = draft_name
        result_data["fingerprint_match"] = fingerprint_match
        result_data["hard_constraint_violations"] = all_violations
        result_data["suspicion_warnings"] = suspicion_warnings
        result_data["variance_warnings"] = variance_warnings
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
        fingerprint_match=fingerprint_match if not reset_baseline else None,
    )

    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Text-only benchmark (mlx-lm path)")
    parser.add_argument("--model-path", default=None, help="Model path or HF repo ID")
    parser.add_argument("--runs", type=int, default=None, help="Measured runs per prompt (overrides config)")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup runs per prompt (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generation tokens (overrides config)")
    parser.add_argument("--reset-baseline", action="store_true", help="Re-establish baseline")
    parser.add_argument("--draft-model-path", default=None, help="Draft model path/id for speculative decoding")
    parser.add_argument("--spec-decode", action="store_true", help="Enable speculative decoding using [bench.text].draft_model from config")
    parser.add_argument("--config", default=None, help="Path to bench_config.toml")
    args = parser.parse_args()

    if args.reset_baseline and (args.draft_model_path or args.spec_decode):
        print("Refusing to reset the baseline WITH speculative decoding on -- the "
              "baseline must be non-speculative so the spec-decode run has something "
              "lossless to compare against.", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = load_config(Path(args.config) if args.config else None)
    bench_params = get_bench_params(config)
    scoring_weights = get_scoring_weights(config)
    constraints = get_constraints(config)

    # CLI args override config
    runs = args.runs if args.runs is not None else bench_params["runs"]
    warmup = args.warmup if args.warmup is not None else bench_params["warmup"]
    max_tokens = args.max_tokens if args.max_tokens is not None else bench_params["max_tokens"]
    seed = bench_params["seed"]

    model_path = resolve_model_path(args.model_path, config)
    draft_model_path = resolve_draft_path(args.draft_model_path, config, args.spec_decode)
    run_benchmark(
        model_path=model_path,
        runs=runs,
        warmup=warmup,
        max_tokens=max_tokens,
        seed=seed,
        reset_baseline=args.reset_baseline,
        scoring_weights=scoring_weights,
        constraints=constraints,
        draft_model_path=draft_model_path,
    )


if __name__ == "__main__":
    main()
