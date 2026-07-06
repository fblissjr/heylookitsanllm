# Optloop-Lib Guide

Last updated: 2026-07-06

Optloop-lib is the repo's only inference-optimization benchmark harness. A Claude Code agent (or a human, manually) edits local forks of mlx-lm and mlx-vlm, benchmarks each change against a deterministic baseline, and keeps only verified improvements. It targets `apps/optloop-lib/repos/` -- editable-install fork clones of mlx-lm and mlx-vlm -- never `src/heylook_llm/`.

## Why "lib-only"

The app-level `apps/optloop` harness was deleted 2026-07-06. Its bench scripts loaded models via `mlx_lm.utils.load()` / `mlx_vlm.utils.load()` directly and never imported `heylook_llm`, so a change to `src/heylook_llm/` -- the thing it was chartered to optimize -- scored identically to no change at all. Optloop-lib doesn't have this problem: it benchmarks editable-install fork repos, and the thing under test genuinely IS the library being edited, so direct library calls are the correct measurement here.

Serving-path (HTTP) benchmarking is a separate, currently out-of-scope problem -- see the measurement section of `internal/backend/plan_2026-07.md`. One caveat matters more than usual right now: no optimization cycle has ever completed end-to-end with this harness. The first real run (a speculative-decoding attempt, per `apps/optloop-lib/CLAUDE.md`) doubles as harness validation, not just an optimization session.

## Quick start

```bash
cd apps/optloop-lib
uv sync
uv run scripts/bench_text.py --reset-baseline   # once, before experiments
uv run scripts/bench_text.py                    # compare vs baseline
uv run scripts/bench_vlm.py                     # VLM variant
uv run scripts/bench_analysis.py                # summarize cycles
uv run pytest tests/ -q                         # harness unit tests (65)
```

Or drive it with the agent:

```
/optloop-lib mar16-lib
```

Model paths resolve in this order: CLI arg > `bench_config.toml` model id > the repo root `models.toml` (local path, no download) > HF download fallback. The `models.toml` lookup was added 2026-07-06 so benches reuse whatever the server already has on disk instead of re-downloading multi-GB weights.

## Prerequisites

### Hardware
- Apple Silicon Mac (MLX requires Metal). Target hardware for the shipped configs: Mac Studio M2 Ultra, 192GB unified memory.
- 64GB+ unified memory recommended for the default 27B bench models (~30GB peak).

### Software
- Python 3.12+ managed by uv.
- `cd apps/optloop-lib && uv sync` -- own venv, own pyproject.toml, separate from the repo root. The editable installs resolve mlx-lm/mlx-vlm to the `repos/` forks; do NOT touch the forks' `setup.py`/`pyproject.toml` (they track upstream), and never commit fork changes into heylookitsanllm.
- Claude Code with the `/optloop-lib` skill (agent-driven loop) and `/optloop-analysis` skill (results summary).

### Models

Two models are wired into `apps/optloop-lib/bench_config.toml`:

| Bench | Model | Purpose |
|-------|-------|---------|
| Text | Gemma-3-27B bf16 | Pure text generation throughput |
| VLM | Qwen3.5-27B mxfp8 | Vision + text generation throughput |

They resolve via the CLI > bench_config.toml > models.toml > HF download order above. Check the root `models.toml` has entries with valid local `model_path`s before running -- otherwise the bench falls through to an HF download.

## Setup (before your first session)

1. Agree on a run tag (e.g. `mar16-lib`).
2. Create a branch in EACH fork repo:
   ```bash
   cd apps/optloop-lib/repos/mlx-lm && git checkout -b optloop-lib/<tag>
   cd apps/optloop-lib/repos/mlx-vlm && git checkout -b optloop-lib/<tag>
   ```
3. Read tier-1 files: `repos/mlx-lm/mlx_lm/generate.py`, `repos/mlx-lm/mlx_lm/models/cache.py`, `repos/mlx-lm/mlx_lm/sample_utils.py`.
4. Load `/mlx-skills:mlx` and `/mlx-skills:mlx-lm` for MLX domain knowledge.
5. Read `apps/optloop-lib/bench_config.toml` (off-limits to edit) and `docs/optimization_log.md` (the cross-session knowledge base -- read it before repeating work someone already tried).
6. Establish baselines:
   ```bash
   cd apps/optloop-lib && uv run scripts/bench_text.py --reset-baseline
   cd apps/optloop-lib && uv run scripts/bench_vlm.py --reset-baseline
   ```
7. Initialize `results.tsv` with the baseline row.
8. Run fork tests once to confirm a clean starting state.

## How the loop works

Each cycle has four phases:

### 1. Change

Snapshot both fork HEADs (`bench_common.snapshot_coderef()`), pick a tier-guided target, record a hypothesis, and edit source directly in `repos/mlx-lm/mlx_lm/` or `repos/mlx-vlm/mlx_vlm/`. Commit in the fork repo -- never in heylookitsanllm.

### 2. Test

If the changed module has fork tests, run them (`cd apps/optloop-lib/repos/mlx-lm && python -m pytest tests/ -x -q`). Any new failure blocks the benchmark step.

### 3. Benchmark

Both bench scripts run against current baselines:
- **Text bench**: loads via `mlx_lm.utils.load()`, runs 3 fixed prompts with greedy decoding.
- **VLM bench**: loads via `mlx_vlm.utils.load()`, runs 2 text + 2 vision prompts (synthetic PIL renders).

Each produces a composite score where 1.0 = baseline. Scores above 1.0 indicate improvement.

### 4. Verify

A 6-step verification runs before the keep/discard decision:

1. **Diff inspection** -- changes stay within `allowed_paths`, no `banned_diff_patterns` matches.
2. **Output fingerprints** -- SHA-256 of token IDs must match baseline.
3. **Fork test suite** -- no new failures.
4. **Per-prompt regression** -- no individual prompt regresses beyond threshold.
5. **Suspicion flags** -- >30% single-metric gain, sampler changes, or anything that looks like skipped work gets flagged (not auto-rejected).
6. **Variance check** -- coefficient of variation across runs must be below 15%.

If both scores are >= 1.0, fingerprints match, and tests pass: **KEEP**. Otherwise: **DISCARD**, reset via `git reset --hard <snapshot_commit>` in each fork repo.

## Understanding the benchmarks

### Text benchmark

Loads via `mlx_lm.utils.load()`, runs 3 fixed prompts (short, medium, long) through `mlx_lm.generate.stream_generate()` with greedy decoding (temp=0, seed=42). Metrics: `gen_tps` (decode throughput), `ttft_ms` (time to first token), `prefill_tps` (prompt processing throughput), `peak_memory_gb`. Warmup runs are unmeasured; measured runs are averaged per prompt, then across prompts.

### VLM benchmark

Loads via `mlx_vlm.utils.load()`, runs 4 prompts: 2 text-only, 2 vision (synthetic 224x224 gradient / 448x448 geometric images). Vision prompts use the pre-filled cache pattern: chat template + input prep, one full forward pass fills the KV cache with vision + prompt context, first token samples from that output, remaining tokens generate via the standard decode loop against the pre-filled cache. Additional metric: `avg_vision_ms`.

## The scoring system

### Composite score formula

```
score = gen_tps_weight   * (current_gen_tps / baseline_gen_tps)
      + ttft_weight      * (baseline_ttft / current_ttft)           # inverted: lower is better
      + prefill_weight   * (current_prefill / baseline_prefill)
      + memory_weight    * (baseline_memory / current_memory)       # inverted: lower is better
```

Baseline always scores exactly 1.0.

### Default weights

| Weight | Value | Metric |
|--------|-------|--------|
| `gen_tps_weight` | 0.40 | Generation tokens/sec |
| `ttft_weight` | 0.25 | Time to first token |
| `prefill_tps_weight` | 0.20 | Prompt processing tokens/sec |
| `memory_weight` | 0.15 | Peak GPU memory |

### Worked example

Baseline: gen_tps=45, ttft=300ms, prefill=900tps, memory=24GB.
After optimization: gen_tps=47, ttft=290ms, prefill=905tps, memory=24GB.

```
score = 0.40 * (47/45)  + 0.25 * (300/290) + 0.20 * (905/900) + 0.15 * (24/24)
      = 0.418 + 0.259 + 0.201 + 0.150
      = 1.028   (2.8% composite improvement)
```

### Decision logic

- Both scores >= 1.0, fingerprints match, tests pass: **KEEP**.
- One improved (>3%) but the other regressed (<2%): **KEEP** (tolerable cross-regression).
- Both regressed: **DISCARD**.
- Fingerprint mismatch: **DISCARD**.
- New test failure: **DISCARD**.

## Output fingerprinting

With greedy decode (temp=0, seed=42), the model produces a deterministic token sequence per prompt. The bench scripts SHA-256 hash the token IDs and compare against baseline.

**Why it exists**: prevents optimizations that accidentally change model behavior. An optimization should make the same computation faster, not produce different output.

**What breaks it**: any change to logit processing, sampler selection, or KV cache population. Even reordering floating-point operations can change results (IEEE 754 isn't associative).

**Honesty caveat -- read this before trusting a KEEP**: the fingerprint freezes behavior relative to the harness's OWN baseline. It does NOT certify that the baseline output was good. There is no ground-truth quality metric anywhere in this harness -- a KEEP means "faster and behaviorally identical to what we started with," not "correct" or "high quality." Don't present composite scores as quality claims.

**How to re-baseline**: if you intentionally changed output behavior, re-run with `--reset-baseline`. Old fingerprints are overwritten.

## Reading results

### results.tsv

One row per experiment at `apps/optloop-lib/results.tsv`. Key columns: `mlx_lm_commit`/`mlx_vlm_commit` (fork commit hashes), `composite_score` (1.0 = baseline), `status` (baseline/keep/discard/crash), `description`.

### Cycle JSONs

Rich structured data at `apps/optloop-lib/data/cycles/cycle_NNNN.json`. Key fields: `decision` + `decision_reason`; `optimizer.description` + `optimizer.hypothesis`; `results.text` / `results.vlm` (per-bench scores, per-prompt breakdown); `verification` (test results, fingerprint status, suspicion flags); `cumulative` (running delta vs original baseline).

### progress.png

Generated by `bench_analysis.py` (if matplotlib is available) at `apps/optloop-lib/data/progress.png`. Shows score trajectories over cycles.

## Data artifacts

| Artifact | Path | Gitignored | Persists across sessions | Audience |
|----------|------|------------|--------------------------|----------|
| baseline.json | `apps/optloop-lib/data/{text,vlm}/` | Yes | Yes (until `--reset-baseline`) | Agent + Human |
| run_*.json | `apps/optloop-lib/data/{text,vlm}/` | Yes | Yes (accumulates) | Agent + Analysis |
| cycle_*.json | `apps/optloop-lib/data/cycles/` | Yes | Yes (accumulates) | Agent + Human |
| results.tsv | `apps/optloop-lib/` | Yes | Yes (accumulates) | Agent + Human |
| progress.png | `apps/optloop-lib/data/` | Yes | Regenerated on analysis | Human |
| optimization_log.md | `docs/` | No (committed) | Yes (on main) | Both |
| bench_config.toml | `apps/optloop-lib/` | No (committed) | Yes (locked, agent cannot edit) | Both |
| fork repos | `apps/optloop-lib/repos/` | Yes (own git history per fork, not tracked by heylookitsanllm) | Yes | Agent + Human |

### Full reset

```bash
rm -rf apps/optloop-lib/data/
rm -f apps/optloop-lib/results.tsv
cd apps/optloop-lib
uv run scripts/bench_text.py --reset-baseline
uv run scripts/bench_vlm.py --reset-baseline
```

## Cross-session memory

`docs/optimization_log.md` is the ONLY cross-session knowledge base -- read it at the start of every session, update it at the end. It has:

- **Performance baselines**: current and historical numbers per model.
- **Performance ceilings**: theoretical max based on hardware bandwidth.
- **What works**: confirmed improvements with magnitude.
- **What doesn't work**: failed approaches with reasoning, so nobody retries them.
- **Technical findings**: gotchas, library quirks, workarounds.
- **Open questions**: untested ideas for future sessions.

As of 2026-07-06 the "what works" section is empty -- no cycle has completed end-to-end yet.

## What gets committed after a session

Optloop-lib never touches heylookitsanllm's `src/`, so there's no branch to merge on this repo. What actually happens:

- **Fork repos** (`apps/optloop-lib/repos/mlx-lm/`, `repos/mlx-vlm/`): kept experiments are commits on `optloop-lib/<tag>` branches inside the fork clones. The agent can push to `origin` (your fork) when asked, never to `upstream`.
- **On heylookitsanllm main**: `docs/optimization_log.md` (new baselines, findings), `CHANGELOG.md` (if the session is worth a version bump), `internal/log/log_YYYY-MM-DD.md` (session log).
- **Not committed anywhere**: `results.tsv`, `apps/optloop-lib/data/`.

Worth internalizing: the root `pyproject.toml` pins `mlx-lm`/`mlx-vlm` to the real upstream repos (`ml-explore/mlx-lm`, `Blaizzy/mlx-vlm`), not to these forks. A gain proven here reaches the actual server only if it's upstreamed to those projects or the pin is deliberately repointed -- neither optloop-lib nor this harness does that automatically.

## Performance ceiling analysis

For autoregressive text generation, each token requires reading all model weights from memory once. The throughput ceiling is:

```
max_gen_tps = memory_bandwidth / model_size_bytes
```

For M2 Ultra (192GB, ~800 GB/s bandwidth):

| Model | Size | Theoretical Max | Typical Baseline | Utilization |
|-------|------|-----------------|------------------|-------------|
| Gemma-3-27B bf16 | ~54 GB | ~14.8 tps | ~11.7 tps | ~79% |
| Qwen3.5-27B mxfp8 | ~27.5 GB | ~29.1 tps | ~21.0 tps | ~72% |

What this means for optimization:

- Python overhead is <0.1% of per-token time. Removing Python-level inefficiencies has negligible impact on gen_tps.
- The remaining gap (21-28%) comes from GPU scheduling overhead, cache coherency, memory access patterns, and attention/KV cache lookup (not a simple sequential read).
- Real gains require reducing GPU operations, cache-friendly memory access patterns, or fused operations (fewer kernel launches).
- Smaller quantized models have higher theoretical max (less data to read per token) but more compute per byte (dequantization).

When to stop: if you're consistently above 90% of theoretical max, further gen_tps gains will be marginal. Shift focus to TTFT, memory efficiency, or vision-pipeline-specific bottlenecks.

## VLM-specific gotchas

### Position state bleeding (Qwen3.5)

Qwen3.5-style mRoPE models cache `_position_ids` and `_rope_deltas` on the `LanguageModel` instance. These must be reset between fresh generations when not using a pre-filled cache, or subsequent requests get stale position embeddings -- shape mismatches or wrong output. The server handles this via `generation_core._reset_vlm_positions()`; fork edits that bypass that code path must handle it themselves.

### torchvision dependency

Transformers 5.x's `Qwen3VLVideoProcessor` imports torchvision at module load time, which crashes standalone VLM scripts without torchvision installed. The server-side fix lives in `mlx_provider._apply_transformers_patches()`, which only runs when the server starts -- standalone bench scripts need the same suppression applied independently if this bites.

### Pre-filled cache pattern

VLM generation is two phases: a vision forward pass fills the KV cache, then standard autoregressive decode continues from that warm cache. Optimizing phase 1 needs vision-encoder-specific knowledge (image grid processing, encoder architecture); phase 2 is ordinary text optimization starting from a warm cache.

## Failure modes and recovery

### Benchmark crashes

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during model load | Model too large for available memory | Use a smaller model or close other apps |
| `_metal.Device` errors | Metal context corruption | Restart terminal |
| `FileNotFoundError` on model path | `models.toml` entry points to a nonexistent directory | Update path or re-download |
| `KeyError` in config parsing | `bench_config.toml` references a model ID missing from `models.toml` | Check the ID matches |
| Infinite hang during generation | Broken stop token logic or sampler bug in the fork edit | Kill process, check the edit for stop-token interference |

### Fingerprint mismatches

The edit changed model output. Possible causes: float operation reordering, KV cache corruption from a prior run, sampler modification (even "no-op" refactors can change float precision), or a different token count (truncation/early-stop behavior changed). If the change is intentional, re-baseline with `--reset-baseline`.

### Agent keeps discarding everything

1. Noisy baselines -- check `variance_warnings` in recent cycle JSONs; if CV > 0.15, bump `runs` to 5 in `bench_config.toml` and re-baseline.
2. Tight constraints -- loosen `max_single_metric_regression`.
3. Wrong target -- confirm the edit is actually in `repos/mlx-lm/mlx_lm/` or `repos/mlx-vlm/mlx_vlm/`, not somewhere the bench never touches.
4. Near ceiling -- see the ceiling analysis above; if baseline is already >90% of theoretical max, expect marginal gains.
5. Cross-regression -- text improves but VLM regresses (or vice versa) beyond `cross_regression_tolerance`.

### Fork repos left mid-experiment

If a session ends without a clean KEEP/DISCARD, reset both forks to the last known-good commit:

```bash
cd apps/optloop-lib/repos/mlx-lm && git reset --hard <snapshot_commit>
cd apps/optloop-lib/repos/mlx-vlm && git reset --hard <snapshot_commit>
```

Snapshot commits come from `bench_common.snapshot_coderef()` calls recorded in the cycle JSONs.

### Stale baselines

Re-establish baselines after: switching fork branches, significant upstream changes to the forks, changing models in `bench_config.toml`, or `uv sync` picking up new mlx/mlx-lm/mlx-vlm versions.

### Sandbox permission issues with uv run

If `uv run` prompts for approval every time despite `autoAllowBashIfSandboxed: true`, uv likely needs to write to its cache directory (`<HOME>/.cache/uv/`) and the sandbox filesystem allowlist doesn't include it, so commands fall back to `dangerouslyDisableSandbox: true` and bypass the auto-allow. Fix: add the uv cache path and `apps/optloop-lib/.venv` to `additionalWritePaths` in `.claude/settings.local.json`.

### Metal context conflicts in tests

MLX embedding and sampler tests can fail when run in the full test suite (Metal context conflicts) but pass individually. This is a pre-existing environment issue, not a regression -- the verification phase allowlists it.

## Analysis deep dive

### bench_analysis.py

```bash
cd apps/optloop-lib && uv run scripts/bench_analysis.py
```

Reads `results.tsv`, per-run JSONs, and cycle JSONs. Output includes: experiment summary (total/kept/discarded/crashed), top improvements ranked by combined score delta, per-metric history for text and VLM, cumulative drift from original baseline, and a score chart (`apps/optloop-lib/data/progress.png` if matplotlib is available). Or run `/optloop-analysis` in Claude Code for the same summary plus a synthesized readout.

### Reading cycle JSONs programmatically

```python
import orjson
from pathlib import Path

cycles = sorted(Path("apps/optloop-lib/data/cycles").glob("cycle_*.json"))
for path in cycles:
    cycle = orjson.loads(path.read_bytes())
    decision = cycle["decision"]
    desc = cycle["optimizer"]["description"]
    text_score = cycle["results"]["text"]["composite_score"]
    vlm_score = cycle["results"]["vlm"]["composite_score"]
    print(f"Cycle {cycle['cycle_id']}: {decision} | "
          f"text={text_score:.4f} vlm={vlm_score:.4f} | {desc}")
```

## Configuration reference

All configuration lives in `apps/optloop-lib/bench_config.toml` (off-limits to the optimizer agent). Key sections:

- `[bench]`: run parameters (runs, warmup, max_tokens, seed, timeout).
- `[bench.text]` / `[bench.vlm]`: model selection.
- `[scoring]`: metric weights (must sum to 1.0).
- `[scoring.decision]`: keep/discard thresholds.
- `[constraints]`: hard limits (max regression, variance, fingerprint matching).
- `[optimizer]` / `[optimizer.scope]`: allowed/banned paths and the tier-1/2/3 target lists.

## Tips

- **Check the ceiling first**: calculate `bandwidth / model_size` for your hardware. If current performance is already >90% of theoretical max, gains will be marginal.
- **Read the optimization log**: `docs/optimization_log.md` before each session -- the agent reads it automatically, but understanding the history helps you set realistic expectations.
- **Start fresh when confused**: full reset (above), then re-baseline.
- **Increase runs for stability**: if results are noisy (high CV warnings), bump `runs` in `bench_config.toml` from 3 to 5.
- **Monitor memory**: shouldn't grow beyond 30% of baseline across cycles; watch for leaks.
- **Vision prompts are synthetic**: the VLM bench uses generated PIL images, not real photographs, and there's no long-context prompt yet. Both are planned additions (`apps/optloop-lib/CLAUDE.md`) that will change fingerprint baselines -- add them before establishing new baselines, not after.
- **A KEEP is not a quality certification**: re-read the honesty caveat under Output Fingerprinting before reporting results externally.
