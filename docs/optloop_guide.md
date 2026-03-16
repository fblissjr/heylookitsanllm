# Optloop User Guide

Optloop is an automated inference optimization system for the heylookitsanllm backend. A Claude Code agent modifies the MLX inference pipeline, benchmarks each change against a deterministic baseline, and keeps only verified improvements. It runs in a continuous loop until interrupted.

Based on dual benchmarks (text + VLM), output fingerprinting for correctness, and per-cycle structured logging.

## When to use optloop vs optloop-lib

| | optloop | optloop-lib |
|---|---|---|
| **Scope** | `src/heylook_llm/` (application layer) | `repos/mlx-lm/` and `repos/mlx-vlm/` (library internals) |
| **Mechanism** | Monkey patches via `.pth` activation | Direct source edits on local forks |
| **Potential** | Limited by bench activation gap | Higher -- changes directly affect hot paths |
| **Complexity** | Lower -- single repo, single branch | Higher -- multiple fork repos, separate venvs |
| **Use when** | You want server-side improvements | You want raw inference speed gains |
| **Git** | Commits on `optloop/<tag>` branch | Commits in fork repos at `apps/optloop-lib/repos/` |

Use **optloop** first to exhaust application-layer optimizations. Use **optloop-lib** when you need to go deeper into mlx-lm/mlx-vlm internals for raw performance gains.

You cannot run both in the same session (different venvs, different scope).

## Quick start

```bash
# 1. Sync dependencies
uv sync

# 2. Verify models exist (paths must be in models.toml)
grep -A5 'id = "google_gemma-3-27b-it-mlx-bf16"' models.toml
grep -A5 'id = "Qwen3.5-27B-mxfp8-mlx"' models.toml

# 3. Start the optimization loop in Claude Code
/optloop mar16
```

The agent handles everything from there: branching, baselines, optimization, benchmarking, verification, and logging.

## How it works

Each cycle has four phases:

### 1. Change

The agent reads the codebase, selects an optimization target, records a hypothesis, and implements the change. For optloop, this typically means adding or modifying monkey patches in `src/heylook_llm/providers/common/patches.py` that alter mlx-lm/mlx-vlm behavior at runtime.

### 2. Test

Unit and contract tests run (`uv run pytest tests/unit/ tests/contract/ -v`). Any new test failure blocks the benchmark step -- the agent must fix the issue or revert before proceeding.

### 3. Benchmark

Both bench scripts run against the current baselines:
- **Text bench**: Loads a model via `mlx_lm.utils.load()`, runs 3 fixed prompts with greedy decoding
- **VLM bench**: Loads a model via `mlx_vlm.utils.load()`, runs 2 text + 2 vision prompts

Each produces a composite score where 1.0 = baseline performance. Scores above 1.0 indicate improvement.

### 4. Verify

A 6-step verification runs before the keep/discard decision:

1. **Diff inspection** -- all changes within allowed paths
2. **Output fingerprints** -- SHA-256 of token IDs must match baseline (byte-identical)
3. **Test suite** -- no new failures
4. **Per-prompt regression** -- no individual prompt regresses beyond threshold
5. **Suspicion flags** -- >30% gain flagged as suspicious
6. **Variance check** -- coefficient of variation across runs must be below 15%

If both scores are >= 1.0, fingerprints match, and tests pass: **KEEP**. Otherwise: **DISCARD** (reverted via `git reset --hard`).

## Prerequisites

### Hardware

- Apple Silicon Mac (MLX requires Metal)
- 64GB+ unified memory recommended (default 27B models need ~30GB peak)

### Software

- Python 3.12+ managed by [uv](https://docs.astral.sh/uv/)
- `uv sync` from the repo root (installs mlx, mlx-lm, mlx-vlm, etc.)
- Claude Code with `/optloop` skill configured

### Models

The benchmarks need two models defined in `bench_config.toml` and resolved via `models.toml`:

| Bench | Model | Purpose |
|-------|-------|---------|
| Text | Gemma-3-27B bf16 | Pure text generation throughput |
| VLM | Qwen3.5-27B mxfp8 | Vision + text generation throughput |

Models must be local MLX-format weight directories registered in `models.toml`. See `apps/optloop/README.md` for model resolution details and download instructions.

## Step-by-step walkthrough

### Before your first session

1. **Sync dependencies**: `uv sync` from repo root
2. **Verify models**: Check `models.toml` has entries for both bench models with valid local paths
3. **Read prior findings**: Check `docs/optimization_log.md` for accumulated cross-session knowledge

### Starting a session

In Claude Code:

```
/optloop mar16
```

The agent will:

1. Read CLAUDE.md, bench_config.toml, and all in-scope source files
2. Read `docs/optimization_log.md` for prior session findings
3. Load MLX domain knowledge via `/mlx-skills:mlx` and `/mlx-skills:mlx-lm`
4. Create branch `optloop/mar16` from current HEAD
5. Set up `.pth` file for monkey patch activation (if using patches)
6. Run both benchmarks with `--reset-baseline`
7. Initialize `results.tsv` with baseline row
8. Run tests to confirm clean starting state
9. Begin the optimize-test-bench-verify loop

Each cycle takes roughly 5-10 minutes depending on model size and change complexity.

### Monitoring

While the agent works:

- **results.tsv**: One row per experiment with scores, decision, and description
- **Cycle JSONs**: `apps/optloop/data/cycles/cycle_NNNN.json` with full structured data per experiment
- **Git log**: Kept experiments are commits on the optloop branch

### Ending a session

When you interrupt the agent (Ctrl+C), it should:

1. Remove the `.pth` file (prevents errors on other branches)
2. Run analysis if data exists
3. Distill findings into `docs/optimization_log.md` on main
4. Commit the updated optimization log

If the agent doesn't complete teardown, do it manually:

```bash
rm -f .venv/lib/python3.13/site-packages/heylook_llm_patches.pth
git checkout main
# Review and update docs/optimization_log.md with session findings
```

### After a session

```bash
# Analyze results
/optloop-analysis app

# Merge improvements to main
git checkout main
git merge optloop/mar16

# Or cherry-pick specific commits
git cherry-pick <commit-hash>

# Update CHANGELOG.md with merged changes
# Update docs/optimization_log.md with new baselines (if improved)
```

## Understanding the benchmarks

### Text benchmark

Loads a model via `mlx_lm.utils.load()` and runs 3 fixed prompts (short, medium, long) through `mlx_lm.generate.stream_generate()` with greedy decoding (temp=0, seed=42).

Metrics:

- **gen_tps**: Tokens generated per second (decode throughput)
- **ttft_ms**: Time to first token in milliseconds
- **prefill_tps**: Prompt tokens processed per second
- **peak_memory_gb**: Peak GPU memory from `mx.get_peak_memory()`

Each prompt gets warmup runs (not measured) followed by measured runs, averaged per prompt, then averaged across prompts.

### VLM benchmark

Loads a model via `mlx_vlm.utils.load()` and runs 4 prompts: 2 text-only and 2 vision prompts with synthetic test images (224x224 gradient, 448x448 geometric shapes).

Vision prompts use the pre-filled cache pattern:

1. Chat template and input preparation
2. Full forward pass through VLM fills KV cache with vision + prompt context
3. First token sampled from output logits
4. Remaining tokens generated via `lm_stream_generate()` using pre-filled cache

Additional metric: `avg_vision_ms` (vision encoding time).

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
      = 0.40 * 1.044    + 0.25 * 1.034     + 0.20 * 1.006     + 0.15 * 1.000
      = 0.418 + 0.259 + 0.201 + 0.150
      = 1.028   (2.8% composite improvement)
```

### Decision logic

- Both scores >= 1.0, fingerprints match, tests pass: **KEEP**
- One improved (>3%) but other regressed (<2%): **KEEP** (tolerable cross-regression)
- Both regressed: **DISCARD**
- Fingerprint mismatch: **DISCARD**
- New test failure: **DISCARD**

## Output fingerprinting

With greedy decode (temp=0, seed=42), the model produces a deterministic token sequence for each prompt. The bench scripts SHA-256 hash the token IDs and compare against baseline.

**Why it exists**: Prevents optimizations that accidentally change model behavior. An optimization should make the same computation faster, not produce different output.

**What breaks it**: Any change to how logits are processed, how the sampler selects tokens, or how the KV cache is populated. Even reordering floating-point operations can change results due to non-associativity of IEEE 754 arithmetic.

**How to re-baseline**: If you intentionally changed output behavior, re-run with `--reset-baseline`. The old fingerprints are overwritten.

## Reading results

### results.tsv

One row per experiment. Key columns:

- `composite_score`: 1.0 = baseline, >1.0 = improvement
- `status`: baseline, keep, discard, or crash
- `description`: what was attempted

### Cycle JSONs

Rich structured data at `apps/optloop/data/cycles/cycle_NNNN.json`. Key fields:

- `decision` + `decision_reason`: what happened and why
- `optimizer.description` + `optimizer.hypothesis`: what was tried
- `results.text` / `results.vlm`: per-bench scores and per-prompt breakdown
- `verification`: test results, fingerprint status, suspicion flags
- `cumulative`: running delta from original baseline

### progress.png

Generated by `bench_analysis.py` (if matplotlib is available). Shows score trajectories over cycles.

## Data artifacts

| Artifact | Path | Gitignored | Persists across sessions | Audience |
|----------|------|------------|--------------------------|----------|
| baseline.json | `data/{text,vlm}/` | Yes | Yes (until --reset-baseline) | Agent + Human |
| run_*.json | `data/{text,vlm}/` | Yes | Yes (accumulates) | Agent + Analysis |
| cycle_*.json | `data/cycles/` | Yes | Yes (accumulates) | Agent + Human |
| results.tsv | `apps/optloop/` | Yes | Yes (accumulates) | Agent + Human |
| progress.png | `data/` | Yes | Regenerated on analysis | Human |
| optimization_log.md | `docs/` | No (committed) | Yes (on main) | Both |
| bench_config.toml | `apps/optloop*/` | No (committed) | Yes (locked) | Both |
| .pth file | `.venv/.../` | N/A | Must be removed at teardown | Agent |

### Full reset

```bash
rm -rf apps/optloop/data/
rm -f apps/optloop/results.tsv
uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1
uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
```

## Cross-session memory

`docs/optimization_log.md` is committed to main and accumulates findings across sessions:

- **Baselines table**: Current and historical performance numbers per model
- **Performance ceilings**: Theoretical max based on hardware bandwidth
- **What works**: Confirmed improvements with magnitude
- **What doesn't work**: Failed approaches with reasoning (so the agent doesn't retry them)
- **Technical findings**: Gotchas, library quirks, and workarounds
- **Open questions**: Untested ideas for future sessions

The agent reads this file at the start of each session and updates it at the end. Human edits are welcome -- add context, correct errors, or flag new ideas.

## What gets merged to main

After an optloop session, you merge the branch. What belongs on main:

**From the optloop branch:**
- `src/heylook_llm/providers/common/patches.py` (the monkey patches that improved bench scores)
- Any other `src/heylook_llm/` changes

**Updated on main after merge:**
- `docs/optimization_log.md` (new baselines, findings)
- `CHANGELOG.md` (version bump describing improvements)
- `internal/log/log_YYYY-MM-DD.md` (session log)

**Not merged (gitignored):**
- `results.tsv`, `data/` directory, `.pth` file

## Configuration reference

All configuration lives in `bench_config.toml`. See `apps/optloop/README.md` for the full reference table.

Key sections:

- `[bench]`: Run parameters (runs, warmup, max_tokens, seed, timeout)
- `[bench.text]` / `[bench.vlm]`: Model selection
- `[scoring]`: Metric weights (must sum to 1.0)
- `[scoring.decision]`: Keep/discard thresholds
- `[constraints]`: Hard limits (max regression, variance, fingerprint matching)
- `[optimizer]`: Scope controls (allowed paths, banned patterns)

## Tips

- **Check the ceiling first**: Calculate `bandwidth / model_size` for your hardware. If current performance is already >90% of theoretical max, gains will be marginal.
- **Read the optimization log**: Before each session, check what's been tried. The agent reads it automatically, but understanding the history helps you set realistic expectations.
- **Start fresh when confused**: `rm -rf apps/optloop/data/ && rm -f apps/optloop/results.tsv` then re-baseline.
- **Increase runs for stability**: If results are noisy (high CV warnings), bump `runs` in bench_config.toml from 3 to 5.
- **Monitor memory**: The agent shouldn't increase memory beyond 30% of baseline, but watch for leaks across cycles.
- **The .pth file is critical**: Without it, monkey patches have no effect on bench scores. With it on the wrong branch, imports fail. Always clean up at session end.

## Next steps

See [optloop_advanced.md](./optloop_advanced.md) for the bench activation gap, monkey patching mechanics, performance ceiling analysis, failure modes, and FAQ.
