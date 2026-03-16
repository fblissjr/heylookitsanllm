# optloop -- Inference Optimization Loop

optloop is an automated benchmark-and-verify system for the heylookitsanllm inference backend. A Claude Code agent reads the codebase, makes targeted changes to the generation pipeline, benchmarks each change against a deterministic baseline, and keeps only verified improvements. It runs indefinitely until interrupted.

There are two benchmarks: text (mlx-lm path) and VLM (mlx-vlm path, text + vision prompts). Both must maintain or improve performance for a change to be kept.

Target hardware: Apple Silicon (Mac Studio M2 Ultra, 192GB unified memory).

## How It Works

The loop has four phases per cycle:

1. **Change** -- The agent picks an optimization idea, records a hypothesis, and implements it in `src/heylook_llm/`.
2. **Test** -- Unit and contract tests run. Failures block the benchmark step.
3. **Benchmark** -- Both bench scripts run against the current baselines. Each produces a composite score (1.0 = baseline, >1.0 = improvement).
4. **Verify** -- Diff inspection, output fingerprint matching, per-prompt regression checks, variance analysis. The agent decides KEEP or DISCARD based on scores and constraint checks. Discarded changes are reverted via `git reset --hard`.

Every cycle is logged to `results.tsv` (one-line summary) and `data/cycles/cycle_NNNN.json` (full structured data).

## Prerequisites

### Hardware

- Apple Silicon Mac (MLX requires Metal)
- 64GB+ unified memory recommended (the default 27B models need ~30GB peak during inference)

### Software

- Python 3.12+ managed by [uv](https://docs.astral.sh/uv/)
- `uv sync` from the repo root (installs mlx, mlx-lm, mlx-vlm, etc.)
- No frontend tooling needed -- optloop is Python-only

### Models

The bench scripts need two models. Model IDs in `bench_config.toml` are short names that get resolved to local filesystem paths via the project root `models.toml`.

| Bench | Config ID | What It Resolves To |
|-------|-----------|---------------------|
| Text | `google_gemma-3-27b-it-mlx-bf16` | The `model_path` field from the matching `[[models]]` entry in `models.toml` |
| VLM | `Qwen3.5-27B-mxfp8-mlx` | Same lookup, different model |

#### Model resolution order

The bench scripts resolve a model ID through this chain (see `resolve_model_from_toml()` in `bench_common.py`):

1. CLI `--model-path` argument (takes priority over everything)
2. Look up the `[bench.text]` or `[bench.vlm]` model ID from `bench_config.toml`
3. Search `models.toml` at the repo root for a `[[models]]` entry with a matching `id` field
4. If found, use that entry's `config.model_path` (a local filesystem path like `/Users/you/Storage/llms/google/google_gemma-3-27b-it-mlx-bf16`)
5. If not found, attempt `huggingface_hub.snapshot_download()` as a fallback

This means you need the models registered in `models.toml` with valid `model_path` values pointing to local directories containing MLX-format weights.

If your models live elsewhere, either:
- Update `models.toml` to point to your paths, or
- Pass `--model-path /path/to/model` directly to the bench scripts

## Step-by-Step Guide

### 1. Verify project state

```bash
# From repo root
uv sync
git status  # should show no uncommitted changes to tracked files
```

### 2. Confirm model paths

Check that `models.toml` has entries for both models and that the paths exist:

```bash
# These should print valid directory paths
grep -A5 'id = "google_gemma-3-27b-it-mlx-bf16"' models.toml
grep -A5 'id = "Qwen3.5-27B-mxfp8-mlx"' models.toml
```

If a model isn't in `models.toml`, download it first:

```bash
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/google_gemma-3-27b-it-mlx-bf16'))"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.5-27B-mxfp8-mlx'))"
```

Then add the resulting cache paths to `models.toml`.

### 3. Establish baselines

Run each benchmark with `--reset-baseline` to record the starting performance. All commands run from the repo root.

```bash
uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1
uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
```

Each prints a grep-friendly results block to stdout:

```
---
composite_score:  1.0000
avg_gen_tps:      45.2
avg_ttft_ms:      312.4
avg_prefill_tps:  890.1
peak_memory_gb:   24.3
runs:             9
model:            google_gemma-3-27b-it-mlx-bf16
bench:            text
```

Baselines are saved as JSON files:
- `apps/optloop/data/text/baseline.json`
- `apps/optloop/data/vlm/baseline.json`

### 4. Run benchmarks manually (optional)

Before starting the automated loop, you can run individual benchmarks to compare against the baseline:

```bash
# Compare current code against existing baseline
uv run apps/optloop/scripts/bench_text.py
uv run apps/optloop/scripts/bench_vlm.py

# Override run count (more runs = more stable, but slower)
uv run apps/optloop/scripts/bench_text.py --runs 5

# Override max generated tokens
uv run apps/optloop/scripts/bench_text.py --max-tokens 512

# Use a specific model directly (bypasses bench_config.toml and models.toml)
uv run apps/optloop/scripts/bench_text.py --model-path /path/to/local/model

# VLM: text prompts only (skip vision encoding)
uv run apps/optloop/scripts/bench_vlm.py --text-only

# VLM: vision prompts only (skip text-only prompts)
uv run apps/optloop/scripts/bench_vlm.py --vision-only

# Use a custom config file
uv run apps/optloop/scripts/bench_text.py --config /path/to/custom_config.toml
```

### 5. Start the optimization loop

In Claude Code, invoke the optloop skill with a run tag:

```
/optloop mar15
```

This triggers the agent to:

1. Read `CLAUDE.md`, `bench_config.toml`, and all in-scope source files
2. Read `docs/optimization_log.md` for cross-session knowledge (baselines, prior findings, known dead-ends)
3. Load `/mlx-skills:mlx` and `/mlx-skills:mlx-lm` domain knowledge
4. Create branch `optloop/mar15` from current HEAD
5. Run both benchmarks with `--reset-baseline` to set starting points
6. Initialize `results.tsv` with headers and a baseline row
7. Run existing tests to confirm a clean starting state
8. Read `references/loop-protocol.md` and `references/optimization-guide.md`
9. Begin the optimize-test-bench-verify loop (runs indefinitely until `Ctrl+C`)

There are two additional skills for related workflows:

- **`/optloop-lib <tag>`** -- Library-level optimization targeting mlx-lm/mlx-vlm fork internals at `apps/optloop-lib/repos/`
- **`/optloop-analysis [app|lib]`** -- Post-run analysis of benchmark results (defaults to app-level)

Each cycle takes roughly 5-10 minutes depending on model size and the nature of the change.

### 6. Monitor while it runs

**results.tsv** -- one row per experiment:

```
commit    text_score  vlm_score  text_tps  vlm_tps  text_ttft  vlm_ttft  memory_gb  status    description
abc1234   1.0000      1.0000     45.2      38.1     312.4      890.2     24.3       baseline  initial baseline
def5678   1.0234      1.0112     46.3      38.5     305.1      882.0     24.1       keep      cache attribute lookup
```

**Cycle JSONs** -- detailed structured data per cycle at `apps/optloop/data/cycles/cycle_NNNN.json`. Key fields:

- `decision`: "keep", "discard", or "crash"
- `decision_reason`: why
- `optimizer.description`: what was tried
- `optimizer.hypothesis`: expected outcome
- `results.text.composite_score` / `results.vlm.composite_score`
- `verification`: test results, fingerprint matches, constraint violations

**Git log** -- kept experiments are commits on the branch:

```bash
git log --oneline optloop/mar15
```

### 7. Analyze results

After stopping the agent, use the analysis skill or run manually:

```bash
# Via skill (in Claude Code):
/optloop-analysis app

# Or manually:
uv run apps/optloop/scripts/bench_analysis.py
```

This reads `results.tsv`, per-run JSONs, and cycle JSONs. It prints:
- Experiment summary (total, kept, discarded, crashed)
- Top improvements ranked by combined score delta
- Per-metric history for text and VLM
- Cumulative drift from original baseline

If matplotlib is installed, it saves `apps/optloop/data/progress.png`:

```bash
# Skip chart generation
uv run apps/optloop/scripts/bench_analysis.py --no-chart
```

### 8. Act on results

```bash
# Keep everything
git checkout main
git merge optloop/mar15

# Cherry-pick specific improvements
git checkout main
git cherry-pick <commit-hash>

# Start over
rm -rf apps/optloop/data/
rm -f apps/optloop/results.tsv
# Then re-baseline and start a new /optloop run
```

## What the Optimizer Can and Cannot Modify

### Allowed

- Any file under `src/heylook_llm/` -- generation loop, sampling, caching, provider layer, router, API
- Configuration defaults in `config.py`
- New files (e.g., `src/heylook_llm/providers/common/patches.py` for monkey-patching mlx-lm/mlx-vlm at runtime)

### Off-limits

- `models.toml` -- model registry, not optimization code
- `pyproject.toml` -- dependency definitions
- `apps/optloop/bench_config.toml` -- eval configuration
- `apps/optloop/scripts/` -- the benchmark harness itself
- `apps/optloop/data/` -- benchmark results data
- `tests/` -- test files must continue passing, but the optimizer cannot change them

The `[optimizer]` section in `bench_config.toml` defines these boundaries formally. The verification phase checks `git diff` output against `banned_diff_patterns` and auto-rejects changes that touch off-limits paths.

## Scoring

### Composite score formula

The composite score is a weighted sum of metric ratios relative to baseline:

```
score = gen_tps_weight   * (current_gen_tps / baseline_gen_tps)
      + ttft_weight      * (baseline_ttft / current_ttft)           # inverted: lower is better
      + prefill_weight   * (current_prefill / baseline_prefill)
      + memory_weight    * (baseline_memory / current_memory)       # inverted: lower is better
```

Baseline always scores exactly 1.0. A score of 1.02 means a 2% composite improvement.

### Default weights

| Weight | Value | Metric |
|--------|-------|--------|
| `gen_tps_weight` | 0.40 | Generation tokens/sec (decode throughput) |
| `ttft_weight` | 0.25 | Time to first token (prefill + first decode latency) |
| `prefill_tps_weight` | 0.20 | Prompt processing tokens/sec |
| `memory_weight` | 0.15 | Peak GPU memory |

Weights must sum to 1.0. A warning prints to stderr if they don't.

To shift optimization pressure, increase the weight of the metric you care about most. Higher `gen_tps_weight` favors raw throughput. Higher `ttft_weight` favors latency. Higher `memory_weight` favors efficiency.

### Worked example

Baseline: gen_tps=45, ttft=300ms, prefill=900tps, memory=24GB.
After optimization: gen_tps=47, ttft=290ms, prefill=905tps, memory=24GB.

```
score = 0.40 * (47/45)  + 0.25 * (300/290) + 0.20 * (905/900) + 0.15 * (24/24)
      = 0.40 * 1.044    + 0.25 * 1.034     + 0.20 * 1.006     + 0.15 * 1.000
      = 0.418 + 0.259 + 0.201 + 0.150
      = 1.028
```

A 2.8% composite improvement.

### Decision thresholds

The agent uses these rules (defined in `.claude/skills/optloop/references/loop-protocol.md`, values in `[scoring.decision]`):

- Both scores >= 1.0 AND fingerprints match AND tests pass: **KEEP**
- One improved (>3%) but other regressed (<2%): **KEEP** (tolerable cross-regression)
- Both regressed: **DISCARD**
- Fingerprint mismatch: **DISCARD** (output correctness violated)
- New test failure: **DISCARD**

## Constraints

Hard limits that auto-reject regardless of composite score:

| Constraint | Default | Effect |
|------------|---------|--------|
| `max_single_metric_regression` | 0.30 | Any single metric regressing >30% = auto-fail |
| `max_token_deviation` | 0.20 | Token count deviating >20% from baseline = correctness concern |
| `per_prompt_regression_check` | true | Regression threshold applied per-prompt, not just averages |
| `require_fingerprint_match` | true | Greedy decode must produce byte-identical token sequences |
| `require_tests_pass` | true | Unit and contract tests must pass |
| `max_suspicion_gain` | 0.30 | >30% gain in one cycle = flagged (warn, not auto-reject) |
| `variance_max_cv` | 0.15 | Coefficient of variation >15% across runs = noisy results warning |

## Verification Phase

After each benchmark run, the agent runs a 6-step verification before deciding:

1. **Diff inspection** -- Confirms all changed files are within `[optimizer].allowed_paths`. Checks for `banned_diff_patterns` matches. Changes to `samplers.py`, `stop_tokens.py`, or `config.py` are flagged but not auto-rejected.

2. **Output fingerprints** -- With greedy decode (temp=0, seed=42), token sequences must be SHA-256 identical to baseline. Mismatch = auto-reject.

3. **Test suite** -- `uv run pytest tests/unit/ tests/contract/ -v`. Known pre-existing failures are allowlisted. Any new failure = auto-reject.

4. **Per-prompt regression** -- When enabled, the regression threshold applies to each prompt individually. A single prompt regressing beyond the threshold while others improve is rejected (can't game the average).

5. **Suspicion flags** (warn, not auto-reject) -- >30% gain in one cycle, changes to sampler construction code, monkey patches applied.

6. **Variance check** -- Coefficient of variation (stddev/mean, using sample variance N-1) of gen_tps across runs. CV > 15% = noisy results warning.

## Configuration Reference

All configuration lives in `bench_config.toml`. This file is off-limits to the optimizer agent.

### `[bench]` -- Run parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `runs` | int | 3 | Measured runs per prompt (averaged for final score) |
| `warmup` | int | 1 | Warmup runs per prompt (not measured, primes caches) |
| `max_tokens` | int | 256 | Max tokens generated per prompt |
| `seed` | int | 42 | Random seed for sampler init (greedy decode, but needed for sampler) |
| `timeout_seconds` | int | 600 | Per-bench timeout (10 minutes) |

### `[bench.text]` / `[bench.vlm]` -- Model selection

| Key | Type | Description |
|-----|------|-------------|
| `model` | string | Model ID looked up in `models.toml` for a local path. Falls back to HF download if not found. |

After changing models, re-establish baselines with `--reset-baseline`.

### `[scoring]` -- Metric weights

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gen_tps_weight` | float | 0.40 | Weight for generation tokens/sec |
| `ttft_weight` | float | 0.25 | Weight for time-to-first-token (lower = better) |
| `prefill_tps_weight` | float | 0.20 | Weight for prompt processing tokens/sec |
| `memory_weight` | float | 0.15 | Weight for peak GPU memory (lower = better) |

### `[scoring.decision]` -- Keep/discard thresholds

These values are read by the agent from `.claude/skills/optloop/references/loop-protocol.md`, not by bench scripts. Changing values here has no effect without also updating the skill references.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `both_improved_min` | float | 1.00 | Both scores must meet this minimum to KEEP |
| `cross_regression_tolerance` | float | 0.02 | Max regression in one bench when the other improves |
| `cross_improvement_min` | float | 0.03 | Min improvement needed to tolerate regression in the other bench |

### `[constraints]` -- Hard limits

See the [Constraints](#constraints) section above for the full table.

### `[optimizer]` -- Scope controls

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allowed_paths` | list[str] | `["src/heylook_llm/"]` | Directories the optimizer may modify |
| `monkey_patch_allowed` | bool | true | Whether runtime patches of mlx_lm/mlx_vlm are permitted |
| `banned_files` | list[str] | (see config) | Specific files the optimizer must not touch |
| `banned_paths` | list[str] | (see config) | Directories the optimizer must not touch |
| `banned_diff_patterns` | list[str] | (see config) | Regex patterns in diffs that auto-reject |

## Directory Structure

```
apps/optloop/
  README.md              this file
  program.md             human-readable reference (agent uses .claude/skills/optloop/ instead)
  bench_config.toml      eval configuration (off-limits to the optimizer)
  results.tsv            per-experiment summary (created at runtime, gitignored)
  scripts/
    bench_text.py        text-only benchmark (mlx-lm path, 3 prompts)
    bench_vlm.py         vision benchmark (mlx-vlm path, 2 text + 2 vision prompts)
    bench_common.py      shared code: scoring, fingerprinting, config loading, baseline mgmt, cycle logging
    bench_analysis.py    post-run analysis of results.tsv and cycle data
  tests/
    conftest.py          sys.path setup for importing from scripts/
    test_bench_common.py unit tests for bench_common pure functions (34 tests)
  data/                  runtime data (gitignored)
    text/                text bench baselines and run JSONs
    vlm/                 VLM bench baselines and run JSONs
    cycles/              per-cycle structured logs (cycle_NNNN.json)
    progress.png         score chart (generated by bench_analysis.py)
```

## Running Tests

The bench scoring and validation logic has its own test suite (34 tests). Run from the repo root:

```bash
uv run pytest apps/optloop/tests/ -v
```

These tests cover the pure functions in `bench_common.py`: composite scoring, variance checking (sample variance N-1), hard constraints, per-prompt constraints, suspicion detection, output fingerprinting, and config extraction. They do not load any model.

## Benchmarks in Detail

### Text benchmark (`bench_text.py`)

Loads a model via `mlx_lm.utils.load()` and runs 3 fixed prompts (short, medium, long) through `mlx_lm.generate.stream_generate()` with greedy decoding (temp=0). Measures:

- **gen_tps**: Tokens generated per second (decode throughput). Measured from first token to last token.
- **ttft_ms**: Time to first token in milliseconds. Covers tokenization, prompt processing, and first decode step.
- **prefill_tps**: Prompt tokens processed per second. Derived from prompt token count divided by TTFT.
- **peak_memory_gb**: Peak GPU memory from `mx.get_peak_memory()`.

Each prompt gets `warmup` unmeasured runs followed by `runs` measured runs. Results are averaged per prompt, then averaged across prompts.

### VLM benchmark (`bench_vlm.py`)

Loads a model via `mlx_vlm.utils.load()` and runs 4 prompts: 2 text-only and 2 vision prompts with synthetic test images (224x224 gradient, 448x448 geometric shapes).

Text prompts go through the same path as the text benchmark but via the VLM's language model (wrapped in `_LogitsWrapper` for logit extraction).

Vision prompts use the pre-filled cache pattern:
1. `vlm_apply_chat_template()` and `vlm_prepare_inputs()` prepare the multimodal input
2. A full forward pass through the VLM fills the KV cache with vision + prompt context
3. The first token is sampled from the output logits
4. Remaining tokens are generated via `lm_stream_generate()` using the pre-filled cache

The VLM benchmark reports an additional metric: `avg_vision_ms` (vision encoding time).

## Anti-Gaming Protections

- **Output fingerprinting**: Greedy decode (temp=0, seed=42) must produce byte-identical token sequences across runs. SHA-256 hash of token IDs is compared. Mismatch = auto-reject.
- **Per-prompt regression checks**: Individual prompt regression is checked, not just averages.
- **Config is off-limits**: Scoring weights, thresholds, and model paths cannot be modified by the optimizer.
- **Banned diff patterns**: Changes outside `allowed_paths` are auto-rejected via regex matching on `git diff` output.
- **Zero-token guard**: If generation produces 0 tokens (broken code path), the bench raises immediately.
- **Weight validation**: Scoring weights that don't sum to 1.0 produce a stderr warning.
- **Variance flagging**: High coefficient of variation across runs is flagged as noisy.

## Troubleshooting

### Model resolution fails

- Check that `models.toml` has a `[[models]]` entry with a matching `id` field
- Check that the entry's `config.model_path` points to an existing directory with MLX weights
- Use `--model-path /path/to/model` as a workaround

### Bench crashes

- **OOM**: Model too large for available memory. Try a smaller model.
- **Metal init failure**: Restart terminal, check macOS updates.
- **Corrupt model cache**: Delete `~/.cache/huggingface/hub/models--<model-id>` and re-download.

### Agent keeps discarding

- **Noisy baselines**: Check `variance_warnings` in recent cycle JSONs. If CV > 0.15, increase `runs` to 5 and re-baseline.
- **Tight constraints**: Loosen `max_single_metric_regression` from 0.30 to 0.40.
- **Wrong target**: Check that the agent is modifying code paths that actually run during the benchmark, not startup or config code.

### Fingerprint mismatch

Output changed means the optimization altered model behavior, not just speed. Auto-rejected by design. If you intentionally changed output logic, re-baseline with `--reset-baseline`.

### Stale data from previous runs

```bash
rm -rf apps/optloop/data/
rm -f apps/optloop/results.tsv
uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1
uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
```

## Library-Level Optimization

By default, `/optloop` only modifies `src/heylook_llm/`. The hot paths (stream_generate, KVCache, samplers) live in mlx-lm and mlx-vlm.

For library-level optimization of those internals, use the separate `/optloop-lib` skill which operates on fork repos at `apps/optloop-lib/repos/`. See `apps/optloop-lib/program.md` for details.

## Cross-Session Memory

`docs/optimization_log.md` accumulates findings across optloop sessions. It is committed to main (not gitignored) and persists between branches and sessions.

Contents:

- **Performance baselines**: Historical numbers per model and hardware
- **Performance ceilings**: Theoretical max based on hardware bandwidth
- **What works**: Confirmed improvements with magnitude and context
- **What doesn't work**: Failed approaches with reasoning
- **Technical findings**: Library quirks, gotchas, and workarounds
- **Open questions**: Untested ideas for future sessions

The optloop agent reads this file during setup and updates it at session end. See [docs/optloop_guide.md](../../docs/optloop_guide.md) for the full user walkthrough.

## Data Artifacts

| Artifact | Path | Gitignored | Persists Across Sessions | Audience |
|----------|------|------------|--------------------------|----------|
| baseline.json | `data/{text,vlm}/` | Yes | Yes (until --reset-baseline) | Agent + Human |
| run_*.json | `data/{text,vlm}/` | Yes | Yes (accumulates) | Agent + Analysis |
| cycle_*.json | `data/cycles/` | Yes | Yes (accumulates) | Agent + Human |
| results.tsv | `apps/optloop/` | Yes | Yes (accumulates) | Agent + Human |
| progress.png | `data/` | Yes | Regenerated on analysis | Human |
| optimization_log.md | `docs/` | No (committed) | Yes (on main) | Both |
| AGENTS.md | `apps/optloop-lib/` | No (committed) | Yes | Agent |
| bench_config.toml | `apps/optloop*/` | No (committed) | Yes (locked) | Both |
| .pth file | `.venv/.../` | N/A | Must be removed at teardown | Agent |

## Session End

When stopping an optloop session:

1. **Remove .pth file**: `rm -f .venv/lib/python3.13/site-packages/heylook_llm_patches.pth`
   The .pth file is branch-agnostic (shared across branches). Leaving it active on a branch without `patches.py` causes import errors.

2. **Run analysis**: `uv run apps/optloop/scripts/bench_analysis.py` (or `/optloop-analysis app` in Claude Code)

3. **Distill findings**: Switch to main and update `docs/optimization_log.md` with:
   - New baselines (if improved)
   - Approaches that worked or failed
   - Technical discoveries
   - Updated open questions

4. **Commit**: `git add docs/optimization_log.md && git commit -m "chore(docs): update optimization log"`

5. **Write session log**: Create `internal/log/log_YYYY-MM-DD.md` with session details

## What Gets Merged to Main

**From the optloop branch:**
- `src/heylook_llm/providers/common/patches.py` and any other `src/` changes

**Updated on main after merge:**
- `docs/optimization_log.md` (new baselines, findings)
- `CHANGELOG.md` (version bump describing improvements)
- `internal/log/` session log

**Not merged (gitignored):**
- `results.tsv`, `data/` directory, `.pth` file

## User Guides

- [docs/optloop_guide.md](../../docs/optloop_guide.md) -- step-by-step user walkthrough, scoring, monitoring, configuration
- [docs/optloop_advanced.md](../../docs/optloop_advanced.md) -- bench activation gap, monkey patching, performance ceilings, failure modes, FAQ
