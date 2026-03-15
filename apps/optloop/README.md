# Inference Optimization Loop

Autonomous optimization of heylookitsanllm's inference pipeline. A Claude Code agent iterates on the codebase, benchmarks each change against a deterministic baseline, and keeps only verified improvements.

The agent reads code, makes targeted changes to the generation pipeline, runs both benchmarks, checks that output correctness is preserved (via output fingerprinting), and either commits the improvement or reverts. It runs indefinitely until interrupted.

Target: Mac Studio M2 Ultra, 192GB unified memory.

## Prerequisites

### Hardware

- Apple Silicon Mac (MLX requires Metal)
- Recommended: M2 Ultra or M4 Max with 64GB+ unified memory
- The default models (27B parameters) need ~30GB during inference; both loaded sequentially means ~30GB peak, not ~60GB

### Software

- Python 3.12+ managed by uv
- `uv sync` from repo root (installs mlx, mlx-lm, mlx-vlm, etc.)
- bun is NOT needed -- optloop is Python-only

### Models

Two models, both from mlx-community on Hugging Face:

| Bench | Model ID | Size on disk |
|-------|----------|-------------|
| Text | `mlx-community/google_gemma-3-27b-it-mlx-bf16` | ~15 GB |
| VLM | `mlx-community/Qwen3.5-27B-mxfp8-mlx` | ~15 GB |

The bench scripts auto-download via `huggingface_hub.snapshot_download()`, but downloading a 27B model mid-bench is painful. Pre-download them (see Step 1).

### Project state

- `uv sync` from repo root must succeed
- Clean git status (`git status` shows no uncommitted changes to tracked files)

## End-to-End Tutorial

### Step 1: Pre-download models

Run from the repo root:

```bash
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/google_gemma-3-27b-it-mlx-bf16'))"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.5-27B-mxfp8-mlx'))"
```

Each prints the local cache path when done. Verify they're cached:

```bash
ls ~/.cache/huggingface/hub/models--mlx-community--google_gemma-3-27b-it-mlx-bf16/
ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-mxfp8-mlx/
```

You should see `snapshots/` directories with model weights inside.

### Step 2: Understand the benchmarks (recommended first time)

Before letting the agent loose, run each bench manually to see what they produce.

#### Establish baselines

```bash
# Text benchmark -- loads Gemma 27B, runs 3 prompts x 3 measured runs
uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1

# VLM benchmark -- loads Qwen3.5 27B, runs 2 text + 2 vision prompts x 3 runs
uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
```

#### Read the output

Each bench prints a grep-friendly results block to stdout:

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

What each field means:

| Field | Meaning |
|-------|---------|
| `composite_score` | Weighted combination of all metrics relative to baseline. 1.0 = baseline, >1.0 = improvement, <1.0 = regression |
| `avg_gen_tps` | Generation tokens per second (higher = better). The main throughput metric |
| `avg_ttft_ms` | Time to first token in milliseconds (lower = better). Measures prefill + first decode latency |
| `avg_prefill_tps` | Prompt processing tokens per second (higher = better). How fast the model ingests the prompt |
| `peak_memory_gb` | Peak GPU memory usage in GB (lower = better) |
| `runs` | Total measured runs (prompts x runs_per_prompt) |
| `model` | Model identifier |
| `bench` | Benchmark type (`text` or `vlm`) |
| `fingerprint_match` | Whether output token sequences match baseline exactly (only shown when comparing to baseline) |
| `avg_vision_ms` | VLM only -- vision encoding time in ms |

The VLM bench also shows `avg_vision_ms` for vision encoding time.

#### Where data lives

- Baselines: `apps/optloop/data/text/baseline.json` and `apps/optloop/data/vlm/baseline.json`
- Run results: `apps/optloop/data/text/run_*.json` and `apps/optloop/data/vlm/run_*.json`

### Step 3: Review configuration

Configuration lives in `apps/optloop/bench_config.toml`. This file is off-limits to the optimizer agent.

#### `[bench]` -- Run parameters

```toml
runs = 3                    # measured runs per prompt (averaged for final score)
warmup = 1                  # warmup runs (not measured, primes caches)
max_tokens = 256            # max tokens to generate per prompt
seed = 42                   # random seed (greedy decode, but needed for sampler init)
timeout_seconds = 600       # per-bench timeout (10 min)
```

More `runs` = more stable results but slower cycles. 3 is the minimum for meaningful variance checks.

#### `[bench.text]` and `[bench.vlm]` -- Model selection

```toml
[bench.text]
model = "mlx-community/google_gemma-3-27b-it-mlx-bf16"

[bench.vlm]
model = "mlx-community/Qwen3.5-27B-mxfp8-mlx"
```

Change these to use different models. After changing, re-establish baselines with `--reset-baseline`.

#### `[scoring]` -- Metric weights

```toml
gen_tps_weight = 0.40       # generation speed
ttft_weight = 0.25          # time to first token
prefill_tps_weight = 0.20   # prompt processing speed
memory_weight = 0.15        # peak memory usage
```

Weights must sum to 1.0 (a warning is printed to stderr if they don't). To shift optimization pressure:
- Want faster generation? Increase `gen_tps_weight`
- Want lower latency? Increase `ttft_weight`
- Want smaller memory footprint? Increase `memory_weight`

#### `[scoring.decision]` -- Keep/discard thresholds

```toml
both_improved_min = 1.00
cross_regression_tolerance = 0.02
cross_improvement_min = 0.03
```

These thresholds are read by the agent from `program.md` instructions, NOT by bench scripts. Changing values here has no effect without also updating `program.md`.

#### `[constraints]` -- Hard limits

```toml
max_single_metric_regression = 0.30   # 30% regression in any one metric = auto-fail
max_token_deviation = 0.20            # 20% token count change = correctness concern
per_prompt_regression_check = true    # check each prompt individually, not just averages
require_fingerprint_match = true      # greedy decode must produce identical tokens
require_tests_pass = true             # unit/contract tests must pass
max_suspicion_gain = 0.30             # >30% gain in one cycle = suspicious, flag for review
variance_max_cv = 0.15                # coefficient of variation > 15% = noisy results
```

When would you change these?
- Loosen `max_single_metric_regression` if the agent keeps discarding changes that trade one metric for another
- Disable `per_prompt_regression_check` if one prompt is inherently noisy
- Set `require_fingerprint_match = false` if using non-greedy sampling (not recommended)

#### `[optimizer]` -- Scope controls

```toml
allowed_paths = ["src/heylook_llm/"]
banned_files = ["models.toml", "pyproject.toml", "apps/optloop/bench_config.toml"]
banned_paths = ["apps/optloop/scripts/", "apps/optloop/data/", "tests/"]
banned_diff_patterns = ["apps/optloop/", "models\\.toml", "tests/"]
```

The agent reads these to know its boundaries. `banned_diff_patterns` are regex patterns checked against `git diff` output during the verification phase.

### Step 4: Start the optimization loop

In Claude Code:

```
/optloop mar15
```

What happens step by step:
1. Agent creates branch `optloop/mar15` from current HEAD
2. Reads `program.md` (agent instructions) and `bench_config.toml` (eval config)
3. Reads all in-scope source files (`src/heylook_llm/providers/`, `router.py`, etc.)
4. Loads `/mlx-skills:mlx` and `/mlx-skills:mlx-lm` skills for domain knowledge
5. Runs both benchmarks with `--reset-baseline` to establish starting points
6. Initializes `results.tsv` with headers and baseline row
7. Runs existing tests to confirm clean starting state
8. Begins the optimize-test-bench-verify loop

Each cycle:
1. Agent picks an optimization idea (guided by MLX skills and code analysis)
2. Implements the change
3. Runs tests -- if they fail, fixes or reverts before proceeding
4. Commits the change
5. Runs both benchmarks
6. Runs verification phase (diff inspection, fingerprints, tests, constraints)
7. Decides: KEEP (commit stands) or DISCARD (`git reset --hard` to previous commit)
8. Logs results to `results.tsv` and `data/cycles/cycle_NNNN.json`

Each cycle takes roughly 5-10 minutes depending on model size and change complexity.

The agent runs indefinitely until you interrupt it (`Ctrl+C`).

### Step 5: Monitor while it runs

#### Watch results.tsv

The agent maintains `apps/optloop/results.tsv` with one row per experiment:

```
commit    text_score  vlm_score  text_tps  vlm_tps  text_ttft  vlm_ttft  memory_gb  status    description
abc1234   1.0000      1.0000     45.2      38.1     312.4      890.2     24.3       baseline  initial baseline
def5678   1.0234      1.0112     46.3      38.5     305.1      882.0     24.1       keep      cache attribute lookup
```

#### Read cycle JSONs

Each cycle writes `apps/optloop/data/cycles/cycle_NNNN.json` with detailed data:

```bash
# Check the latest cycle
cat apps/optloop/data/cycles/cycle_0001.json | python -m json.tool
```

Key fields to look at:
- `decision`: "keep", "discard", or "crash"
- `decision_reason`: human-readable explanation of why
- `optimizer.description`: what the agent tried
- `optimizer.hypothesis`: why it expected improvement
- `results.text.composite_score` / `results.vlm.composite_score`: the scores
- `verification`: test results, fingerprint matches, constraint violations

#### Watch git log

Kept experiments create commits on the branch:

```bash
git log --oneline optloop/mar15
```

Each commit has a conventional commit message (e.g. `perf(generation): cache attribute lookups in hot loop`).

#### If the agent seems stuck

- **Keeps discarding**: Check `variance_warnings` in recent cycle JSONs. Noisy baselines cause legitimate improvements to look like noise. Try increasing `runs` in config and re-baselining.
- **All crashes**: Check stderr output. Usually a model loading or memory issue.
- **Same score every time**: The change might not affect the measured code paths. Check what files are being modified.

### Step 6: Analyze after stopping

After interrupting the agent (or letting it run for a while):

```bash
uv run apps/optloop/scripts/bench_analysis.py
```

This reads `results.tsv`, per-run JSONs, and cycle JSONs to produce:

#### Experiment summary

```
=== Experiment Summary ===
  Total:     15
  Baseline:  1
  Kept:      4
  Discarded: 9
  Crashed:   1
```

#### Top improvements

```
=== Top Improvements (by combined delta) ===
  #    Delta     Text      VLM  Description
  1  +0.0346   1.0234   1.0112  cache attribute lookups in hot loop
  2  +0.0198   1.0098   1.0100  reduce sync barriers in decode
```

#### Per-metric history

```
=== TEXT Metric History ===
Timestamp              Score   GenTPS       TTFT    Prefill   Memory
2026-03-15T10:30:00   1.0000     45.2      312.4      890.1     24.3
2026-03-15T10:42:00   1.0234     46.3      305.1      895.2     24.1
```

#### Cumulative drift

Shows total change from original baseline across all kept experiments.

#### Progress chart

If matplotlib is installed, saves `apps/optloop/data/progress.png` with text and VLM scores over time.

```bash
# Skip chart generation
uv run apps/optloop/scripts/bench_analysis.py --no-chart
```

### Step 7: Act on results

#### Keep everything

```bash
git checkout main
git merge optloop/mar15
```

#### Cherry-pick specific improvements

```bash
git checkout main
git cherry-pick <commit-hash>   # from git log of the optloop branch
```

#### Start over with different config

```bash
# Data is gitignored, just delete it and re-baseline
rm -rf apps/optloop/data/
rm -f apps/optloop/results.tsv
# Edit bench_config.toml as desired, then start a new /optloop run
```

#### Re-run on same branch

If you want to continue optimizing from where you left off, just run `/optloop` again on the same branch. Baselines from the previous run still exist in `data/`.

## Manual Bench Runs

All commands run from repo root.

```bash
# Run text bench (compares against existing baseline)
uv run apps/optloop/scripts/bench_text.py

# Run VLM bench
uv run apps/optloop/scripts/bench_vlm.py

# Override runs per prompt
uv run apps/optloop/scripts/bench_text.py --runs 5

# Override max tokens
uv run apps/optloop/scripts/bench_text.py --max-tokens 512

# Override warmup
uv run apps/optloop/scripts/bench_text.py --warmup 3

# Use a different model
uv run apps/optloop/scripts/bench_text.py --model-path mlx-community/Llama-3.3-70B-Instruct-4bit

# Reset baseline (required after model change or code change that affects output)
uv run apps/optloop/scripts/bench_text.py --reset-baseline

# Use custom config file
uv run apps/optloop/scripts/bench_text.py --config /path/to/custom_config.toml

# VLM: text prompts only (skip vision)
uv run apps/optloop/scripts/bench_vlm.py --text-only

# VLM: vision prompts only (skip text)
uv run apps/optloop/scripts/bench_vlm.py --vision-only

# Analyze all cycle data
uv run apps/optloop/scripts/bench_analysis.py

# Analyze without chart
uv run apps/optloop/scripts/bench_analysis.py --no-chart
```

## Configuration Reference

### `[bench]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `runs` | int | 3 | Measured runs per prompt. Averaged for final score. Min 2 for variance checks. |
| `warmup` | int | 1 | Warmup runs per prompt (not measured). Primes model/GPU caches. |
| `max_tokens` | int | 256 | Maximum tokens to generate per prompt. |
| `seed` | int | 42 | Random seed for sampler init. With temp=0 (greedy), output is deterministic. |
| `timeout_seconds` | int | 600 | Per-bench timeout in seconds (10 min default). |

### `[bench.text]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `mlx-community/google_gemma-3-27b-it-mlx-bf16` | HF repo ID or local path for text model. |

### `[bench.vlm]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `mlx-community/Qwen3.5-27B-mxfp8-mlx` | HF repo ID or local path for VLM model. |

### `[scoring]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gen_tps_weight` | float | 0.40 | Weight for generation tokens/sec in composite score. |
| `ttft_weight` | float | 0.25 | Weight for time-to-first-token (lower = better). |
| `prefill_tps_weight` | float | 0.20 | Weight for prompt processing tokens/sec. |
| `memory_weight` | float | 0.15 | Weight for peak GPU memory (lower = better). |

### `[scoring.decision]`

These values are read by the agent from `program.md`, NOT by bench scripts.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `both_improved_min` | float | 1.00 | Both scores must meet this minimum to KEEP. |
| `cross_regression_tolerance` | float | 0.02 | Max regression in one bench when the other improves. |
| `cross_improvement_min` | float | 0.03 | Min improvement needed in one bench to tolerate regression in the other. |

### `[constraints]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_single_metric_regression` | float | 0.30 | Any single metric regressing more than 30% = auto-fail. |
| `max_token_deviation` | float | 0.20 | Token count deviating more than 20% from baseline = correctness concern. |
| `per_prompt_regression_check` | bool | true | Apply regression threshold per-prompt, not just on averages. |
| `require_fingerprint_match` | bool | true | Output token sequences must be byte-identical to baseline. |
| `require_tests_pass` | bool | true | Unit and contract tests must pass. |
| `max_suspicion_gain` | float | 0.30 | Gains exceeding 30% in a single cycle are flagged as suspicious (warn, not reject). |
| `variance_max_cv` | float | 0.15 | Coefficient of variation > 15% across runs = noisy results warning. |

### `[optimizer]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allowed_paths` | list[str] | `["src/heylook_llm/"]` | Directories the optimizer may modify. |
| `monkey_patch_allowed` | bool | true | Whether runtime patches of mlx_lm/mlx_vlm are permitted. |
| `banned_files` | list[str] | (see config) | Specific files the optimizer must not touch. |
| `banned_paths` | list[str] | (see config) | Directories the optimizer must not touch. |
| `banned_diff_patterns` | list[str] | (see config) | Regex patterns in diffs that auto-reject. |

## How the Scoring Works

The composite score is a weighted sum of metric ratios relative to baseline:

```
score = gen_tps_weight * (current_gen_tps / baseline_gen_tps)
      + ttft_weight    * (baseline_ttft / current_ttft)         # inverted: lower = better
      + prefill_weight * (current_prefill / baseline_prefill)
      + memory_weight  * (baseline_memory / current_memory)     # inverted: lower = better
```

Baseline always scores exactly 1.0. A score of 1.02 means a 2% composite improvement.

### Example with numbers

Suppose baseline is: gen_tps=45, ttft=300ms, prefill=900tps, memory=24GB.
An optimization gives: gen_tps=47, ttft=290ms, prefill=905tps, memory=24GB.

```
score = 0.40 * (47/45)  + 0.25 * (300/290) + 0.20 * (905/900) + 0.15 * (24/24)
      = 0.40 * 1.044    + 0.25 * 1.034     + 0.20 * 1.006     + 0.15 * 1.000
      = 0.418           + 0.259            + 0.201            + 0.150
      = 1.028
```

A 2.8% composite improvement.

### What shifting weights does

- Increasing `gen_tps_weight` favors raw throughput -- the agent will prioritize decode loop optimizations
- Increasing `ttft_weight` favors latency -- the agent will focus on prefill speed and reducing sync barriers
- Increasing `memory_weight` favors efficiency -- the agent will look for memory pool tuning, cache trimming, type downcasting

## How Verification Works

After each benchmark run, the agent runs a 6-step verification phase before deciding to keep or discard.

### 1. Diff inspection

Reads `git diff HEAD~1` and checks:
- No files modified outside `[optimizer].allowed_paths` -- auto-reject if violated
- No matches against `[optimizer].banned_diff_patterns` -- auto-reject if violated
- Changes to sensitive files (`samplers.py`, `stop_tokens.py`, `config.py`) are flagged but not auto-rejected

### 2. Output fingerprints

With greedy decode (temp=0, seed=42), token sequences must be byte-identical to baseline. The bench scripts compute SHA-256 hashes of token ID sequences and compare them. Mismatch = auto-reject.

### 3. Test suite

`uv run pytest tests/unit/ tests/contract/ -v`

Known pre-existing failures (allowlisted -- do not count as regressions):
- 5 router tests (YAML config vs TOML parser)
- 3 mlx_perf tests (removed mlx_batch_vision module)
- MLX embedding/sampler tests in full suite (Metal context conflicts)

Any NEW failure = auto-reject.

### 4. Per-prompt regression

If `per_prompt_regression_check = true`, the regression threshold is applied to each prompt individually. A single prompt regressing beyond the threshold while others improve is rejected -- can't game the average.

### 5. Suspicion flags (warn, don't auto-reject)

- Any single metric improved >30% in one cycle (likely measurement noise or gaming)
- Changes to sampler/processor construction code
- Monkey patches applied

Logged prominently in cycle JSON for human review.

### 6. Variance check

If the coefficient of variation (stddev/mean) of gen_tps across runs exceeds 15%, results are flagged as noisy. The bench uses sample variance (N-1 denominator) for accurate estimation with small run counts.

## Troubleshooting

### Model won't load

- **Disk space**: Each 27B model needs ~15GB. Check `df -h`.
- **HF auth**: Some models need `huggingface-cli login`. The default models are public.
- **Wrong format**: Must be MLX-format models (safetensors with `config.json`). GGUF won't work.

### Bench crashes

```bash
# Check the last 50 lines of output
tail -n 50 /path/to/stderr/output
```

Common causes:
- Model too large for available memory (try a smaller model)
- MLX Metal initialization failure (restart terminal, check macOS updates)
- Corrupted model cache (`rm -rf ~/.cache/huggingface/hub/models--<model-id>` and re-download)

### Agent keeps discarding

- **Noisy baselines**: Check `variance_warnings` in cycle JSONs. If CV > 0.15, increase `runs` to 5 and re-baseline.
- **Tight constraints**: Loosen `max_single_metric_regression` from 0.30 to 0.40.
- **Wrong optimization target**: Check that the agent is modifying code paths that actually affect the benchmark (the generation loop, not startup code).

### Fingerprint mismatch

- Output changed means the optimization altered model behavior, not just speed. Auto-rejected.
- If using non-greedy sampling: set `require_fingerprint_match = false` (not recommended).
- If you intentionally changed output (e.g., fixed a bug): re-baseline with `--reset-baseline`.

### Data from old runs interfering

```bash
rm -rf apps/optloop/data/
rm -f apps/optloop/results.tsv
# Then re-baseline
uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1
uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
```

### Baseline JSON corrupted

If a crash interrupted a write, you may see JSON parse errors. Delete the corrupted file and re-baseline. (The bench scripts use atomic writes via tmp+rename to prevent this, but pre-existing files from older versions may not have this protection.)

## Running Tests

The bench scoring and validation logic has its own unit test suite (34 tests). Run from the repo root:

```bash
uv run pytest apps/optloop/tests/ -v
```

These tests cover the pure functions in `bench_common.py`: composite scoring, variance checking (sample variance N-1), hard constraints, per-prompt constraints, suspicion detection, output fingerprinting, and config extraction. They do not require a loaded model -- they test the math and validation logic only.

## File Layout

```
apps/optloop/
  README.md              # This file (human guide)
  program.md             # Agent instructions (read by the optimizer)
  bench_config.toml      # Eval configuration (off-limits to optimizer)
  results.tsv            # Per-experiment summary (created at runtime)
  scripts/
    bench_text.py        # Text-only benchmark (mlx-lm path)
    bench_vlm.py         # Vision benchmark (mlx-vlm path, text + vision prompts)
    bench_common.py      # Shared: scoring, fingerprinting, config, baseline mgmt, cycle logging
    bench_analysis.py    # Post-hoc analysis of results.tsv and cycle data
  tests/
    conftest.py          # sys.path setup for importing from scripts/
    test_bench_common.py # Unit tests for bench_common pure functions (34 tests)
  data/                  # Runtime data (gitignored)
    text/                # Text bench baselines and run JSONs
    vlm/                 # VLM bench baselines and run JSONs
    cycles/              # Per-cycle structured logs (cycle_NNNN.json)
    progress.png         # Score chart (generated by bench_analysis.py)
```

## Anti-Gaming Protections

- **Output fingerprinting**: greedy decode (temp=0, seed=42) must produce
  byte-identical token sequences across runs. Mismatch = auto-reject.
- **Per-prompt regression checks**: can't game averages by regressing one
  prompt while improving another.
- **Verification phase**: diff inspection, test suite, suspicion flags,
  variance checks (CV > 0.15 = noisy, flag for review).
- **Config is off-limits**: scoring weights, thresholds, model paths
  cannot be modified by the optimizer.
- **Banned diff patterns**: changes outside allowed_paths are auto-rejected.
- **Zero-token guard**: if generation produces 0 tokens (broken code path), the bench raises immediately instead of recording silent zeros.
- **Weight validation**: if scoring weights don't sum to 1.0, a warning is printed to stderr.

## Local Source Mode (experimental)

By default, the optimizer can only modify code in `src/heylook_llm/`. The actual hot paths (stream_generate, KVCache, samplers) live in mlx-lm and mlx-vlm, which are installed from GitHub HEAD.

Local source mode switches to editable installs from local clones, allowing the optimizer to modify library internals directly.

Local clones:
- `coderef/mlx-lm` -> local mlx-lm checkout
- `coderef/mlx-vlm` -> local mlx-vlm checkout

To enable (Phase B, not yet fully implemented):
1. Switch `pyproject.toml` sources to local paths with `editable = true`
2. Uncomment the expanded `allowed_paths` in `bench_config.toml`
3. Re-establish baselines (`--reset-baseline`)

See the "Local Source Mode" section in `program.md` for details.
