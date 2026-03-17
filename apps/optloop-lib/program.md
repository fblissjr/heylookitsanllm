# Library-Level Inference Optimization Loop

Human-readable reference for library-level optimization of mlx-lm and mlx-vlm internals. Fork repos are cloned locally at `apps/optloop-lib/repos/`. Two benchmarks (text + VLM), continuous loop with verification, output fingerprinting, and per-cycle structured logging. Benchmarks include single-turn and multi-turn prompts to test both raw generation speed and context-carry performance with growing KV caches.

Target hardware: Mac Studio M2 Ultra, 192GB unified memory.

> **Agent invocation**: Use `/optloop-lib <run-tag>` in Claude Code. The skill handles setup,
> pre-flight reads, and references loading automatically. This file is the human-readable
> companion; the agent-consumable instructions live in `.claude/skills/optloop-lib/`.

## Setup

1. Agree on a run tag (e.g. `mar15-lib`)
2. Create branches in EACH fork repo:
   ```
   cd apps/optloop-lib/repos/mlx-lm && git checkout -b optloop-lib/<tag>
   cd apps/optloop-lib/repos/mlx-vlm && git checkout -b optloop-lib/<tag>
   ```
3. Read tier-1 files from fork repos:
   - `repos/mlx-lm/mlx_lm/generate.py`
   - `repos/mlx-lm/mlx_lm/models/cache.py`
   - `repos/mlx-lm/mlx_lm/sample_utils.py`
4. Load `/mlx-skills:mlx` and `/mlx-skills:mlx-lm` skills for MLX optimization knowledge
5. Read `apps/optloop-lib/bench_config.toml` -- this is your config. Do NOT modify it.
5b. Read `docs/optimization_log.md` -- cross-session knowledge base (shared with optloop). Check baselines and prior findings before repeating work.
5c. Read `apps/optloop-lib/AGENTS.md` -- library internals knowledge base (populated during optimization cycles).
6. Establish baselines:
   ```
   cd apps/optloop-lib && uv run scripts/bench_text.py --reset-baseline 2>&1
   cd apps/optloop-lib && uv run scripts/bench_vlm.py --reset-baseline 2>&1
   ```
7. Initialize `results.tsv` with header:
   ```
   mlx_lm_commit	mlx_vlm_commit	text_score	vlm_score	text_tps	vlm_tps	text_ttft	vlm_ttft	memory_gb	status	description
   ```
   Add baseline row from grep output.
8. Run fork tests once to establish baseline pass/fail:
   ```
   cd apps/optloop-lib/repos/mlx-lm && python -m pytest tests/ -x -q 2>&1 | tail -5
   ```
9. Note the baseline fingerprints from stderr output -- these are the correctness anchors.

## Scope

Read scope from `[optimizer]` section of `bench_config.toml`.

### What you CAN modify

- Anything in `repos/mlx-lm/mlx_lm/` -- generate loop, sampling, caching, model architectures
- Anything in `repos/mlx-vlm/mlx_vlm/` -- VLM pipeline, prompt utils, model code

### What you CANNOT modify

- `repos/mlx-lm/setup.py` or `repos/mlx-vlm/pyproject.toml` (build configs)
- `repos/mlx-lm/tests/` or `repos/mlx-vlm/mlx_vlm/tests/` (test suites)
- `apps/optloop-lib/scripts/` (bench harness)
- `apps/optloop-lib/bench_config.toml` (eval config)
- `apps/optloop-lib/data/` (bench data)
- `src/heylook_llm/` (that's optloop's job, not yours)

### Tier-guided optimization

Read `[optimizer.scope]` from bench_config.toml:

- **Tier 1** (hot paths): `generate.py`, `cache.py`, `sample_utils.py` -- start here
- **Tier 2** (model architectures): `models/*.py`, `quant/*.py`
- **Tier 3** (VLM pipeline): `mlx_vlm/generate.py`, `prompt_utils.py`, `utils.py`

## The Loop

```
LOOP FOREVER:
1. Snapshot fork HEADs:
   Use bench_common.snapshot_coderef() conceptually -- record current branch + commit
   for both repos before making changes.
1b. Read results.tsv -- check what's been tried this session, identify patterns
1c. On first iteration: also read docs/optimization_log.md and AGENTS.md for cross-session context

2. Choose optimization target (tier-guided, prefer tier 1 early)

3. Record hypothesis: what you're changing and why you expect it to help

4. Implement change in fork repo:
   Edit files directly in repos/mlx-lm/mlx_lm/ or repos/mlx-vlm/mlx_vlm/

5. Quick-check fork tests if touching tested code:
   cd apps/optloop-lib/repos/mlx-lm && python -m pytest tests/ -x -q 2>&1 | tail -10
   (only if the changed module has corresponding tests)

6. Commit in the fork repo:
   cd apps/optloop-lib/repos/mlx-lm && git add -A && git commit -m "perf: <description>"

7. Run BOTH benchmarks from apps/optloop-lib/:
   cd apps/optloop-lib && uv run scripts/bench_text.py 2>&1
   cd apps/optloop-lib && uv run scripts/bench_vlm.py 2>&1

8. Read results:
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:\|^fingerprint_match:" <text_output>
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:\|^avg_vision_ms:\|^fingerprint_match:" <vlm_output>

9. If either grep empty: crashed. tail -n 50 <output>, attempt fix or skip.

10. Run VERIFICATION PHASE (see below)

11. Decision:
    - BOTH scores >= 1.0 AND fingerprints match AND tests pass: KEEP
    - One improved but other regressed: evaluate magnitude
      Small regression (<2%) with significant gain (>3%) in other = KEEP
      Otherwise DISCARD
    - Both regressed: DISCARD
    - Fingerprint mismatch: DISCARD (output correctness violated)
    - New test failures: DISCARD

12. If DISCARD: reset fork repos to pre-cycle state
    cd apps/optloop-lib/repos/mlx-lm && git reset --hard <snapshot_commit>
    cd apps/optloop-lib/repos/mlx-vlm && git reset --hard <snapshot_commit>

13. Append to results.tsv (with fork commit hashes from both repos)

14. Write cycle JSON via bench_common helpers (see Cycle Logging below)

15. NEVER STOP
```

## Verification Phase

Run these checks AFTER benchmarks, BEFORE the keep/discard decision. You are verifying your own work.

### 1. Diff Inspection

Read the diff in the modified fork repo and check:
- No files modified outside `[optimizer].allowed_paths`
- No matches against `[optimizer].banned_diff_patterns`

### 2. Output Fingerprints

Already checked by bench scripts. If `fingerprint_match: false` in grep output: auto-reject.
With greedy decode (temp=0, seed=42), token sequences must be byte-identical to baseline.

### 3. Fork Test Suite

If you touched code with corresponding tests, run them:
```
cd apps/optloop-lib/repos/mlx-lm && python -m pytest tests/ -x -q 2>&1 | tail -10
```

Any NEW failure = auto-reject.

### 4. Per-Prompt Regression

If `[constraints].per_prompt_regression_check = true` in config, apply the regression threshold to each prompt individually. A single prompt regressing beyond the threshold while others improve = reject.

### 5. Suspicion Flags (warn, don't auto-reject)

- Any single metric improved >30% in one cycle
- Changes to sampler construction code
- Changes that look like they reduce computation (might be skipping work)

Log these prominently in the cycle JSON and results.tsv.

### 6. Variance Check

If stddev of gen_tps across runs > 15% of mean (CV > 0.15), results are noisy. Re-run or flag.

## Cycle Logging

After every experiment (kept, discarded, or crashed), write a cycle JSON file to `apps/optloop-lib/data/cycles/cycle_NNNN.json`.

The cycle JSON must include:
- `cycle_id`: sequential integer
- `timestamp`: ISO 8601 UTC
- `git`: info about the fork repos (branch, commit hash for each)
- `git.coderef_changes`: per-repo commit info and diff stat
- `optimizer`: description of what you changed, your hypothesis
- `config_snapshot`: current scoring weights, max_tokens, seed, bench runs
- `results`: text and vlm composite scores, per-metric and per-prompt data, fingerprint match status
- `verification`: tests passed, new failures, fingerprints match, suspicion flags
- `cumulative`: delta vs original baseline for each key metric, total kept/discarded/crashed
- `decision`: keep/discard/crash
- `decision_reason`: human-readable explanation

Use `bench_common.next_cycle_id()`, `bench_common.save_cycle()`, and `bench_common.build_cycle_data()` helpers.

## Optimization Categories

### Tier 1: Hot paths (start here)
- `generate.py` decode loop: fewer Python attribute lookups, fewer conditionals per token
- `cache.py` KV cache: faster operations, memory layout optimization
- `sample_utils.py`: skip no-op processors, pre-compute constants
- Strategic GPU synchronization placement
- Type promotion avoidance (float32 vs float16 operations)
- Multi-turn KV cache efficiency: prefill performance with prior conversation context already in cache

### Tier 2: Model architectures
- Attention implementations: fused operations, memory-efficient attention
- Quantization kernels: faster dequant paths
- Model-specific optimizations (Gemma, Llama, Qwen)

### Tier 3: VLM pipeline
- Vision encoding efficiency
- Image processing pipeline
- Prompt template construction
- Pre-filled cache pattern optimization
- Multi-turn vision conversations: text follow-up generation after image encoding with shared KV cache

## Git Hygiene

- Commits go in fork repos (`repos/mlx-lm/`, `repos/mlx-vlm/`), NEVER in heylookitsanllm
- Agent can push to `origin` (fork) when asked, never to `upstream`
- Each fork repo gets its own branch: `optloop-lib/<tag>`
- Use `bench_common.snapshot_coderef()` and `bench_common.reset_coderef()` for rollbacks

## Key Constraints

- Never break the streaming API contract
- Token output quality must remain identical (greedy decode at temp=0 in bench)
- Output fingerprints must match baseline (byte-identical token sequences)
- Memory usage can increase only within 30% of baseline
- Fork tests must continue passing
- Do NOT modify bench scripts, test files, or bench_config.toml
- Log every experiment, even crashes

## Session End

When the session ends (interrupted, wrapping up):

1. Run analysis if cycle data exists
2. Distill findings into `docs/optimization_log.md` on main:
   - New baselines achieved (if improved over prior best)
   - Approaches that worked (with magnitude)
   - Approaches that failed (with reasoning)
   - Technical discoveries and gotchas
3. Update `apps/optloop-lib/AGENTS.md` with any new library internals knowledge
4. Commit updates on main
5. Write session log to `internal/log/log_YYYY-MM-DD.md`

## Never stop

Run indefinitely until manually interrupted.
