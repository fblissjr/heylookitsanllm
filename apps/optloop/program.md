# Inference Optimization Loop

Human-readable reference for the app-level optimization loop. Two benchmarks (text + VLM), continuous loop with verification, output fingerprinting, and per-cycle structured logging.

Target hardware: Mac Studio M2 Ultra, 192GB unified memory.

> **Agent invocation**: Use `/optloop <run-tag>` in Claude Code. The skill handles setup,
> pre-flight reads, and references loading automatically. This file is the human-readable
> companion; the agent-consumable instructions live in `.claude/skills/optloop/`.

## Setup

1. Agree on a run tag (e.g. `mar15`)
2. Create branch `optloop/<tag>` from current HEAD
3. Read all in-scope files:
   - `src/heylook_llm/providers/mlx_provider.py`
   - `src/heylook_llm/providers/common/generation_core.py`
   - `src/heylook_llm/providers/common/samplers.py`
   - `src/heylook_llm/providers/common/model_wrappers.py`
   - `src/heylook_llm/providers/common/vlm_inputs.py`
   - `src/heylook_llm/providers/common/cache_helpers.py`
   - `src/heylook_llm/providers/common/prompt_cache.py`
   - `src/heylook_llm/providers/common/radix_cache.py`
   - `src/heylook_llm/providers/common/stop_tokens.py`
   - `src/heylook_llm/router.py`
   - `src/heylook_llm/api.py`
   - `src/heylook_llm/config.py`
4. Load `/mlx-skills:mlx` and `/mlx-skills:mlx-lm` skills for MLX optimization knowledge
5. Read `apps/optloop/bench_config.toml` -- this is your config. Do NOT modify it.
5b. Read `docs/optimization_log.md` -- cross-session knowledge base with baselines, findings, and known dead-ends from prior sessions. Do not repeat failed approaches without a new angle.
6. Run both baselines:
   ```
   uv run apps/optloop/scripts/bench_text.py --reset-baseline 2>&1
   uv run apps/optloop/scripts/bench_vlm.py --reset-baseline 2>&1
   ```
7. Initialize `results.tsv` with header:
   ```
   commit	text_score	vlm_score	text_tps	vlm_tps	text_ttft	vlm_ttft	memory_gb	status	description
   ```
   Add baseline row from grep output.
8. Run existing tests: `uv run pytest tests/unit/ tests/contract/ -v`
9. Note the baseline fingerprints from stderr output -- these are the correctness anchors.

## Scope

Read scope from `[optimizer]` section of `bench_config.toml`.

### Cross-session knowledge

`docs/optimization_log.md` accumulates findings across sessions: baselines, what worked, what failed, technical gotchas. Read it during setup; update it at session end (on main, after merge).

### What you CAN modify

- Any file in `src/heylook_llm/` -- generation loop, sampling, caching, provider layer, router, API
- Configuration defaults in `config.py` (not `models.toml`)
- You CAN create monkey-patch files (e.g. `src/heylook_llm/providers/common/patches.py`) but must log them prominently

### What you CANNOT modify

- `models.toml` or code that creates/populates it
- `apps/optloop/scripts/` (bench harness)
- `apps/optloop/bench_config.toml` (eval config)
- `apps/optloop/data/` (bench data)
- Test files in `tests/` (tests must continue passing)
- Dependencies in `pyproject.toml`
- Can only *recommend* sampler/cache settings, not bake them into code that reads `models.toml`

### Local Source Mode

When `pyproject.toml` uses path sources (editable installs) and `[optimizer].allowed_paths` includes `coderef/` directories, the optimizer can modify mlx-lm and mlx-vlm internals directly instead of being limited to monkey patches.

Requirements for local source mode:
- `bench_config.toml` uses the expanded `allowed_paths` (uncomment the coderef line)
- `pyproject.toml` `[tool.uv.sources]` points to local paths with `editable = true`
- Baselines MUST be re-established after switching between git and local source modes
- Commits in coderef repos are separate from the main repo -- track them in cycle JSON under `git.coderef_changes`

## The Loop

```
LOOP FOREVER:
1. git status / git log --oneline -5
1b. Read results.tsv -- check what's been tried this session, identify patterns in kept/discarded experiments
1c. On first iteration: also read docs/optimization_log.md for cross-session findings
2. Choose an optimization idea (use /mlx and /mlx-lm skills for guidance)
3. Record your hypothesis: what you're changing and why you expect it to help
4. Implement the change
5. Run tests:
   uv run pytest tests/unit/ tests/contract/ -v
   - If tests fail: fix or revert, do not proceed to bench
6. git commit (conventional commit message)
7. Run BOTH benchmarks:
   uv run apps/optloop/scripts/bench_text.py 2>&1
   uv run apps/optloop/scripts/bench_vlm.py 2>&1
8. Read results:
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:\|^fingerprint_match:" text_run.log
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:\|^avg_vision_ms:\|^fingerprint_match:" vlm_run.log
9. If either grep empty: crashed. tail -n 50 <log>, attempt fix or skip.
10. Run VERIFICATION PHASE (see below)
11. Append to results.tsv (both scores)
12. Write cycle JSON via bench_common helpers (see Cycle Logging below)
13. Decision:
    - BOTH scores >= 1.0 AND fingerprints match AND tests pass: KEEP
    - One improved but other regressed: evaluate magnitude
      Small regression (<2%) with significant gain (>3%) in other = KEEP
      Otherwise DISCARD
    - Both regressed: DISCARD
    - Fingerprint mismatch: DISCARD (output correctness violated)
    - New test failures: DISCARD
14. If discard: git reset --hard to previous commit
15. Update internal/log/log_YYYY-MM-DD.md with experiment results
```

## Verification Phase

Run these checks AFTER benchmarks, BEFORE the keep/discard decision. You are verifying your own work.

### 1. Diff Inspection

Read `git diff HEAD~1` and check:
- No files modified outside `[optimizer].allowed_paths`
- No matches against `[optimizer].banned_diff_patterns`
- Flag (log prominently, don't auto-reject) changes to:
  - `samplers.py`
  - `stop_tokens.py`
  - `config.py`

### 2. Output Fingerprints

Already checked by bench scripts. If `fingerprint_match: false` in grep output: auto-reject.
With greedy decode (temp=0, seed=42), token sequences must be byte-identical to baseline.

### 3. Test Suite

`uv run pytest tests/unit/ tests/contract/ -v`

Known pre-existing failures (allowlisted):
- 5 router tests (YAML config vs TOML parser)
- 3 mlx_perf tests (removed mlx_batch_vision module)
- MLX embedding/sampler tests in full suite (Metal context conflicts)

Any NEW failure = auto-reject.

### 4. Per-Prompt Regression

If `[constraints].per_prompt_regression_check = true` in config, apply the regression threshold to each prompt individually. A single prompt regressing beyond the threshold while others improve = reject.

### 5. Suspicion Flags (warn, don't auto-reject)

- Any single metric improved >30% in one cycle
- Changes to sampler/processor construction code
- Monkey patches applied

Log these prominently in the cycle JSON and results.tsv.

### 6. Variance Check

If stddev of gen_tps across runs > 15% of mean (CV > 0.15), results are noisy. Re-run or flag.

## Cycle Logging

After every experiment (kept, discarded, or crashed), write a cycle JSON file to `apps/optloop/data/cycles/cycle_NNNN.json`.

The cycle JSON must include:
- `cycle_id`: sequential integer
- `timestamp`: ISO 8601 UTC
- `git`: commit hash, parent hash, files modified, diff stat
- `optimizer`: description of what you changed, your hypothesis
- `config_snapshot`: current scoring weights, max_tokens, seed, bench runs
- `results`: text and vlm composite scores, per-metric and per-prompt data, fingerprint match status
- `verification`: tests passed, new failures, fingerprints match, suspicion flags
- `cumulative`: delta vs original baseline for each key metric, total kept/discarded/crashed
- `decision`: keep/discard/crash
- `decision_reason`: human-readable explanation

Use `bench_common.next_cycle_id()`, `bench_common.save_cycle()`, and `bench_common.build_cycle_data()` helpers.

## Optimization Categories

### Shared path (benefits both benches)
- `generation_core.py` hot loop: fewer Python attribute lookups, fewer conditionals per token
- `samplers.py`: skip no-op processors, pre-compute constants
- `cache_helpers.py`: faster snapshot/restore, avoid redundant copies
- Strategic GPU synchronization placement (M2 Ultra has high bandwidth -- minimize sync points)
- Lock contention reduction
- Type promotion avoidance (float32 vs float16 operations)
- Memory pool tuning for 192GB unified memory

### Text-specific (mlx-lm path)
- Radix tree lookup optimization (block size, fewer allocations)
- Prompt cache hit ratio improvements
- Speculative decoding acceptance tuning
- DraftTuner window size / threshold tuning

### VLM-specific (mlx-vlm path, more headroom)
- `LanguageModelLogitsWrapper` overhead reduction (attribute caching, __call__ path)
- `vlm_apply_chat_template` optimization (template caching, fewer string ops)
- Image processing pipeline (`vlm_inputs.py`)
- Pre-filled cache pattern efficiency
- Vision encoding forward pass (minimize Python overhead around VLM call)

### Monkey patches (allowed, must log)
- Can patch mlx_lm / mlx_vlm at runtime
- Must go in a dedicated file: `src/heylook_llm/providers/common/patches.py`
- Must be logged prominently in cycle JSON

### Recommendations only (cannot code-change)
- KV cache type/quantization settings in models.toml
- Sampler hyperparameter defaults
- Draft model configurations

## Understanding the Code Paths

### Text (mlx-lm)
```
tokenizer.apply_chat_template() -> tokenizer.encode()
  -> generate_text() -> build_sampler()
  -> run_generation() -> radix cache lookup -> lm_stream_generate()
  -> yield responses
```

### Vision (mlx-vlm)
```
vlm_apply_chat_template() -> vlm_prepare_inputs()
  -> build_sampler()
  -> model(input_ids, cache=request_cache, pixel_values=...) [vision forward]
  -> sample first token
  -> run_generation(pre_filled_cache=request_cache) [text decode]
  -> yield responses
```

### Shared code
- `generation_core.py:run_generation()` -- unified decode loop (both paths)
- `samplers.py:build()` -- sampler construction (both paths)
- `model_wrappers.py:LanguageModelLogitsWrapper` -- VLM text decode adapter

## Key Constraints

- Never break the streaming API contract
- Token output quality must remain identical (greedy decode at temp=0 in bench)
- Output fingerprints must match baseline (byte-identical token sequences)
- Memory usage can increase only within 30% of baseline
- All unit/contract tests must pass
- Do NOT modify bench scripts, test files, or bench_config.toml
- Log every experiment, even crashes

## Session End

When the session ends (interrupted, wrapping up, or switching branches):

1. Remove .pth file: `rm -f .venv/lib/python3.13/site-packages/heylook_llm_patches.pth`
2. Run analysis if cycle data exists: `uv run apps/optloop/scripts/bench_analysis.py`
3. Switch to main: `git checkout main`
4. Distill findings into `docs/optimization_log.md`:
   - New baselines achieved (if improved over prior best)
   - Approaches that worked (with magnitude)
   - Approaches that failed (with reasoning)
   - Technical discoveries and gotchas
   - Updated open questions
5. Commit `docs/optimization_log.md`
6. Write session log to `internal/log/log_YYYY-MM-DD.md`

## Never stop

Run indefinitely until manually interrupted.
