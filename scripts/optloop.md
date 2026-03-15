# Continuous Inference Optimization Loop

Agent instructions for autonomous inference optimization. Two benchmarks (text + VLM), continuous loop.

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
4. Load `/mlx` and `/mlx-lm` skills for MLX optimization knowledge
5. Run both baselines:
   ```
   uv run scripts/bench_text.py --reset-baseline > text_run.log 2>&1
   uv run scripts/bench_vlm.py --reset-baseline > vlm_run.log 2>&1
   ```
6. Initialize `results.tsv` with header:
   ```
   commit	text_score	vlm_score	text_tps	vlm_tps	text_ttft	vlm_ttft	memory_gb	status	description
   ```
   Add baseline row from grep output.
7. Run existing tests: `uv run pytest tests/unit/ tests/contract/ -v`

## Scope

### What you CAN modify

- Any file in `src/heylook_llm/` -- generation loop, sampling, caching, provider layer, router, API
- Configuration defaults in `config.py` (not `models.toml`)

### What you CANNOT modify

- `models.toml` or code that creates/populates it
- `scripts/bench_text.py`, `scripts/bench_vlm.py`, `scripts/bench_common.py`, `scripts/bench_analysis.py`
- Test files (tests must continue passing)
- Dependencies in `pyproject.toml`
- Can only *recommend* sampler/cache settings, not bake them into code that reads `models.toml`

## The Loop

```
LOOP FOREVER:
1. git status / git log --oneline -5
2. Choose an optimization idea (use /mlx and /mlx-lm skills for guidance)
3. Implement the change
4. Run tests:
   uv run pytest tests/unit/ tests/contract/ -v > test.log 2>&1
   - If tests fail: fix or revert, do not proceed to bench
5. git commit (conventional commit message)
6. Run BOTH benchmarks:
   uv run scripts/bench_text.py > text_run.log 2>&1
   uv run scripts/bench_vlm.py > vlm_run.log 2>&1
7. Read results:
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:" text_run.log
   grep "^composite_score:\|^avg_gen_tps:\|^avg_ttft_ms:\|^avg_vision_ms:" vlm_run.log
8. If either grep empty: crashed. tail -n 50 <log>, attempt fix or skip
9. Append to results.tsv (both scores)
10. Decision:
    - BOTH scores >= 1.0 (or one improved, other unchanged): KEEP
    - One improved but other regressed: evaluate magnitude
      Small regression (<2%) with significant gain (>3%) in other = KEEP
      Otherwise DISCARD
    - Both regressed: DISCARD
11. If discard: git reset --hard to previous commit
12. Update internal/log/log_YYYY-MM-DD.md with experiment results
```

## Optimization Categories

### Shared path (benefits both)
- `generation_core.py` hot loop: fewer Python attribute lookups, fewer conditionals
- `samplers.py`: skip no-op processors, pre-compute constants
- `cache_helpers.py`: faster snapshot/restore, avoid redundant copies
- Strategic GPU synchronization placement
- Lock contention reduction
- Type promotion avoidance (float32 vs float16 operations)

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
- Memory usage can increase only within 30% of baseline
- All unit/contract tests must pass
- Do NOT modify bench scripts or test files

## Never stop

Run indefinitely until manually interrupted.
