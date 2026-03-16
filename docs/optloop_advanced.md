# Optloop Advanced Guide

Deep dives into how optloop works under the hood, common failure modes, and FAQ. Read the [optloop user guide](./optloop_guide.md) first.

## The bench activation gap

This is the most important thing to understand about optloop.

The bench scripts (`bench_text.py`, `bench_vlm.py`) load models directly via `mlx_lm.utils.load()` and `mlx_vlm.utils.load()`. They never import `heylook_llm`. This means:

- Changes to `generation_core.py`, `samplers.py`, `prompt_cache.py`, etc. have **zero effect on bench scores**
- Only code that runs when `mlx_lm` or `mlx_vlm` is imported can affect benchmarks
- Two mechanisms cross this gap: monkey patches (Phase A) and direct source edits (Phase B)

### Phase A: Monkey patches via .pth file

A `.pth` file in the site-packages directory causes Python to execute code at startup, before any script runs. This is how patches reach the bench scripts:

```
.venv/lib/python3.13/site-packages/heylook_llm_patches.pth
    -> imports heylook_llm.providers.common.patches
    -> patches.py monkey-patches mlx_lm.generate, etc.
    -> bench_text.py runs with patched mlx_lm
```

The `.pth` file contains one line:

```
import heylook_llm.providers.common.patches
```

Python processes all `.pth` files during interpreter startup (`site.py`), so the patches are active before `bench_text.py` even begins importing.

### Phase B: Direct source edits (optloop-lib)

The higher-potential approach: edit mlx-lm and mlx-vlm source directly. This is what `/optloop-lib` does, operating on fork repos at `apps/optloop-lib/repos/`. Changes to `generate.py`, `cache.py`, and `sample_utils.py` directly affect what the bench scripts measure.

## Monkey patching mlx-lm

### The sys.modules gotcha

```python
# WRONG -- gives the generate FUNCTION, not the module
import mlx_lm.generate as gen_mod
gen_mod.stream_generate = patched_version  # patches the function object, not the module

# RIGHT -- get the actual module
import mlx_lm.generate  # ensure loaded
gen_mod = sys.modules['mlx_lm.generate']
gen_mod.stream_generate = patched_version  # patches the module's attribute
```

This happens because `mlx_lm/__init__.py` does `from .generate import generate`, which makes `mlx_lm.generate` resolve to the function when imported via `as`.

### Module shadowing

When you do `import mlx_lm.generate`, Python loads the module and caches it in `sys.modules`. But `mlx_lm.__init__` re-exports the function, so `mlx_lm.generate` (the attribute) is the function, while `sys.modules['mlx_lm.generate']` is the module. Always use `sys.modules` to get the module.

### What can be patched

Anything in the module's namespace. Common targets:

- `mlx_lm.generate.stream_generate` -- the core decode loop
- `mlx_lm.models.cache.KVCache` -- KV cache implementation
- `mlx_lm.sample_utils` -- sampler utilities

### Import timing

The bench scripts do `from mlx_lm.generate import stream_generate` (importing the function by name). If you patch `sys.modules['mlx_lm.generate'].stream_generate` AFTER the importing code runs, the already-imported reference won't see the patch. The `.pth` file ensures patches apply at startup, before any imports.

## Performance ceiling analysis

### The math

For autoregressive text generation, each token requires reading all model weights from memory once. The throughput ceiling is:

```
max_gen_tps = memory_bandwidth / model_size_bytes
```

For M2 Ultra (192GB, ~800 GB/s bandwidth):

| Model | Size | Theoretical Max | Typical Baseline | Utilization |
|-------|------|-----------------|------------------|-------------|
| Gemma-3-27B bf16 | ~54 GB | ~14.8 tps | ~11.7 tps | ~79% |
| Qwen3.5-27B mxfp8 | ~27.5 GB | ~29.1 tps | ~21.0 tps | ~72% |

### What this means for optimization

- Python overhead is <0.1% of per-token time. Removing Python-level inefficiencies has negligible impact on gen_tps.
- The remaining gap (21-28%) comes from GPU scheduling overhead, cache coherency, memory access patterns, and attention/KV cache lookup (which isn't a simple sequential read).
- Real gains require: reducing GPU operations (skip unnecessary work), better memory access patterns (cache-friendly layouts), or fused operations (fewer kernel launches).
- Smaller quantized models have higher theoretical max (less data to read per token) but also more compute per byte (dequantization).

### When to stop

If you're consistently above 90% of theoretical max, further gen_tps gains will be marginal. Optimization focus shifts to:

- Reducing TTFT (first-token latency) instead of gen_tps
- Memory efficiency (serving more concurrent requests)
- Vision pipeline optimizations (which have their own bottlenecks beyond memory bandwidth)

## VLM-specific gotchas

### Position state bleeding (Qwen3.5)

Qwen3.5-style mRoPE models cache `_position_ids` and `_rope_deltas` on the `LanguageModel` instance. These must be reset between fresh generations when not using a pre-filled cache. Without reset, subsequent requests get stale position embeddings, causing shape mismatches or incorrect output.

The server handles this via `generation_core._reset_vlm_positions()`. Monkey patches that bypass the server code path must handle it themselves.

### torchvision dependency

Transformers 5.x introduced `Qwen3VLVideoProcessor` which imports torchvision at module load time. This crashes standalone VLM scripts that don't have torchvision installed. The fix in `mlx_provider._apply_transformers_patches()` suppresses the import, but this only runs when the server starts. Standalone bench scripts need the same patch applied at startup via `.pth`.

### Pre-filled cache pattern

VLM generation is a two-phase process:

1. **Vision forward pass**: full model forward with image data fills KV cache
2. **Text decode**: standard autoregressive generation using the pre-filled cache

Optimizing phase 1 requires VLM-specific knowledge (image grid processing, vision encoder architecture). Optimizing phase 2 is the same as text optimization but starts with a warm cache.

## Failure modes and recovery

### Benchmark crashes

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during model load | Model too large for available memory | Use a smaller model or close other apps |
| `_metal.Device` errors | Metal context corruption | Restart terminal |
| `FileNotFoundError` on model path | `models.toml` entry points to nonexistent directory | Update path or re-download |
| `KeyError` in config parsing | `bench_config.toml` references missing model ID | Check model ID matches a `models.toml` entry |
| Infinite hang during generation | Broken stop token logic or sampler bug | Kill process, check patches for stop_token interference |

### Fingerprint mismatches

The optimization changed model output. Possible causes:

- Float operation reordering (non-associativity of IEEE 754)
- KV cache corruption (partial state from prior run)
- Sampler modification (even "no-op" refactors can change float precision)
- Different token count (truncation or early stop behavior changed)

If the output change is intentional (you deliberately improved quality), re-baseline with `--reset-baseline`.

### Agent keeps discarding everything

Common causes:

1. **Noisy baselines**: Check `variance_warnings` in recent cycle JSONs. If CV > 0.15, increase `runs` to 5 and re-baseline.
2. **Tight constraints**: Loosen `max_single_metric_regression` in bench_config.toml.
3. **Wrong target**: The agent is modifying code paths that don't run during benchmarks. Check the bench activation gap -- only monkey patches and direct source edits affect scores.
4. **Near ceiling**: If baseline is already >90% of theoretical max, marginal improvements are within measurement noise.
5. **Cross-regression**: Text improves but VLM regresses (or vice versa). The agent discards if the regression exceeds `cross_regression_tolerance`.

### Stale baselines

Baselines should be re-established:

- After switching between git and local source modes
- After significant code changes on main that affect inference
- After changing models in bench_config.toml
- After `uv sync` updates mlx-lm or mlx-vlm versions

### .pth file left on wrong branch

If you switch branches without removing the `.pth` file, Python tries to import `heylook_llm.providers.common.patches` at startup. If `patches.py` doesn't exist on the new branch, every `uv run` command fails with an ImportError.

Fix: `rm -f .venv/lib/python3.13/site-packages/heylook_llm_patches.pth`

### Sandbox permission issues with uv run

If `uv run` commands prompt for approval every time despite `autoAllowBashIfSandboxed: true`:

1. uv needs to write to its cache directory (`~/.cache/uv/`)
2. The sandbox filesystem allowlist may not include this path
3. Without sandbox access, commands run with `dangerouslyDisableSandbox: true`
4. This bypasses `autoAllowBashIfSandboxed` and falls through to explicit allow-lists

Fix: Add uv cache path and `.venv/` to `additionalWritePaths` in `.claude/settings.local.json`:

```json
"sandbox": {
  "enabled": true,
  "autoAllowBashIfSandboxed": true,
  "additionalWritePaths": [
    "~/.cache/uv",
    "<project-root>/.venv"
  ]
}
```

### Metal context conflicts in tests

MLX embedding and sampler tests fail when run in the full test suite (Metal context conflicts) but pass individually. This is a pre-existing issue, not a regression. The optloop agent's verification phase allowlists these failures.

## FAQ

### The agent ran 10 cycles and kept nothing

Check the ceiling analysis first. If baseline is already >90% of theoretical max, there isn't much room for improvement. Also check:

- Is the agent modifying bench-affecting code? (See bench activation gap)
- Are baselines noisy? (Check variance warnings)
- Are constraints too tight? (Check bench_config.toml thresholds)

### Scores improved but fingerprints don't match

The optimization changed model output behavior, not just speed. The bench scripts use greedy decode (temp=0, seed=42) and SHA-256 hash the token IDs. Any change to the token sequence -- even one token -- fails the fingerprint check.

If the output change is intentional, re-baseline. Otherwise, investigate: float precision, operation order, sampler behavior, or stop token logic.

### How do I know when to stop?

- **Ceiling analysis**: Calculate `bandwidth / model_size` for your hardware. Above 90%, diminishing returns.
- **Diminishing returns pattern**: If the last 5+ cycles all discarded, the low-hanging fruit is gone.
- **Optimization log**: Check `docs/optimization_log.md` for what's been tried across all sessions.

### Can I run optloop and optloop-lib in the same session?

No. They use different venvs and different scope. The `.pth` mechanism (optloop) conflicts with direct source edits (optloop-lib) because both change how `mlx_lm` behaves. Run them in separate sessions.

### What if I want to test a specific idea?

Modify the query in the skill invocation or tell the agent directly:

```
/optloop mar16

> Focus on KV cache memory layout optimization for M2 Ultra bandwidth characteristics.
> Try prefetching the next layer's KV cache while computing attention on the current layer.
```

### How do I compare results across sessions?

- **Optimization log**: `docs/optimization_log.md` has a baselines table with historical numbers
- **Cycle JSONs**: Compare `cumulative.delta_vs_original` across sessions (if starting from the same baseline)
- **bench_analysis.py**: Run on specific session data for detailed analysis

### The bench takes forever

- Reduce `runs` in bench_config.toml (fewer measured runs per prompt)
- Reduce `max_tokens` (shorter generations)
- Check model size vs available memory (if swapping to disk, everything slows)
- Close other GPU-intensive apps

### I merged the optloop branch but performance regressed on the server

The bench scripts measure raw mlx-lm/mlx-vlm performance. The heylookitsanllm server adds additional layers (radix cache, prompt cache, streaming, API handling) that can interact with optimizations in unexpected ways. Test the server path separately after merging.

### "Can I run this on a MacBook Air?"

Optloop itself runs anywhere, but the default benchmark models (27B) need ~30GB during inference. A MacBook Air with 24GB can run smaller models but will need modified bench_config.toml entries. A 16GB machine should use 7B-class models.

## Analysis deep dive

### bench_analysis.py

Run after an optloop session to analyze results:

```bash
uv run apps/optloop/scripts/bench_analysis.py
```

It reads `results.tsv`, per-run JSONs, and cycle JSONs. Output includes:

- Experiment summary (total, kept, discarded, crashed)
- Top improvements ranked by combined score delta
- Per-metric history for text and VLM
- Cumulative drift from original baseline
- Score chart (saved as `data/progress.png` if matplotlib is available)

### Reading cycle JSONs programmatically

```python
import orjson
from pathlib import Path

cycles = sorted(Path("apps/optloop/data/cycles").glob("cycle_*.json"))
for path in cycles:
    cycle = orjson.loads(path.read_bytes())
    decision = cycle["decision"]
    desc = cycle["optimizer"]["description"]
    text_score = cycle["results"]["text"]["composite_score"]
    vlm_score = cycle["results"]["vlm"]["composite_score"]
    print(f"Cycle {cycle['cycle_id']}: {decision} | "
          f"text={text_score:.4f} vlm={vlm_score:.4f} | {desc}")
```

## Extending optloop

### Adding new prompts

Prompts are defined in the bench scripts (`bench_text.py`, `bench_vlm.py`). These are off-limits to the optimizer agent, but you can modify them for your own testing:

- Add entries to the prompts list
- Maintain greedy decode settings (temp=0, seed=42) for fingerprint compatibility
- Re-baseline after changing prompts

### Changing models

Update `[bench.text]` or `[bench.vlm]` in `bench_config.toml` with a model ID that exists in `models.toml`. Re-baseline after changing models.

### Adjusting weights

Modify `[scoring]` weights in `bench_config.toml` to shift optimization pressure. Higher `gen_tps_weight` favors raw throughput. Higher `ttft_weight` favors latency. Weights must sum to 1.0.

### Custom constraints

Modify `[constraints]` in `bench_config.toml`:

- Loosen `max_single_metric_regression` for more tolerance
- Disable `per_prompt_regression_check` to allow average-level evaluation
- Adjust `variance_max_cv` if your environment is noisy

## optloop-lib specifics

### Fork management

The `/optloop-lib` skill operates on local forks at `apps/optloop-lib/repos/`:

- `repos/mlx-lm/` -- fork of mlx-lm
- `repos/mlx-vlm/` -- fork of mlx-vlm

Each fork gets its own branch (`optloop-lib/<tag>`). Commits go in the fork repos, never in heylookitsanllm.

### AGENTS.md

`apps/optloop-lib/AGENTS.md` is a knowledge base populated during optimization cycles. It records what the agent learns about mlx-lm and mlx-vlm internals: code patterns, constraints, and optimization opportunities. The agent reads it at session start and updates it during cycles.

### Tier system

The optimizer follows a tier-guided approach:

1. **Tier 1** (hot paths): `generate.py`, `cache.py`, `sample_utils.py` -- start here
2. **Tier 2** (model architectures): `models/*.py`, `quant/*.py`
3. **Tier 3** (VLM pipeline): `mlx_vlm/generate.py`, `prompt_utils.py`, `utils.py`

### Editable installs

The fork repos use editable installs. Changes to source files are immediately visible to the bench scripts without reinstalling. This is configured via `[tool.uv.sources]` in the optloop-lib pyproject.toml.

### Coderef snapshots

The agent uses `bench_common.snapshot_coderef()` to record fork HEAD commits before each cycle and `bench_common.reset_coderef()` to roll back discarded changes. This ensures clean rollback across multiple repos.
