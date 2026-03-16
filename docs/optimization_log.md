# Optimization Log

Accumulated findings from optloop inference optimization sessions.
Updated at the end of each session (on main, after merge).

## Performance Baselines

| Date | Model | gen_tps | ttft_ms | prefill_tps | memory_gb | Notes |
|------|-------|---------|---------|-------------|-----------|-------|
| 2026-03-16 | Gemma-3-27B bf16 | 11.7 | 679.8 | 74.9 | 53.2 | First text baseline with patches |
| 2026-03-16 | Qwen3.5-27B mxfp8 | 21.0 | 652.3 | 104.0 | 27.5 | First VLM baseline |

## Performance Ceilings (M2 Ultra 192GB, ~800 GB/s bandwidth)

- Gemma-3-27B bf16 (~54GB): theoretical max = 14.8 gen_tps. Current 11.7 = 79%
- Qwen3.5-27B mxfp8 (~27.5GB): theoretical max = 29.1 gen_tps. Current 21.0 = 72%
- Python overhead is <0.1% of per-token time. GPU forward pass dominates.

## What Works

(Nothing confirmed bench-measurable yet. Mar16 patches established the activation
mechanism but baselines were recorded WITH patches active, so no before/after delta.)

## What Doesn't Work (or has negligible impact)

- **Radix cache debounce** (mar16): Attempted debouncing _check_memory_pressure() every
  N inserts. Broke test expectations and the check is already infrequent (max 128/lifetime).
- **Python-level overhead reduction** (mar16 analysis): wired_limit cache, skip logsumexp,
  skip peak_memory -- all <0.1% of per-token time. Real gains require reducing GPU work.

## Technical Findings

- Bench scripts never import heylook_llm. Only monkey patches via .pth activation
  or Phase B (local source edits) affect bench scores.
- `import mlx_lm.generate as gen_mod` gives the function, not the module.
  Use `sys.modules['mlx_lm.generate']`.
- Qwen3.5 caches _position_ids and _rope_deltas on LanguageModel. Must reset
  between fresh generations (no pre-filled cache) when using radix cache.
- Transformers 5.x video processor crashes without torchvision. Need patches
  for standalone VLM scripts.
- .pth file must be removed when switching away from optloop branch.

## Open Questions / Untested Ideas

- Phase B (direct mlx-lm source edits) -- highest potential, not yet enabled
- mx.compile on decode step
- Shape bucketing for variable-length prefill
- Custom Metal kernels for fused attention
- KV cache memory layout optimization for M2 Ultra bandwidth characteristics
