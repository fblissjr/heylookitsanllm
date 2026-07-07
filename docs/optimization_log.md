# Optimization Log

Last updated: 2026-07-07

Accumulated findings from inference optimization sessions.
Updated at the end of each session (on main, after merge).

Note (2026-07-07): FIRST real optloop-lib run -- text baseline re-established on
mlx 0.32.0 with the target repo id fixed (`mlx-community/gemma-3-27b-it-bf16`;
the old config id did not exist) and a new `long_context` workload (~1k prompt
tokens) added. Numbers match the Mar-16 baseline (11.7 gen_tps) -- continuity
holds. First experiment: classic speculative decoding (see its section below).
Harness validated: it correctly flagged the spec-decode regressions and
fingerprint divergences. VLM baseline still pending real photos.

Note (2026-07-06): the app-level `apps/optloop` harness was retired -- its
benchmarks never exercised the server path (a fact first recorded in the
Technical Findings below on 2026-03-16). Sessions now target
`apps/optloop-lib` (fork-level) only; this file remains the single
cross-session knowledge base. The 2026-03-16 baselines below were measured
via direct library calls and remain valid for optloop-lib comparisons.

## Performance Baselines

| Date | Model | gen_tps | ttft_ms | prefill_tps | memory_gb | Notes |
|------|-------|---------|---------|-------------|-----------|-------|
| 2026-03-16 | Gemma-3-27B bf16 | 11.7 | 679.8 | 74.9 | 53.2 | First text baseline with patches |
| 2026-03-16 | Qwen3.5-27B mxfp8 | 21.0 | 652.3 | 104.0 | 27.5 | First VLM baseline |
| 2026-07-07 | gemma-3-27b-it-bf16 | 11.7 | 1322.0 | 151.9 | 53.9 | mlx 0.32.0; 6 text prompts incl. long_context; canonical text baseline (avg over prompts; ttft/prefill higher because the prompt mix now includes long-prefill workloads) |
| 2026-07-07 | gemma-4-31b-it-8bit-mlx (VLM) | 15.3 | 1428.4 | 167.5 | 33.3 | FIRST real-vision baseline. mlx-vlm 0.6.5/mlx-lm 0.31.3; 14 prompts (2 text + 3 synthetic + 9 real photos), avg vision-encode 1592ms. Dense = bandwidth-bound. |
| 2026-07-07 | gemma-4-26b-a4b-it-8bit-mlx (VLM) | 48.1 | 483.4 | 494.7 | 27.3 | Same suite/date. MoE (~4B active): 3.1x faster decode + 3x faster vision-encode (524ms) than the dense 31B despite similar total params -- dispatch-bound, not bandwidth-bound. |

## Performance Ceilings (M2 Ultra 192GB, ~800 GB/s bandwidth)

- Gemma-3-27B bf16 (~54GB): theoretical max = 14.8 gen_tps. Current 11.7 = 79%
- Qwen3.5-27B mxfp8 (~27.5GB): theoretical max = 29.1 gen_tps. Current 21.0 = 72%
- Python overhead is <0.1% of per-token time. GPU forward pass dominates.

## Speculative decoding (classic draft-model) -- 2026-07-07

First real experiment. Target `gemma-3-27b-it-bf16` (the multimodal repo; mlx-lm
loads only its text tower via gemma3.py), draft `gemma-3-1b-it-bf16` (same
gemma-3 tokenizer). Greedy, mlx 0.32.0, 6 prompts, vs the non-speculative
baseline. Per-prompt gen_tps:

| prompt | baseline | num_draft=2 | num_draft=4 |
|--------|---------:|------------:|------------:|
| short             | 11.7 | 10.3 | **12.9** |
| medium            | 11.8 |  9.4 | **12.0** |
| long              | 11.7 |  9.6 | 11.2 |
| multi_turn_short  | 11.7 |  9.6 | 11.3 |
| multi_turn_long   | 11.7 |  —   |  8.9 |
| long_context      | 11.6 |  7.3 |  7.0 |
| **avg / composite** | **11.7 / 1.00** | **9.1 / 0.91** | **10.5 / 0.96** |

Findings:
- **num_draft_tokens dominates the outcome.** The fork default (2) is uniformly
  bad (0.91). At 4, SHORT-context prompts turn POSITIVE (short +10%, medium
  +2%) -- concluding from the default alone would have been wrong. Added a
  `--num-draft-tokens` flag so the knob is sweepable.
- **The win is strongly context-length-dependent and NET-NEGATIVE overall.**
  Benefit shrinks monotonically with context and craters on long context
  (long_context -40% at both settings; multi_turn_long -24% at nd=4). As the KV
  cache grows, draft acceptance drops (a 1B model diverges from the 27B's greedy
  path on longer/harder content) and the batched multi-token verify over a large
  cache gets expensive. Overall composite stays <1.0 (0.96 best).
- **Greedy spec-decode is NOT bit-identical here.** Fingerprints diverge from
  baseline on every multi-token prompt (short is the only match) -- the target's
  BATCHED verify forward has a different float-accumulation order than sequential
  single-token decode, flipping borderline argmaxes. This is numerics, not a
  correctness bug, but it means the harness's `require_fingerprint_match` guard
  CANNOT certify a speculative run; spec-decode needs a distributional/quality
  gate instead.
- **Verdict:** classic 1B->27B draft speculative decoding does not beat plain
  decode on this bandwidth-bound bf16 target on M2 Ultra (11.7 gen_tps = 79% of
  the ~14.8 bandwidth ceiling -- little headroom for a draft to exploit). It can
  help short-generation/short-context calls at num_draft>=4. This matches the
  Direction thesis: the real decode win is verification-based decoding
  (DFlash/DSpark), not classic draft. Next experiments if pursued: a stronger
  draft (gemma-3-4b) to raise acceptance; a num_draft sweep isolated to
  short-context; measure acceptance rate directly (needs the fork to surface it).

## VLM vision baselines -- gemma-4 dense + MoE, mrope RESOLVED (2026-07-07)

First real-vision baselines (9 owner photos + synthetic renders). The path to
here surfaced three things, all now resolved:

- **The bench's VISION path had never run against a real VLM.** The Mar-16 "VLM
  baseline" (Qwen3.5-27B, 21.0 gen_tps) was a TEXT model through the loader's
  text path. Two dead config ids (Qwen3.5-27B-mxfp8, then a mrope wall on
  Qwen3-VL) were dead ends.
- **transformers-needs-torchvision** (torch-free venvs): fixed by porting the
  server's two soft-patches into bench_vlm.py (v1.34.13) -- still in place.
- **mrope RESOLVED by updating the forks.** The stale Mar-15 mlx-vlm fork
  (`#820`, 0.4.0) had the multimodal-RoPE bug (cos/sin broadcast mismatch on
  image-token expansion). Pulling the owner's synced forks -> **mlx-vlm 0.6.5
  (`#1529`), mlx-lm 0.31.3 (`#1431`)** fixed it: gemma-4 dense/MoE AND Qwen3-VL
  all run the manual pre-filled-cache vision path clean, no wrap_language_model
  port needed. `uv sync` was clean (only dropped soundfile; mlx stayed 0.32.0).
  gemma-3/gemma-4 use standard RoPE, so the fork bump alone unblocked them.

Baselines (8-bit, 14 prompts = 2 text + 3 synthetic + 9 real photos, runs=3):

| metric | dense 31B | MoE 26B-A4B |
|--------|----------:|------------:|
| gen_tps        | 15.3  | **48.1** |
| vision_ms      | 1592  | **524**  |
| prefill_tps    | 167.5 | **494.7**|
| ttft_ms        | 1428  | 483      |
| peak_mem_gb    | 33.3  | 27.3     |

**Profile contrast (the reason to bench both):** the MoE is ~3.1x faster decode
AND ~3x faster vision-encode/prefill than the dense 31B despite similar TOTAL
params -- because only ~4B are active per token. Dense is bandwidth-bound (reads
all weights every token, like gemma-3-27b); MoE shifts the bottleneck to expert
dispatch/gather + the (small, fast) active set. An mlx-lm/mlx-vlm optimization
will score very differently on the two -- bandwidth tricks help the dense, expert
dispatch/gather help the MoE. Per-model baselines (v1.34.14) keep both +
fingerprints without clobbering. Next: the MTP experiment (MoE +
`gemma-4-26B-A4B-it-assistant-bf16-mlx` drafter via mlx-vlm's `draft_kind="mtp"`)
-- the shot at the verification-based-decoding win.

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
