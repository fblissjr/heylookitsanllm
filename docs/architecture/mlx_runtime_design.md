# heylook MLX runtime (proposed design)

Last updated: 2026-09-30

Status: proposed, not built; the open decisions in section 10 are the owner's.
Grounded in VISION.md principle 6 as amended 2026-09-30 (commit 15516d28): heylook owns the runtime around
mlx-vlm's models where it serves the owner's use; model code and kernels stay upstream.

## 1. The idea in one paragraph

Today every MLX request builds a throwaway `BatchGenerator` inside `vlm_engine.generate`
and runs alone behind a process-wide gate. The runtime replaces that with one
long-lived engine per loaded model that heylook schedules itself, plus a small set of
policies (admission, prefill chunking, drafting, caching, sampling) and a per-model
profile of measured settings. mlx-vlm stays the library that runs the model. The
optimizations live in the policies and profiles, which apply to every model; the
model-specific work is limited to measured profile values and upstream PRs.

## 2. Boundary: what we use, what we own

Used from mlx-vlm as a pinned library (surface watched by
`tests/contract/test_mlxvlm_surface.py` at every pin move):
- model loading, processors, `get_input_embeddings`, the model forward;
- `make_cache` and the cache classes (plain, batch, rotating, quantized);
- the prefix-cache store and coordinator (`APCManager`, `APCCoordinator`);
- drafters and the speculative round loops (mtp, dflash, eagle3). This is the
  fastest-moving upstream area, so we reuse it and own only the policy around it;
- `sample_utils`, `convert` and quantization.

Owned by heylook:
- the engine loop and scheduler (section 3);
- the policies (section 4);
- model profiles (section 5);
- carried patches (`mlx_vlm_patches.py`, existing pattern, with retirement tests);
- instrumentation (section 6).

Never owned: model definitions, kernels, a copy of any mlx-vlm module.

## 3. The engine

One engine per loaded model, one worker thread that alone touches MLX (the existing
pinned-worker rule).

```
requests ──> admission queue ──> engine.step() loop
                                   ├─ decode one step for all active rows
                                   ├─ prefill one chunk of the next admitted prompt
                                   ├─ emit tokens per row (stream to its request)
                                   └─ apply cancels (remove row), finishes, harvest
```

- Rows carry their own sampler, logits processors, thinking budget, stop set,
  max_tokens and telemetry.
- Build on, don't clone: stage 1 drives mlx-vlm's `BatchGenerator` as the inner step
  machine, long-lived instead of per request. It already interleaves one decode step
  with one prefill chunk per `next()` (`generate/ar.py` ~2990-3040) and supports insert
  and remove. We replace only the pieces that block us, starting with the per-row
  sampler (mlx-lm's design, `mlx_lm/generate.py` ~1389; offered upstream, carried until
  merged).
- `vlm_engine.generate` keeps its signature and becomes "submit a row and stream it", so
  `mlx_provider.py` and everything above it do not change.
- The process gate becomes per-model capacity (`parallel`, default 1 = today's
  behaviour). gguf gets the same admission object in front of llama-server's own slots
  (`-np N --kv-unified`), which is how concurrency stays one design across engines.

## 4. Policies

Plain functions over plain state, unit-testable without a model. Each one is where a
class of optimization lives, for every model at once.

| Policy | Decides | First version | Later |
|---|---|---|---|
| Admission | who runs, in what order | FIFO up to `parallel` | interactive before batch-job rows (a request class on the wire) |
| Prefill chunk | chunk size per step | `prefill_step_size` from the profile | shrink chunks while other rows decode, so a new prompt does not stall a running chat |
| Drafting | draft or not, per row | draft only when the row is alone; a newcomer waits (drafted batches cannot take rows) | finish the round, drop to batched when a second row arrives; resume drafting when alone |
| Cache | checkpoints, retention, eviction | today's `capture_lengths` + `refresh_snapshots`, per row | keep the reply's cache when the template reproduces it; evict by conversation, pin system prompts |
| Sampling | per-row token choice | per-row samplers, one batched call when shared | per-row random keys (reproducible under concurrency) |

## 5. Model profiles

Per-model measured settings in the model's `model.heylook.toml` (existing sidecar
mechanism), each carrying where it came from, so the engine report can say why a value
is in force (VISION principle 1):

- weights variant (bf16 / 8-bit affine / mixed / 4-bit) chosen by the precision ladder;
- drafter and block size, with the measured acceptance record;
- `prefill_step_size`, checkpoint interval, `parallel`, KV quantization (default off);
- `measured: <date> <record path>` for every value that came from a measurement.

A value with no record is a default, and the report says so.

## 6. Instrumentation

- Per step: wall time, time blocked in `mx.eval` (GPU) vs host time, rows active,
  prefill tokens this step. Host syncs show up here (v2.0.92's identity-processor sync
  was exactly this class).
- Per request: prompt, reused and prefilled tokens, drafting rounds and accepted drafts
  (or why it did not draft), concurrency level while it ran. Throughput numbers from
  contended and solo runs never mix.
- On demand: Metal capture of a few steps for kernel-level questions.

## 7. Onboarding a model (how it goes beyond one model)

Every model in daily use goes through the same pipeline; Glimmer is the first.

0. Coverage: check upstream's `model_cases.json` has the model in `cases` (forward pass)
   and `apc`. If not, adding it is the first upstream contribution: our own changes (the
   #2356 rework's output-neutral test, for one) can only cover models upstream can build.
1. Correctness baseline at full precision: `vlm_parity_probe` (vision), `chain_probe`
   (cache), eval bank tier.
2. Precision ladder: bf16, 8-bit affine, mixed_4_8, 4-bit. Speed by `perf_ab`, quality
   by eval-bank delta against bf16. Pick the smallest that holds quality.
3. Drafter: every compatible drafter, acceptance on the owner's own prompts, pick one.
4. Runtime knobs: prefill chunk, checkpoint interval, `parallel`.
5. Step profile: find model-specific hot spots, PR upstream, carry until merged.
6. Record the profile with provenance.

At each pin move, rerun a short `perf_ab` against every daily model's recorded profile;
a regression is a finding, not noise.

### Pilot: Muse Glimmer 30B (code audit 2026-09-30, read-only, pin fdcdaf46)

Checkpoint: bf16 throughout; parameter counts per component are derivable from the
safetensors headers. From its `config.json`: dense (no MoE), GQA with 2 KV heads, and a
layer pattern of three sliding-window layers to one full-attention layer
(`layer_types`, `sliding_window`).

What the audit found (`mlx_vlm/models/muse_glimmer/`, citations in the audit):
- Nothing structural. Fused SDPA, no decode mask on plain caches, compiled norm and
  residual chains, no host syncs, one targeted perf pass upstream (f4a5a67f).
- About twice the kernels per layer of a llama-style model, mostly parity-driven: eager
  RoPE (the only model in mlx-vlm that uses it), float32 centered norms, an uncompiled
  qk-scale round trip, unfused q/gate/k/v and gate/up projections.
- The floor is weights: the MLP (3 x hidden x intermediate) dominates each layer's
  weights, derivable from `config.json`. So the
  order is precision ladder, then the drafter (the publisher's DFlash assistant, which
  mlx-vlm supports), then kernel trimming.
- Prefix cache: rotating layers make it checkpoint mode, so the #2356 rework applies.
- KV quantization: not worth it here (the KV is small); uniform `kv_bits` cannot
  quantize rotating caches anyway.
- Coverage gap: no `cases` or `apc` entry upstream, only speculative tests. Step 0
  applies.

Kernel-trimming candidates (step 5; effect sizes are the audit's guesses, unmeasured):
1. Fold `qk_scale_factor` into the SDPA scale (SDPA already scales in float32).
2. Fuse q/gate/k/v and gate/up into single quantized matmuls at load (same bytes, fewer
   launches; the 6656->256 k/v matmuls underfill the GPU).
3. One-kernel centered norm with a float32 weight, keeping parity.
4. Swapping eager RoPE for `mx.fast.rope` changes numerics against the reference; that
   is an owner call on parity versus speed, not a free win.
Each is an upstream PR if it holds up under `perf_ab`, not heylook model code.

## 8. Stages and gates

| Stage | Delivers | Gate |
|---|---|---|
| 0 (now, no code) | #2356 rework; 8-bit vs bf16 on the qwen3.8 27b fine-tune; DFlash2 vs MTP; Glimmer onboarding steps 1-3 | measurements recorded |
| 1 | long-lived engine per model, per-row samplers, per-model capacity, per-step instrumentation | chain_probe clean on each model class; single-row speed equal to today (perf_ab) |
| 2 | gguf slots under the same admission object | two concurrent chats + a batch job, both engines |
| 3 | drafting policy v2 (drop to batched, resume) | drafted and batched rows both correct under chain_probe |
| 4 | cache policy v2 (reply retention, eviction by conversation) | measured reuse gain on real conversations |
| 5 | profile-driven hot-path work per daily model | per-fix A/B, upstream PR filed |

## 9. Risks

- `BatchGenerator` internals churn upstream. Mitigation: depend on insert/remove/next and
  the cache classes only; contract tests at every pin move.
- Concurrency changes timing and numerics (a row in a batch is not bitwise a row alone).
  Mitigation: per-request concurrency recorded; correctness checks compare like with like.
- Complexity. Mitigation: policies are small pure functions; anything that grows toward
  model code is an upstream PR instead.

## 10. Open decisions for the owner

1. Drafting under concurrency: exclusive-with-queue first, or per-model "drafting vs
   parallel", until stage 3?
2. Request classes: should a batch job be marked on the wire so interactive chats go first?
3. Per-row sampler: offer upstream now, or carry first and offer once it has run here?
4. Where the design lives once agreed: `docs/architecture/` as the design record.
