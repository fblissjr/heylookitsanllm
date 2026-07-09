# J-space (Jacobian lens) integration — build + verifier plan

Last updated: 2026-07-09

Status: **ALL PHASES DONE (easy + medium tiers).** Phase 0-1 (V1 gpt2 + V2 gemma-2-2b apply-parity
cos 1.0), Phase 4 router (V4 AUC 0.795/0.815), Phase 2 endpoint (`/v1/jspace/analyze`), Phase 3 v3
`jspace` page -- all built, tested, and verified end-to-end on the served gemma-4-26b-a4b 8-bit MoE
VLM (raw "...city of" -> Paris in the workspace). Deferred to future work: live per-token streaming
instrumentation, VLM vision residuals, our own lens fitting, calibrated live risk (needs per-model
traffic normalizer), generation-gate coordination. Scope was easy + medium tiers only. Reproducible spike harness +
fixtures live in the gitignored `coderef/jspace_scratch/` (see the repro section).

## What this is

Anthropic's July 2026 paper "Verbalizable Representations Form a Global Workspace in Language
Models" shipped an interpretability tool, the **Jacobian lens** (J-lens): a per-layer linear map
that reads which vocabulary tokens a residual-stream activation is *disposed toward* — the model's
silent "workspace" (j-space). We are wrapping the lens *apply* path into this MLX server as a
**post-hoc analysis feature** (not live-stream instrumentation), with an explore-page
visualization and a bonus hallucination-risk score.

This is an **inspection** feature: zero capability/perf gain. Roadmap-wise it is a Phase-5-ish
research item and must NOT displace v3 hardening. Its value is the repo's measure-first ethos —
a "what is the model silently tracking" view — not a speedup.

### Goal & non-goals

In scope: load pre-fit lenses; port the lens *apply* to MLX; capture residual stream via a
dedicated non-streaming forward; `POST /v1/jspace/analyze` (prompt+completion → layer×position
workspace); an explore-page heatmap; bonus router hallucination-risk score.

Deferred (hard tier, not this plan): live per-token instrumentation of the streaming hot path;
VLM/vision residual capture; fitting our own lenses (we download; no backward-pass harness).

## Provenance & assets (what exists, verified 2026-07-09)

- **Reference code**: `anthropics/jacobian-lens` (Apache-2.0, PyTorch + `transformers`). Package
  `jlens`. Cloned locally to the gitignored `coderef/jacobian-lens/` and read directly — the
  build is grounded in the real source, not blog summaries.
- **Pre-fit lenses on HF**:
  - `solarkyle/jspace-lenses` — fit on the **exact served models**: `gemma-4-26b-a4b-it` (the
    MoE), `gemma-4-12b-it`, `gemma-4-e4b-it`, `qwen3.6-27b`, `huihui-gemma-4-12b-it-abliterated`.
    Also ships a hallucination-risk **router** (see bonus) and eval traces incl. an NF4-4bit
    "lens survives quantization" set (`uncertainty_shapeq4_*`).
  - `neuronpedia/jacobian-lens` — larger zoo incl. tiny models (`gpt2-small`, `pythia-70m`) for
    cheap parity testing, plus gemma-2/3/4 and qwen families. A full mirror of this set is
    available in a local store outside the repo (no re-download needed for V2+).
- **Served-model quantization**: the MoE is served at **8-bit** MLX (`gemma-4-26b-a4b-it-8bit-mlx`),
  which is *closer* to the bf16 lens-fit basis than the NF4 case solarkyle already validated — so
  the quantization-transfer risk flagged in the first draft is largely retired.

## CONFIRMED priors (replaces the first draft's inferences)

### Lens file format (CONFIRMED — read a real `.pt` + the reference `save`/`load`)

A lens `.pt` is a plain dict (loaded with `weights_only=True`):

```
{ 'J': {layer_idx: Tensor[d_model, d_model]},   # dense, one per source layer
  'source_layers': [0, 1, ..., n_layers-2],     # final layer omitted (~identity)
  'd_model': int,
  'n_prompts': int }
```

Dense per-layer transport matrices — **NO low-rank, NO whitening/centering, NO bias, NO stored
unembed**. `JacobianLens.load` upcasts every `J` to **float32**. Memory ≈ a model-fraction of
unified RAM (gemma-4-12b lens = 1.39 GB = 3840² × 2B × 48L; the 8-bit MoE lens ≈ 0.46 GB).
`solarkyle`'s files may wrap this dict in a `JacobianLens` object (`.load`/`.merge`); the `J`
payload is identical.

### Reference apply semantics (CONFIRMED — read `jlens/lens.py`, `hf.py`, `hooks.py`)

- **Capture point**: `ActivationRecorder` registers a **forward hook** on each block and stores
  its **output** (`output[0]` for tuple-returning blocks). So `activations[l]` = the residual
  stream **after block `l`**. Proven by the reference's own invariant: `model_logits =
  unembed(activations[final_layer])` with no transport reproduces the model's real logits.
- **Transport**: `residual @ J[l].T` (i.e. `J[l] @ h` in column convention).
- **Unembed**: `softcap(lm_head(final_norm(x)))` — final norm THEN head THEN optional
  `final_logit_softcapping` via `cap * tanh(logits / cap)`.
- **Full apply**: `lens_logits_l = softcap(lm_head(final_norm(h_l @ J[l].T)))`.
- **Tokenization gotcha**: `from_hf` sets `tokenizer.add_bos_token = True`. Sidestepped entirely
  by reusing the oracle's returned `input_ids` on the MLX side — tokenization cannot diverge.
- **Layouts** the reference auto-detects: Llama/Qwen/Mistral/Gemma/OLMo/Phi (`model`, `norm`,
  `embed_tokens`, `lm_head`), GPT-2 (`transformer.h`, `ln_f`, `wte`), GPT-NeoX
  (`gpt_neox`, `final_layer_norm`, `embed_in`, `embed_out`).

### Architecture mapping (mlx-lm ↔ reference), the two spike models

| | gpt2-small (V1) | gemma-2-2b (V2) — proxy for served gemma-4 |
|---|---|---|
| mlx-lm module | `gpt2.py` | `gemma2.py` (served MoE = `gemma4.py`/`gemma4_text.py`, `model_type=gemma4`) |
| block output (capture) | `x + mlp(...)` | `h + post_feedforward_layernorm(mlp(...))` |
| final norm | `ln_f` (LayerNorm) | `model.norm` (RMSNorm, `1+weight`) |
| head | `wte.as_linear` (tied) | `embed_tokens.as_linear` (tied) |
| embed scaling | none | `× sqrt(hidden_size)` (inside forward) |
| final_logit_softcapping | none | **30.0** (served gemma-4 = 30.0, confirmed) |
| notes | — | alt local/global attn (sliding window) — irrelevant for short probes |

mlx-lm 0.31.3 ships `gpt2`, `gpt_neox`, `gemma2`, `gemma3(_text)`, `gemma4(_text)`, `gemma3n`.
The gemma-4 **e-series** ("nano") likely uses the gemma3n altup/PLE architecture — a poor residual
stand-in for the dense-residual MoE; hence V2 uses **gemma-2-2b** (same unembed shape: softcap
30 + RMSNorm + √d + tied head) rather than a nano model.

## The framework decision

Lens `.pt` + `jlens` are PyTorch; the server is MLX. So:

1. **Offline, once per lens** (torch, dev-time only): load `lens.pt`, extract the `J` dict, save
   per-layer `J[l]` as `mx`-loadable **safetensors** + a JSON sidecar (`source_layers`, `d_model`,
   `final_logit_softcapping`). Converter is implemented (`coderef/jspace_scratch/make_oracle.py`).
2. **In-server apply (MLX)**: `softcap(head(final_norm(h_l @ J[l].T)))` — route through the
   model's REAL final norm + head + softcap so gemma's caps and tied embeddings are correct by
   construction (also a correctness anchor).
3. **Residual capture**: a dedicated forward over prompt+completion recording `h_l` per source
   layer, decoupled from `run_generation`/the FIFO gate — post-hoc, no streaming-latency risk.
   Prototype capture = replicate the mlx-lm model forward loop (transparent) or monkeypatch block
   `__call__` (arch-agnostic; what the real `capture.py` should generalize to).

## Placement (backend + frontend)

- `src/heylook_llm/jspace/` — `lens.py` (JSpaceLens: load safetensors+sidecar, transport, apply)
  **[built]**, `capture.py` (ModelAdapter arch-introspection + `capture_residuals` via a temporary
  block wrapper) **[built]**, `features.py` (workspace_readout / router_feature_vector /
  HallucinationRouter) **[built]**. The offline
  torch→safetensors converter is dev-time only (not in the runtime; server loads safetensors via
  `mx.load`). Unit tests: `tests/unit/test_jspace.py` (download-free, tiny random-weight models).
- API: `api/jspace.py`, `APIRouter(tags=["JSpace"])`, tag added to `openapi_tags` +
  `app.include_router()` in `api.py` (repo convention).
- Lens assets: `huggingface_hub` download to a gitignored cache; NEVER committed (LFS, 100s
  MB–GB). A `model_id → lens repo/path` mapping. Gemma lenses inherit Gemma Terms of Use —
  cache-only, document provenance.
- Frontend: new view under the v3 `explore` page — read `js/page.js` `createPage` first; reuse
  `probabilityToColor()` + Token-Explorer patterns for the layer × token heatmap.

## Build phases

- **Phase 0 — recon + conversion (DONE).** Lens format confirmed; converter written.
- **Phase 1 — MLX apply + residual capture (DONE, V1 GREEN on gpt2-small).** cos 1.00000 on
  residuals, lens logits, and model logits vs genuine `jlens.apply()`, top-5 overlap 5/5, all 11
  source layers, both probes. Semantic preview: multihop workspace L10 = `[yen, ¥, Yen, Japanese,
  Osaka]` (currency concepts surface late; gpt2-small leans Japan-prior over boot→Italy, expected
  at its size — the *mechanism* is right).
- **Phase 2 — API endpoint (medium, DONE).** `GET /v1/jspace/models` + `POST /v1/jspace/analyze`
  (`jspace_api.py`, tag `JSpace`, registered in `api.py`). `jspace/analyze.py` reuses the provider's
  exact prompt formatting (chat template + `<bos>`), greedy-generates the answer, captures the
  residual stream, returns onset top-k strip + optional layer×position heatmap + features + risk.
  `jspace/registry.py` = `HEYLOOK_JSPACE_DIR/<model_id>/` lens cache (offline-converted safetensors;
  optional `normalizer.json`/`router.json` for risk). Registry unit + endpoint contract tests.
  Compute runs in a threadpool (blocking/Metal-bound); it does NOT yet coordinate with the FIFO
  generation gate — low-frequency analysis endpoint, but concurrent use with generation is a known
  gap.
- **Phase 3 — frontend view (medium, DONE).** New v3 page `apps/heylook-frontend-v3/js/pages/jspace.js`
  (dedicated `jspace` nav route, not surgery on explore.js): model picker (lens-gated via
  `/v1/jspace/models`), prompt, `raw`/`chat` + heatmap toggles, calls `/v1/jspace/analyze`, renders
  the layer×top-k "silent words" strip (colored by within-layer rank), an optional layer×position
  heatmap (colored by confidence), and a risk badge. Reuses the explore-chip OKLCH formula.
  **Verified end-to-end on the served 26B MoE**: raw completion "...city of" -> answer Paris, onset
  workspace surfaces [Amsterdam, Paris, paris, PARIS, Kolkata]. KEY: read at a CONTENT token (raw
  completion, default) not the chat generation-prompt boundary (formatting junk) -- `chat=true`
  reserved for the risk features.
- **Phase 4 — hallucination router (bonus, `features.py` [built]).** 10 workspace features
  (entropy stats, ignition frac/depth, log-rank, band agreement, hedge rank) + 4 output-confidence
  baselines; z-score per model (`FeatureNormalizer`); `sigmoid(w·z + b)` (`HallucinationRouter`).
  Weights shipped by `solarkyle` (`router/workspace_router_e4b.json`). **V4 PASS**: e4b TriviaQA
  trace AUC 0.795 workspace-only / 0.815 combined, both > the 0.771 first-token-logprob baseline.

## Verifier ladder (with actual results)

- **V0 — Reference oracle (DONE).** Genuine `jlens.apply()` (from the local `coderef/jacobian-lens`
  clone) on a fixed prompt set → per-(layer,pos) lens logits + top-k + **residuals** committed as
  fixtures. Dumping residuals lets V1 check capture-correctness separately from the transport/
  unembed math (isolates the MoE capture-point risk).
- **V1 — Numerical parity (DONE, PASS).** MLX apply vs V0: cos > 0.99, top-5 overlap ≥ 4/5.
  gpt2-small fp32↔fp32 → cos **1.00000**, overlap **5/5**. `resid_cos = 1.0` proves mlx-lm gpt2 ==
  HF gpt2 numerically (capture point correct).
- **V2 — gemma parity (DONE, PASS).** gemma-2-2b fp32↔fp32, validates the gemma unembed path
  shared with the served gemma-4: **softcap 30 + RMSNorm final norm + √d embed scaling + tied
  head**. cos **1.00000**, top-5 **5/5**, all 25 source layers. Semantic sanity already strong:
  eiffel workspace L24 = `[Paris, paris, Paris, France, París]`. (Does NOT cover MoE capture or
  8-bit transfer — those are later, on the real MoE.)
- **V3 — Semantic sanity.** Canonical probes surface expected workspace tokens mid-to-late layers.
- **V4 — Router replication (DONE, PASS).** Reproduced on solarkyle's e4b TriviaQA trace with the
  shipped weights: AUC 0.795 workspace-only / 0.815 combined > 0.771 logprob baseline. Validates
  the router + entropy-trajectory features. **V4b (deferred):** validate `workspace_readout`'s
  rank/ignition/hedge scalars from OUR lens logits (needs running the module on e4b) — folds into
  the served-gemma-4 integration.
- **V5 — E2E + regression.** Endpoint latency + unified-memory headroom on the Mac; assert the
  normal generation path is byte-identical with the feature off. Optional Metal-gated browser E2E.

## Risks / open unknowns (updated)

- **Quantization transfer — downgraded.** Served MoE is 8-bit (not 4-bit); solarkyle already shows
  NF4 survival. Still worth an explicit measurement on the real MoE (V2-on-MoE), but no longer a
  headline risk. gpt2/gemma-2-2b spikes are fp32↔fp32 (isolate port correctness first).
- **MoE residual capture point (VALIDATED 2026-07-09).** Ran the module on the served
  `gemma-4-26b-a4b-it-8bit-mlx` (mlx-vlm VLM): the late-band workspace surfaces the correct entity
  ("Eiffel Tower ... city of" -> Paris/cities), confirming block-output capture through the
  128-expert routing + 8-bit-lens transfer. mlx-vlm's own forward comments that the recorded hidden
  matches HF's `Gemma4TextDecoderLayer` output (the lens's fit convention). Two harness fixes were
  needed: `_Recorder` must proxy `layer.layer_type` (mask construction), and **gemma requires an
  explicit `<bos>`** or the residual stream degrades to multilingual-token garbage. Instruct models
  on RAW completion prompts are noisy (base gemma-2-2b was clean) -> the endpoint must format
  prompts properly. Remaining: full V4b (workspace-readout AUC from OUR logits, vs the trace's
  stored scalars).
- **gemma-4 e-series architecture.** Nano (altup/PLE) is not a clean residual stream; avoid as a
  proxy. Use gemma-2-2b (V2) then the real MoE.
- **Unified-memory pressure.** A 0.46–1.4 GB lens competes with the KV cache under
  `max_loaded_models=1`. Mitigation: lazy-load on first analyze, evict with the model, start with
  the 0.46 GB MoE lens.
- **Final-norm / softcap fidelity.** Mitigated by routing through the model's real head; V1/V2 are
  the guards. (gpt2 has no softcap, so V2 is the first real softcap test.)

## Sequencing recommendation

Phase 1 proved the core linear-algebra port on a tiny model. The remaining risk is entirely in the
gemma-specific unembed (V2, in progress) and later the MoE capture point. Finish V2, then build the
real `src/heylook_llm/jspace/` module + endpoint against a proven-safe model, then scale to the
served gemma-4 MoE with the same parity gates before wiring the explore view.

## Reproduce (spike harness — gitignored `coderef/jspace_scratch/`)

- `make_oracle.py` — throwaway torch venv (torch + transformers + editable `jlens` from
  `coderef/jacobian-lens`); env-parameterized (`HF_MODEL`/`LENS_PT`/`PREFIX`/`JSPACE_OUT`).
  Produces genuine-`jlens.apply()` fixtures and converts the lens → safetensors + sidecar.
- `mlx_apply.py` — project venv; mlx-lm gpt2 forced fp32; replicates the forward capturing block
  outputs; applies `wte.as_linear(ln_f(h @ Jᵀ))`; prints the V1 gate.
- A per-model gemma variant follows the same shape with RMSNorm + √d + softcap-30 unembed.

## Installing a lens (convert + register)

Git-tracked helper (torch + jlens, run in a SEPARATE env — not the MLX server venv):

    uv run --with torch --with safetensors --with huggingface_hub \
        --with "jlens @ git+https://github.com/anthropics/jacobian-lens" \
        python scripts/jspace_convert_lens.py \
        --hf-repo solarkyle/jspace-lenses --hf-file gemma-4-26b-a4b-it/lens.pt \
        --model-id gemma-4-26b-a4b-it-8bit-mlx --softcap 30

Writes `adapters/jspace/<model_id>/lens.safetensors` + `lens.sidecar.json`. `adapters/` is
git-tracked (`.gitkeep`) with gitignored contents (like `modelzoo/`); the registry
(`LensRegistry.from_env`) defaults there, so the model then appears in `GET /v1/jspace/models`
with zero config (override via `HEYLOOK_JSPACE_DIR`). For risk scores, also drop a
`normalizer.json` (per-model feature mean/std) and `router.json` (solarkyle spec) in that dir.

## Deferred / follow-ups (not in the easy+medium scope)

- **Calibrated live risk.** A single request can't be z-scored; risk is `null` until a per-model
  `normalizer.json` is placed. Future: a running-stats normalizer that accrues over analyze calls
  so risk emerges after N samples, or a one-shot calibration pass. (V4 proved the router math;
  full V4b = workspace-readout AUC from OUR lens logits on e4b, not just the trace scalars.)
- **Generation-gate coordination (DONE, v1.34.33).** Analyze now pins the model and runs its
  forwards under the process-global FIFO generation gate, so it serializes with generation and
  other analyze calls — no concurrent Metal graphs, no racing block-list mutation. (Backpressure
  via `check_capacity` / a client cancel path is still a possible refinement.)
- **Live per-token streaming instrumentation** (hard tier) — read the workspace during generation,
  not just post-hoc.
- **VLM vision residuals** — jspace is text-only; the image path is untouched.
- **Fitting our own lenses** — we download + convert; no backward-pass fitting harness in-repo.
- **DONE this pass:** the convert+register helper (`scripts/jspace_convert_lens.py` + `adapters/`)
  and an E2E page check (`tests/e2e/suites/pages.mjs`, lens-gated) — both to make the next
  iteration cheap (fewer 26B reloads).
