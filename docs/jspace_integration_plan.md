# J-space (Jacobian lens) integration — build + verifier plan

Last updated: 2026-07-10

> **Part 2 -- go-forward plan (2026-07-10), at the bottom of this doc,** supersedes the
> "Deferred / follow-ups" list: fit our own lens (new `jlens-mlx` sibling repo), the
> modular fitter + customizable corpus, tooling to adopt from the references, and the
> visualizer track. The repo boundary + `jspace_scratch` dissolution are decided there.

Status: **ALL PHASES DONE (easy + medium tiers).** Phase 0-1 (V1 gpt2 + V2 gemma-2-2b apply-parity
cos 1.0), Phase 4 router (V4 AUC 0.795/0.815), Phase 2 endpoint (`/v1/jspace/analyze`), Phase 3 v3
`jspace` page -- all built, tested, and verified end-to-end on the served gemma-4-26b-a4b 8-bit MoE
VLM (raw "...city of" -> Paris in the workspace). Deferred to future work: live per-token streaming
instrumentation, VLM vision residuals, our own lens fitting, calibrated live risk (needs per-model
traffic normalizer), generation-gate coordination. Scope was easy + medium tiers only. The Phase-1 spike harness + fixtures were relocated 2026-07-10 to the `jlens-mlx`
sibling repo (`migrated_from_scratch/`); see Part 2.

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
   `final_logit_softcapping`). Converter is implemented (relocated 2026-07-10 to the `jlens-mlx` repo's `migrated_from_scratch/make_oracle.py`).
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

## Reproduce (spike harness — relocated 2026-07-10 to the `jlens-mlx` repo, `migrated_from_scratch/`)

> The paths below said `coderef/jspace_scratch/`; that dir was dissolved into the
> `jlens-mlx` sibling repo's `migrated_from_scratch/` on 2026-07-10 (see Part 2). The
> file names are unchanged.

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

---

# Part 2 — go-forward plan (2026-07-10): fit our own lens + a real visualizer

Supersedes the "Deferred / follow-ups" list above. Context: a three-way study of our
apply-only impl vs two references — `WeZZard/jlens-qwen36` (MLX, fits on-device,
`qwen3_5`-arch only, code-quality verified: analytic Jacobians checked vs `mx.vjp`,
custom GDN Metal kernel checked vs the ops fallback, golden regression gate with
mutation-checking) and **Neuronpedia** (torch / nnsight+nnterp + transformer_lens,
model-agnostic, the interp OG, ships steer/swap/ablate) — plus the owner's two goals:
(a) **fit our own lenses** on Apple silicon, and (b) make the jspace feature (backend +
visualizer) materially better.

## Repository boundary (DECIDED 2026-07-10)

The fitting pipeline is heavy, dev-time, mlx-only research code that produces an
artifact the server consumes; it does NOT belong in this server (which stays a lean
scheduler — same reasoning as Q6's RLM extraction). Split by kind of artifact:

| Kind | Home | Notes |
|---|---|---|
| Fitting/research code (modular fitter, Metal kernels, corpus tooling, parity/fidelity harnesses, lens diffing) | **`jlens-mlx`** — NEW standalone sibling repo (like `rlm-heylook`, `batch-labeler`) | mlx-only deps; borrows from two Apache-2.0 upstreams |
| Lens weights (`lens.safetensors` ~0.5–3 GB + sidecars) | **A HuggingFace repo**, LFS-backed — **GATED on first own-fit** (caveat below) | ecosystem convention; server already downloads models from HF |
| Apply-only runtime (`src/heylook_llm/jspace/`, `jspace_api.py`, `jspace.js`, docs, golden gate + tiny fixtures) | **this repo** (stays) | consumes lens artifacts via `adapters/jspace/<model_id>/` (gitignored contents, like `modelzoo/`) |
| PyTorch reference | **`fblissjr/jacobian-lens` fork** — kept THIN (reference mirror + optional torch-side lenses), rebaseable on Anthropic | cited by `jlens-mlx`; the MLX work does NOT go here (mixing torch+MLX kills rebase-ability) |

**HF-publish caveat (why it's gated, not now):** we have not fit our own lens yet;
today's `adapters/jspace/*` lenses were CONVERTED from third-party pre-fits
(solarkyle/neuronpedia) and the gemma ones inherit the **Gemma Terms of Use** —
republishing them would be a licensing problem. The HF lens repo is for OUR fitted
lenses; stand it up after the first own-fit in `jlens-mlx`. Until then the server keeps
consuming locally-converted lenses via `adapters/` (never committed). Bonus once own-fits
are published as MLX-native safetensors: `scripts/jspace_convert_lens.py` collapses from
"torch+jlens throwaway venv converts a `.pt`" to a pure HF download — no torch on the
server side at all.

## `jspace_scratch` dissolution (DECIDED)

`coderef/jspace_scratch/` was the gitignored Phase-1 verifier spike — five kinds of thing
in one coat. **DONE 2026-07-10:** relocated into `jlens-mlx/migrated_from_scratch/` (code +
sidecars tracked; GB binaries preserved but gitignored) and the `coderef/jspace_scratch/`
dir removed. The sort mapping:

| Scratch file(s) | Destination |
|---|---|
| `make_oracle.py`, `convert_lens.py` (torch converter + oracle gen) | `jlens-mlx` (research converter; the user-facing installer is already `scripts/jspace_convert_lens.py` here) |
| `mlx_apply.py`, `mlx_apply_gemma.py` (V1/V2 parity harness) | `jlens-mlx`; its assertions become THIS repo's golden gate |
| `validate_moe.py`, `verify_router.py`, `verify_module.py` (research verification) | `jlens-mlx` |
| `verify_endpoint.py`, `probe_thread.py` (server-integration checks) | THIS repo `tests/` (real tests, not scratch) |
| `oracle_*.npz/json` fixtures | tiny gpt2 → `tests/golden/`; larger gemma → `jlens-mlx` or regenerate on demand |
| `lens_gpt2.safetensors` (tiny) | `tests/` fixture (golden gate). `lens_gemma22b.*` → HF (post own-fit) |
| `README.md` | salvaged into `jlens-mlx` README + this doc |

## The modular fitter (design — avoids a single-arch trainer)

**PIVOTED 2026-07-10** to Anthropic's `jacobian-lens` design after a 3-way cross-check
(Anthropic fits via `torch.autograd.grad`; `solarkyle/jspace` builds on the same Anthropic
reference; Neuronpedia is apply-only). Simpler than jlens-qwen36's approach AND it designs
away BOTH bug classes we hit porting it (the `rms²` seed bug and the chain-indexing off-by-one).

- **Baseline fitter (correct-by-construction, arch-agnostic).** For each source layer `l`,
  `J_l = d(h_final_block)/d(h_l)` via a **direct end-to-end `mx.vjp`** (one-hot cotangents at
  valid output positions, position-averaged) — porting Anthropic's `fitting.py` autograd loop
  to MLX. **No chain** of per-layer `M_l`, **no closed-form norm seed**: the final norm stays
  OUTSIDE `J`, applied as the real nonlinear module at decode (`unembed(final_norm(J_l·h))` —
  the apply path our server already uses, so fit-here/apply-there is consistent by construction).
  Works on any differentiable MLX model; the only error surface is autodiff's (i.e. none).
  Rademacher probing is an optional readout-grade speedup.
- **GDN speed accelerator (optional, per-arch, DEFERRED).** A direct VJP through qwen3_5's 48
  Gated-DeltaNet layers is slow (MLX's fused GDN kernel has no VJP). jlens-qwen36's custom Metal
  GDN backward + analytic assembly is a ~30-60× speedup — we PORT it (verified vs `mx.vjp`,
  attributed, **NOT vendored**) only when the baseline is too slow on the real 27B. Small-model
  baselines (gpt2 / gemma-2-2b) need none of it.
- **Coverage:** `qwen3_5` (our served `Qwen3.5-27B-abliterated` IS this arch — 64 layers,
  `full_attention_interval=4` → 48 GDN + 16 full-attn, `d_model` 5120) is the fit target; the
  baseline works on any arch; the GDN accelerator lands when the 27B baseline's speed demands it.
  `Qwen3_5ForConditionalGeneration` is a multimodal/MTP wrapper — reach the text stack via
  `.language_model.model` (same walk as `capture.py`).

## Customizable fitting corpus (design — corpus choice is load-bearing)

Fitting a lens is closest to **quantization calibration** (estimate a moment of the
activation distribution, deploy on a possibly-different one) with a **control-vector**
twist (a circuit the corpus never activates contributes ~0 to the averaged Jacobian → the
lens is structurally blind to it). WikiText is a bad default here; for our **abliterated**
model it is actively wrong — it contains ~0 refusal-triggering content, so the
refusal/safety circuitry is dormant across the whole corpus and the lens goes blind along
exactly the directions abliteration edited (the directions we most want to read). And the
lens is a chained product through depth, so early-layer lenses inherit every downstream
mismatch and need the most data. First-class, swappable:

- **corpus recipe as config + provenance** stamped on the lens artifact (recipe + model SHA + position policy). No lens without provenance.
- **chat-templated by default** (keep one raw-prose control arm).
- **position policy as a pluggable mask** — average over assistant / think-span tokens; explicitly drop BOS/sink/role tokens (high-norm Jacobian outliers). NOT a hardcoded "skip first 4" (that heuristic is calibrated for raw-text BOS sinks; wrong under ChatML).
- **on-policy corpus builder** — fit on the model's own sampled generations at generated-token positions, mixed ~50–70% with human-written diversity.
- **held-out fidelity gate** — per-layer KL / top-k agreement vs true logits on held-out target-distribution data; refuse to save a lens without it. Never grade a lens on its own fitting corpus.
- **lens diffing** as a first-class op — WikiText-fit vs chat-fit, stock-Qwen vs abliterated. For the abliteration case, the diff IS the finding.
- **VLM:** image tokens are projector outputs off the text-embedding manifold (larger norms, different attention topology). STRATIFY — a modality-conditioned image-position lens — rather than pool (averaging two separated clusters linearizes around a faithful-to-neither midpoint); validate fidelity per-modality; image positions get their own mask.

## Tooling to adopt from `jlens-qwen36` (the bench and the gate travel together)

Its test/perf discipline is more mature than ours; the value is the *process*, not the
fitting scripts:

1. **Standing golden gate for `jspace/analyze`** (highest-leverage) — `analyze()` is
   deterministic (greedy), so freeze `onset_strip` top-k ids + `features` into a golden
   JSON with a tie-aware **calibrated** epsilon (measure the worst tie-gap over N cells,
   pick ~4.6× headroom so matmul-batch-shape ulp noise doesn't false-positive), and
   **mutation-check** it (deliberately break the code, confirm the gate fails, revert).
   Turns the one-time V1/V2 parity into a wired regression gate. →
   `scripts/gen_jspace_gate_golden.py` + `tests/unit/test_jspace_gate.py` + `tests/golden/`.
2. **Perf ledger** (Target / Gate / Baseline / ranked-Backlog-with-REJECTED-hypotheses /
   History), per campaign (per model+workload), as the working memory for `fast-mlx`
   sessions. Antidote to the "perf claim rots within a day" failure MEMORY.md records.
3. **In-process, stage-attributed generation bench** (`scripts/benchmark.py` is HTTP-only,
   can't separate forward vs sampler vs detok) → `scripts/bench_generation.py`, stage
   timers routed through the existing `perf_collector.ChunkTelemetry`; must run on
   `_executor_pool` + gen gate or document the bypass.
4. **Branch-pinning `LensRegistry` test** (never-silent-fallback; assert the error names the
   bad input + lists what's available + offers the escape hatch; single-candidate
   auto-select is loud).
5. **Merge `workspace_range.py` with our hardcoded `band_layers`** (`features.py` hardcodes
   the 0.25–0.75 fraction) → a data-driven per-model band + a converted-lens smoke test
   (we have none).
6. **`sitecustomize.py` shim pattern** — a self-removing root shim for mlx-lm-vs-transformers-v5
   import breaks. Pocket it; adopt only when we actually hit one (we have the same collision
   surface: HEAD mlx-lm + `transformers` override).

## Frontend visualizer (GATED on DESIGN.md / impeccable)

The reference visualizers are far richer than our static strip. Do NOT clone
`jlens-qwen36`'s glass aesthetic — seed a v3 `DESIGN.md` first (plan Phase 4 item 2; v3
already has an implicit OKLCH strength/chip system to formalize). Then, cheapest-high-value
first (each mostly reuses data we already return, or a small per-cell top-k backend
extension):

1. **click-to-pin per-cell top-N readout** (jlens-qwen36's core interaction; directly answers "go layer by layer").
2. **layer slider / focus** to walk depth (Neuronpedia's `LayerRangeSlider` — a drag range that re-scopes the readout).
3. **live streaming** (per-token workspace rows) — needs a new streaming analyze endpoint.
4. **interventions (steer/swap/ablate)** UI — needs real backend (residual-stream hook at layer L + forward-from-layer re-gen); sequence AFTER streaming. Both references converge on the same math: transport a token's unembed direction by `J_lᵀ`, then add (steer) / project-out (ablate) / swap source→target.

## Attribution (both MLX/torch upstreams Apache-2.0; Neuronpedia MIT, ideas only)

`jlens-mlx` ships `NOTICE` + per-file provenance headers crediting
`anthropics/jacobian-lens` (via the fork) and `WeZZard/jlens-qwen36`; Neuronpedia credited
as design inspiration. `docs/jspace_guide.md` already credits the lineage — consolidate to
one acknowledgements pointer.

## Observations & watch-items (2026-07-10)

Folded in from the study + scaffold pass; captured so they aren't re-derived, not yet actioned.

- **DESIGN PIVOTED to Anthropic's fitter (2026-07-10, evidence-driven).** A 3-way cross-check
  (Anthropic `jacobian-lens` fits via `torch.autograd.grad`; `solarkyle/jspace` builds on the same
  reference; Neuronpedia is apply-only) settled the fit design: **direct end-to-end `mx.vjp`, norm
  outside `J`, no chain, no closed-form seed** — see "The modular fitter" above. jlens-qwen36 is
  now scoped to one optional thing: the GDN speed kernel. We do **not** vendor it.
- **CONCRETE BUG caught by porting.** jlens-qwen36's `analytic.py::rms_norm_jacobian` rank-1 term
  is over `rms**2` where the correct derivative is `rms**3` (contradicts its own docstring).
  Verified vs `mx.vjp`: rms^3 matches autodiff to ~5e-7, the vendored rms^2 diverges ~0.35. The
  Anthropic pivot avoids it entirely (no closed form). This is why we PORT + verify, never vendor +
  trust. Writeup + repro: `internal/research/upstream_jlens_qwen36_rmsnorm_seed_bug.md` /
  `jlens-mlx/scripts/check_rmsnorm_seed.py`.
- **`coderef/jspace` = solarkyle's replication — the ORIGIN of assets we already ship** (the gemma
  lenses `solarkyle/jspace-lenses` and the hallucination router in `features.py` / V4). torch/Modal,
  fits via the Anthropic reference (confirms the pivot). Its value is the downstream **analysis
  layer** (hallucination anatomy, lie detection, uncertainty/emotion probes, cross-model transfer) —
  a research backlog for what to do with our fitted lenses, plus a corpus-recipe reference.

- **`jlens-qwen36` is trustworthy — verified first-hand, with one caveat.** Its analytic
  Jacobians are checked vs `mx.vjp`, its custom GDN Metal kernel vs the ops fallback
  (< 1e-4), the full assembled layer vs the exact VJP on real activations, and it has a
  golden regression gate with mutation-checking + an honest perf ledger (rejects its own
  optimizations on real-weight measurement). Caveat: it's a 79-commit / 3-day solo sprint,
  and its validation covers **`qwen3_5` only**. The vendored seed's golden gate does NOT
  transfer — when we wire the generic-VJP path or add a new-arch accelerator, re-validate
  independently (parity vs `mx.vjp` on a tiny model).

- **The references are moving targets — don't freeze the seed.** `jlens-qwen36` last
  committed the day we studied it; Neuronpedia shipped Jacobian Lens ~2 weeks ago and is
  iterating fast (gpt-oss support, more models, headvis). Periodically re-pull: Neuronpedia
  for model-agnostic + intervention patterns, `jlens-qwen36` for kernel/perf updates.

- **The Qwen lens we serve TODAY is probably mismatched.**
  `adapters/jspace/Qwen3.5-27B-abliterated-8bit-mlx/lens.sidecar.json` is `n_prompts=672`,
  `hf_model_name=""` — unknown provenance, almost certainly fit on **stock** Qwen, not the
  abliterated (abliterated) weights we serve. So it reads the wrong function precisely where
  abliteration edited it. Concrete motivation for the first own-fit; until then, treat the
  served Qwen readouts as provisional.

- **Visualizer: decide aggregation-vs-matrix in the DESIGN.md pass.** Neuronpedia
  deliberately uses a sidebar **aggregation** (most-common readout tokens over a layer
  range), NOT a position×layer matrix, because a full matrix gets unwieldy on long
  transcripts; `jlens-qwen36` uses a virtualized matrix + spring-glide. For our vanilla-JS
  v3, `jlens-qwen36`'s grid is the more directly liftable (same tech); Neuronpedia is the
  better *design* reference. Pick the paradigm before building.

- **Don't build three benches.** The proposed in-process stage-attributed
  `bench_generation.py` (tooling item 3) overlaps optloop-lib's charter AND Phase 5 item 2's
  "thin HTTP serving-path bench" in `plan_2026-07.md`. Reconcile before building:
  `bench_generation` = in-process stage attribution (fast-mlx working memory); the HTTP
  bench = serving-path; optloop-lib = library-fork level. Three lanes, one each.

- **`verify_endpoint.py` / `probe_thread.py` belong back here.** They test the running
  heylook endpoint + MLX thread semantics, and are currently parked in
  `jlens-mlx/migrated_from_scratch/`. Re-home them as real server `tests/` (tracked in the
  `jlens-mlx` `MIGRATION.md`; mirrored here so the server side remembers).

- **Recommended porting order: thin vertical slice first.** Before the hard GDN Metal work,
  wire the **generic-VJP** path on a tiny model (the migrated gpt2/pythia oracles already
  cover it) so the chain driver + save + apply + held-out fidelity gate are proven
  end-to-end. De-risks the driver independently of the kernel.

- **Fidelity-gate semantics — don't miscalibrate it (2026-07-10).** Two lessons from grading the
  first own-fits on the served 8-bit 27B. (a) The identity/target-layer tripwire must grade on
  **KL≈0**, not top-1==1.0: on a QUANTIZED model, fp32-vs-native rounding swaps near-tied tokens so
  identity top-1 is <1.0 (~0.97) even when the apply path is exactly correct (identity KL~0.006,
  top-10~0.99). A top-1 threshold false-alarms; KL still catches a genuinely broken apply path (it
  goes large). (b) Agreement-with-FINAL-logits is a strict requirement ONLY at the identity layer;
  **earlier layers SHOULD diverge** — that divergence is the lens's signal (the "silent" tokens a
  layer is disposed toward before later layers revise). So the production gate should assert
  identity-exact + monotonic improvement toward the target, NOT an absolute top-k floor on early
  layers, or it penalizes exactly what makes the lens useful. Measured depth gradient on the fuller
  fit: top-1 0.44→0.16 as the source layer moves from 3 to 11 blocks back.

- **Quantization ⇒ its own lens (why our own-fit matters, restated concretely).** Only the fitted
  `J_l` matrices carry a fit-time-quant assumption; the final norm + head stay OUTSIDE `J` and are
  applied from the live served model, so they are always the correct quant (this is why the identity
  lens reproduces the served model's true logits). A lens fit on a different quant is approximately
  transferable (quantization preserves the function) but degraded, worse at deeper layers. We fit on
  the exact served 8-bit checkpoint via `mlx_lm.load` (the VJP runs through that dequantized forward),
  so we are matched — the same reason the inherited stock-fit lens is only provisional.

## Sequencing

1. **Scaffold `jlens-mlx`** (DONE 2026-07-10) — structure, attribution, `MIGRATION.md`, `DESIGN.md`.
2. **Apply path GREEN** (DONE 2026-07-10) — mirrored capture/lens; gpt2 V1 parity cos 1.0
   (`scripts/check_gpt2_parity.py`).
3. **Baseline fitter GREEN + cross-checked vs the reference** (DONE 2026-07-10) — direct
   end-to-end `mx.vjp` (Anthropic design; norm outside J; no chain, no closed-form seed). gpt2:
   J_target==I exact, apply-parity cos 1.0 (`scripts/fit_gpt2_baseline.py`); AND our MLX fit ==
   Anthropic's torch `jlens` on the same corpus — J cos **1.000000**, max_abs_err ~5e-6
   (`scripts/xcheck_fit_{torch,mlx}.py`). So the fitter is verified against the reference, not
   just autodiff-asserted. **The cross-check now spans TWO arch families:** gpt2 (LayerNorm) +
   **gemma-2-2b** (RMSNorm + logit-softcap-30, via a small gemma array-mask tail -- gemma's
   attention reads `mask.dtype`) -- J cos **1.000000**, max_abs_err ~5e-4 -- so RMSNorm+softcap
   generalization is verified. No vendored code; jlens-qwen36's rms² seed bug caught + avoided.
3b. **`qwen3_5` GDN tail GREEN** (DONE 2026-07-10) — the last arch. `jlens_mlx/providers/qwen3_5_gdn.py`:
   per-layer fa/ssm mask dispatch (mirrors `Qwen3_5TextModel.__call__`) + the jlens-qwen36 Metal GDN
   backward PORTED (not vendored) as an `mx.custom_function` VJP over the STOCK fused forward, swapped
   in via a reentrant context manager (both `gated_delta_update` refs) so the forward is byte-identical
   to mlx-lm. Re-verified vs `mx.vjp` at every grain (`scripts/check_qwen3_5_synthetic.py`): kernel
   dq/dk/dv/dg/dbeta rel err ~3e-7 / cos 1.000000 (incl. GQA rf=3, B=2, T=128 boundary), forward parity
   bit-exact, whole-fit J (kernel) vs pure-autodiff J cos 1.000000. Adaptations: atomic buffers as kernel
   OUTPUTS (`init_value=0`), gate grads always on. The custom kernel makes each GDN layer's backward one
   launch (~8x over the ops loop on a tiny model; bigger with depth) — but it does NOT change the outer
   cost: a direct VJP fits `J_l` with **one backward per output dim = d_model (5120) VJPs per source
   layer per prompt** (Anthropic's estimator).
3c. **Cotangent dim-batching DONE** (2026-07-10) — `providers/generic_vjp.py` now batches C output-dim
   rows through the tail's NATIVE batch axis (C independent copies of the primal, each hot at a different
   output dim; one `mx.vjp` per chunk). Avoids `mx.vmap` over the GDN `custom_function` (no vmap rule) —
   the batch axis is one the GDN kernel already handles. Verified batched J == chunk_size=1 J to fp32
   round-off (synthetic gate [4], rel 2e-7). Measured on the served 27B: **2.4× at chunk≥64** (33.3→13.8s
   for a 1-block tail; saturates by 64, memory 29GB@256). Deeper tails cost ~17s/added-block at chunk=128
   (J_52 tail=11 = 188s, peak 39.5GB). Same FLOPs as one-at-a-time — the win is GPU utilization. The
   analytic assembly (a further, FLOP-reducing speedup) remains unported.
4. **Corpus recipe** (chat + safety, NOT WikiText) + **own-fit** on the served abliterated Qwen3.5-27B
   (Metal-gated; stop the server for a full-depth run). `scripts/fit_served_qwen_bootstrap.py`: bootstrap
   (layers 61-62) + a graded late-band fit (layers 60/56/52, chunk 128, ~24min) both GREEN on real
   weights; the fit runs the fidelity gate + stamps per-layer scores + provenance on the sidecar.
   **TARGET THE PRODUCT BAND (layers 16-47).** The server only reads `features.band_layers` = the
   `[0.25,0.75)` slice; a fit outside it (the 52-62 late-band runs) serves NOTHING in the product. Band
   layers are the deep end (long tails) → the real production fit is the server-stopped big run, driven by
   `corpus.py::build_corpus` (**IMPLEMENTED** 2026-07-10: streaming HF load + weighted strata + chat-
   template + role-aware position masks; on-policy generation is a separated GPU step; `datasets` already
   in the venv) + `fit.fit_corpus` (averages J over the corpus using each item's mask). `build_corpus`
   needs the mlx-lm TokenizerWrapper (jinja template) — a raw AutoTokenizer from a model dir can miss it.
   Preview composition offline first with `scripts/build_corpus_preview.py`. The v3 visualizer is a fast
   before/after read once an own-fit is installed at `adapters/jspace/<model_id>/` (a user-driven swap of
   the provisional lens).
5. **Held-out fidelity gate** (DONE — `verify.py::fidelity_gate`: per-layer top-1/top-k/KL vs true logits
   on held-out prompts, identity-layer tripwire, save-refusal) + **lens diff** (DONE — `verify.py::diff`:
   two lenses on the same activations → per-layer top movers). The abliterated-vs-stock diff is the first
   real finding — still needs a STOCK Qwen3.5-27B fit to diff against (separate model, not local yet).
6. Back here: the standing **golden gate**; then the **visualizer** once `DESIGN.md` lands.
7. **HF lens repo** once there's an own-fit worth publishing.
