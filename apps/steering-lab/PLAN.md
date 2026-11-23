# MLX Steering & Mechanistic Lab (`apps/steering-lab`)

Date: 2026-11-23  
Status: Active standalone lab (MLX-only, targeting `Swift-1.5-Qwen3.8-27B-MLX-8bit` first; zero coupling to `heylook` daily-driver chat routes)

## 1. Purpose & Design Philosophy

`steering-lab` is a **white-box activation steering and mechanistic exploration workbench** for Apple Silicon MLX models. It exists to explore *non-1D* steering — subspace erasure, multi-vector composition, phase-split thinking/content control, decoupled `IF X -> THEN Y` conditional neurons, and visual-vs-text token masking — without bloating `heylookitsanllm`'s daily-driver chat server, GGUF subprocesses, presets, or DuckDB conversation store.

If any operator or vector pack eventually earns a place in daily chat, its `.safetensors` + `meta.json` sidecar plugs cleanly into `heylook`'s reserved `engine.steering` slot (Plan W14) without rewriting the server contract.

### Why MLX-Only & Why `Qwen 3.8 27B` First

1. **MLX gives full white-box control at zero scale-0 cost:**
   - No custom C++ capture binary, no per-quantization LoRA weight baking, and zero compute cost when scales are `0`.
   - Works identically across dense (`Qwen 3.8 27B`, `Gemma 4 31B`) and MoE (`Gemma 4 26B-A4B`) architectures.
   - Keeps MTP speculative decoding mathematically exact under lossless Leviathan rejection sampling, and turns MTP acceptance itself into a live **off-manifold detector**.
2. **`Swift-1.5-Qwen3.8-27B-MLX-8bit` combines four rich subsystems in one model:**
   - **`3:1` Gated DeltaNet (`48`) + Full-Attention (`16`) hybrid (`L00..L63`):** Probes where concepts actually live across layer depths and layer types (GDN recurrent state vs. FA layers vs. terminal `L63`).
   - **Native `<think>` + `content` phases:** Splits inner reasoning from visible answers (e.g., cold rigorous `<think>` with stylized `content`, or vice versa).
   - **Native Vision (`<|image_pad|>` tokens):** Enables VLM Dual-Lock experiments — masking interventions strictly to visual patch tokens, prompt text tokens, `<think>` tokens, or visible `content` tokens.
   - **Fast 8-bit + MTP execution:** Practical for live multi-arm sweeps and interactive token-level microscopy on 192 GB unified RAM.

---

## 2. The Four Non-1D Steering Operators

All operators share one pure-math engine (`src/steering_lab/operators.py`) and one sidecar pack format (`vector.safetensors` + `meta.json`):

### 2.1. Regime A — Multi-Slider Manifold Replacement (`FA` Band `L39, L43, L47, L51`, `~60–80%` Depth)
For concepts with a coherent target pole (register, persona, analytical vs. analogic prose, visual critique lens):

- **Single-vector action (local-centered Pole-A erasure + target push):**
  $$\Delta h = -s_{\text{erase}}\,\big((h - \bar{h})\,\mathbf{V}_r^\top \mathbf{D}\,\mathbf{V}_r\big) + s\,\hat{\boldsymbol{\mu}}_d$$
- **Multi-slider composition (`compose_regime_a`, ported from H3 `control_vectors.py`):**
  1. **`peer_ros` (peer target orthogonalization):** Orthogonalize each active slider's Pole-A erasure basis $\mathbf{V}_{r,i}$ against the target poles $\{\hat{\boldsymbol{\mu}}_{d,j}\}_{j \neq i}$ of every *other* active slider so stacking 2–4 concepts never cancels peer targets.
  2. **Clamped `[0, 1]` union SVD:** Form the combined positive semi-definite erasure operator $\mathbf{E} = \sum_i w_i\,\mathbf{V}_{r,i}^\top \mathbf{D}_i\,\mathbf{V}_{r,i}$, eigendecompose / SVD it, and clamp eigenvalues into `[0, 1]` so overlapping concepts never overshoot (`> 1.0` sign flip).
  3. **Capability `holds` projection:** Optionally project both the erasure operator and net target push into the orthogonal complement of a clean capability hold basis $\mathbf{H}_{\text{hold}}$ (captured from math/code/JSON/tool-call spans) so style sliders cannot damage structured reasoning.
  4. **Smooth `tanh` RMS knee:** Bound total intervention norm smoothly (`knee * tanh(||Δh|| / knee)`) so extreme multi-slider sums never collapse off-manifold.
- **Depth & layer-type rule (Qwen 3.8 `3:1` hybrid map):** On Qwen 3.8 27B (`L00..L63`), the `16` Full-Attention (`FA`) layers sit strictly at **`idx % 4 == 3`** (`3, 7, ..., 39, 43, 47, 51, 55, 59, 63`), whereas `L48, L49, L50` are **`GDN` recurrent state layers** (`layer.is_linear == True`). `runtime.py` derives `FA` vs. `GDN` dynamically from `lm.model.layers`, defaults multi-slider Regime-A concepts to the `~60–80%` FA band **`[39, 43, 47, 51]`** before `L52..L63`'s autoregressive funnel crushes concept rank, and explicitly tags `L49`/`L62` as `GDN` contrast probes.

### 2.2. Regime B — Active Noise Cancellation (`L63` or Mid-Layer)
For pure failure-mode suppression where `ADDS = noise` and there is no single Pole-B target (AI-slop, corporate hedges, sycophancy, anxious filler):

$$g_t = \operatorname{clip}\!\left(\frac{\cos(h_t - \bar{h}_t,\; -\hat{\boldsymbol{\mu}}_d) - m_0}{w},\; 0,\; 1\right), \qquad \Delta h_t = -\operatorname{clip}(s, 0, 1)\cdot g_t \cdot e_{\text{cap}} \cdot \big((h_t - \bar{h}_t)\,\mathbf{V}_{\text{diff}}^\top \mathbf{D}\,\mathbf{V}_{\text{diff}}\big)$$

- **Local centering with `mu_clean` warmup prior (`prior_ema` / `mu_clean` / `running`):** Pure causal running mean collapses to `h_1 - h_bar_1 = 0` on token 1, is noisy on tokens `2..10`, and mixes `<think>` with `content` unless reset per phase. `RunningCenter` defaults to `prior_ema` (seeding `(sum_h, count)` with `prior_weight * mu_clean` and resetting cleanly on `<think> -> content`), with `mu_clean` (fixed) and `running` (raw) kept as live lab toggles, all rollback-safe under speculative rejection.
- **Normalized half-wave deadzone ramp (`m0 ~ 0.08`, `w ~ 0.22`):** Matches the `+0.21..+0.39` real cosine band in ~5k-D residual spaces; stays at exact `0.0` on clean technical tokens.
- **Multi-rank paired-shift basis (`vr_diff`, `r = 8`, `boost = 0`, upper cap only):** Excludes neutral co-occurring words from $\mathbf{V}_r$ and avoids rank-1 loop instability.

### 2.3. Conditional `IF X -> THEN Y` Neurons (`ReLU` Read/Write Decoupling)
Decouples the trigger axis from the edit axis:

$$g_t = \operatorname{clip}\!\left(\frac{\cos(h_t - \bar{h}_t,\; \hat{\mathbf{w}}_{\text{read}}) - m_0}{w},\; 0,\; 1\right), \qquad \Delta h_t = s \cdot g_t \cdot \hat{\mathbf{w}}_{\text{write}}$$

Uses one-sided clipped ReLU (never `silu(z)*z`, which has a spurious positive bump for negative `z`). Enables context-conditional behaviors such as *"IF prose hedging -> shift toward crisp physical nouns"* or *"IF describing an image -> shift toward optical/lighting detail"* while sitting at exact `0.0` outside the trigger.

### 2.4. `S3` Random-Direction Control
Generates a matched-norm, matched-rank orthogonalized random basis $\mathbf{V}_{\text{rand}}$ and random unit pole $\hat{\boldsymbol{\mu}}_{\text{rand}}$ (seeded deterministically) orthogonal to the real direction — one-click double-dissociation control proving observed shifts come from the learned subspace rather than perturbation noise.

---

## 3. The Six Lab Instruments

1. **Cached Multi-Layer Span Bank (`bank.py`, `runtime.py`):**
   - Teacher-forced capture across a configurable layer ladder (default Qwen 3.8 probe set: `[15, 27, 39, 43, 47, 49, 51, 55, 62, 63]`, dynamically tagging each layer `FA` (`idx % 4 == 3`) vs. `GDN` (`is_linear == True`)).
   - Tags and saves raw activations separately for **`image_pad`**, **`prompt_text`**, **`think`**, and **`answer`** spans under `apps/steering-lab/data/banks/<bank-id>/` (gitignored).
   - Once captured, rebuilding vectors with `mean_diff`, `SVD(A_c)`, paired `vr_diff`, `peer_ros`, `holds`, or `IF->THEN` takes milliseconds on CPU without reloading model weights.
2. **Subspace Geometry & Tangent-RMSNorm `lm_head` X-Ray Bench (`bank.py`, `operators.py`):**
   - Cross-concept pairwise cosine matrix, bank effective rank (`exp(entropy)` of singular values), top-5 PC energy, and `p90 |cos|` at every captured layer (directly visualizing `L51 -> L63` rank collapse with `FA`/`GDN` layer badges).
   - **Tangent-RMSNorm `lm_head` Adds / Erases:** Applies Qwen 3.8's zero-centered final RMSNorm (`1 + weight`) as an exact directional tangent `(norm(h_ref + eps*v) - norm(h_ref)) / eps` before `lm_head` so `L63` top-k words are not scrambled, alongside `split_half_cosine` and `flip_null`. Mid-layer readouts (`< L63`) are explicitly labeled **Advisory Logit Lens** because direct projection skips downstream GDN/FA rotation.
3. **Non-1D Operator Playground (`ui/` + `server.py`):**
   - Multi-slider mixer with live toggles for `peer_ros`, clamped union SVD, capability `holds`, `tanh` knee, `r=1` vs. `r=8`, `S3` random control, centering mode (`prior_ema` / `mu_clean` / `running`), layer subset (`FA`-only `idx % 4 == 3` / `GDN`-only / `< L63` / `L63`), and token mask (`image_pad` / `prompt_text` / `think` / `content`).
4. **Token-Level Microscope (Per-Token Telemetry Heatmap, `KL`, 4-Gram Loop Meter & Click-a-Token Causal `Δ logprob`):**
   - Records per-token `{token_id, text, phase, g_t, ray_cos, rel_delta, entropy, kl_vs_base, mtp_accepted}` on every decode step and computes exact `KL(p_steered || p_s0)` + rolling 4-gram repetition alongside entropy and MTP share.
   - MTP acceptance alone misses degenerate loops (which are high-probability and keep high MTP share — as seen in Round 2a `B_L63_ray_r1`). Reading all four together cleanly separates `nominal`, `in_manifold_shift` (low entropy + low repetition + real KL/shift), `loop_collapse` (high 4-gram repetition even with high MTP), and `manifold_collapse` (spiking entropy).
   - UI highlights each `<think>` and `content` token by live telemetry metric (`g_t`, `rel_delta`, `kl_vs_base`, `ray_cos`, `entropy`) and supports click-a-token causal `Δ logprob` vs. an unsteered `s = 0` forward.
5. **Seed-Locked Side-by-Side Sweep Strip (`runtime.py`, `ui/`):**
   - Runs the same prompt (+ optional image) and exact seed with **freshly reset GDN/FA caches per arm** across an unsteered `s = 0` baseline and 2–3 operator arms (or a scale ladder).
   - Forces **`MTP = off`** whenever `temperature > 0.0` (preventing speculative draft-length differences from desynchronizing PRNG draws across arms), reporting token counts, gate-fire fraction, mean `rel_delta`, `KL`, 4-gram repetition, manifold badge, and inline token heatmaps side by side.
6. **The Chain Lens — J-Lens Transported Contrastive MTP Readout (`chain_lens.py`):**
   - Inspired by *The Chain Lens [interim]* (`vgel` & Raymond Douglas, LessWrong): extends single-token X-rays into short (`3–5` token, avoiding >5-step hallucination drift) multi-token latent phrases by:
     1. Fitting a fast CPU **residual Ridge J-Lens transport** (`L_mid -> L63`) from the cached `SpanBank`'s paired states,
     2. Adding `± s * v` onto the **corpus-mean prior** `h_prior` and running plausibility-filtered contrastive beam search (`log P_MTP(t | h_prior + s*v) - alpha * log P_MTP(t | h_prior)`) on Qwen 3.8's native 1-layer MTP head (`*-mtp`), and
     3. Scoring each decoded phrase with a **§6 round-trip cosine fidelity check** back against the probed direction (`POST /api/chain-lens`).

---

## 4. Runtime Engineering Invariants (MLX / Qwen 3.8)

1. **Unrolled verify-loop safety (`qwen3_5_runtime.py:511-645`):** Because `qwen3_5_runtime.speculative_verify_hidden` inlines the layer loop without calling `layer.__call__`, `runtime.py` hooks `lm.model.layers[idx]`, exposes `lm._steering_lab_hook`, and **disables unrolled verify when steering is active** unless the LM explicitly marks `_supports_steering_lab_hook = True` (preventing silent hook bypass).
2. **Per-position phase masking & boundary clamping:** Token phase (`prompt_text` / `image_pad` -> `think` -> `content`) is tracked per position (`[1, seq_len, 1]`), so even a multi-token speculative verify batch straddling `</think> -> content` masks each position cleanly and clamps commit at the phase boundary.
3. **Rollback-safe running mean ($\bar{h}_t$):** Speculative verification computes candidate prefix means functionally from saved `(sum_h, count)` and commits only `[:commit_accepted]` so rejected draft tokens never leak into local centering.
4. **Seed-locked RNG parity:** `sweep()` forces `MTP = off` whenever `temperature > 0.0` (only allowing MTP under greedy `temperature == 0.0` with hook-aware verify) so PRNG streams never butterfly-diverge across arms.
5. **Prefix-cache safety:** Terminal `L63`-only steering leaves all KV/GDN caches untouched (100% cache reuse across scale changes); mid-layer steering (`< L63`) folds the canonical non-zero steering signature into cache lookup keys.
6. **Model fingerprinting & GPU coexistence:** Every saved bank and vector pack records `{model_id, hidden_size, num_hidden_layers, config_sha256}` and refuses cross-architecture loads loudly. The lab runs on port `8111`, loads weights only on explicit user action (`POST /api/model/load`), and runs all unit tests and offline bank/geometry/Chain-Lens math on CPU (`0%` Metal contention with `heylook`).

---

## 5. Experiment Playbook (When GPU Is Free)

1. **Experiment 1 — Answer-Span Multi-Slider Bank (`Swift-1.5-Qwen3.8-27B-MLX-8bit`):**
   - Capture raw spans at `[15, 27, 39, 43, 47, 49, 51, 55, 62, 63]` on 4 starter contrastive corpora (`clinical_vs_poetic`, `blunt_vs_hedged`, `socratic_vs_declarative`, `anti_slop`) plus `capability_holds` (math/code/JSON).
   - Compare bank effective rank and `p90 |cos|` across the `FA` band (`L39, L43, L47, L51`) vs. `L63` to verify the ~75%-depth transition directly on Qwen 3.8, and inspect vectors via J-Lens Chain Lens (`3–5` tokens).
   - Sweep 2- and 3-slider blends with `peer_ros` OFF vs. ON and `holds` OFF vs. ON on unseen prompts.
2. **Experiment 2 — Phase-Split Thinking vs. Output:**
   - Keep `<think>` unsteered (`phase="content"`) and dial heavy style/register shifts on visible answers — verify reasoning accuracy stays pinned while voice transforms cleanly.
3. **Experiment 3 — Hybrid FA vs. GDN Probing:**
   - Apply the same mid-depth vector to **FA-only layers (`idx % 4 == 3`: `L39, L43, L47, L51`)**, **GDN-only layers (`L48, L49, L50`)**, and all layers; measure style persistence, `KL(p_steered || p_s0)`, and 4-gram repetition over 500-token generations.
4. **Experiment 4 — VLM Dual-Lock (`image_pad`-only vs. `content`-only):**
   - Feed an image + neutral prompt and steer *only* `<|image_pad|>` token positions at `L27 / L47` vs. *only* decode `content` tokens — measuring whether Qwen 3.8's visual representations can be steered before text decoding begins.
5. **Experiment 5 — `IF X -> THEN Y` Conditional Neurons:**
   - Build a decoupled neuron (e.g., `w_read` from prose hedging / uncertainty, `w_write` toward crisp concrete phrasing) and inspect the Token Microscope heatmap to verify selective firing (`g_t == 0` on code/math, `g_t > 0` on hedging prose).

---

## 6. Graduation Path to `heylook` (Future, Optional)

Nothing in `src/heylook_llm/` is touched while experimenting here. If an operator and vector pack prove themselves on held-out daily prompts:
- Vector packs already use `<direction-id>/{vector.safetensors, meta.json}` with model fingerprints and `{operator, phase, layers, scale}`.
- `heylook`'s `engine.steering` slot (`src/heylook_llm/providers/contract.py`) can adopt the proven subset (`steering.py` + `b1_runner` hook + `qwen3_5_runtime` 2-line verify hook) without inheriting any lab-only UI or telemetry machinery.
