# MLX Steering Lab — Capabilities & Operator Guide

**Steering Lab** (`apps/steering-lab/`) is a standalone white-box activation-steering and mechanistic interpretability workbench for Apple Silicon MLX models (starting with `Swift-1.5-Qwen3.8-27B-MLX-8bit` + its sibling `-mtp` drafter).

It stays deliberately decoupled from `heylook`'s daily-driver `/v1/messages` server, presets, GGUF subprocesses, and SQLite store, while sharing `heylook`'s canonical MLX overlays, `heylook.toml` model discovery, and `~/.heylook/api_key` access gate.

For the full mathematical derivations and Qwen 3.8 runtime invariants, see [apps/steering-lab/PLAN.md](../apps/steering-lab/PLAN.md).

---

## Quick Launch

```bash
# Launch the interactive web workbench on http://127.0.0.1:8111/
# (auto-attaches the newest local SpanBank + built vectors under apps/steering-lab/data/)
uv run heylookllm --steering-lab

# List all companion apps under apps/
uv run heylookllm --apps
```

- **Zero GPU on startup:** Bank geometry, built vectors, and demo readouts load on CPU immediately. The MLX checkpoint only loads onto Metal when you click **Model… → Load Checkpoint** (or invoke `capture`).
- **Shared access control:** When `~/.heylook/api_key` (or `HEYLOOK_API_KEY`) is configured, Steering Lab mounts `heylook`'s `HostCheckMiddleware`, `AccessGateMiddleware`, `/login` session cookie, `x-api-key` / `Bearer` auth, and always-on Origin/CSP guards automatically. Binding `--host 0.0.0.0` fails closed unless an API key is configured.

---

## What Steering Lab Can Do

### 1. Capture Reusable Multi-Layer Activation Banks (Capture Once, Iterate on CPU)
Instead of re-running a 27B model every time you tweak an SVD rank, orthogonalization rule, or hold subspace, Steering Lab splits work into two stages:

- **Capture (`capture`):** Runs unsteered greedy generation (accelerated by the model's native MTP speculative drafter when present, with live per-pole progress and ETA), then runs a single teacher-forced pass to extract hidden states across a configurable layer ladder (default Qwen 3.8 probe set: `L15, L27, L39, L43, L47, L49, L51, L55, L59, L62, L63`).
- **Span isolation & hygiene:** Saves separate tensors per `(concept, pole, span_kind, layer)` for **`image_pad`**, **`prompt_text`**, **`think`**, and **`answer`**. Position `0` (attention sink) and chat-control special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, EOS) are excluded from text/think/answer rows while `<|image_pad|>` tokens remain intact for vision spans. Truncated `<think>` traces are never mislabeled into `answer`.
- **Fast CPU rebuilds (`build`):** Rebuilds Regime A, Regime B, Conditional, and capability-hold packs (`.safetensors` + `meta.json`) directly from a cached `SpanBank` in under a second on CPU without touching Metal.

### 2. Steer Beyond 1D Vector Addition (Four Non-1D Operators)
Classic 1D steering (`h + s * v`) breaks down as soon as you stack multiple concepts, mix layers with different residual norms, or try to erase an unwanted habit without corrupting clean prose. Steering Lab implements four operators in [`operators.py`](../apps/steering-lab/src/steering_lab/operators.py):

| Operator | Best For | How It Works |
| :--- | :--- | :--- |
| **Regime A** (Multi-Slider Manifold Replacement) | Concepts with a coherent target style/register/lens (`clinical_vs_poetic`, `blunt_vs_hedged`, `socratic_vs_declarative`, `vision_optical_vs_mood`) | Local-centered Pole-A subspace erasure + calibrated target push on the mid-depth Full-Attention band (`idx % 4 == 3`: `L39, L43, L47, L51`). Supports multi-slider **`peer_ros`** (orthogonalizing each concept's erasure basis against peer targets), **clamped `[0, 1]` union SVD** (preventing overlapping erasures from sign-flipping), **capability `holds`** (projecting edits orthogonal to math/code/JSON spans), **`mu_d_norm` vs. `norm_matched`** push scaling, **`layer_scale_mode`** (`divide_k` / `divide_sqrt_k` / `raw`) so multi-layer stacks match single-layer budgets, and a smooth relative **`tanh` RMS knee**. |
| **Regime B** (Active Noise Cancellation) | Pure failure-mode suppression where there is no single target pole (`anti_slop`, hedging, filler) | Deadzone cosine gate (`m0`, `w`) over a local-centered residual (`prior_ema` / `mu_clean` / `running`, reset cleanly at `<think> -> content`) driving a multi-rank paired-shift erasure basis (`vr_diff`, `boost = 0`). Stays at exact `0.0` on clean tokens and fires selectively when the failure mode activates. |
| **Conditional `IF X -> THEN Y`** | Context-dependent rules (`cond_hedge_to_concrete`: *when hedging appears, ground in concrete physical nouns*) | Decouples a 4-pole read trigger (`read_neg` vs. `read_pos`) from a separate write push (`write_neg` vs. `write_pos`) via a one-sided clipped ReLU ramp (`0.0` outside the trigger). |
| **`S3` Matched-Rank Random Control** | Scientific sanity checks / double dissociation | Replaces real concept bases with a deterministic, matched-norm, matched-rank orthogonal random basis — one toggle to confirm a behavioral shift comes from the learned subspace rather than generic activation noise. |

### 3. Surgical Phase, Modality & Hybrid Layer Control
Qwen 3.8 is a `3:1` hybrid of **48 Gated DeltaNet (`GDN`)** recurrent layers and **16 Full-Attention (`FA`)** layers (`idx % 4 == 3`), with native `<think>` reasoning and `<|image_pad|>` vision tokens. Steering Lab detects layer types dynamically from `lm.model.layers` and lets you isolate:

- **Phase masks:** Steer only **`content`** (keeping `<think>` bit-exact unsteered), only **`think`**, **`prompt`**, or **`all`** — including exact per-position masking across the `</think> -> content` transition.
- **Token masks (VLM Dual-Lock):** Restrict steering strictly to **`image_pad`** visual tokens, **`prompt_text`**, **`decode`**, or **`all`**, letting you test whether a visual concept acts during image encoding vs. text decoding.
- **Layer filters:** Restrict active sliders to **`FA` only** (`idx % 4 == 3`), **`GDN` only**, **mid-layers (`< L63`)**, or **terminal `L63` only**.

### 4. Six Interactive Mechanistic Instruments (Web UI & REST API)

1. **Non-1D Operator Mixer:** Stack sliders across multiple concepts, toggle `peer_ros`, clamped union SVD, capability `holds`, `norm_matched`, `S3` random control, `layer_scale_mode`, centering mode, phase/token/layer filters, and inspect composed blend stats (effective rank, hold overlap, net push) before generating.
2. **Seed-Locked Side-by-Side Sweep Strip:** Run the same prompt (and optional image) with an identical seed and freshly reset GDN/FA caches across a baseline (`s = 0`) and multiple scale or operator arms.
   - Prevents PRNG drift across arms when `temperature > 0` by keeping live speculative verify off, while still computing real **post-hoc teacher-forced MTP agreement (`mtp_share` + per-token `mtp_accepted`)** in a single steered forward pass.
   - Reports exact **`KL(p_steered || p_s0)`**, **4-gram repetition**, **entropy**, **relative hidden shift (`rel_delta`)**, **`mtp_share`**, and an automatic manifold status badge (**`nominal`**, **`in_manifold_shift`**, **`loop_collapse`**, or **`manifold_collapse`**) — catching degenerate high-MTP repetition loops that naive acceptance-rate checks miss.
3. **Token-Level Microscope & Click-a-Token Causal `Δ logprob`:** Color-codes every `<think>` and `content` token by live `g_t`, `rel_delta`, `kl_vs_base`, `ray_cos`, or `entropy`. Clicking any token runs an exact steered-vs-unsteered causal forward at that position and shows the top boosted and erased next-token candidates.
4. **The Chain Lens (Multi-Token Contrastive MTP Readout):** Inspired by *The Chain Lens* (vgel & Raymond Douglas):
   - Transports mid-layer vectors (`L_mid -> L63`) via an FP64 residual Ridge **J-Lens** map whose regularization is auto-selected by cross-validation on held-out `val_r2` over pooled `answer + think` states (with an optional exact upper-layer forward toggle).
   - Contrastively unrolls short (`3–5` token) latent phrases through Qwen 3.8's native 1-layer MTP head (`log P_MTP(t | h_prior + s*v) - alpha * log P_MTP(t | h_prior)`).
   - Scores every decoded phrase with a **round-trip residual fidelity check** (`encode_phrase_hidden` at the target layer: cosine similarity, peer-concept rank, and calibration-corpus percentile).
5. **Subspace Geometry Bench:** Plots cross-layer effective rank, top-5 PC energy, and concept-pair cosine matrices with `FA`/`GDN` badges so you can directly see where concepts separate cleanly vs. where late layers collapse rank.
6. **Tangent-RMSNorm `lm_head` X-Ray:** Projects directions through Qwen 3.8's zero-centered `(1 + w)` final RMSNorm directional tangent before `lm_head`, flagging mid-layer projections (`< L63`) explicitly as **Advisory Logit Lens** (pointing you to Chain Lens and click-a-token causal `Δ logprob` for faithful mid-layer readouts).

---

## Built-In Starter Corpora

All built-in corpora live in [`apps/steering-lab/src/steering_lab/corpora.py`](../apps/steering-lab/src/steering_lab/corpora.py) and balance Pole A / Pole B across shared neutral topics so topic nouns do not leak into style subspaces:

| Corpus ID | Kind / Operator | Default Layers | What It Probes |
| :--- | :--- | :--- | :--- |
| `clinical_vs_poetic` | `regime_a` (`content`) | `L39, L43, L47, L51` | Crisp engineering prose vs. metaphorical literary prose |
| `blunt_vs_hedged` | `regime_a` (`content`) | `L39, L43, L47, L51` | Direct conclusions vs. tentative diplomatic hedging |
| `socratic_vs_declarative` | `regime_a` (`content`) | `L39, L43, L47, L51` | Probing diagnostic questions vs. declarative answers |
| `anti_slop` | `regime_b` (`content`) | `L63` | Active noise cancellation for corporate-AI clichés |
| `vision_optical_vs_mood` | `regime_a` (`image_pad`) | `L27, L47` | VLM Dual-Lock: concrete optics/geometry vs. abstract mood (synthetic images or `--image-dir`) |
| `cond_hedge_to_concrete` | `conditional` (`content`) | `L43, L47` | 4-pole `IF hedging -> THEN concrete physical measurements` |
| `capability_holds` | `hold` | All captured | Math, Python algorithms, and strict JSON prompts protecting structured reasoning |
| `jlens_calibration` | `calibration` | All captured | Diverse cross-domain prompts for J-Lens transport fitting and Chain Lens percentiles |

---

## Common CLI Recipes

All commands work directly from the repo root via `uv run heylookllm --steering-lab ...`:

```bash
# 1. Capture a focused 4-corpus core bank (uses MTP + medium reasoning effort by default)
uv run heylookllm --steering-lab capture \
  --model Swift-1.5-Qwen3.8-27B-MLX-8bit \
  --corpus clinical_vs_poetic,anti_slop,capability_holds,jlens_calibration \
  --out apps/steering-lab/data/banks/swift_qwen38_core

# 2. Capture all 8 corpora (optionally supply real photos for vision_optical_vs_mood)
uv run heylookllm --steering-lab capture \
  --model Swift-1.5-Qwen3.8-27B-MLX-8bit \
  --corpus all \
  --image-dir /path/to/photos \
  --out apps/steering-lab/data/banks/swift_qwen38_full

# 3. Build vector + hold packs from a cached bank (CPU)
uv run heylookllm --steering-lab build \
  --bank apps/steering-lab/data/banks/swift_qwen38_core \
  --out apps/steering-lab/data/vectors/swift_qwen38_core

# 4. Inspect cross-layer FA/GDN geometry (CPU)
uv run heylookllm --steering-lab inspect-bank \
  --bank apps/steering-lab/data/banks/swift_qwen38_core

# 5. Launch the web UI (auto-attaches newest bank/vectors under apps/steering-lab/data/)
uv run heylookllm --steering-lab --host 127.0.0.1 --port 8111

# 6. Zero-GPU sandbox mode (synthetic demo bank + UI preview)
uv run heylookllm --steering-lab demo-bank --out apps/steering-lab/data/banks/demo_synth
uv run heylookllm --steering-lab serve --bank apps/steering-lab/data/banks/demo_synth
```

### Remote Access (Optional)
1. Ensure `~/.heylook/api_key` exists (`uv run heylookllm key status`).
2. Add your chosen hostname (e.g. `steering.example.com`) to top-level `allowed_hosts` in `heylook.toml`.
3. Reverse-proxy HTTPS traffic from that hostname (or use `tailscale serve --bg --https=8111 http://127.0.0.1:8111`) to `http://127.0.0.1:8111`. Browser sessions authenticate at `/login` with the same key and cookie rules as `heylook`.

---

## Practical Guardrails & Current Boundaries

- **Mind shared Metal memory:** Keep `heylook` idle or unload models (`Unload` button in the top bar) if running other heavy Metal jobs concurrently.
- **Reaching `content` when thinking is enabled:** If a slider has `phase="content"` and `enable_thinking=True`, set `max_tokens` high enough to finish `<think>` (or switch thinking off / `reasoning_effort="low"`), otherwise the trace stays inside `<think>` where content-only sliders deliberately apply zero perturbation.
- **Keep Chain Lens short (`3–5` steps):** Contrastive MTP chains become speculative past ~5 tokens; treat Chain Lens and mid-layer X-Ray as hypothesis generators, and trust **click-a-token causal `Δ logprob`** + **seed-locked sweeps** as ground truth.
- **Experimental lab, not production chat:** Steering Lab is intentionally separate from `heylook`'s daily chat server so you can mutate operators freely. Only vector packs that prove themselves across sweeps and holds graduate into production.
