# Cross-Repo Pollination & Ideation: SVD, Subspace Erasure, and General Steering (`ComfyUI-h3-explorations` <-> `heylookitsanllm`)

Date: 2026-11-23  
Status: Research / ideation note (dated diagnostic snapshot; runtime throughput/latency measurements live in local run logs, not tracked docs)

> **Context:** Combines empirical diagnostics on `Qwen3.8-27B` (64-layer `3:1` Gated DeltaNet + Full-Attention hybrid, 1-layer stateful MTP drafter) with cross-repo peer review from `ComfyUI-h3-explorations` (`control_vectors.py`, `lora_branch.py`, `docs/wiki/steering/how_pole_a_erasure_works.md`). Incorporates all three peer-review fixes (`ACE` clean-anchor + Regime B ReLU gate, per-component FP64 SVD vs. `_pack_3band_block_diag`, and the 8-bit/BF16 activation dtype caveat), the full `L03..L63` phase-transition & `L63` `lm_head` logit-lens findings, and bidirectional brainstorming for general steering across both repos.

---

## 1. Shared DNA & November 2026 Prior-Art / Novelty Ledger (`LLM -> DiT -> LLM`)

The linear algebra between `ComfyUI-h3-explorations` and `heylookitsanllm` keeps converging because these operators traveled **LLM -> Diffusion (H3) -> LLM**. Looking across late-2026 SOTA in all three silos, most individual algebraic primitives exist in isolation; what is genuinely novel lives at the seams:

| Domain / Primitive | Established Nov 2026 Prior Art (Do Not Claim as New) | What Is Genuinely Net-New Across Our Two Repos |
| :--- | :--- | :--- |
| **DiT Steering & Unmerged LoRAs** (`control_vectors.py`, `steer_attention_hooks.py`, `lora_branch.py`) | Training-free SVD/subspace erasure (`Safree` ICLR, `CURE` NeurIPS, `IPErase`), OT style sliders (`Linear-AcT`, Apple 2025), single-target retain orthogonalization (`RASteer`, `arXiv:2610.01969`), video-only DiT steering (`LA-LQR`, `STAS`, `MUTE`, `FADE`, `CLEAR`), audio-only DiT steering (`TADA!`, ICLR 2026), unmerged compute-dtype LoRAs over INT8 + step-windowing | **1) Unified single-stream AV DiT causal anatomy:** First depth/stream map of joint `24 fps` video + `40 Hz` stereo audio (`blocks.0..19` layout/face lock, `blocks.20..29` visual breakout, `10`-block audio lag in `31..47`, terminal `block 49`).<br>**2) Compiler-safe `1.00x`-NFE query slicing (`1x Q_aud -> K_txt_base`):** Exploits token asymmetry (`130–810` audio vs. `23k+` video rows) with `+0` `malloc_graph` breaks across attention backends.<br>**3) Multi-slider compositional algebra (`Theorem 1`):** `50`-stem audit uncovering `50.9%` shared live-action scaffolding, `> 1.0` union eigenvalue spikes, and the `4.0 MP` diptych norm cliff — solved via `peer_ros`, `[0, 1]` clamped union SVD, and smooth RMS knee.<br>**4) Multimodal Dual-Lock Law (`Proposition 1`):** Why fixed-prompt `<Picture 1>` pairs collapse `V_r` and why VAE `reference_latent` chroma overrides both text and `ColQwen` vision-token erasure until released. |
| **Late-Interaction & Static Retrieval -> Generation** (`j_lens.py`, `docs/wiki/steering/research/2026-11-20_multivector_retrieval_sota_to_dit_steering.md`) | `ColBERTv2` / `WARP` (SIGIR 2025), `PyLate` Ward pooling (`arXiv:2511.00444`), Google `MUVERA`, `ColPali` / `ColQwen`, Mixedbread *Latent Terms* (`arXiv:2605.29384`), `Model2Vec` / `POTION` — all built for similarity search (`MaxSim(Q, D)`), not in-place generative conditioning edits | **1) Retrieval-to-generation boundary laws:** Algebraic collapse of balanced Sinkhorn OT mean-pooling (`8.88e-16`) back to `v_1d`; `MUVERA`/appended bags breaking `s=0` identity via attention partition drift; asymmetric `ColBERT` row-softmax (`tau = 0.15`) as length-asymmetry rescue while `97%–99.6%` of `PCRR` transport comes from `-P_{V_r,λ}`.<br>**2) Zero-GPU Wiener Jacobian X-Ray + `Model2Vec` vibe search without an `lm_head`:** Distilling `j_lens` across truncated `Qwen3-VL` `L49` + bidirectional `5376`-D `TokenRefiner` to read `Adds`/`Erases` and synthesize `@against` erasers on CPU. |
| **LLM Reasoning Control & Hybrid Forensics** (`heylookitsanllm` `Qwen3.8-27B`) | `mergekit-extract-lora`, `peft`, **QLoRA** (Dettmers et al., 2023), Punica/S-LoRA, `mlx-lm`; `INLP` (2020), `RLACE` (2022), **`LEACE`** (NeurIPS 2023), **Abliteration** (2024), **`ACE`** (Marshall, Scherlis & Belrose, 2024), **`SEAL`** (2025); **`NoWait`** (`arXiv:2506.08343`, 2025), **`EAT`** (2025), **`DEER`** / **`CoDE-Stop`** / **`PUMA`** (2025–2026) | **1) Mechanistic unification of `NoWait` & RL conciseness (`Swift 1.5`):** Teacher-forcing `3,195` `<think>` tokens across `L03..L63` showed distributed contextual computation (`L03..L59`, `effR_D ~ 570`) collapsing `11x` at `L63` (`effR_D = 52.6`, `E_cos = +0.826`) into a pure suppressor (`ADDS = noise`, `ERASES = exact top-20 self-doubt pivots` by `-4.0..-4.3` nats, `</think>` flat at `+0.0021`).<br>**2) Hybrid `3:1` GDN+FA RL forensics & `<think>`-only LoRA phase gating:** Proving `Swift 1.5` left **all 48 Gated DeltaNet layers untouched** (`0.000000` on `A_log`/`dt_bias`/`norm`, rank-32 knee on the 7 FA/MLP matrices), and gating that unmerged branch strictly to `in_thinking == True` so CoT gets `-55%` RL conciseness while visible answers run pure stock weights (preserving APC + MTP). |

**Paper Scope Discipline:** Keep the H3 manuscript tightly focused on its net-new **Causal 4D Joint Audio-Video DiT Routing** contribution (`blocks.20..29` visual breakout, `10`-block audio lag, `1x Q_aud -> K_txt_base`, Dual-Lock VLM/VAE boundaries). Keep Qwen3.8 / LLM findings here as a separate cross-repo research note / write-up rather than diluting H3's headline claims.

---

## 2. Empirical Results on `Qwen3.8-27B`

### A. 4-Bucket Behavioral Baselines (`14` Verifiable Prompts, `seed=42`, `temp=1.0`, MTP Active)

Prompts span 4 verifiable buckets: **Easy** (`4`: TCP/UDP, capital, 1-word chemical symbol, git command), **Hard** (`5`: bat-and-ball `\$0.05`, `7^100 mod 13 = 9`, knights & knaves, Python list-mutation bug, grid combinatorics `34`), **Structured** (`3`: strict JSON, 6 word-capped bullets, executable `is_balanced` code), and **Open Technical** (`2`: CoW vs. paged KV for hybrid LLMs, asyncio/Metal thread-local stream bug).

| Arm | Pass | Total Think Tok | Mean Think | Total Ans Tok | MTP Draft Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stock 8-bit `xhigh`** | 14/14 | **7,494** | 535.3 | 983 | 0.596 |
| **Stock 8-bit `medium`** | 14/14 | 5,359 (-28.5%) | 382.8 | **2,310** (+135%) | 0.624 |
| **Stock 8-bit `low`** | 14/14 | 4,338 (-42.1%) | 309.9 | **2,063** (+110%) | 0.620 |
| **Swift 1.5 BF16 `xhigh`** | 13/14* | **3,380** (-54.9%) | 241.4 | **939** (-4%) | 0.609 |

*\*Swift 1.5 had one 1-char typo in `code_brackets` (`if ch in "([":` omitting `{`). Dropping Stock to `medium`/`low` more than doubled visible answer length while still burning `1,205` think tokens on a 4-bullet prompt (`sys_design`), whereas Swift 1.5 cut `sys_design` `2,312 -> 570`, `code_bug` `1,573 -> 543`, and `math_mod` `856 -> 217` while preserving crisp `xhigh` answers. (Wall-clock timing and tok/s depend on BF16 vs. 8-bit bandwidth and are kept in local run JSONs.)*

### B. Exact BF16 Weight-Delta SVD (`BF16 Swift-1.5 - BF16 Stock Qwen3.8`, `851` Language-Model Tensors)

Computed on exact unquantized BF16 checkpoints (accounting for HF-to-MLX `conv1d` axis transpose and zero-centered RMSNorm `+1.0` sanitize offset), with chunked Gram `eigvalsh` spectra across 32 sampled layers (all 16 Full-Attention + 16 paired GDN layers):

- **Active 7-module LoRA targets (`rel_frob = ||ΔW||_F / ||W||_F`):**
  - `mlp.gate_proj` (`0.00499`), `mlp.up_proj` (`0.00496`), `self_attn.q_proj` (`0.00408`), `self_attn.k_proj` (`0.00400`), `self_attn.o_proj` (`0.00320`), `self_attn.v_proj` (`0.00290`), `mlp.down_proj` (`0.00262`).
- **Untouched subsystems:** All **48 Gated DeltaNet layers** (`A_log`, `dt_bias`, `linear_attn.norm` = **`0.000000`**; GDN linear projections `0.00036..0.00058` flat isotropic noise, `eff_r ≈ 2,507..2,609 / 5,120`), `embed_tokens` (`0.00011`), `lm_head` (`0.00008`). *(RMSNorm `0.00184` is 1 BF16 ULP from `+1.0` sanitize rounding.)*
- **Razor-sharp rank-32 knee:** `k_proj` (`r16=69.1%`, **`r32=85.3%`**, `r64=86.4%`), `o_proj` (`r16=62.4%`, **`r32=78.5%`**, `r64=79.2%`), `v_proj` (`r16=70.3%`, **`r32=78.5%`**, `r64=80.5%`). The remaining ~15–20% across `r33..5120` is BF16 merge rounding dust.
- **Extraction caveat (peer fix):** Do **not** port H3's `_pack_3band_block_diag` (hardcoded to 3 equal-height MHA bands). Qwen3.8 uses asymmetric GQA (`q=12288`, `k=v=1024`) and 2-band `gate+up`, and MLX keeps all 7 projections as separate modules. What transfers 1-to-1 is `bench/convert_pika_lora.py`'s **per-component FP64 SVD truncation** so larger `q`/`k` norms never starve `v`.

### C. Teacher-Forced Logit & Activation/PCRR X-Ray (`3,195` `<think>` Tokens)

Replayed the exact token IDs of the 14 Stock `xhigh` `<think>` traces (capped at `384` think tokens each, identical `xhigh` system prompt) through both Stock and Swift 1.5:

- **Ground-truth logit shift (`Swift 1.5 - Stock`):**
  - Mean `<think>` entropy: `0.5315 -> 0.3978` (`-25.2%`). Paragraph-boundary `</think>` (`id=248069`): `d_logit = +0.0021`.
  - **Top 20 suppressed active tokens (`Δ logprob` in nats):** `' actually' (-4.32)`, `' wait' (-4.26)`, `' however' (-4.21)`, `' alternatively' (-4.14)`, `' hold' (-4.14)`, `' possibly' (-4.13)`, `' instead' (-4.11)`, `' perhaps' (-4.10)`, `' nonetheless' (-4.06)`, `' hmm' (-4.06)`, `' nevertheless' (-4.06)`, `' might' (-4.05)`, `' either' (-4.05)`, `' error' (-4.02)`, `' mistake' (-4.02)`, `' rather' (-4.01)`, `' otherwise' (-3.99)`, `' rethink' (-3.99)`, `' different' (-3.99)`, `' doubt' (-3.99)`.
- **Full-depth activation & PCRR profile (`L03..L63`, row-0 attention sink excluded):**
  *(Dtype caveat from peer review: `A_p` was Stock 8-bit while `B_p` was Swift 1.5 BF16, so 8-bit quantization residual inflates `effR_D` where `rel_d` is small (`L03..L59`); re-check 2–3 layers on `Stock BF16 vs Swift BF16` or `Stock 8-bit vs Stock 8-bit + LoRA` to separate true state-conditional `B(Ax)` rank from 8-bit noise.)*

| Layer | Type | `rel_d` | `r1_E%` | `sh_cos` | `flipN` | `effR_A` | `effR_D` | `dR8_E` | `PCRR_R2` | `E_cos` | Row-0 Sink |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L03` | FA | 0.005 | 0.196 | +0.985 | 0.247 | 59.3 | 22.8 | 0.676 | +0.014 | -0.088 | 3.6x |
| `L15` | FA | 0.007 | 0.081 | +0.937 | 0.083 | 309.8 | 533.3 | 0.258 | +0.027 | -0.032 | 3.5x |
| `L27` | FA | 0.012 | 0.070 | +0.925 | 0.176 | 450.1 | 838.4 | 0.176 | +0.042 | -0.021 | 4.3x |
| `L39` | FA | 0.020 | 0.062 | +0.899 | 0.196 | 467.9 | 673.1 | 0.191 | +0.037 | -0.075 | 3.4x |
| `L47` | FA | 0.026 | 0.073 | +0.892 | 0.236 | 519.5 | 577.4 | 0.172 | +0.037 | -0.037 | 2.2x |
| `L51` | FA | 0.044 | 0.168 | +0.925 | 0.351 | 520.6 | 622.1 | 0.147 | +0.067 | +0.270 | 1.1x |
| `L55` | FA | 0.051 | 0.152 | +0.906 | 0.375 | 530.7 | 546.0 | 0.166 | +0.074 | +0.278 | 0.8x |
| `L59` | FA | 0.056 | 0.133 | +0.880 | 0.306 | 579.3 | 568.4 | 0.166 | +0.069 | +0.210 | 0.8x |
| `L62` | GDN | 0.084 | 0.299 | +0.945 | 0.353 | 575.5 | 355.1 | 0.277 | +0.173 | +0.277 | 1.1x |
| **`L63`** | **FA** | **0.187** | **0.515** | **+0.984** | **0.354** | **343.4** | **52.6** | **0.627** | **+0.340** | **+0.826** | **1.9x** |

- **The `L63` phase transition (`lm_head` logit-lens readout) & causal-topology catch:**
  - **`L63 mu_d ERASES`:** `' actually' (-16.3)`, `' maybe' (-16.0)`, `' wait' (-16.0)`, `' perhaps' (-15.5)`, `' possibly' (-15.2)`, `' instead' (-15.0)`, `' either' (-15.0)`, `' however' (-14.9)`
  - **`L63 mu_d ADDS`:** pure noise (`'' (+7.4)`, `'和' (+6.9)`, `'-Cds' (+6.7)`) — proving that at `L63` right before `lm_head`, 64 layers of distributed rank-32 LoRA updates funnel into a **massive, nearly rank-1 (`51.5%`) unidirectional hesitation-erasure event (`E_cos = +0.826`)**.
  - **Crucial causal-topology caveat (from H3 peer review):** Only `final_norm` + `lm_head` sit after `L63` (no transformer layers read `h_63` into KV/GDN on token `t`). So `L63` is where upstream `L00..L59` *contextual decisions* ("is this `'Wait'` catching a real sign error, or spinning on `2+2` for the 4th time?") converge onto the low-rank unembedding subspace of the 20 doubt words. Intervening *only* at `L63` cannot access the RL LoRA's upstream error-detection circuit, making `L63`-only subtraction an internal cousin of `NoWait` logit damping.
  - **Decisive 3-arm test for `heylook`:** Benchmark **(1) `<think>`-gated Rank-32 LoRA (`L00..L63`)** vs. **(2) `L63`-only Half-Wave Ray-Gated `vr_diff` suppressor (`~160 KB`)** vs. **(3) 20-token `NoWait` logit damper (`0 KB`)** on tricky self-correction prompts (`code_bug`, `logic_knights`, `math_mod`, `code_brackets`, `sys_design`) to measure exactly how much upstream `L00..L62` calibration buys over terminal/logit suppression.

---

## 3. Regime-B Erasure Math & The `~75%`-Depth Routing Law (H3 Disk Verification)

### A. Why Naive `h - mu_a` AND Naive `h - mu_clean` Both Fail on Full Prompts (and the Exact Fix)

Zero-GPU checks on `ComfyUI-h3-explorations`'s `50` shipped `fl2va` bank stems and cached `5376`-D full-scene prompts exposed **two traps** in naive Regime-B erasure:

1. **Backwards Pole-A centering (`h - mu_a` without `+mu_d`):** Zeroes out at the unwanted center and perturbs clean states.
2. **Full-prompt Part-of-Speech trap on dataset `h - mu_clean`:** Clean-anchored `ACE` (`h - mu_clean`) self-damps cleanly by `8.6x–30.5x` on *isolated attribute clauses*, **but falsely fires on `100.0%` of tokens (`RMS = 12.2–31.6`) on full `383`-token prompts** (`covered_market.pt`) because `mu_clean` is an adjective centroid and `h_noun - mu_clean` carries a large part-of-speech offset. (The exact same trap hits mixed code/math/digit/punctuation tokens inside an LLM `<think>` trace.)
3. **Unsupervised `SVD(A_c)` drags in neutral co-occurring words:** In H3's `de_llm_slop` (`mu_d_positive_words = []`, identical to Qwen3.8 `L63`'s `ADDS = noise`), unsupervised `SVD(A_c)` placed `focus (+1.20)` and `focused (+0.93)` into `vr[2]` right alongside `blur`, cutting clean sharpness by `-33%..-39%`.

**Disk-verified fix (`Local-Centered Half-Wave Ray Gate + Paired-Shift Basis vr_diff`):**

$$g_t = \operatorname{clip}\!\left(\frac{\cos(h_t - \bar{h},\; -\hat{\boldsymbol{\mu}}_d) - m_0}{w},\; 0,\; 1\right), \qquad \Delta h_t = -\operatorname{clip}(s, 0, 1) \cdot g_t \cdot e_{\text{cap}} \cdot \big((h_t - \bar{h})\,\mathbf{V}_{\text{diff}}^\top \mathbf{D}\,\mathbf{V}_{\text{diff}}\big)$$

- **Local centering ($h_t - \bar{h}$):** Uses prompt-local mean $\bar{h}$ in DiT or a running `<think>` mean $\bar{h}_t$ in causal LLM decode, canceling scene / part-of-speech scaffolding.
- **Half-wave ray gate with deadzone ($m_0 > 0$):** Only fires when a token aligns positively with the unwanted Pole-A ray $-\hat{\boldsymbol{\mu}}_d$; clean math/code/digit/scene tokens sit below $m_0 \implies g_t = 0$. On H3's clean scene prompts (`covered_market`, `indoor_evening`, `indoor_neutral`), this dropped `de_llm_slop` RMS from **`67.50 -> 1.36–1.77` (`38x–50x` quieter!)** with zero rebuild of existing banks.
- **Paired-shift basis $\mathbf{V}_{\text{diff}}$ + `boost = 0` + upper energy clamp only:** Extracting $\mathbf{V}_r$ from paired diffs $\mathbf{D}_p = \mathbf{B}_p - \mathbf{A}_p$ (`vr_diff`) instead of unsupervised $\mathbf{A}_c$ keeps neutral co-occurring concepts (like `focus`) out of the eraser; dropping `+s·mu_d` and the RMS floor prevents amplifying noise on clean inputs.

### B. The Universal `~75%`-Depth Phase Transition (`Qwen3-VL-32B` & `Qwen3.8-27B`)

Projecting H3's `N = 50` shipped `fl2va` control vectors across `Qwen3-VL-32B`'s `L49`, H3's `TokenRefiner`, and `L63` / `lm_head` (via `J_49_to_63` and `M_union`) reveals the **exact same `~75%`-depth phase transition** we measured on `Qwen3.8-27B`:

| Stage (`N = 50` H3 `fl2va` Stems, Max Rank = `49`) | Dim | Bank Eff. Rank | Top-5 PC Energy | Mean `\|cos\|` | `p90 \|cos\|` | Geometric Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. `Qwen3-VL L49`** (`~75%` depth, `vector.encoder`) | 5,120 | **31.48 / 49** | 37.9% | 0.2438 | 0.4244 | Peak world-model breadth; concepts stay orthogonal |
| **2. `H3 Post-Refiner`** (`vector.post_refiner`) | 5,376 | **30.04 / 49** | 38.2% | **0.2235** | **0.4185** | Preserves `95.4%` rank while improving orthogonality |
| **3. `Qwen3-VL L63` pre-head** (`J_49_to_63 @ v_enc`) | 5,120 | **18.99 / 49** (`-40%`) | 57.1% | 0.3423 | **0.6247** | Autoregressive funnel crushes `40%` of concept rank |
| **4. `Qwen3-VL L63 lm_head`** (`M_union @ v_enc`) | 27,407 | **14.04 / 49** (`-55%`) | 64.4% | 0.4416 | **0.7633** | Next-token BPE bottleneck (`p90 \|cos\| = 0.76`) |

**Resulting Depth-Routing Law for LLM Steering:**
- **Regime A multi-slider semantic/tone composition** must tap **`~75%` depth (`L43..L49`)**, *before* `L50..L63` crushes 40–55% of independent concept dimensions into `p90 |cos| = 0.62–0.76` collisions.
- **Regime B single-trait lexical/reflex suppressors** (hesitation, sycophancy, AI-slop) tap **`L63` (or output logits)** precisely *because* the autoregressive funnel has already collapsed the failure mode onto a single clean direction (`r1 = 51.5%`, `E_cos = +0.826`).
- **J-Lens advisory rule:** Because `j_lens` projects forward into `L63`/`lm_head`, two vectors whose English readouts overlap (`p90 |cos| = 0.763`) can be `~2x` more orthogonal inside the model (`p90 |cos| = 0.419`); treat word-readout overlap as an advisory diagnostic, not a hard collision gate.

---

## 4. Broadening to Steering in General: Cross-Repo Pollination & Ideation

### Direction A: `ComfyUI-h3-explorations` -> `heylookitsanllm` (General LLM Steering)

1. **Multi-Slider Steering Mixer for Local LLMs (`peer_ros` + Clamped Union SVD + Capability `holds`):**
   - Most open-source LLM steering stops at one rank-1 vector (`abliteration` / `CAA` / `ACE`). Stacking 2–3 vectors (*Concise Thinking* + *Anti-AI-Slop / Anti-Sycophancy* + *Direct Tone* + *No Refusal*) typically cancels out or collapses off-manifold.
   - H3's `control_vectors.py` equilibrium math (`peer_ros` orthogonalization against peer targets + clamped `[0, 1]` union SVD `comp16` + `tanh` norm knee + shared capability `hold_basis` over clean math/code/JSON/`<tool_call>` activations) combined with the `ACE` $\boldsymbol{\mu}_{\text{clean}}$ anchor + Regime B ReLU gate behaves like **active noise cancellation for RLHF habits**: silent during clean technical work, engaging automatically on sycophancy, corporate filler, anxious `<think>` loops, or spurious refusals.
2. **Phase-Gated Steering & Phase-Gated LoRAs (H3 `_Window` & `1x Q_aud` Split -> LLM Generation Phases):**
   - In `src/heylook_llm/providers/common/b1_runner.py` / `speculative_session.py`, generation naturally splits into **3 clean structural phases**:
     - **`<think>` phase:** Run anti-overthinking (Swift-Delta LoRA, `L63` hesitation eraser, or empirical `NoWait`) while keeping style sliders off.
     - **Visible `content` phase:** Drop `<think>` steering to `0` and engage voice / tone / anti-slop sliders — full-rigor thinking, custom voice on output.
     - **`<tool_call>` / structured JSON phase:** Hard-zero *all* style/persona sliders (`s = 0`) so personality controls can never corrupt tool syntax or JSON schemas.
3. **Depth-Banded Routing on Hybrid GDN+FA LLMs (H3 `blocks.20..29` & `L49` -> Qwen3.8 Split-Depth Routing):**
   - Route **Regime A multi-slider semantic/tone steering** into the **`~75%`-depth Full-Attention band (`L43..L49`)** where concepts stay `~2x` more orthogonal (`p90 |cos| = 0.42`), and route **Regime B single-trait lexical/reflex suppressors** (hesitation, slop, sycophancy) into **`L63` (or logits)** — keeping early reasoning scaffolding (`L00..L39`) and all 48 GDN recurrent states untouched.
4. **Composable Unmerged Behavioral LoRA Bank on One 8-bit Base (`MiniMaxH3LoRABranch` -> `heylook`):**
   - Instead of storing multiple full 28 GB quantized copies of Qwen3.8 (stock, Swift 1.5, uncensored/heretic, coding fine-tunes), extract per-component FP64 SVD deltas against the single BF16 base and load them as compact unmerged sidecar branches on top of one 8-bit MLX checkpoint — continuously dialable (`s in [0, 1]`), phase-gated (`<think>` vs. `content`), and composable.
5. **Ad-Hoc On-the-Fly Concept Faders (`MiniMaxH3WordVector` -> `heylook` UI):**
   - Mirroring H3's CPU `against_words` SVD synthesizer: synthesize an instant `L63` `ACE` erasure payload from user-typed "against" phrases via one short prefill + `lm_head` X-ray check, with zero offline dataset collection.
6. **Scientific Hygiene Ported Straight from H3:**
   - Every LLM steering vector gets `lm_head` **Adds / Erases** X-ray, `split_half_cosine`, `flip_null`, `channel_census`, and `S1_own_Vr` vs. `S3_random_Vr` double-dissociation verification before saving to a bank.

### Direction B: `heylookitsanllm` / Qwen3.8 -> `ComfyUI-h3-explorations` (Reverse Pollination)

1. **Closed-Loop Adaptive `meter_score` Steering per Denoising Step:**
   - Just as `heylook` can monitor live internal state (`meter_score` + MTP acceptance + entropy) to turn steering on *only* when drifting into a loop, `MiniMaxH3Steer` could measure a lightweight per-step `meter_score` on the evolving latent/cross-attention readout to **adaptively modulate `strength(t)` across denoising steps** — pushing hard early against the prompt veto and throttling back once the target look locks in (preventing over-cook collapse past norm `~110`).
2. **Terminal-Layer Funneling Check Across H3's 50 Blocks (`L63` Analog):**
   - Teacher-forcing identical noisy latents through Pika/Turbo/FlashGen vs. Base across `blocks.0..49` and plotting `rel_d`, `effR_D`, and `PCRR` per block shows whether H3 distills similarly concentrate in a tight block band — identifying blocks where `lora_branch.py` can skip `B(Ax)` compute for free speedup.
3. **Dead-Subsystem Pruning via `rel_frob` + Rank-Knee Census:**
   - Running an automated `rel_frob` + rank-knee audit on any new H3 distill/LoRA (dual-stream attention, MLP, refiner, audio branch, AdaLN) flags noise-floor modules that can be dropped from `lora_branch.py` cleanly.
4. **Teacher-Forced Behavioral Distillation:**
   - Teacher-forcing Base's latent trajectories through a strong aesthetic/de-slop LoRA and inspecting `v_ft - v_base` via `j_lens` / frequency-band / chroma decomposition could distill black-box fine-tune gains into lightweight, model-portable training-free steering vectors.
5. **Three Immediate Concrete Impacts Back on H3 (Post–Wave 4):**
   - **Faster, non-truncated Qwen3.8 for H3's `/lab`, Easy Button, & AutoEdit (`heylook_client.py` / `llm_pairs.py`):** Once `<think>`-gated Swift-Delta LoRA lands in `heylook`, H3 clients can use server-side concise thinking instead of setting tight client-side `thinking_budget` guillotines.
   - **Direct empirical validation for H3's deferred `Regime B` ("pure suppressor", `de_llm_slop`):** `L63`'s `ERASES = 20 doubt words / ADDS = pure noise` mirrors `de_llm_slop` exactly — confirming that pure failure-mode suppressors need the `ACE` clean-anchor + one-sided ReLU + upper-clamp-only (`boost = 0`) operator rather than forced `|s| * 67.5` RMS rescaling.
   - **Explains why H3 truncates `Qwen3-VL-32B` at `L49/64`:** Broad semantic representations hold through `L47` (`~75%` depth) before collapsing onto next-token `lm_head` logits across `L51..L63` (`effR_D: 622 -> 52.6`), validating H3's `L49 + TokenRefiner` J-Lens tap without needing to chase `layers.50..63`.

---

## 5. Cross-Domain Synthesis: The 3-Regime Control Law & Known Traps

Across both **H3 DiT** and **Qwen3.8 LLM**, our experiments isolate a clean **3-Regime Control Law** separating when each operator is mathematically appropriate:

| Regime | Examples (H3 DiT & Qwen3.8 LLM) | Required Operator | Why |
| :--- | :--- | :--- | :--- |
| **Regime A — Attribute Manifold Replacement** *(coherent target pole)* | H3 visual/audio styles (`noir`, `charcoal`, `reverb_cathedral`); explicit LLM tone/persona shifts under a static prompt (`~75%`-depth `L43..L49` band) | **Span-isolated Pole-A erasure** $-\mathbf{P}_{\mathbf{V}_r,\lambda}(\mathbf{H} - \bar{\mathbf{h}})$ **+ target push** $+s\hat{\boldsymbol{\mu}}_d$ **+ `peer_ros` + clamped `[0, 1]` union SVD + `holds`** | Beats the frozen-anchor veto, protects capabilities via `holds`, avoids `L50..L63` concept collision, and stacks multiple sliders cleanly without cancellation or eigenvalue sign flips. |
| **Regime B — Pure Failure-Mode Suppression** *(no single Pole-B target; `ADDS = noise`)* | H3 `de_llm_slop`; LLM self-doubt/hesitation/sycophancy/refusal reflexes at `L63` (or output-space empirical `NoWait` + MTP spinning sensor / transactional `EAT`) | **Local-centered half-wave ray gate ($\cos(h - \bar{h}, -\hat{\boldsymbol{\mu}}_d)$ with deadzone $m_0$) + paired-shift basis $\mathbf{V}_{\text{diff}}$ + upper-cap-only (`boost = 0`)** (or targeted logit damping on the 20 doubt pivots) | Eliminates both forced-RMS noise amplification and full-prompt POS false triggers (`38x–50x` quieter on clean H3 prompts); prevents neutral co-occurring words (`focus`) from entering `V_r`; near-zero compute overhead, 100% APC-safe. |
| **Regime C — Multi-Layer Contextual Policy** *(state-dependent upstream circuit)* | H3 Pika 6-step trajectory curvature; Qwen3.8 `Swift 1.5` `L00..L59` *"real sign error vs. anxious 4th re-check"* discrimination; uncensored/domain fine-tunes | **Unmerged per-component FP64 SVD `lora_branch` over quantized weights with step/phase gating (`_Window` / `in_thinking == True`)** | Cannot be reduced to a static Prelude/readout vector without losing upstream state sensitivity (`effR_D ~ 570`). Exact rank-32 weight recovery (`79–85%`), zero GDN hooks, continuous `s in [0, 1]`, zero requantization loss over 8-bit. |

### Decisive Next Experiment in `heylookitsanllm`
First run a quick 60s verification on `mlp`/`q_proj` SVD rank knees, `Swift-1.5-mtp` vs. `Stock-mtp`, and a clean BF16 `L47`/`L63` spot-check; then run a **3-arm benchmark** on the 14-prompt 4-bucket suite (focusing on tricky self-correction prompts: `code_bug`, `logic_knights`, `math_mod`, `code_brackets`, `sys_design`) comparing:
1. **Regime C:** `<think>`-gated unmerged Rank-32 Swift-Delta LoRA (`L00..L63`, 7 FA/MLP modules, 0 GDN hooks),
2. **Regime B (residual):** `L63`-only local-centered half-wave ray-gated `vr_diff` suppressor (`~160 KB`, `boost = 0`, deadzone margin $m_0$), and
3. **Regime B (logit):** 20-token empirical `NoWait` logit damper (`0 KB`, applied to both target + MTP drafter logits).

### Known Traps to Avoid
- **Anchoring Regime-B erasure at `mu_a` without `+mu_d`** (inverts self-damping) **OR applying raw dataset `h - mu_clean` across mixed-POS full prompts** (falsely triggers on `100%` of nouns/code/digits due to POS offset; use local centering + half-wave ray gate with deadzone $m_0$).
- **Extracting Regime-B `V_r` via unsupervised `SVD(A_c)` over unsegmented traces** (drags neutral co-occurring words like `focus` or topic/syntax variance into `V_r`; use paired-shift `vr_diff` and isolate opposing spans first).
- **Stacking multi-slider Regime-A concepts at `L63`** (`L50..L63` collapses `40%–55%` of concept rank with `p90 |cos| = 0.62–0.76`; stack at `~75%` depth `L43..L49` instead).
- **Static global `</think>` bias, high `presence_penalty` in `<think>`, greedy `temp=0`, prompt word limits, solo 1-layer MTP thinking, GDN decay-gate hacking, or continuous embedding-mixture Soft Thinking on GDN** — all either backfire on quality/loops or break MTP + APC + hybrid state stability.
