# J-Space guide — reading a model's verbalizable workspace

Last updated: 2026-07-09

This is the how-to-use guide for the **j-space** (Jacobian-lens) interpretability
feature. For the design rationale, verifier ladder, and status, see
[jspace_integration_plan.md](./jspace_integration_plan.md).

## 1. What it is (in one minute)

Anthropic's July-2026 paper *"Verbalizable Representations Form a Global Workspace
in Language Models"* introduced the **Jacobian lens** (`anthropics/jacobian-lens`):
a per-layer linear map that reads, out of a model's residual stream, **which
vocabulary tokens an activation is disposed toward** — the model's silent
"workspace" (j-space). Applying it at each layer shows the concepts the model is
tracking *before* it commits to an answer.

This server wraps the lens **apply** path (reading, not fitting) as a post-hoc
analysis feature:

- `GET /v1/jspace/models` — which served models have a lens installed.
- `POST /v1/jspace/analyze` — for a prompt: greedily answer, capture the residual
  stream, and return the layer-by-layer workspace + features + optional
  hallucination-risk.
- A v3 **J-Space** page (`/v3#/jspace`) that renders it.

It is an **inspection** tool — zero capability or speed gain. Its value is seeing
what the model is silently considering.

## 2. How it works (the pipeline)

```
 offline (once per model, torch+jlens)          in the server (MLX)
 ────────────────────────────────────           ─────────────────────────────
 fit lens (Anthropic/HF)  ──►  lens.pt           format prompt  (chat template + <bos>)
                                  │                     │
 scripts/jspace_convert_lens.py   │              greedy generate short answer
   torch.load → mx safetensors    ▼                     │  (first answer token + logprobs)
 adapters/jspace/<model>/lens.safetensors        capture residual stream per band layer
   + lens.sidecar.json                                  │  (block outputs, via a temporary hook)
                                                  apply lens:  unembed(h_l @ J_lᵀ)
                                                        │  (through the model's REAL final-norm
                                                        │   + tied head + softcap)
                                                  read out: top-k silent tokens per layer,
                                                        │   entropy trajectory, answer-token rank
                                                  features → optional hallucination-risk
```

Key modules (`src/heylook_llm/jspace/`):

- **`lens.py`** — `JSpaceLens`: loads the converted safetensors + sidecar, holds the
  per-layer `J_l` matrices, and applies `unembed(residual @ J_lᵀ)`.
- **`capture.py`** — `ModelAdapter` locates the text decoder / final-norm / tied-or-untied
  head / soft-cap inside an mlx-lm **or** mlx-vlm model (it walks
  `model → .model / .language_model → .model`). `capture_residuals` runs one forward and
  records each requested block's **output** residual (the same tensor the lens was fit on).
- **`features.py`** — `workspace_readout` (ignition / rank / entropy / hedge-rank over the
  "band"), `router_feature_vector` (the 10 named features), and `HallucinationRouter` +
  `FeatureNormalizer` (per-model z-scored logistic regression predicting P(answer wrong)).
- **`registry.py`** — `LensRegistry` maps a served `model_id` to its lens under
  `adapters/jspace/<model_id>/`.
- **`analyze.py`** — orchestrates the pipeline above.

### Concepts you need

- **Band** — the workspace lives in the *middle half* of the network
  (layers `[0.25·L, 0.75·L)`); the read-out uses those layers, not single ones.
- **Answer-onset** — the workspace is read at the **last prompt position** (the state just
  before the model answers), using the token it actually generated first.
- **Raw vs chat prompting (important!)** — the analyze endpoint defaults to `chat=false`
  (**raw completion**: `<bos>` + your text). That makes the last position a real *content*
  token, so the top-k surfaces sensible words (e.g. "…the city of" → Paris). With
  `chat=true` (full chat template) the last position is the generation-prompt boundary, where
  the top-k is dominated by *formatting* tokens — useful for the risk features (which use the
  answer token's rank, not the top-k) but poor for the visualization.
- **`<bos>` is load-bearing** — gemma's residual stream degrades to multilingual-token garbage
  without the BOS attention sink; the pipeline always prepends it.

## 3. End-to-end tutorial

### Step 1 — Install a lens for a served model

Lens *fitting* is offline and needs PyTorch + `jlens` (NOT in the MLX server venv). The
git-tracked helper downloads a pre-fit lens and converts it in one step. Run it in a
throwaway env:

```bash
uv run --with torch --with safetensors --with huggingface_hub \
    --with "jlens @ git+https://github.com/anthropics/jacobian-lens" \
    python scripts/jspace_convert_lens.py \
    --hf-repo solarkyle/jspace-lenses \
    --hf-file gemma-4-26b-a4b-it/lens.pt \
    --model-id gemma-4-26b-a4b-it-8bit-mlx \
    --softcap 30
```

- `--model-id` is the **served** model id — the directory name the endpoint looks up. Point
  it at whatever id you serve (a lens fit on the bf16 checkpoint transfers to your 8-bit MLX
  quant; validated on the served 26B MoE).
- Or use a local file: `--lens-pt /path/to/lens.pt`.

This writes `adapters/jspace/<model-id>/lens.safetensors` + `lens.sidecar.json`. `adapters/`
is git-tracked (a `.gitkeep`) with **gitignored contents** (like `modelzoo/`); the lens files
are never committed. Where lenses exist (HF): `solarkyle/jspace-lenses` (fit on the gemma-4
served models incl. the 26B MoE) and `neuronpedia/jacobian-lens` (a large zoo incl. tiny
`gpt2-small`/`pythia-70m` for testing).

The registry defaults to `<repo>/adapters/jspace`; override with `HEYLOOK_JSPACE_DIR`.

### Step 2 — Confirm it's registered

```bash
curl -s localhost:8080/v1/jspace/models | jq
# { "models": ["gemma-4-26b-a4b-it-8bit-mlx"], "base_dir": ".../adapters/jspace" }
```

If `models` is empty, the lens isn't where the registry looks — check the directory name
matches the served model id and that `lens.safetensors` is present.

### Step 3 — Analyze a prompt (API)

```bash
curl -s localhost:8080/v1/jspace/analyze -H 'content-type: application/json' -d '{
  "model": "gemma-4-26b-a4b-it-8bit-mlx",
  "prompt": "The Eiffel Tower is located in the city of",
  "heatmap": true
}' | jq
```

Response shape:

```jsonc
{
  "answer": "Paris, Paris, Paris,",         // greedy continuation (raw mode)
  "first_answer_token": " Paris",
  "prompt_tokens": ["<bos>", "The", " E", ...],
  "band_layers": [7, 8, ..., 21],
  "onset_strip": [                           // one entry per band layer
    { "layer": 21, "entropy": 3.1,
      "top_k": [ { "token": " Paris", "logit": 12.4 }, { "token": " Amsterdam", ... } ] },
    ...
  ],
  "heatmap": [ { "layer": 21, "cells": [ { "token": "…", "entropy": 2.8 }, ... ] } ],
  "features": { "ws_mean_entropy": ..., "ws_ignition_frac": ..., "bl_first_token_logprob": ... },
  "risk": null                               // P(wrong); null unless a normalizer is configured
}
```

Useful body fields: `messages` (chat-style, instead of `prompt`), `chat` (default `false`;
`true` = chat template — for risk, not viz), `max_answer_tokens` (default 8), `top_k`
(default 8), `heatmap` (default `false` — the heavier layer×position grid).

### Step 4 — Analyze in the UI

Open `/v3`, go to the **J-Space** tab. Pick a lens-enabled model, type a *completion-style*
prompt ("The Eiffel Tower is located in the city of"), and click **Analyze**. You get:

- The **workspace strip** — one row per band layer (deep → shallow), each showing the top-k
  silent tokens, colored green→red by their within-layer rank. Late layers should converge on
  the answer concept (cities, with Paris prominent).
- The generated **answer** and, if configured, a **risk** badge.
- Toggle **layer×token heatmap** for the per-position grid (top-1 token per cell, colored by
  confidence), and **chat mode** to switch to chat-template prompting.

### Step 5 (optional) — Enable the hallucination-risk score

Risk is `null` by default because a single request can't be z-scored. To enable it, drop two
files next to the lens:

- `adapters/jspace/<model_id>/normalizer.json` — `{"mean": {feat: μ}, "std": {feat: σ}}`,
  the per-model feature statistics over your traffic (the transfer trick).
- `adapters/jspace/<model_id>/router.json` — a solarkyle-style spec
  `{"models": {"combined": {"features": [...], "weights": [...], "bias": ...}}}`
  (ships in `solarkyle/jspace-lenses/router/`).

With both present, `analyze` returns `risk` = P(answer wrong) (use `chat=true` so the
answer-token-rank features are meaningful). On solarkyle's TriviaQA traces this router hit
AUC ~0.80, beating the output-logprob baseline.

## 4. Interpreting the output

- **A late-band layer whose top-1 is the answer concept** = the model "knows" it internally.
- **Low `entropy` + the answer rising in rank across layers** (high `ws_ignition_frac`) = a
  confident internal trajectory. Diffuse, high-entropy late layers, or the answer never
  entering the top-10 (`ws_ignition_frac` ≈ 0), correlate with wrong answers — that's what the
  risk router keys on.
- The **hedge-rank** feature tracks how near hedging tokens (" maybe", "?", " unsure") sit in
  the workspace.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/analyze` 404 | No lens for that model id. Check `GET /v1/jspace/models`; the `adapters/jspace/<dir>` name must equal the served model id. |
| Workspace top-k is garbage (rare multilingual/code tokens) | You're in `chat=true` mode (reads the format boundary) — use raw completion (default). If raw is also garbage, the prompt's last token isn't content; use a completion-style prompt. |
| Empty picker / "No j-space lenses installed" | Run the convert helper; confirm `adapters/jspace/<model_id>/lens.safetensors` exists (or set `HEYLOOK_JSPACE_DIR`). |
| `risk` always `null` | Expected until you add `normalizer.json` + `router.json` (Step 5). |
| Slow analyze | It runs `max_answer_tokens` no-cache forwards + per-band-layer unembeds on the full model. It's an analysis endpoint, not a hot path; keep `heatmap=false` and small `max_answer_tokens` for speed. |

## 6. Caveats

- **Text-only.** The image path of a VLM is not read; only its language model.
- **Serialized, not concurrent.** Analyze pins the model and runs its forwards under the
  process-global FIFO generation gate, so it serializes with generation and other analyze calls
  (no concurrent Metal work). While an analyze is in flight the model is pinned, so a request for a
  *different* model briefly can't evict it — expected for this analysis endpoint.
- **Provenance.** Lenses are Anthropic's `jlens` output (Apache-2.0 code; gemma lenses inherit
  the Gemma Terms of Use — keep them out of git, which the `adapters/` gitignore does). The
  MLX apply path was verified to reproduce the reference to cosine ~1.0 (see the plan's V1/V2).
