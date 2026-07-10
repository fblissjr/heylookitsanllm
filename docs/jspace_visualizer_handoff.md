# J-space visualizer — frontend handoff

Last updated: 2026-07-10

> **Progress (2026-07-10, v1.34.36–.37):** the DESIGN.md gate is CLEARED
> (`apps/heylook-frontend-v3/DESIGN.md` — OKLCH system formalized; paradigm =
> matrix-first with the layer slider + aggregation sidebar as the growth path).
> SHIPPED: **item 1** (click-to-pin readout, Esc/arrow walk, echo highlight,
> onset marker), the **per-cell top-N analyze extension** (`heatmap_top_k` —
> every heatmap cell pins its full readout), **item 2** (slot-based layer-range
> slider + most-common-tokens aggregation in the unpinned detail panel), and a
> **"provisional lens" badge** off `/v1/jspace/models` `meta` (sidecar
> provenance). All E2E-covered. Next: item 3 (live streaming endpoint), then
> interventions.

A self-contained brief for a Claude starting the **j-space (Jacobian-lens) visualizer**
track in the v3 frontend. This track runs **in parallel** with the lens-fitting work (that
lives in the `jlens-mlx` sibling repo; you do not need it). You are building UI on top of a
backend read-out API that **already exists and works today**.

## TL;DR — your mission

The `jspace` page (`apps/heylook-frontend-v3/js/pages/jspace.js`) currently renders a
**static** workspace strip: one POST to `/v1/jspace/analyze`, then a full render of per-layer
top-k "silent" tokens + an optional layer×position heatmap + a hallucination-risk badge. It is
correct but inert. Make it **interactive**, in the cheapest-value-first order below. Everything
you need to build and test against is live right now — do **not** wait on the production lens
fit.

**Hard gate: seed a v3 `DESIGN.md` before building any new visual UI.** v3 has an implicit
OKLCH strength/chip system (in `css/app.css` comments) but no written design language. The plan
(`docs/plan_2026-07.md` Phase 4 item 2) requires formalizing it first so the visualizer has a
visual vocabulary to build on. This is itself a frontend task and has no dependency on anything
else — do it first.

## Ground truth you must not re-derive

### The v3 frontend contract (read these before touching a line)
- **`apps/heylook-frontend-v3/js/page.js`** — the page lifecycle. Pages export
  `createPage({ setup(ctx), teardown?(ctx) })`. Per mount `ctx` gives you: `el` (root), `state`,
  `signal` (AbortSignal, aborted on teardown — pass to every fetch), `alive` (check after every
  `await`), `guard(fn)` (no-ops after teardown), `throttle(fn)` (frame-throttled, auto-cancels),
  `linkedController()`, `onTeardown(fn)`. **Read this first — it governs everything.**
- **Vanilla JS, no build step.** No React, no bundler. DOM via `createEl(...)` from `js/utils.js`.
  Served at `/v3`. Package manager for any tooling is **bun**, never npm.
- **`js/api.js`** hand-writes the API layer (there is no generated TS client in v3). The two
  jspace methods already exist: `api.jspaceModels()` and `api.jspaceAnalyze(body, {signal})`.
- **Design tokens live in `css/app.css` comments** (no DESIGN.md yet — you seed it).
- **Orientation docs:** `docs/frontend_v3.md` (map + backend coupling) and
  `docs/frontend_v3_spec.md` (§4 = the authoritative API contract — update it in the same commit
  as any contract change).

### The apply API (exists today, unchanged by any of this)
`GET /v1/jspace/models` → `{ models: [id...], base_dir }` (only models with an installed lens).

`POST /v1/jspace/analyze` with `{ model, prompt | messages, max_answer_tokens=8, top_k=8,
heatmap=false, chat=false }` returns (see `src/heylook_llm/jspace/analyze.py::analyze`):
```
{
  "model", "answer", "first_answer_token",
  "prompt_tokens": ["tok", ...],                    // decoded prompt tokens (positions)
  "band_layers": [int, ...],                         // the depth band the lens is read at
  "onset_strip": [ { "layer", "entropy",             // one row per band layer, answer-onset
                     "top_k": [ {"token","logit"}...] } ],
  "heatmap": [ { "layer", "cells":[ {"token","entropy"}... ] } ] | null,   // layer×position, top-1/cell
  "heatmap_positions": [int...] | null,
  "features": {...},                                 // workspace features
  "risk": float | null                               // hallucination-risk, if a router is configured
}
```
Backend notes that matter: analyze is deterministic (greedy), pins the model, and runs on the
pinned MLX executor thread under the global generation gate — so it **serializes** with
generation and other analyze calls (one at a time, no concurrent Metal). Latency is real
(seconds); design for a busy/disabled state, which the page already has.

## The reference visualizers (study these for interaction ideas)

We looked at two production/research viz's for interactivity; both live in the **gitignored
`coderef/`** dir (local-only reference clones — read them, don't vendor):

- **`coderef/neuronpedia`** (MIT) — the interp OG. The **design + interaction** reference.
  Key ideas: a **`LayerRangeSlider`** (drag a range to re-scope the readout), and a deliberate
  **sidebar aggregation** (most-common readout tokens over a layer range) *instead of* a full
  position×layer matrix, because a full matrix gets unwieldy on long transcripts. Ships
  steer/swap/ablate interventions.
- **`coderef/jlens-qwen36`** (WeZZard, Apache-2.0) — MLX, same arch family as our served model.
  Its viz uses a **virtualized position×layer matrix** with spring-glide animation and a
  **click-to-pin per-cell top-N readout** (its core interaction — directly answers "walk it
  layer by layer"). Its glass aesthetic is **not** to be cloned; lift the *interaction*, not the
  look.

**Load-bearing decision for the DESIGN.md pass — aggregation vs. matrix.** Neuronpedia's sidebar
aggregation scales to long transcripts; jlens-qwen36's matrix is more directly liftable into our
vanilla-JS v3 (same tech, and we already render a heatmap grid). Pick the paradigm *before*
building, and write the choice into DESIGN.md. Recommendation to weigh: start from our existing
heatmap grid (matrix) for the cheap wins, but borrow Neuronpedia's layer-range slider and
aggregation sidebar as the scalable reading mode — they compose.

## Sequenced work (cheapest value first)

Each item names the backend dependency. The first two need **no** backend change — pure
frontend on data we already return.

1. **Click-to-pin per-cell top-N readout** (jlens-qwen36's core interaction; highest value).
   Today each heatmap cell shows only top-1 + entropy in a tooltip. Make a cell click **pin** a
   detail panel showing that (layer, position)'s top-N silent tokens.
   - *Backend:* the strip already returns top-k per band layer at the answer-onset; per-*cell*
     top-N across all positions is a small analyze extension (return top-N per heatmap cell, not
     just top-1). Until then, the answer-onset column can be pinned from `onset_strip` with zero
     backend work — ship that first, then the general per-cell version behind the extension.
2. **Layer slider / focus** (Neuronpedia's `LayerRangeSlider`). A drag-range control that
   re-scopes which `band_layers` rows are shown / aggregated. *Backend:* none — reuses the
   response; it's a client-side filter + an aggregation view over the selected range.
3. **Live streaming** (per-token workspace rows as the answer generates). *Backend:* needs a
   **new streaming analyze endpoint** (SSE, per-generated-token workspace rows). Sequence after
   1–2. Follow the existing SSE + `ctx.signal` patterns; the analyze path already runs on the
   pinned executor thread under the gen gate.
4. **Interventions (steer / swap / ablate)** UI. *Backend:* the biggest piece — needs a real
   residual-stream hook at layer L + a forward-from-layer re-gen endpoint. **Sequence last**,
   after streaming. Both reference viz's converge on the same math: transport a token's unembed
   direction by `J_lᵀ`, then add (steer) / project-out (ablate) / swap source→target.

## What you can build against RIGHT NOW

- A **provisional lens is already installed** for the served abliterated Qwen3.5-27B (under
  `adapters/jspace/<model_id>/`, gitignored contents) and a gemma lens too, so
  `/v1/jspace/models` returns real entries and analyze returns real data. The visualizer does
  **not** depend on lens *quality* — build and test the UI against the provisional lens now; when
  the production own-fit lands (fitting track) it's a drop-in file replacement, no frontend
  change. (The served readouts are "provisional" until the own-fit; fine for UI work.)
- **Local E2E:** the v3 browser E2E harness is `tests/e2e/` (puppeteer-core + system Chrome —
  claude-in-chrome refuses localhost). `cd tests/e2e && bun install`, then `node run.mjs pages`.
  Must run UNSANDBOXED. Add jspace-interaction checks here.

## Gotchas

- **Never break the page lifecycle**: check `ctx.alive` after every `await`; pass `ctx.signal`
  to every fetch; register cleanup with `ctx.onTeardown`. Stale callbacks after teardown are the
  classic v3 bug.
- **Don't clone jlens-qwen36's glass aesthetic** — lift interaction, seed our own look in
  DESIGN.md.
- **Don't start interventions first** — it's the deepest backend work; it waits behind streaming.
- **Update `docs/frontend_v3_spec.md` §4 in the same commit** as any API-contract change (e.g.
  the per-cell top-N extension or the streaming endpoint).
- **Keep the served model's brand name out of git-tracked content** — refer to it as "the served
  abliterated Qwen3.5-27B"; its exact id lives only in the gitignored `models.toml`.

## First moves

1. Read `js/page.js`, `js/pages/jspace.js`, `js/utils.js`, `docs/frontend_v3.md`,
   `docs/frontend_v3_spec.md` §4.
2. Read `coderef/neuronpedia` and `coderef/jlens-qwen36` for interaction patterns; read
   `docs/jspace_integration_plan.md` Part 2 "Frontend visualizer" + the "Observations &
   watch-items" note on aggregation-vs-matrix.
3. Seed `apps/heylook-frontend-v3/DESIGN.md` (formalize the OKLCH strength/chip system + pick
   aggregation-vs-matrix).
4. Ship item 1 against the answer-onset column (no backend), then scope the per-cell top-N
   analyze extension.
