# Frontend v3 -- orientation & backend coupling

Last updated: 2026-07-23 (preset bar extracted to shared `preset-bar.js`,
notebook gets the same section; files list + drawer contribution kinds
reconciled)

The single map for the **current** frontend. The older React-frontend docs that
used to sit beside this file (architecture, applet catalog, migration plan,
design system, api schema) described the retiring v2/legacy app and were retired
to the internal archive on 2026-07-09.

## Where the authoritative docs are (this file is a map, not a copy)

| Concern | File | Notes |
|---------|------|-------|
| **Build contract / API contract** | `docs/frontend_v3_spec.md` | §4 = the authoritative backend API contract. Update it in the SAME commit as any contract change (standing rule). |
| **Roadmap** | `docs/project/plan_2026-07.md` | Phased 0-5. §"v3 frontend guardrails" + Phase 4 (v3 hardening) + Phase 3b (Messages-API migration) are the v3-facing parts. |
| **Graded status (done/left)** | `docs/project/CURRENT.md` §1-2 | Authoritative SOLID/HALF-BAKED/UNCERTAIN/STUB grading. The snapshot below is a convenience copy and will drift -- trust CURRENT.md. |
| **Backlog** | `docs/project/TODO.md` | v3 follow-ups live under "Presets/system-prompt follow-ups". |

## What v3 is

Vanilla JS, **no framework, no bundler, no build step** -- one
`<script type="module">` bootstraps everything. Served by the FastAPI backend at
`/v3` (a ~15-line mount block in `src/heylook_llm/api.py`, cloned from `/v2`).
Source of truth for conversations/notebooks/presets is the **server-side DuckDB
store** (`db.py`); the browser persists only sampler settings in localStorage.
Desktop + iPhone Safari are co-primary.

Read `js/page.js` (the `createPage` lifecycle) before touching any page.

### Files (`apps/heylook-frontend-v3/`)

```
index.html
css/app.css                 # design tokens + rationale live in comments here; DESIGN.md is the written form
js/
  app.js                    # bootstrap + hash router + crash-guard error panel
  page.js                   # createPage lifecycle (READ FIRST)
  api.js                    # table-generated endpoint wrappers
  streaming.js              # SSE over /v1/chat/completions (keepalive, reader.cancel, abort-as-completion, 503 retry, mid-stream {"error"})
  settings.js               # sampler store + global display-pref store (buildDisplayPanel/getDisplayPref/onDisplayChange); null = backend-cascade; snapshotSettings()/applySettings()
  settings-drawer.js        # app-shell global slide-over settings drawer; registerSettings(contribution) shared by all pages (sections/sampling/display/extras)
  preset-bar.js             # shared drawer section (createPresetBar): select is inert, Apply is an explicit armed-confirmed copy, live drift line; used by chat + notebook
  markdown.js, utils.js
  vendor/                   # marked.esm.js, purify.es.mjs (only vendored deps)
  pages/  chat.js  notebook.js  models.js  perf.js  explore.js  jspace.js
```

Batch was dropped from v3 scope on purpose (spec §6); the backend endpoint remains.

## Status snapshot (grading is authoritative in CURRENT.md §1)

**Done / SOLID**: chat (conversations CRUD, streaming w/ thinking blocks,
edit/regenerate/delete via position truncation, stop=partial saved, status
telemetry line, mobile drawer); shared layer (page.js, hash router, api.js,
streaming.js, settings cascade); notebook, models, perf, explore; **images in
chat** (attach incl. iPhone camera roll + paste, thumbnail strip, rendered from
the content-block store, v1.34.20); **per-conversation system prompt + saved
presets** (v1.34.22); **DRY shared settings drawer** (2026-07-11: chat settings
extracted into an app-shell global slide-over shared by all 6 pages --
sampling / global display prefs / per-page extras; `js/settings-drawer.js`;
code-reviewed); browser E2E in `tests/e2e/` (66 checks, drawer-driven; green bar
the load-sensitive streaming-cadence guard); **composer icons + multi-image
hardening** (2026-07-20, v1.34.60-.61: attach + thinking-toggle are now
`.btn--icon` buttons styled off `aria-pressed`, multi-image attach capped at 8
with an aria-live announcement + per-image "Remove image N" labels); **vision
token budget** (2026-07-20, v1.34.64: `vision_tokens` drawer control,
cap-gated on the model's `vision` capability, mapped server-side by
duck-typing the loaded processor -- gemma-4 buckets / qwen pixel budget);
**thinking now actually works broadly** (2026-07-20, v1.34.60-.64: gemma-4's
canonical template thinking channel and Qwen3.5's prefilled-`<think>` template
both split into the collapsible thinking block correctly; the checkbox/icon
auto-appear from template detection, no `models.toml` flag needed).

**Done (was "Left")**:
- **DONE -- visual design (2026-07-11)**: the impeccable audit + polish pass ran
  across all 6 pages + shell + drawer (slop-clean, scored 17/20). Fixed a mobile +
  a11y cluster -- notably **delete/rename were unreachable on iPhone** (hover-gated,
  no touch fallback) -- plus aria-live status, `<label for>` association, a real
  drawer focus-trap, and the mobile settings gear (FAB -> bottom-nav item; a FAB
  collided with chat's Send). The load-bearing a11y/mobile-parity rules new UI must
  honor are `apps/heylook-frontend-v3/DESIGN.md` §7. iPhone-17-Pro verified via
  viewport + touch-media emulation (19/19), not a real device. (plan Phase 4 item 2)

**Left**:
- **NOT DONE -- cutover**: retiring v2 & promoting v3 is deliberately open until
  the owner has lived in `/v3` daily. Nothing blocks it. (plan Phase 3; the older
  legacy React app was already deleted in v1.34.25)
- Small backlog: `show_special_tokens` render-consumer wiring (pref exists but
  gated `wired:false` until a surface honors it); `enable_thinking` tri-state.
  (TODO.md) (The "panel drifted from preset" indicator shipped in v1.39.2 --
  live drift line + explicit armed Apply, selection inert, apply a copy -- and
  v1.39.3 extracted the bar to the shared `preset-bar.js` and gave notebook
  the same section. v1.39.1 fixed a data-loss bug where the old blur-only
  commit lost the chat system prompt when the drawer closed under focus
  -- v1.39.4 generalized the fix, blurring the focused field on drawer
  close for every commit-on-change field, not just the sysprompt textarea.)

## Backend <-> v3 coupling (the "tightly coupled" part)

### Endpoints each page consumes today

| Page | Endpoints |
|------|-----------|
| chat | `/v1/conversations` CRUD, `/v1/chat/completions` (SSE), `/v1/presets` CRUD |
| notebook | `/v1/notebooks` CRUD, `/v1/chat/completions` |
| models | `/v1/models`, `/v1/capabilities`, `/v1/admin/models` (+ `/import`, `/scan`, load/unload) |
| perf | `/v1/performance/profile/`, `/v1/system/metrics` |
| explore | `/v1/chat/completions` **with logprobs** |
| jspace | `/v1/jspace/models`, `/v1/jspace/analyze` (Jacobian-lens workspace read-out) |
| shared | `/v1/data/clear` (danger zone; presets are EXCLUDED from it -- config, not data) |

Chat/notebook/explore stream over the **OpenAI wire** (`/v1/chat/completions`)
today. `/v1/messages` exists and emits the right SSE grammar but v3 does not use
it yet (the reference in `chat.js:799` is a migration marker).

### Load-bearing contracts that MUST survive any backend change (plan guardrails)

1. **Logprobs** (Token Explorer / explore.js) -- response via logprob fields on
   the stream. Messages spec has no logprobs -> they ship as namespaced
   extensions.
2. **Streaming telemetry** (status line + perf) -- timing/KV fields ride the
   usage chunk (needs `stream_options.include_usage=true`).
3. **Sampler cascade** -- v3 sends only non-null keys; **null = backend cascade**.
   Don't make the backend require fields v3 omits.
4. **Server-side persistence** is a product pillar (what makes iPhone+desktop
   co-primary work; position-based truncation builds on it). The DuckDB store is
   that pillar.

### Backend work v3 still needs (sourced from the plan)

- **Messages-API migration (plan Phase 3b)** -- chat/notebook/explore move from
  `/v1/chat/completions` to `/v1/messages`. **Order-critical**: port the logprobs
  collector + thinking-parser wiring + telemetry plumbing onto the Messages
  translator **BEFORE** the completions bridge dies, or Explore breaks (guardrail
  #1). Update spec §4 in the same commits.
- **Native image content blocks (plan Phase 4 item 5)** -- v3 currently converts
  stored content blocks -> `image_url` on the OpenAI wire (marked in `chat.js`).
  Swap for native Messages image blocks when 3b lands (one function). The store
  already persists the Anthropic-spec nested image shape. An E2E image check and
  the Q8 server-side upload-resize spike are pending.
- **`enable_thinking` tri-state (auto/on/off)** -- a contract change deliberately
  deferred to 3b's extension design so it's designed once with the Messages
  `thinking` mapping (guardrail #3).
- **Perf-page analytics (Q5 analytics half)** -- DuckDB-querying-JSONL so the perf
  page gets real analytics; still planned (the app-state half of Q5 shipped in
  v1.34.20).
- **Radix single-slot (Q7)** has a named v3 UX cost: sidebar conversation switches
  re-prefill (TTFT on big models). Decided knowingly; measure switch frequency if
  it hurts.
- **Model-management page promotion (plan Phase 6)** -- the `models` page today
  lists + loads/unloads via `/v1/admin/models`. Phase 6's redesign
  (registry-over-scan, add-by-path-anywhere, non-clobbering toml merge) makes
  this page the real model-management UI: add-by-path, edit id/tags/config,
  enable/disable, dedupe, re-scan-as-merge -- so `models.toml` becomes an
  implementation detail, not the interface. Coupled: j-space **lens
  management** (fit/convert + "which models have a lens") lands on the same
  surface, since a lens is another per-model artifact
  (`adapters/jspace/<model_id>/`). Direction captured in the plan; not yet scoped.
- **J-space visualizer enhancements (Phase-5-ish)** -- gate cleared + items 1-2
  shipped v1.34.36-.37 (click-to-pin readout w/ per-cell top-k via the
  `heatmap_top_k` analyze extension; layer-range slider + aggregation view;
  "provisional lens" badge off `/v1/jspace/models` `meta`). Remaining: live
  streaming rows (new SSE analyze endpoint) -> steer/swap/ablate interventions
  (needs real backend; last). Detail: `docs/jspace_integration_plan.md` Part 2
  "Frontend visualizer" + the progress note atop `docs/jspace_visualizer_handoff.md`.

## Verifying v3 in a browser

puppeteer-core + system Chrome via `tests/e2e/` (claude-in-chrome refuses
localhost by policy). `cd tests/e2e && bun install`, then `node run.mjs [chat|pages]`.
MUST run UNSANDBOXED (Chrome profile dir + Metal). See `tests/e2e/README.md`.
