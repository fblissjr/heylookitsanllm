# Frontend v3 -- orientation & backend coupling

> For how the UI BEHAVES from a user's seat -- presets vs ad-hoc settings, the
> generation lifecycle, editing, and the rough edges writing it exposed --
> see [frontend_v3_user_guide.md](./frontend_v3_user_guide.md).

Last updated: 2026-08-26 (v1.79.9-.15. Beyond the painter below: staged
images are capped at 2048px on the longest edge before upload and base64 is
minted at SEND rather than held (`image-prep.js`); chat and notebook share one
document write path (`document-writer.js`) carrying the keepalive PUT-ordering
rule that used to live in two hand-maintained copies; `/v3` assets answer
If-None-Match with a 304 and gzip text (the `no-cache` policy is unchanged --
what changed is that revalidating no longer re-sends the file). BACKEND
COUPLING, the reason this file exists: `GET /v1/conversations` no longer
carries `system_prompt` or `params` (fetch the conversation for either), and
both it and the conversation body now carry `generating` -- a generation
OUTLIVES the response that started it, so a run survives a tab or conversation
switch and the composer has a third state that offers Stop for a run this page
never subscribed to.

The streaming painter is INCREMENTAL. It used
to re-parse the whole accumulated response through marked+DOMPurify and assign
the result to innerHTML on every animation frame -- so the per-frame cost grew
with the response (marked's parse is superlinear in length) and a long
generation saturated the main thread, which on a phone is heat and battery.
`markdown-stream.js` now splits the text at a boundary no markdown construct can
span, renders each segment ONCE into a committed prefix whose nodes are then
left alone, and re-renders only the tail; the painter is rate-limited to ~15/s
instead of one run per frame; and "is the reader at the tail" is measured at the
TOP of the painter, before it mutates, where the reads are cache hits rather
than the forced full-list layout the old per-frame measurement caused (on iOS
the list is never skipped -- `content-visibility` is gone entirely as of
v1.79.18). A
cached flag fed by scroll events was tried first and rejected: pinning coalesces
those events to a handful per generation, so it went stale whenever the viewport
changed underneath it -- a phone keyboard, every time. Notebook's painter got the same rate
limit and explore's thinking box now appends its delta instead of rewriting the
whole string per token. Its growth with response length went from superlinear to
near-linear, which is the durable part; the measured figures live in the
CHANGELOG entry rather than here. Previously 2026-08-17 (v1.72.0-.1): chat attachments grew a third input: drag-and-drop
onto the thread, with a drop overlay whose label names what THIS model accepts.
Picker, paste and drop now funnel through one `addFiles` -> `addPendingFiles`
routine, so paste can stage audio (it was image-only) and every input is gated
once. Paste moved off the composer textarea to the page root -- a paste after
clicking in the thread used to land nowhere -- and reads `clipboardData.files`
with an `items` fallback for Safari. Cap gating moved to STAGING time: a drop or
paste onto a model without the cap refuses loudly and stages nothing; the
send-side block stays, but its case is now only "staged on a capable model, then
switched away". v1.72.1 fixed five review findings on top, one of them the headline
case: paste listened on the chat ROOT, but clicking a message leaves focus on
`document.body` (an ancestor of that root), so it never arrived -- and the
check covering it dispatched at `.chat__messages` and passed anyway. Paste now
listens on `document`; the check clicks for real and dispatches at
`document.activeElement`, asserting that target is outside the chat root. Also
fixed: capability chrome refreshed before `modelSelect.value` moved (so the
overlay described the conversation being LEFT), `preventDefault` before the cap
check (a text+image clipboard payload lost its text at a text-only model), the
pending array orphaned across the FileReader await by a concurrent send, and a
silent discard of drops carrying nothing attachable. Guarded by 36 checks in
`tests/e2e/render.mjs`. Previously 2026-08-11: per-model config editor -- the Models page grew a
schema-driven Configure panel consuming the v1.52-1.53 admin surface --
`GET /v1/admin/model-options` (field type/bounds/enum/default + the six-class
`effect` metadata + `arg`/`ui`/`shape`/`reason` hints) and
`PATCH /v1/admin/models/{id}` (typed values, null = reset-to-default,
`reload_required_fields` in the response). Shared editor module
`js/model-config.js`; grouped by effect class; gguf plumbing
host/port/server_binary/startup_timeout_s deliberately not rendered; non-default
rows carry a mono summary chip. Design + remaining arc (fit meter, editorial
groups, model-switch hardening):
`internal/research/expert_offload_design_frontend.md`. Previously 2026-08-07:
catch-up pass against the backend changes since
2026-07-26 -- v1.45.0-1.49.9. Four fixes, all live-verified: omitting
`enable_thinking` now means OFF on gguf too (it meant ON, so the toggle
lied and thinking could not be turned off at all); the Models page can
scan LOCAL folders, which is the only way to reach the entire GGUF import
arc; scan rows report what the importer actually found (modalities incl.
audio, thinking, a paired drafter and the `--spec-type` it needs) and stop
claiming `vision:false` for everything; Load sends `?warm=true`. Plus:
`/v1/admin/models` now DERIVES capabilities through the shared
`capabilities.py` instead of the always-empty stored override, so the
Models page shows them at all. Previously 2026-07-26: attach button
capability-gated like the
thinking toggle -- vision and/or audio caps drive visibility AND the
picker's accept list; audio attach/render shipped for `audio`-capable
models (gguf); perf trends grew a draft-acceptance column. DECIDED:
v3 keys ALL modality gating off `capabilities` from /v1/models --
`modalities` stays a server-side description and is deliberately unread
here (caps are what the server will actually serve; e.g. an MLX gemma
declares the audio modality but never gets the audio cap). Previously
2026-07-23: system-prompt editor extracted to shared
`prompt-section.js`; preset provenance persisted as `applied_preset_id`;
preset bar shared via `preset-bar.js`, notebook gets the same sections)

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
  settings.js               # sampler store + global display-pref store (buildDisplayPanel/getDisplayPref/setDisplayPref; displayWireFields() = display prefs that ride the WIRE, never the sampler bag -- show_special_tokens); null = backend-cascade; snapshotSettings()/applySettings()
  settings-drawer.js        # app-shell global slide-over settings drawer; registerSettings(contribution) shared by all pages (sections/sampling/display/extras)
  preset-bar.js             # shared drawer section (createPresetBar): select is inert but FOLLOWS the document's applied_preset_id until an explicit pick, Apply and Save are BOTH armed-confirmed (Save overwrites the unrecoverable side -- v1.79.20), read-only preview of the selected preset's own prompt, live drift line; used by chat + notebook
  prompt-section.js         # shared drawer section (createPromptSection): the per-document system-prompt editor -- commits state per keystroke, debounces the PUT, flushes on blur AND on teardown; used by chat + notebook
  markdown.js                 # the ONLY text->HTML path (marked + DOMPurify; raw HTML is SHOWN, never rendered)
  markdown-stream.js          # incremental render for a message still streaming: safe-boundary split, committed prefix, tail-only re-render (+ appendPlainText for thinking boxes)
  image-prep.js               # staging-time resolution cap (EXIF-aware, passes small images through untouched) + blobToBase64, minted at send
  document-writer.js          # the per-document write path shared by chat + notebook: system-prompt PUT chain and the keepalive ordering rule, applied_preset_id stamp
  utils.js                    # createEl/debounce/autoGrow + throttleToFrame (cheap work) and throttleToInterval (painters whose cost scales with the document)
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
code-reviewed); browser E2E in `tests/e2e/` (75 checks, drawer-driven; green bar
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
  close for every commit-on-change field, not just the sysprompt textarea.
  v1.39.6 added the applied-preset chip beside each page's model select --
  name + "(edited)" once drifted, click opens the drawer; provenance is
  session-local: explicit apply/save stamps only, with an exact state match
  labeled by live inference, never stored.)

## Backend <-> v3 coupling (the "tightly coupled" part)

### Endpoints each page consumes today

| Page | Endpoints |
|------|-----------|
| chat | `/v1/conversations` CRUD, **`/v1/conversations/{id}/generate` (Messages SSE -- v1.66.0, the server-side saga; DELETE = Stop)**, `/v1/presets` CRUD, `/v1/admin/models` (residency dots + the cold-send load status; the load-cost CONFIRM was removed v1.62.3 -- cost is disclosed, only loss gates) + `load?warm=true` (the bar's Load button) |
| notebook | `/v1/notebooks` CRUD, `/v1/messages` (v1.74.0, Phase 3b) |
| models | `/v1/models`, `/v1/capabilities`, `/v1/admin/models` (+ `/import`, `/scan` **with local `paths`**, `load?warm=true`/unload, `PATCH /{id}` config edit), `/v1/admin/model-options` (option schema for the Configure panel + row chips) |
| perf | `/v1/performance/profile/`, `/v1/system/metrics` |
| explore | `/v1/messages` **with the `heylook_logprobs` extension** (v1.74.0) |
| jspace | `/v1/jspace/models`, `/v1/jspace/analyze` (Jacobian-lens workspace read-out) |
| shared | `/v1/data/clear` (danger zone; presets are EXCLUDED from it -- config, not data) |

Chat generates over the **conversation-scoped generate endpoint** (Messages
SSE grammar + the `heylook_saved` extension -- v1.66.0, plan Phase 2; the
server builds the request from the store and owns persistence). Notebook and
explore stream `/v1/messages` since v1.74.0 (Phase 3b): logprobs ride the
namespaced `heylook_logprobs` events, timing rides `message_stop.performance`,
and no v3 page speaks `/v1/chat/completions` anymore (the endpoint stays for
external consumers).

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

### Known gap: gguf entries imported before 2026-08-07 under-report (owner action)

`modalities` and `supports_thinking` are STORED on a gguf entry at import time
(unlike MLX, which derives them at load), so an entry written before v1.49.4/.6
carries neither the audio modality nor the thinking flag. `/v1/models` reports
capabilities from those stored values, and v3 gates all modality UI on
capabilities — so those models show no thinking toggle and no audio attach even
though the server serves both. Verified 2026-08-07 on the local fleet: every
gemma-4 GGUF reports `["chat","vision"]` from its entry while a fresh scan of the
same files derives `["text","vision","audio"]` + `supports_thinking: true`.

A stored value is indistinguishable from a deliberate override, so nothing should
silently rewrite it: the remedy is to re-import (or delete those two keys from) the
affected `models.toml` entries. The durable fix is to derive them at load like the
MLX path does, which belongs with the Wave 1 derive-at-load arc rather than bolted
on here.

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
- **Fit meter for the config editor** -- the load-options design names the
  memory readout (weights vs Metal working set vs KV headroom, FAIL-for-MLX /
  WARN-for-gguf asymmetry) as the surface's real payoff; it needs the fit
  endpoint (`ram_report.py`'s `check_fit`/`size_config_gb` behind HTTP --
  backend ask #2 in `internal/research/expert_offload_design_frontend.md` §9),
  which does not exist yet. Do NOT compute fit client-side. The §15
  model-switch arc SHIPPED 2026-08-11 (G1 drop-with-disclosure for history
  media -- block stays for staged attachments, the asymmetry is commented at
  both sites; G2 stop-stream-on-switch; G3 residency dots + pre-switch
  warning with Cancel/Switch-anyway + Load now button; a send with an
  unconfirmed target selected commits the switch). Still open from §15: G4
  context-length estimate, G5 per-message model attribution (waits for a
  `_SCHEMA_VERSION` bump), F14 switch-lock during a pending load.
- **Model-management page promotion (plan Phase 6)** -- the `models` page today
  lists + loads/unloads via `/v1/admin/models`, and (2026-08-11) edits per-model
  config through the schema-driven Configure panel. Phase 6's redesign
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
localhost by policy). `cd tests/e2e && bun install`, then `bun run e2e[:chat|:pages]`.
MUST run UNSANDBOXED (Chrome profile dir + Metal). See `tests/e2e/README.md`.
