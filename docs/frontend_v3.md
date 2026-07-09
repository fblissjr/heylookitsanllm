# Frontend v3 -- orientation & backend coupling

Last updated: 2026-07-09 (v1.34.25)

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
css/app.css                 # all design tokens + rationale live in comments here (no DESIGN.md yet)
js/
  app.js                    # bootstrap + hash router + crash-guard error panel
  page.js                   # createPage lifecycle (READ FIRST)
  api.js                    # table-generated endpoint wrappers
  streaming.js              # SSE over /v1/chat/completions (keepalive, reader.cancel, abort-as-completion, 503 retry, mid-stream {"error"})
  settings.js               # sampler panel; null = backend-cascade; snapshotSettings()/applySettings(); `lead` hook on buildSettingsPanel
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
presets** (v1.34.22); browser E2E in `tests/e2e/` (55 checks live-green).

**Left**:
- **UNCERTAIN -- visual design**: impeccable design gates never ran; no
  `DESIGN.md` seeded; tokens/rationale live only in `css/app.css` comments. iPhone
  Safari checked via viewport emulation, not a real device. (plan Phase 4 item 2)
- **NOT DONE -- cutover**: retiring v2 & promoting v3 is deliberately open until
  the owner has lived in `/v3` daily. Nothing blocks it. (plan Phase 3; the older
  legacy React app was already deleted in v1.34.25)
- Small backlog: notebook preset bar; "panel drifted from preset" indicator;
  `enable_thinking` tri-state. (TODO.md)

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

## Verifying v3 in a browser

puppeteer-core + system Chrome via `tests/e2e/` (claude-in-chrome refuses
localhost by policy). `cd tests/e2e && bun install`, then `node run.mjs [chat|pages]`.
MUST run UNSANDBOXED (Chrome profile dir + Metal). See `tests/e2e/README.md`.
