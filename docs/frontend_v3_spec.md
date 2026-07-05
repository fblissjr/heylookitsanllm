# Frontend v3 — build spec

Last updated: 2026-07-05

A from-scratch, MUCH-simpler rewrite of `apps/heylook-frontend-v2/`, preserving its functionality
and vision. This doc is the **complete build contract**: a fresh session should be able to build v3
from this file alone, without re-reading the ~10k lines of v2 source. Distilled by parallel extraction
agents (per-page + core + backend + pretext), each claim spot-verified against source.

## Vision (unchanged from v2)

A personal MLX inference-server frontend, served by the FastAPI backend at `/v2` (v3 can reuse that
mount or take a new one). Vanilla JS, **no framework, no bundler, no node_modules, no build step** —
a single `<script type="module">` bootstraps everything. Source of truth for conversations/notebooks is
server-side SQLite; the browser persists only sampler settings in localStorage.

## What "much simpler" means here (decided)

The user's goals, in priority order:
1. **Drop pretext virtualization** — biggest single win (see §1).
2. **Reduce per-page boilerplate** — one shared page-lifecycle layer instead of every page re-implementing
   mount/teardown/state/throttle/guard ceremony (see §3).
3. **Consolidate files** — fewer, tighter modules.
4. **Trim features** — decided (§6): 5 pages (chat, models, notebook, perf-simplified, token explorer);
   batch dropped.
5. **Framework only if it adds significant value** — verdict: **no framework** (see §2).
6. **Impeccable design** — treat visual quality as a first-class requirement, not an afterthought (§7).

v2 is ~9,630 total lines; ~4,900 is vendored (pretext engine, marked, DOMPurify, bidi tables). Hand-written
app code is ~3,700 lines. Dropping pretext deletes **5,429 lines** outright.

---

## 1. Drop pretext (DECIDED — verified safe)

**Delete:** `js/components/pretext_chat_model.js` (906), `js/components/pretext_chat_renderer.js` (338),
all of `js/vendor/pretext/` (4,185), the `.pretext-canvas`/`.line-row`/`.code-line` CSS block (~65 lines),
and the `usePretext` branch scaffolding in chat.js (state fields, `renderMessagesPretext()`, scroll handler,
cursor positioning — ~100+ lines). **Total ≈ 5,400 lines gone.**

**Why it's safe (three independent findings converged):**
- chat.js **already ships a complete legacy render path** — `renderMessagesLegacy()` (chat.js:412) +
  `buildMessageEl()` (:434) — that runs today whenever a message is being edited. `usePretext` (:38) is a
  toggle over two functionally identical paths. Verified: both exist and are wired.
- Everything pretext computes in JS is already delivered natively: **auto-width bubbles** by CSS flexbox
  (`.message-bubble { max-width: 78% }` user / `100%` assistant — verified in css/app.css:208–226);
  **wrapping/RTL/bidi** by the browser's own text engine (pretext's bidi is an admitted "simplified" pdf.js
  port not even wired into its line-breaker).
- The only unique capability is **virtualizing thousands of DOM rows**, irrelevant at this app's scale
  (personal chat: dozens–low-hundreds of turns). Native substitute: `content-visibility: auto` +
  `contain-intrinsic-size` per `.message`. If enormous transcripts ever matter, paginate (fetch last N,
  "load earlier" on scroll-to-top via `IntersectionObserver`) — far simpler than a layout engine.

**Replacement:** standardize on the legacy path. `renderMarkdown()` (marked + DOMPurify, already vendored,
already used by batch.js) → `.message-content` inside `.message-bubble` inside `#chat-messages` with native
`overflow-y: auto`. This *completes a simplification v2 already started*.

Keep vendored `marked.esm.js` and `purify.es.mjs` — they stay.

---

## 2. No framework (DECIDED)

Against the user's "only if it adds significant value" bar, a framework does not clear it here:
- v2 is already reactive-free vanilla JS that works. The pain is **boilerplate**, not lack of a framework —
  and boilerplate is solved by §3's ~120-line shared layer, not by adding React/Vue/Svelte + a bundler
  (which would reintroduce node_modules and a build step the project deliberately avoids).
- Backend serves static files directly; "no build step" is an architectural feature (see v2 CLAUDE.md).

**Verdict: stay vanilla JS + ES modules.** The "value add" the user wants comes from a tiny disciplined
shared layer, not a framework.

---

## 3. Shared layer (the boilerplate killer)

v2's single biggest source of repeated code: **the mount/teardown/state/throttle/beforeunload/stale-guard
pattern is enforced by nothing but convention** — every page hand-reimplements it, and the extraction found
real latent bugs from that (e.g. models.js `handleToggleLoad`/`handleScan`/`handleImport` miss the
`if (!state) return` post-await guard that other handlers have; batch.js init has an unguarded async race).
A rewrite fixes this by making the pattern *the framework*.

### 3a. `createPage(spec)` lifecycle helper (~80 lines) — NEW
One implementation of the lifecycle contract every page needs. Rough shape:

```js
// pages return createPage({ setup(ctx) {...}, teardown(ctx) {...} })
// ctx provides: el (mount root), state (fresh per mount), signal (AbortSignal aborted on teardown),
//   throttle(fn) (auto-reset on teardown), guard(fn) (no-ops after teardown), onTeardown(fn)
```

- `state` is per-mount (replaces module-singleton + `freshState()`), so most `if (!state) return` guards
  vanish — a torn-down page's closures simply hold a dead `ctx`.
- `signal` (an `AbortController` per mount) is passed to `streamChat` and `fetch`; teardown aborts it,
  which kills in-flight streams and polls without manual controller bookkeeping.
- `throttle(fn)` wraps `throttleToFrame` and auto-nulls on teardown (kills the "null the throttle var, not
  just .reset()" footgun).
This single layer removes the bulk of v2's per-page ceremony and the class of teardown-race bugs.
(No page in v3 polls on an interval — perf is on-demand — so no shared polling helper is needed. `signal`
covers in-flight fetch/stream cancellation on teardown.)

### 3b. Router (`app.js`, ~40 lines) — keep, thin
Hash routes → dynamic `import()` of page modules (keeps code-splitting, no bundler). Lifecycle: teardown
previous page → `replaceChildren()` on `#main` → import route module → `mount(el, name)`. Set
`app.dataset.page = name`, toggle `.nav-item--active`. Fold the nav-active + `data-page` bookkeeping into
route registration rather than post-hoc DOM queries.

### 3c. API layer (`api.js`, ~50 lines) — keep, generate from a table
Core `request(method, path, body)`: builds `X-Request-ID` (`crypto.randomUUID` + fallback), sets
`Content-Type`, JSON-stringifies body, throws `Error(detail)` on non-ok (parse `{detail}`, fall back to
`statusText`). Generate the ~20 wrappers from a route table instead of hand-writing each. **Must also cover
the 3 endpoints v2's api.js omits** (they're called ad hoc today): streaming chat, batch, perf profile —
though streaming stays in `streaming.js`, not `api.js`.

### 3d. Streaming (`streaming.js`, ~110 lines) — keep as-is conceptually
SSE via `fetch` + `ReadableStream` (not `EventSource`). Callbacks: `onToken` (delta.content), `onThinking`
(delta.thinking), `onLogprobs` (delta logprobs — explore only), `onComplete`, `onError`. Adds
`stream: true` + `stream_options: { include_usage: true }`. **Critical gotchas to preserve (verified against
backend):**
- Treat `AbortError` as normal completion (call `onComplete` with partial content).
- Call `reader.cancel()` in the abort/catch path — without it, GC timing causes "Failed to fetch" on the
  next request (documented v2 bug).
- **Ignore SSE comment lines** (`: keepalive\n\n`, emitted every 5s during long prefill) — do not parse as
  data. (v2 may not handle this; v3 must.)
- Usage/timing only arrives if `include_usage: true` was sent; stream always ends with `data: [DONE]`.

### 3e. Settings (`settings.js` + panel, ~200 lines) — keep contract, data-drive the panel
- 10 sampler keys, all default `null` = "use backend cascade" (global → thinking → models.toml → request).
  `samplerParams()` copies only non-null keys (extra: `top_k` requires `>0`, `presence_penalty` requires
  `>0`), so omitted keys respect the backend cascade. **Preserve this exactly** — it's a real integration
  contract, not cosmetics.
- localStorage key `heylook-v2-settings` (pick a v3 key or keep for continuity), 300ms debounced write,
  synchronous in-memory cache.
- `PARAM_META` drives the panel (label/min/max/step/section/type). Data-drive control rendering off it +
  a generic "bind control to settings key" helper instead of v2's per-control `createEl` chains +
  per-param null-branching + full-panel-rebuild-on-reset.
- Params: temperature, max_tokens, top_p, top_k (core); min_p, repetition_penalty, repetition_context_size,
  presence_penalty, seed (advanced); enable_thinking (advanced checkbox, shown only if model caps include
  `thinking`; unchecking sets `null` not `false`).

### 3f. utils / markdown — keep
- `utils.js`: `createEl`, `beforeUnloadGuard`, `throttleToFrame`, `statCard`, `formatBytes`. (v3 folds
  beforeUnloadGuard + throttle handling into `createPage`, but keep the primitives.)
- `markdown.js`: `renderMarkdown(text)` = `marked.parse` → `DOMPurify.sanitize`, falls back to
  `escapeHtml` on parse error. **The only sanctioned path for model/user text → HTML.** Never double-sanitize.
  `marked.use()` (not `setOptions`, removed in marked v5+). `ensureMarked()` is now a no-op — drop it.

---

## 4. Backend API contract (integration surface — verified against source)

Base: same-origin. Two independent, both-opt-in auth gates (loopback exempt from the API-key gate by
default): `HEYLOOK_API_KEY` (`Authorization: Bearer`) on inference; `HEYLOOK_ADMIN_TOKEN`
(`X-Heylook-Admin-Token`) on admin + `/v1/data/clear`. Conversations/notebooks/models-list/metrics/profile
are unauthenticated.

**Chat completions** `POST /v1/chat/completions` (streaming SSE):
- Body: `{ model, messages:[{role,content,thinking?}], ...samplerParams(), stream:true,
  stream_options:{include_usage:true}, logprobs?, top_logprobs? }`, header `X-Request-ID`.
- SSE chunks: `choices[0].delta.content` | `.delta.thinking` | `.logprobs.content`
  (`[{token,logprob,top_logprobs:[{token,logprob}]}]`). Final usage chunk (only if `include_usage`):
  `{ usage:{prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details?},
  timing:{total_duration_ms, peak_memory_gb?, kv_cache_bytes?, queue_wait_ms?, ...},
  generation_config?, stop_reason }`. Terminator `data: [DONE]`.
- **503 backpressure**: `{error:{code:"model_overloaded"}}` + `Retry-After`, `X-RateLimit-*` headers — v3
  should surface this as a friendly "server busy, retrying" state, not a raw error.
- **Trap**: setting `processing_mode` (≠ "conversation") switches the response to a *different* schema
  (`chat.completion.batch`). v3 chat should NOT send `processing_mode` — ignore this path.

**Batch** `POST /v1/batch/chat/completions`: body `{ requests: ChatRequest[] (min 2) }`; all `requests[].model`
must be identical (else 400), none may set `stream` (else 400). Response `{ data: ChatCompletionResponse[],
batch_stats:{total_requests, elapsed_seconds, throughput_tok_per_sec, memory_peak_mb, ...} }`.

**Conversations** (prefix `/v1/conversations`, no auth):
- `GET /` → `{conversations:[{id,title,model_id,system_prompt,created_at,updated_at}], total}` — **no messages**.
- `POST /` (201) `{title,model_id?,system_prompt?}` → full conv incl `messages:[]`.
- `GET /{id}` → conv **with** `messages:[{id,role,content,thinking,position,...}]` ordered by position.
- `PUT /{id}` `{title?,model_id?,system_prompt?}` (only set fields patched; empty→400) → updated conv
  **without messages** (asymmetric — keep your in-memory messages, don't trust PUT to return them).
- `DELETE /{id}` → `{status:"deleted",id}`.
- `POST /{id}/messages` (201) `{role,content,thinking?}` → message.
- `PUT /{id}/messages/{msgId}` `{content?,thinking?}` → message.
- `DELETE /{id}/messages?after={pos}` → deletes `position > pos` (position-based truncation drives
  regenerate/edit-regenerate/delete-cascade).

**Notebooks** (prefix `/v1/notebooks`, no auth): `GET /` list **omits content**; `GET /{id}` full;
`POST /` `{title,content,system_prompt?,model_id?}`; `PUT /{id}` partial (incl content); `DELETE /{id}`.

**Admin models** (`X-Heylook-Admin-Token`): `GET /v1/admin/models` →
`{models:[{id,provider,description?,tags,enabled,capabilities,config,loaded}], total}`;
`POST /{id}/load` → `{status:"loaded",model_id}` (500 on failure); `POST /{id}/unload` →
`{status:"unloaded"|"not_loaded"}` (never errors); `POST /scan` `{paths?:[], scan_hf_cache:bool}` →
`{models:[{id,path,provider,size_gb,vision,quantization?,already_configured,tags,description}], total}`;
`POST /import` `{models:[{id,path,provider}], profile?}` → `{imported:[...], total, warning?}`.
(Backend also exposes toggle/status/validate/profiles/bulk-profile/discovered — out of scope unless a
trimmed feature needs them.)

**Models list** `GET /v1/models` → `{data:[{id,provider?,capabilities?}]}` (enabled models only).
**Metrics** `GET /v1/system/metrics?force_refresh?` → `{system:{ram_used_gb,ram_available_gb,ram_total_gb,
cpu_percent}, models:{[id]:{memory_mb,context_used,context_capacity,context_percent,requests_active,
requests_queued}}}` (30s server cache).
**Perf profile** `GET /v1/performance/profile/{1h|6h|24h|7d}` → `{timing_breakdown:[{operation,avg_time_ms,
count,percentage}], trends:[{hour,response_time_ms,tokens_per_second,requests}], resource_timeline,
bottlenecks}` (in-memory ring buffer, lost on restart; 503/empty if analytics extra not installed).
**Clear** `POST /v1/data/clear` (admin) → `{conversations_deleted, notebooks_deleted}`.
**Capabilities** `GET /v1/capabilities`.

---

## 5. Per-page functional specs (essentials + must-preserve gotchas)

Each page = `createPage({...})`. Rendering: prefer targeted updates over v2's rebuild-everything-on-every-
state-change; but full-rebuild is acceptable where it keeps code simpler (these are small pages).

### chat (v2: 815 lines → target far less once pretext + dual-path branching go)
Conversations sidebar (list/new/select/rename-on-dblclick/delete-with-confirm); model `<select>` (persists
`model_id` to conv, fire-and-forget); gear → sampler panel scoped to model caps. Send (Enter w/o Shift;
auto-create conv if none, title = first 50 chars of first message); textarea auto-grow capped 200px. Stream
assistant response (RAF-throttled DOM writes; blinking cursor; collapsible "Thinking" block above content,
auto-open while streaming, present only if thinking text arrives). Stop (abort = normal completion, partial
saved). On complete: persist assistant message, status line `"N tokens · X.XX GB peak · Y KV"` from
usage/timing. Per-message actions: Edit (inline textarea → Save / Save&Regenerate / Cancel), Regenerate
(assistant only), Delete (confirm), Copy. Edit/regenerate/delete use **position-based truncation**
(`DELETE .../messages?after=position-1`). Near-bottom-aware auto-scroll (only if within 100px, or forced on
send/switch/stream-start). Stale-response guard: capture `targetConvId` at stream start, bail in callbacks if
`activeId` changed. beforeunload during stream. Remove the unused `bus.js` import (dead code in v2).

### notebook (v2: 341 lines) — plain-text, no pretext, no markdown render
Multi-doc: sidebar list (new/select/delete-confirm) + editor (title input, model select, collapsible system-
prompt textarea, large auto-grow content textarea, Generate/Stop). Debounced auto-save 500ms (reads fields
live from DOM in v2 → v3 should hold them in state and drop the querySelectors). Generate: split content at
cursor, messages = optional system + user(text-before-cursor or "Continue writing."), stream tokens inserted
**at cursor** preserving text after + caret, auto-resize. flushSave on select/teardown (don't lose edits).
Content is `textarea.value` only — never innerHTML (XSS-safe by construction).

### models (v2: 250 lines) — admin
List model cards (Loaded/Idle badge, provider/capability/disabled tags, description). Per-model Load/Unload
(per-model busy flag — keep per-row granularity; **fix**: make import per-row too, v2 has a global import
flag asymmetry). Scan (`scan_hf_cache:true`) → results panel (filter `already_configured`, Import each →
removes from panel + reloads list). "Clear all conversations & notebooks" with `confirm()`. **Fix v2's
inconsistent error UX**: pick ONE surfacing strategy (v2 shows errors only on initial list load, silently
console-logs scan/load/import/clear failures). **Fix**: add the missing `if (!ctx) return` post-await guards
(free via `createPage`). Custom scan `paths` UI is a v2 gap — leave out of scope unless requested.

### batch — DROPPED from v3 scope
Not included (user decision). The backend endpoint (`/v1/batch/chat/completions`) stays; if batch is wanted
later it's a self-contained ~150-line page against the §4 batch contract.

### perf (v2: 212 lines) — dashboard, SIMPLIFIED, no constant polling
Single-user tool — keep it very simple. Fetch `GET /v1/system/metrics` **on mount + a manual "Refresh"
button only; NO automatic interval** (the backend caches metrics 30s, so live polling bought nothing).
Show RAM/CPU stat cards + per-loaded-model cards (memory MB, context used/capacity, requests_active).
Time-range selector (1h/6h/24h/7d) fetches `GET /v1/performance/profile/{range}` on demand:
timing_breakdown table + last-8 trends. Graceful empty state if analytics extra absent (profile null/503).
Build stable containers once and update values in place — no full-rebuild-per-refresh. This is the smallest
honest version of the page; do not add auto-refresh, websockets, or live charts.

### explore / token explorer (v2: 378 lines) — power-user
Model select, prompt, stream with `logprobs:true, top_logprobs:5`. Render each content token as an inline
chip colored by `exp(logprob)` (probability→HSL red→green via `probabilityToColor`); whitespace shown as
visible glyphs (·, ↵, →). Click a chip → detail panel: `logprob | prob%`, position, ranked bar chart of
top-5 alternatives. Keyboard nav (←/→ move selection, Esc clears). Thinking tokens in a separate section
(no color). **Simplify**: replace v2's dual incremental-vs-full render with a single RAF-throttled full
re-render; scope the keydown listener to the page container (v2 attaches to `document`). This is the only
page exposing logprobs — see trim question §6.

---

## 6. Feature scope (DECIDED)

v3 ships **5 pages**: chat, models, notebook, perf (simplified), token explorer. **Batch is dropped.**

| Page | v2 lines | Verdict | Notes |
|------|---------|---------|-------|
| chat | 815 | **Keep — core** | The product. Legacy render path only. |
| models | 250 | **Keep — core** | Load/unload/scan/import + clear-data. |
| notebook | 341 | **Keep** | Plain-text, no pretext. |
| perf | 212 | **Keep — simplified** | On-demand only, no polling. See §5. |
| explore | 378 | **Keep** | Only logprobs surface; simplify render per §5. |
| batch | 219 | **DROP** | Endpoint stays; page not built. |

---

## 7. Design quality ("impeccable")

Treat as a requirement. Before writing UI, consult the `frontend-design` skill for intentional typography,
spacing, and a non-templated aesthetic; and `dataviz` for the perf dashboard + explore logprob visualization
(color scales, stat tiles). Keep the single `css/app.css`, light/dark-consistent, mobile responsive
(@768px). Replace blocking native `confirm()`/`alert()` (v2 uses them in models/chat/notebook) with in-app
confirm affordances where it doesn't cost much — but keep a confirm step for destructive actions per repo rule.

---

## 8. Build sequence (for the build session)

1. Scaffold: `index.html` shell (base href, `#sidebar`/`#main`/`#bottom-nav`), `css/app.css` (port v2 minus
   `.pretext-*`), vendored `marked` + `purify`.
2. Shared layer: `createPage` + `pollWhileMounted`, router, `api.js` (table-generated), `streaming.js`
   (with keepalive-comment + reader.cancel + abort-as-complete), `settings.js` + data-driven panel,
   `utils.js`, `markdown.js`.
3. chat page (legacy render path only) — highest risk, do first; verify streaming, edit/regenerate,
   position truncation, stop/abort, thinking blocks.
4. models, notebook, perf (simplified, no polling), token explorer. (No batch page — §6.)
5. Verify against a running backend (`/run` or the `verify` skill): drive each flow, watch the network tab
   for the exact payloads in §4, confirm 503 backpressure + abort behave.

## 9. Delegation note (how this spec was built cheaply)

Extraction was fanned out to parallel sonnet-tier subagents (one per page + core + backend + pretext), each
returning a fixed-schema distillation so the Opus main loop ingested ~1.5k lines of specs instead of ~10k
lines of source. Synthesis + the architecture decisions (drop pretext, no framework, trim) stayed in the
main loop; every load-bearing claim was spot-verified against source before landing here. The base
model-delegation rule now lives at `.claude/rules/model-delegation.md` so future sessions auto-route
mechanical work down-tier. The build itself (step 8) is well-specified enough to hand to a sonnet task-coder
per page, with main-loop verification of each returned page.
