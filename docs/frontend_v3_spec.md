# Frontend v3 — build spec

Last updated: 2026-07-26 (§4: model-routing failure is now 400 not 500 on
chat/completions + messages; startup no longer pre-warms default_model)

> **STATUS: BUILT.** v3 shipped at `/v3` (v1.31.0) and was verified end-to-end;
> graded done/not-done status lives in `docs/project/CURRENT.md`. This doc
> remains the contract of record: §4 is kept current as the backend moves
> (last contract updates 2026-07-06, marked "v1.31.1+" / "v1.32.0" below).

A from-scratch, MUCH-simpler rewrite of `apps/heylook-frontend-v2/`, preserving its functionality
and vision. This doc is the **complete build contract**: a fresh session should be able to build v3
from this file alone, without re-reading the ~10k lines of v2 source. Distilled by parallel extraction
agents (per-page + core + backend + pretext), each claim spot-verified against source.

## Vision (unchanged from v2)

A personal MLX inference-server frontend, served by the FastAPI backend at `/v2` (v3 can reuse that
mount or take a new one). Vanilla JS, **no framework, no bundler, no node_modules, no build step** —
a single `<script type="module">` bootstraps everything. Source of truth for conversations/notebooks is
server-side SQLite; the browser persists only sampler settings in localStorage.

### Where v3 lives (recommended default)

Build in a **new sibling directory**, `apps/heylook-frontend-v3/`, served at a **new mount, `/v3`**,
leaving `/v2` and `apps/heylook-frontend-v2/` untouched until cutover. This is reversible and lets both
run side by side during the rewrite. The `/v2` mount in `src/heylook_llm/api.py` (~line 199-213) is a
~15-line block: `GET /v2` + `GET /v2/{rest:path}` resolve against a static dir and `FileResponse` it,
falling back to `index.html` for SPA routing, with path-traversal protection via
`resolved.is_relative_to(_v2_frontend_dir)`. Duplicate this verbatim for `/v3` pointing at the new dir —
do not generalize/parameterize the existing `/v2` block to also serve v3; two small explicit blocks are
simpler than one parameterized one for a two-mount case. Update `openapi_tags`/description only if v3
introduces new endpoint groups (it doesn't — it reuses the existing backend contract in §4 as-is).
Cut over (retire v2, promote v3 to `/v2` or rename the mount) as a separate, later decision once v3 is
verified end-to-end.

## What "much simpler" means here (decided)

The user's goals, in priority order:
1. **Drop pretext virtualization** — biggest single win (see §1).
2. **Reduce per-page boilerplate** — one shared page-lifecycle layer instead of every page re-implementing
   mount/teardown/state/throttle/guard ceremony (see §3).
3. **Consolidate files** — fewer, tighter modules.
4. **Trim features** — decided (§6): 5 pages (chat, models, notebook, perf-simplified, token explorer);
   batch dropped.
5. **Framework only if it adds significant value** — verdict: **no framework** (see §2).
6. **Impeccable design** — all UI goes through the impeccable plugin's workflow and quality gates (§7);
   project design context lives in root `PRODUCT.md`.

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
  `thinking`; unchecking sets `null` not `false`); vision_tokens (advanced number, shown only if model caps
  include `vision`; target visual tokens per image, backend snaps to the model's own processor support —
  gemma-4 buckets 70/140/280/560/1120, qwen continuous pixel budget; null = processor default).

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
  stream_options:{include_usage:true}, logprobs?, top_logprobs?,
  continue_final_message? }`, header `X-Request-ID`.
- **Continuation / prefill (`continue_final_message`, added v1.61.0):**
  finish the FINAL message instead of opening a new assistant turn. Absent =
  auto: a trailing assistant message is continued (both providers; before
  v1.61.0 this convention only suppressed the generation prompt on MLX and
  the turn still rendered CLOSED -- it did not actually continue). `true` =
  continue whatever the final message is, ANY role -- user-role continuation
  is MLX-only (llama-server prefills assistant turns natively but has no
  user-turn spelling; gguf 400s). `false` = never continue (MLX opens a
  fresh assistant turn; gguf 400s on a trailing assistant message, which
  llama-server always continues). The response carries ONLY the continuation
  text on every provider (llama-server's prefill echo is stripped
  server-side; an all-text parts-list prefill is flattened with the same
  ' '-join rule as MLX and works -- v1.61.1). Not supported with image
  history or diffusion models (400). STREAMING caveat (v1.61.1): these
  refusals fire after HTTP 200 + headers have flushed, so they surface as an
  in-band SSE error frame typed `invalid_request_error` (code
  `invalid_request`; on `/v1/messages` an Anthropic error event of the same
  type) -- never `server_error`. Clients should treat that frame as a 400.
  v3's Save & Continue on the message editor uses this: PUT the edit,
  truncate after, stream with `continue_final_message: true`, append into
  the SAME message row (both roles; the button hides for user messages on
  gguf models).
- SSE chunks: `choices[0].delta.content` | `.delta.thinking` | `.logprobs.content`
  (`[{token,logprob,top_logprobs:[{token,logprob}]}]`). Final usage chunk (only if `include_usage`):
  `{ usage:{prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details?},
  timing:{total_duration_ms, peak_memory_gb?, kv_cache_bytes?, queue_wait_ms?,
  draft_tokens?, draft_accepted?, draft_acceptance?, ...},
  generation_config?, stop_reason }` (the `draft_*` trio appears only when
  speculative decoding was active; v3 renders `draft_acceptance` in the
  post-stream stats line). Terminator `data: [DONE]`.
- **503 backpressure**: `{error:{code:"model_overloaded"}}` + `Retry-After`, `X-RateLimit-*` headers — v3
  should surface this as a friendly "server busy, retrying" state, not a raw error. (As built: the bounded
  retry lives in `streaming.js` itself with an `onRetryWait(seconds, attempt)` callback; pages only render
  the status line.)
- **Mid-stream generation failure (v1.31.1+)**: the backend emits `data: {"error":{"message":..., "type":
  "server_error", "code":"generation_failed"}}` followed by `data: [DONE]` — never as a content delta.
  Non-streaming requests get HTTP 500 with the message in `detail`; the Messages API emits `event: error`.
  `streaming.js` converts the payload to a thrown error routed to `onError`. Clients must never render
  `error.message` as assistant content.
- **Model routing failure = 400 (v1.44.5)**: an unknown/disabled `model`, or omitting `model` when the
  server has no `default_model` configured, returns HTTP 400 with the reason + the available ids in
  `detail` — on both `/v1/chat/completions` and `/v1/messages`. It was a 500 through v1.44.4. A failed
  model *load* (corrupt weights, unsupported architecture) is still a 500: 400 means "pick a different
  model", 500 means "this model is broken". Note `/v1/hidden_states` and `/v1/embeddings` have not been
  converted and still answer 500 for an unknown id.
- **Startup loads nothing (v1.44.4)**: the server no longer pre-warms `default_model` at boot (only an
  explicit `--model-id` does), so the first request to any model pays the load. v3 must not assume a
  model is resident — the models page reflects real state.
- **Server-side defaults (v1.32.0)**: when the request, its preset, and the model config are all silent,
  the effective sampler floor is `temperature 0.7, max_tokens 4096` (was 0.1/512), and imported models
  carry `default_sampler = "balanced"`. The UI's null-means-cascade settings contract is unchanged.
- **Omitting `enable_thinking` means OFF, on every engine (2026-08-07)**. v3's toggle is
  binary — it sends `true` or omits the key, never `false` — so what "omitted" resolves to
  IS the off state. MLX always resolved it to an explicit `False`; gguf only sent
  `chat_template_kwargs` for a non-None value, so an omitted key handed llama-server's
  `--jinja` run to the GGUF's own template default — thinking ON for gemma-4 / Qwen3.6 /
  DeepSeek-V4. One checkbox, opposite meanings per engine, and no way at all to turn
  thinking off on a gguf model. The shared cascade
  (`samplers.resolve_effective_sampling`) now materializes the effective switch
  unconditionally instead of only when the thinking overlay fires, so both providers
  send an explicit bool. A tri-state (auto/on/off) that makes the template default
  reachable again is still the deferred Phase-3b design item.
- **Trap**: setting `processing_mode` (≠ "conversation") switches the response to a *different* schema
  (`chat.completion.batch`). v3 chat should NOT send `processing_mode` — ignore this path.

**Batch** `POST /v1/batch/chat/completions`: body `{ requests: ChatRequest[] (min 2) }`; all `requests[].model`
must be identical (else 400), none may set `stream` (else 400). Response `{ data: ChatCompletionResponse[],
batch_stats:{total_requests, elapsed_seconds, throughput_tok_per_sec, memory_peak_mb, ...} }`.

**Conversations** (prefix `/v1/conversations`, no auth):
- `GET /` → `{conversations:[{id,title,model_id,system_prompt,params,applied_preset_id,created_at,updated_at}], total}` — **no messages**.
- `POST /` (201) `{title,model_id?,system_prompt?,params?,applied_preset_id?}` → full conv incl `messages:[]`.
  `applied_preset_id` at create (added v1.59.0) is the new-document preset
  inheritance: the document explicitly STARTS as that preset, which counts as
  an apply under the stamp rules below.
- `GET /{id}` → conv **with** `messages:[{id,role,content,thinking,position,...}]` ordered by position.
- `PUT /{id}` `{title?,model_id?,system_prompt?,params?,applied_preset_id?}` (only set fields patched; empty→400) → updated conv
  **without messages** (asymmetric — keep your in-memory messages, don't trust PUT to return them).
- `DELETE /{id}` → `{status:"deleted",id}`.
- `POST /{id}/messages` (201) `{role,content,thinking?}` → message.
- `PUT /{id}/messages/{msgId}` `{content?,thinking?}` → message.
- **Content blocks (added v1.34.20, DuckDB store):** `content` accepts a plain
  string OR a Messages-style block list, e.g.
  `[{type:"image",source:{type:"base64",media_type,data}},{type:"text",text}]`.
  Every message response carries BOTH `content` (flattened text of the text
  blocks — back-compatible; render targets that only know strings keep
  working) and `content_blocks` (the full stored list — the image-rendering
  source of truth). Strings normalize to one text block. For generation, v3
  converts stored image blocks to OpenAI `image_url` parts (data URLs) until
  the Messages-first migration makes the stored blocks the wire shape.
- **Audio input (added v1.42.0, gguf models only):** `/v1/chat/completions`
  accepts `{type:"input_audio", input_audio:{data:<RAW base64, NOT a data
  URL>, format?:"wav"|"mp3"|...}}` content parts (or `url` instead of
  `data`); the Messages API accepts the block form `{type:"audio",
  source_type:"base64"|"url", media_type?, data?|url?}`. Only models with
  the `audio` capability (gguf/llama-server) serve it — MLX models return
  400. The v3 attach affordance is capability-gated on `vision`/`audio`
  (shipped v1.43.0: gated attach button + dynamic accept list, audio chips
  in the strip, `<audio controls>` rendering, stored `audio` blocks
  converted to `input_audio` parts at send).
- `DELETE /{id}/messages?after={pos}` → deletes `position > pos` (position-based truncation drives
  regenerate/edit-regenerate/delete-cascade — LEGACY once the client moves to
  `/generate` below; the CRUD stays for edits/deletes without generation).

**Conversation-scoped generation (added v1.65.0 — plan_chat_orchestration.md
Phase 1; the server-side saga that replaces the client-orchestrated
truncate→stream→persist sequences):**
- `POST /{id}/generate` `{mode:"append"|"regenerate"|"continue",
  message_id?, user_content?, overrides?}` → SSE stream.
  - The server builds the provider request FROM THE STORE: the conversation's
    `system_prompt`, `params` (sampler bag, cap-gated keys dropped for the
    target model), `model_id`, and message rows (media blocks the model can't
    take are dropped at the wire and counted — the server-side twin of the
    client's toWireContent).
  - `append`: optional `user_content` (string or block list) persists as a
    new user turn first, then a fresh assistant turn generates.
  - `regenerate`: `message_id` anchors — everything from the anchor onward is
    excluded from the prompt, and the truncation COMMITS atomically with the
    replacement row only. A failed/empty generation leaves the thread
    untouched.
  - `continue`: prefill semantics — the anchor rides as the final message
    (`continue_final_message:true`; user-role is MLX-only, provider 400s
    in-band otherwise) and the continuation merges back onto the anchor row,
    again committing truncation + update together.
  - `overrides` layers one-shot sampler values over `params` (same allowlist
    + cap gates) and may carry `model` to generate with a non-stamped model.
  - Wire: Messages SSE grammar (`message_start` / `content_block_start|delta|
    stop` / `message_delta` / `message_stop`, same translator as
    `/v1/messages`) plus ONE namespaced extension event, always LAST:
    `event: heylook_saved` `data: {type, conversation_id, mode,
    end_reason:"complete"|"aborted"|"error", messages:[<full stored rows>],
    dropped_media:{images,audio}, timing:{peak_memory_gb, kv_cache_bytes,
    queue_wait_ms, draft_acceptance}}` — the client's post-stream state is
    ASSIGNMENT from `messages`, never position arithmetic. In-band typed
    `error` events precede it on failure (invalid_request_error /
    api_error, same grammar as `/v1/messages`).
  - Persistence is server-owned: completion, abort, AND client disconnect
    all persist (disconnect via a detached task — a phone locking mid-stream
    loses nothing). Errors before any output persist nothing.
  - Arbitration: one active generation per conversation — a second POST gets
    409 `{error:{code:"generation_in_progress"}}`. While one is active,
    MESSAGE mutations on the conversation (POST/PUT messages, DELETE
    ?after, conversation DELETE) also 409 (v1.67.0): the generation's
    commit truncates by position in its own transaction, and rows appended
    mid-stream by another client would be silently destroyed at that
    commit. Metadata PUTs (title/prompt/params) stay open.
  - v3 sends `overrides: {model, ...panel snapshot}` on every generate
    (v1.67.0): the store is the request's BASE, but panel writes to it are
    debounced/async — overrides carry the user's live intent past that
    window. A client with no such window can omit them.
- `DELETE /{id}/generate` → aborts the active generation (partial persists;
  the Stop button's server-side spelling). 404 when none is active — on
  404 the v3 client aborts its fetch locally instead (covers the
  503-retry sleep and the pre-claim dispatch window).

**Notebooks** (prefix `/v1/notebooks`, no auth): `GET /` list **omits content**; `GET /{id}` full;
`POST /` `{title,content,system_prompt?,model_id?,params?,applied_preset_id?}` (`applied_preset_id`
at create: same v1.59.0 inheritance contract as conversations); `PUT /{id}` partial (incl content,
`params`, `applied_preset_id`); `DELETE /{id}`.

**Presets** (prefix `/v1/presets`, no auth; added v1.34.22): saved bundles of system prompt + sampler
params, LM-Studio-style. UI-authored and expanded **client-side** (apply = copy `system_prompt` onto the
conversation + copy `params` into the settings panel); NOT the server's TOML preset registry that
`ChatRequest.sampler` names — no wire relationship between the two.
- `GET /` → `{presets:[{id,name,system_prompt,params,created_at,updated_at}], total}` ordered by name.
- `POST /` (201) `{name, system_prompt?, params?:{}}` → preset. Names unique → 409; blank name → 400.
- `PUT /{id}` `{name?,system_prompt?,params?}` (only set fields patched; `system_prompt:null` clears;
  empty→400; rename collision→409) → updated preset.
- `DELETE /{id}` → `{status:"deleted",id}`.
- **`applied_preset_id` (schema v6, added v1.39.16)** — on conversations AND
  notebooks, the preset a document is running. Written on EXPLICIT actions
  only: Apply and Save stamp it, Del clears it for the active document, and
  a NEW document created from the selected/stamped preset stamps at birth
  (v1.59.0 inheritance -- starting-as IS an apply).
  A document whose state merely *matches* a preset is labelled by live
  client-side inference and is NEVER stamped — storing an inference could
  bind stale state to the wrong document after a failed load (the v1.39.7
  lesson). A stamp naming a deleted preset is self-healing: the client
  resolves stamps against the live preset list, so a dangling id simply
  reads as "no stamp". Note this is deliberately NOT carried inside
  `params`: that object is the sampler bag and everything in it reaches the
  model, so non-sampler state may never live there.
- `params` is an open sampler-knob object; the authoritative key vocabulary is `PARAM_META` in the v3
  settings panel (`apps/heylook-frontend-v3/js/settings.js`) — do not re-enumerate it here. A preset
  stores only the knobs it pins — absent keys stay on the null-means-cascade contract when applied.
  Presets survive `POST /v1/data/clear` AND store schema recreates (they're config, not data).

**Admin models** (`X-Heylook-Admin-Token`): `GET /v1/admin/models` →
`{models:[{id,provider,description?,tags,enabled,capabilities,config,loaded,
stale_reload_fields}], total}`.
`config` is the model's STORED keys only (`exclude_unset`, 2026-08-11) — absent IS how a
default is spelled in models.toml, and the config editor + the row's non-default summary
chip depend on telling "explicitly set" from "inherited default". The resolved dump it
used to be made every default look deliberately chosen (chip on every row,
`n_gpu_layers 999` rendered as an explicit choice). NB pydantic validators that ASSIGN
derived values must restore `__pydantic_fields_set__` or their fields leak back into
this dump as "stored" (`_resolve_modalities` does; a test pins it).
`stale_reload_fields` (2026-08-11) is SERVER-derived: the requires_reload keys whose
saved value differs from what the loaded process was built with (always `[]` when not
loaded). It's the truth behind v3's "config changed — reload to apply" row marker —
client-side bookkeeping of the same fact dies on remount and drifts on partial
failures;
`POST /{id}/fit` (added v1.60.0) body `{config_overrides?:{key: value|null}, headroom_gb?:8}` →
`{weights_gb, headroom_gb, reclaimable_gb, working_set_gb?, max_buffer_gb?,
sysctl_wired_mb?, sysctl_suggest_mb?, kv_headroom_gb?, hard_working_set,
verdict:"pass"|"warn"|"fail", lines:[{ceiling,verdict,need_gb,have_gb,note}],
sizing_notes:[], estimated}` — the server-computed memory-fit verdict
(`heylook_llm.ram_fit`, same computation as `scripts/ram_report.py`). The UI
RENDERS this and never re-derives fit client-side. `config_overrides` are
candidate (unsaved) edits over the stored config, `null` = reset-to-default
(the PATCH spelling); read-only, ignores sizing-irrelevant keys.
`hard_working_set` carries the engine asymmetry (provider-derived: over the
Metal working set is FAIL for MLX, WARN for gguf, which loads past it and
degrades into paging). `sysctl_suggest_mb` is non-null ONLY when over the
working set while `iogpu.wired_limit_mb` is at its OS default — show the
sysctl hint iff present. 404 unknown id; 422 `{field:"model_path",...}` when
the (overridden) path does not exist. All numbers are measured today;
`estimated` flips when any component becomes an approximation and estimates
must render in a different visual register (design doc §5).
`POST /{id}/load[?warm=true]` → `{status:"loaded",model_id,warmed?,warm_ms?|warm_error?}` (400 unknown id, 500 load failure; `warm=true` additionally runs a 1-token generation through the real generation path -- the canonical readiness call for spawn harnesses, 2026-07-20);
`POST /{id}/reload[?warm=true]` (added v1.62.0) → unload + load(+warm) as ONE
server-owned operation, exact `load` response shape (shared body, so the warm
contract cannot fork). Reloading an unloaded model is just a load. This is
what v3's "Reload now" sends -- the old browser-driven unload-then-load pair
could strand a model unloaded if the tab died between the calls;
`POST /{id}/unload` →
`{status:"unloaded"|"not_loaded"}` (never errors); `POST /scan` `{paths?:[], scan_hf_cache:bool}` →
`{models:[{id,path,provider,size_gb,vision,quantization?,already_configured,tags,description,
modalities,supports_thinking?,draft_model_path?,draft_spec_type?}], total}`
(`ScannedModelListResponse`, wired as the route's `response_model` 2026-08-07 — it sat
unreferenced for months while the dataclass grew, so the declared contract had no way to
disagree loudly with what shipped);
`POST /import` `{models:[{id,path,provider}], default_sampler?}` → `{imported:[...], total, warning?}`
(field renamed from `profile` 2026-07-20; unknown body keys 422 via extra="forbid").

Three things about `/scan` that bit (all fixed 2026-08-07):
- **`paths` is not optional in practice.** The HF cache is ONE source; every locally
  downloaded model — the whole GGUF fleet — is reachable only through `paths`, and v3
  hardcoded `{scan_hf_cache: true}` with no paths, so none of it could be found from the
  UI. Paths resolve on the SERVER (a relative one against its working directory); a
  missing path is a no-op, not an error.
- **`modalities`/`vision` are DERIVED, not read from a config key.** Only the MLX entry
  builder ever wrote `config["vision"]`, and thin entries stopped writing it — so scans
  reported `vision:false` for every model of both providers. gguf states its modalities in
  the entry (read from the projector's own header, which declares vision and audio
  separately); everything else derives them from the model dir through the shared detector.
- **`draft_spec_type` is reported, never applied.** Import pairs a drafter's PATH but
  leaves `spec_type` unset, because whether speculative decoding pays off is a per-model
  measurement. Which `--spec-type` the drafter requires is a fact about the file, and it
  only ever reached the server log.

`GET /v1/admin/models`'s `capabilities` is DERIVED through the same helper `/v1/models`
uses (`capabilities.py`, extracted 2026-08-07). It previously reported the stored
`ModelConfig.capabilities` OVERRIDE, which is empty on every entry that never hand-wrote
one — so the Models page listed no capabilities at all while chat, one endpoint over,
gated its entire UI on them.

`POST /{id}/load?warm=true` is what v3's Load button sends, so "Loaded" means ready
rather than merely resident; the page renders `warm_ms` as a note and surfaces
`warm_error` without calling the load a failure (the model is loaded either way).
(Backend also exposes toggle/status/validate/samplers/bulk-default-sampler/discovered — the
sampler routes were renamed from profiles/bulk-profile 2026-07-20; out of scope unless a
trimmed feature needs them.)

**Per-model config editing** (consumed since 2026-08-11, backend v1.52-1.53):
`GET /v1/admin/model-options` → `{providers:{[provider]:{fields:[{name,effect,type,
default,required,minimum?,maximum?,exclusiveMinimum?,exclusiveMaximum?,enum?,arg?,ui?,
shape?,reason?}]}}}` — every settable models.toml key per provider, derived from the
provider config classes (a new backend field appears in the UI with no frontend change).
NOT under `/v1/admin/models` (its `{model_id:path}` catch-all would eat the path).
`effect` says WHEN a change lands and drives the editor's layout:
`per_request`/`applies_live`/`descriptive` = immediate; `requires_reload` = saved to
models.toml, a loaded model keeps running as-is until reloaded (the editor offers
"Reload now" = `POST /{id}/reload?warm=true` after such a save, v1.62.0); `load_time_only` = rendered
disabled with the field's `reason`; `identity` fields are never listed. `arg` is the
llama-server flag spelling (shown as a hint, pinned to the emitted argv by a backend
test); `ui:"advanced"` collapses the field into an Advanced group; `ui:"hidden"` means
NO editor renders it (declared on the field — gguf's host/port/server_binary/
startup_timeout_s, mlx's derived `vision` mirror — so the hide policy has one source
instead of a name-list copied into every consumer); `shape:"flag"` marks a bare
boolean flag. Array fields edit as one element per LINE, never comma-joined —
elements legitimately contain commas (`extra_args`: `--tensor-split "3,1"`).
`PATCH /v1/admin/models/{id}` body `{config:{key: value|null}}` →
`{model, reload_required_fields, warning?}`. Values are TYPED JSON (numbers as numbers);
**null means "unset — back to the default"** and removes the key from models.toml (absent
IS how a default is spelled on disk; this is the same null-means-cascade philosophy as
guardrail #3). `reload_required_fields` is the server's provider-aware answer — the
frontend renders it rather than re-deriving reloadiness client-side. `warning` carries a
post-save config-reload failure. A value TOML can't store returns 400, not 500.
CAUTION for harnesses: the PATCH rewrites models.toml through `tomli_w`, which drops
every comment in the file — E2E checks must intercept it, never let it land (the E2E
server isolates only the DB, not models.toml).
`DELETE /v1/admin/models/{id}` → `{status,model_id,warning?}`; **409** when the entry is a
DISABLED override for a file discovery still finds — deleting it would delete the only
record of the "off" decision and the next scan would serve the model again, enabled. The
detail text is the explanation; render it (v1.69.1 — it was an uncaught 500 before).
Every mutating admin route runs in the server's threadpool, not on the event loop: each
one re-runs the `[scan]` discovery walk (twice, counting the reload), so an `async`
handler would freeze in-flight SSE streams for its duration. Expect these calls to take
as long as a scan takes; other requests keep flowing meanwhile.

**Models list** `GET /v1/models` → `{data:[{id,provider?,capabilities?,modalities?}]}` (enabled models only). `modalities` (v1.34.43) is the model's declared capability set (`["text","vision","audio","video"]`); `capabilities` stays gated to what the server actually serves (image input) -- description != served. `thinking` (v1.34.60) is auto-detected from whether the model's chat template references `enable_thinking` (Qwen3 `<think>` blocks, gemma-4 thought channels) -- no `models.toml` flag needed; this is what shows/hides the drawer checkbox and composer icon.
**Metrics** `GET /v1/system/metrics?force_refresh?` → `{system:{ram_used_gb,ram_available_gb,ram_total_gb,
cpu_percent}, models:{[id]:{memory_mb,context_used,context_capacity,context_percent,requests_active,
requests_queued}}}` (30s server cache).
**Perf profile** `GET /v1/performance/profile/{1h|6h|24h|7d}` → `{timing_breakdown:[{operation,avg_time_ms,
count,percentage}], trends:[{hour,response_time_ms,tokens_per_second,requests}], resource_timeline,
bottlenecks}` (in-memory ring buffer, lost on restart; 503/empty if analytics extra not installed).
**Clear** `POST /v1/data/clear` (admin) → `{conversations_deleted, notebooks_deleted}`.
**Capabilities** `GET /v1/capabilities`. Includes `samplers` (2026-07-20):
`{available:[{name,description}], request_field:"sampler", model_default_field:"default_sampler"}` —
the bundled named-sampler registry, for scripted clients; distinct from `/v1/presets`
(saved user prompt+sampler bundles), which is what v3's preset bar uses. `server_version`
now reports the real package version (was hardcoded "1.0.1").
**J-space** `GET /v1/jspace/models` → `{models:[id], meta:{[id]:{provisional,fit_date,fit_source,
n_prompts}}, base_dir}` (models with a fitted lens; `meta` added v1.34.37 — `provisional` means the
lens sidecar has no own-fit provenance stamp and drives the UI's "provisional lens" badge);
`POST /v1/jspace/analyze` body `{model, prompt|messages, chat?, heatmap?, heatmap_top_k?,
max_answer_tokens?, top_k?}` → `{answer, first_answer_token, prompt_tokens, band_layers,
onset_strip:[{layer,entropy,top_k:[{token,logit}]}], heatmap?:[{layer,cells:[{token,entropy,top_k?}]}],
heatmap_positions?, features, risk}`. `heatmap_top_k` (v1.34.37, default 0): when >0 each heatmap
cell also carries `top_k:[{token,logit}]` (descending), enabling per-cell pinned readouts; 0 keeps
the pre-extension `{token,entropy}` cell shape. Lens-gated (404 if no lens for the model).

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

**Model switching (added 2026-08-11, v1.57.0 -- the §15 arc of the load-options
design doc, `internal/research/expert_offload_design_frontend.md`):** the select's
option labels carry residency (`●` resident / `○` idle, from `/v1/admin/models`;
plain ids until the first successful fetch -- the UI never guesses residency;
refreshed at mount, after Load, after the first delta of a cold send, and
after each completed generation, never polled). Switching runs a PRE-switch
check for LOSS only: incompatible history media ("N images... will be
dropped") and -- only alongside a real warning -- a thinking-toggle-will-hide
note. Warnings render inline in the chat status area (never a modal) with
Cancel / Switch anyway; the switch does NOT commit (model_id write, cap
re-gating, drawer rebuild) until confirmed, Cancel reverts the select, and a
Send with the unconfirmed target selected commits it (sending is the
strongest confirmation). A compatible switch commits silently.

**Load cost is disclosed, never gated (owner call 2026-08-11, v1.62.3 --
supersedes the G3 residency confirm).** Choosing a model IS choosing to pay
for it, so there is no decision to put behind a button, and the old confirm
fired hardest on the emptiest state (nothing resident, no conversation) where
it also claimed an eviction that could not happen. Three non-blocking
disclosures replace it: the `○` in the option label, the "Load" button beside
the select (unloaded models only; same `load?warm=true` warm contract, phases
spoken by the status line), a status line on switching to an unloaded target
("... is not loaded — your first message loads it"), and a live status for
the whole pre-first-token wait on Send -- `Loading <id>…` when the target is
cold, `Waiting for the first token…` otherwise -- cleared by the first
content OR thinking delta, and by stream teardown so a zero-token completion
or a Stop mid-load cannot strand it. **History media is caps-gated at
the wire** (`toWireContent(msg, caps)`): blocks the current model cannot take
are dropped from requests -- never from the store; switch back and they ride
again -- with a per-message "N images not sent to this model" disclosure in
the transcript. DELIBERATE ASYMMETRY: staged attachments still BLOCK the send
(just chosen, trivially un-choosable), history DROPS with disclosure (work
already done; forcing its deletion punishes the wrong thing) -- commented at
both sites, do not unify. Mid-stream switch stops the stream BEFORE model_id
changes hands (an abort, not attribution -- G5 waits for messages.model_id).

**Composer image + thinking controls (added post-spec, v1.34.20 + v1.34.60-.61 --
documented here for the current build):** attach and thinking-toggle are icon
buttons (`.btn--icon`, see DESIGN.md §7) beside the textarea, not text buttons.
Attach opens a file picker (`multiple`, iPhone camera roll included) or accepts
paste; images render as a thumbnail strip capped at **8 attachments**, with an
aria-live announcement when the cap is hit and a "Remove image N" label per
thumbnail. The thinking toggle is visible only when the selected model reports
the `thinking` capability (see §4 Models list -- auto-detected from the chat
template, no `models.toml` flag required) and mirrors the drawer checkbox's
true/unset semantics 1:1 via `onSettingsChange`.

**Settings entry points + presets (added post-spec -- 2026-07-11 shared-drawer
extraction, presets client-side bundle v1.34.22, explicit Apply + drift v1.39.2,
shared `preset-bar.js` v1.39.3 -- documented here for the current build):**
sampler + system-prompt controls live in the app-shell shared settings drawer
(`settings-drawer.js`), not an inline per-page panel; chat additionally exposes
an in-context opener (`.chat__settings-btn` in the top bar, via
`drawer.openSettings(btn)`) onto the same singleton drawer. The system-prompt
textarea commits state per keystroke with a 400ms debounced PUT (flushed
immediately on blur) -- not save-on-blur only, which lost text when the drawer
closed under focus. The preset `<select>` is inert (records the selection and
prefills the save-as name, never writes the document); Apply is an explicit
button, armed-confirmed ("Replace prompt?") only when it would replace a
differing non-empty prompt; a live drift line (`role="status"`) reports whether
the selected preset matches the current prompt + sampler state.

**The prompt is an OVERRIDE BOX (owner rule 2026-08-11, v1.62.3).** A preset
OWNS a system prompt and carries it onto whatever it is applied to, but a
preset whose `system_prompt` is empty/null makes NO claim about the prompt:
applying it leaves the conversation's prompt (or the model's own default)
exactly as it was, rather than blanking it. Empty means "does not speak for
the prompt", never "set it to empty". Everything downstream follows from that
one rule -- only a carrying preset can arm "Replace prompt?", and `matchesState`
ignores the prompt for a promptless preset, so drift is decided by the sampler
half alone. It also defuses a data-loss chain: a preset saved while the box was
blank is inert instead of destructive.

**The system-prompt chip** (`.chat__sysprompt-chip`, beside the applied-preset
chip) states what is in force, in four states: "No system prompt" (quiet,
dashed -- always rendered, never hidden, because an absent chip is
indistinguishable from a broken one and that ambiguity is what let a
disappearing prompt go unnoticed), "System prompt: custom", "System prompt:
&lt;preset&gt;", and "&lt;preset&gt; (modified)" once it diverges from the preset it came
from. The full text (capped at 300 chars) rides the tooltip + accessible name,
and clicking any state opens the drawer with the textarea focused and scrolled
into view. It is fed by the preset bar's `promptState()` -- deliberately
separate from `indicatorInfo()`, which answers the whole-document question:
a chip claiming to track the prompt must not flip on a temperature nudge.

**A prompt typed before any conversation exists is parked in localStorage**
(`heylook.v3.chat.draft-prompt`) until a conversation adopts it, and both
create paths clear it. Without this it lived in page state alone, so a reload
or a trip to another page silently ate it -- while sampler params in the same
window survived (settings.js parks those), which is exactly why the loss read
as "everything else loads fine, just not the system prompt". An
applied-preset chip (v1.39.6, `.preset-chip`) sits beside each page's model
select naming the preset the active document is running -- "(edited)" once it
drifts -- and opens the drawer on click. Provenance is session-local: only
explicit Apply/Save stamp it (delete clears it); a document whose state merely
equals a preset is labeled by live inference and never stored, so a stale or
coincidental match cannot persist as a false "(edited)" claim. The server
stores no preset association.

### notebook (v2: 341 lines) — plain-text, no pretext, no markdown render
Multi-doc: sidebar list (new/select/delete-confirm) + editor (title input, model select, collapsible system-
prompt textarea, large auto-grow content textarea, Generate/Stop). Debounced auto-save 500ms (reads fields
live from DOM in v2 → v3 should hold them in state and drop the querySelectors). Generate: split content at
cursor, messages = optional system + user(text-before-cursor or "Continue writing."), stream tokens inserted
**at cursor** preserving text after + caret, auto-resize. flushSave on select/teardown (don't lose edits).
Content is `textarea.value` only — never innerHTML (XSS-safe by construction). The system-prompt textarea and
the preset bar above it (same shared `preset-bar.js` grammar as chat, added v1.39.3) live in the shared
settings drawer as registered sections, not inline in the editor form.

### models (v2: 250 lines) — admin
List model cards (Loaded/Idle badge, provider/capability/disabled tags, description). Per-model Load/Unload
(per-model busy flag — keep per-row granularity; **fix**: make import per-row too, v2 has a global import
flag asymmetry). Scan (`scan_hf_cache:true`) → results panel (filter `already_configured`, Import each →
removes from panel + reloads list). "Clear all conversations & notebooks" with `confirm()`. **Fix v2's
inconsistent error UX**: pick ONE surfacing strategy (v2 shows errors only on initial list load, silently
console-logs scan/load/import/clear failures). **Fix**: add the missing `if (!ctx) return` post-await guards
(free via `createPage`). Custom scan `paths` UI is a v2 gap — leave out of scope unless requested.

Added 2026-08-11: per-row **Configure** expands an inline config-editor card
(`js/model-config.js`, schema-driven off `GET /v1/admin/model-options` — see §4). Grouped
by effect class (immediate / requires-reload / advanced / fixed-with-reason); tri-state
selects for booleans and enums (`default (X)` = unset), empty input = unset, so the
null-means-default contract is honest in every control; Save PATCHes only dirty fields;
after a reload-required save on a loaded model the panel offers Reload now (unload +
warm load, same per-row busy flag), and the ROW shows a
"config changed — reload to apply" marker driven by the response's
`stale_reload_fields` — server truth, so it survives remounts/other tabs and clears
itself on the refetch after a reload or unload (the marker is what prevents "I set it
and nothing happened"). A save REBUILDS the panel (that is how the marker/chip
repaint); the outcome text survives the rebuild as a one-shot note, written into the
`role="status"` live region only after mount so it actually announces. Unsaved edits
live in a per-model draft map keyed off page state so re-renders don't lose them.
Errors go to the page's single status area.

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
(Historical build-time decision, kept as the record. A 6th page — j-space — was added post-build,
~v1.34.31; the live page list is in `docs/frontend_v3.md`.)

| Page | v2 lines | Verdict | Notes |
|------|---------|---------|-------|
| chat | 815 | **Keep — core** | The product. Legacy render path only. |
| models | 250 | **Keep — core** | Load/unload/scan/import + clear-data. |
| notebook | 341 | **Keep** | Plain-text, no pretext. |
| perf | 212 | **Keep — simplified** | On-demand only, no polling. See §5. |
| explore | 378 | **Keep** | Only logprobs surface; simplify render per §5. |
| batch | 219 | **DROP** | Endpoint stays; page not built. |

---

## 7. Design quality — impeccable workflow (REQUIRED)

Design goes through the **impeccable plugin** (`/impeccable`), not ad-hoc taste. Its project context
already exists: **`PRODUCT.md` at the repo root** (register: product; personality: warm minimal; users:
single owner, desktop + iPhone Safari co-primary; pragmatic a11y floor; anti-references incl. desktop-only
layouts and SaaS-dashboard grammar). Every impeccable command reads it before working — keep it current.

How the build uses it:
- **Session setup**: invoking `/impeccable` runs its context script and loads the *product* register
  reference plus its general rules (contrast floors, typography caps, motion rules, absolute bans like
  side-stripe borders / gradient text / identical card grids / eyebrow-kickers). Those bans are hard
  constraints on all v3 UI.
- **Palette**: v3 is a fresh visual start — run the bundled `palette.mjs` for a brand seed color and
  compose bg/surface/ink/accent/muted around it in **OKLCH**. Do not inherit v2's palette by default,
  and mind impeccable's warning against the cream/sand near-white default.
- **Per-surface builds**: use `/impeccable craft <page>` (shape interview → build end-to-end) for each
  page's UI, starting with chat. `craft` owns the design flow; the functional contract stays this spec.
- **DESIGN.md**: seed via `/impeccable document` once the scaffold + first page exist, so later pages and
  variants stay on-system.
- **Gates before calling a page done**: `/impeccable audit <page>` (a11y/perf/responsive — must pass on an
  iPhone-Safari-sized viewport, not just desktop) and `/impeccable polish <page>` pre-ship. The bundled
  `detect.mjs` slop-detector can run over changed files as a cheap check between gates.
- **Iteration**: `/impeccable live` is available for in-browser variant picking once the dev server
  (the FastAPI backend serving `/v3`) is running.

Still applies regardless of impeccable: single `css/app.css`, no build step; replace blocking native
`confirm()`/`alert()` (v2 uses them in models/chat/notebook) with in-app confirm affordances where cheap —
but keep a confirm step for destructive actions per repo rule. Use the `dataviz` skill for the perf stat
tiles and the explore logprob color scale specifically (chart/viz color method), inside impeccable's system.

---

## 8. Build sequence (for the build session)

1. Design system first: invoke `/impeccable` (loads `PRODUCT.md` + product register), run `palette.mjs`
   for the OKLCH brand seed, and write a fresh `css/app.css` token layer + app shell around it. Do NOT
   port v2's CSS wholesale — v2's stylesheet is a reference for what exists, not the starting point.
2. Scaffold: `index.html` shell (base href `/v3/`, `#sidebar`/`#main`/`#bottom-nav`), vendored `marked` +
   `purify`, the `/v3` mount in `api.py` (duplicate the `/v2` block).
3. Shared layer: `createPage`, router, `api.js` (table-generated), `streaming.js` (with keepalive-comment +
   reader.cancel + abort-as-complete), `settings.js` + data-driven panel, `utils.js`, `markdown.js`.
4. chat page via `/impeccable craft chat` (legacy render path only) — highest risk, do first; verify
   streaming, edit/regenerate, position truncation, stop/abort, thinking blocks. Then seed `DESIGN.md`
   (`/impeccable document`) so remaining pages stay on-system.
5. models, notebook, perf (simplified, no polling), token explorer — each through `craft`, gated by
   `audit` (incl. iPhone-Safari viewport) + `polish` per §7. (No batch page — §6.)
6. Verify against a running backend (`/run` or the `verify` skill): drive each flow, watch the network tab
   for the exact payloads in §4, confirm 503 backpressure + abort behave.

## 9. Backend co-evolution (v3 is a native frontend, not a generic client)

v3 is TIGHTLY coupled to the heylookitsanllm core engine, on purpose. It is the server's own frontend,
not an OpenAI-compatible client that happens to point here: it should freely use heylook-specific
surfaces (usage-chunk `timing` telemetry, admin routers, `/v1/capabilities`, conversation/notebook
stores) and — the important part — **when a backend change would make the frontend meaningfully simpler
or better, change the backend** rather than absorbing complexity client-side. The §4 contract is the
current state, not a frozen boundary.

When the build hits a "backend should support X" moment, use the delegation ladder (rule:
`.claude/rules/model-delegation.md`; pre-shaped agents in `.claude/agents/`):

1. **Explore (cheap, read-only)** — before deciding anything, fan out an `Explore` agent to map the
   blast radius: which routers/schema/config the change touches, existing patterns to follow, affected
   tests. Mirror the repo's removing-a-feature checklist in reverse: `api.py` (tags + include_router),
   `config.py`, `schema/`, tests, docs.
2. **Decide (main loop, strongest model)** — API shape, response-model, and compatibility calls stay
   up-tier. The backend serves v2 (React) and v3 during the transition, so changes must
   be additive/back-compatible until v2 retires.
3. **Implement (task-coder)** — hand the change as a complete spec: exact endpoint, request/response
   models, files to touch, tests to write (TDD per repo convention). Mechanical sweeps (renames,
   regenerations) go to `fast-executor`.
4. **Verify (main loop)** — `uv run pytest tests/unit/ tests/contract/` (green is the invariant; any
   failure is a regression). Update §4 of THIS spec when the contract changes, so the spec stays the
   single source of truth for the frontend contract.

Known v2-era gaps already worth this treatment if the build wants them: custom scan `paths` UI
(backend supports it, v2 never exposed it); a slimmer conversation-list payload is NOT needed (already
slim); anything where §5 says "fix" about error surfacing is frontend-only — don't touch the backend
for those.

## 10. Delegation note (how this spec was built cheaply)

Extraction was fanned out to parallel sonnet-tier subagents (one per page + core + backend + pretext), each
returning a fixed-schema distillation so the Opus main loop ingested ~1.5k lines of specs instead of ~10k
lines of source. Synthesis + the architecture decisions (drop pretext, no framework, trim) stayed in the
main loop; every load-bearing claim was spot-verified against source before landing here. The base
model-delegation rule lives at `.claude/rules/model-delegation.md`, with pre-shaped `fast-executor` and
`task-coder` agents in `.claude/agents/`, so future sessions auto-route mechanical work down-tier. The
build itself (§8) is well-specified enough to hand to a `task-coder` per page — and backend changes follow
the §9 ladder — with main-loop verification of everything that comes back.
