# Current Work

Last updated: 2026-08-30. v1.79.43 on the `frontend` branch.

**Verification state, as of the last commit:**

| Suite | Result | When |
|---|---|---|
| unit + contract | 1737 passed (1601 + 136) | re-run at v1.79.43 |
| `bun run e2e:render` (model-free) | 102/102 | at v1.79.42 |
| `tests/smoke/` mlx-lm arm | 26/26, 3 UNCOVERED | at `a274682` |
| `tests/smoke/` mlx-vlm arm | 31/31, 2 UNCOVERED | at `a274682` |
| `tests/smoke/` gguf arm | **30/30 on each of two models** | re-run at v1.79.43 |
| `bun run e2e:chat` | 33/46 -- RUN, still red. See item 2 | re-run at v1.79.43 |

The gguf and chat rows moved this session because both were finally run
against a live server. The two MLX smoke arms still carry the commit they
were measured at; nothing since has been able to move them without a server.

NOTE ON RUNNING THE SUITE LOCALLY: `tests/contract/` opens the real
`data/conversations.duckdb` and DuckDB takes an exclusive lock, so every
contract test errors in setup while any server is running against the default
DB. That is not a regression -- run them with an isolated
`HEYLOOK_DB_PATH`, which needs an unsandboxed invocation (an env-var prefix
does not match the sandbox's `uv run` exemption).

HANDOFF (next session start here): two things are open; neither blocks work.

1. **`uv.lock` stays dirty on purpose -- OWNER RULED 2026-08-29, do not
   re-raise.** The working tree carries an uncommitted re-resolution
   (mlx-vlm, transformers, tokenizers, huggingface-hub, pydantic) with no
   pyproject change. It is deliberately left uncommitted and unreverted. Two
   things follow, both worth knowing rather than acting on: `uv run` syncs
   from the working tree, so local checks run against THOSE versions rather
   than the committed lock -- a green suite here is evidence about the local
   resolution, not about what a clean checkout would install; and the file
   remains git-TRACKED (it must, `scripts/guard_stable_channel.sh` inspects
   its staged blob), so this is "leave the modification alone", never
   "gitignore it".

2. **`bun run e2e:chat` is 33/46 and the v1.79.41 repair was not the fix.**
   That release fixed two real selector rots and claimed they were "the whole
   static gap". Running the suite showed otherwise. The deeper cause is
   architectural rather than a selector: `tests/e2e/lib/browser.mjs` seeds
   sampler settings into `localStorage` and expects the chat settings panel to
   reflect them, but since v1.65-66 chat hydrates that panel from the
   DOCUMENT (`hydrateDocParams` -> `applySettings(doc.params)`), so selecting
   a conversation overwrites the seed before the first assertion runs. The
   `seeded max_tokens` check fails with the document's value, and the preset
   and system-prompt checks rest on the same stale model of where chat state
   lives. This is a suite REWRITE, not a patch. Until it is done the suite is
   not a gate, and the app is not implicated: the model-free render suite
   drives the same real `/v3` page at 102/102 including its uncaught-page-
   error check, and the generate-path rows pass server-side in smoke.

**CLOSED THIS SESSION** (were items 2, 3 and the standing product defect):

- **The gguf smoke arm is covered and needed no code fix.** The recorded
  failure was `llama-server exited with code 1`; the model's architecture is
  `qwen4exp` and the canonical llama.cpp build has since been rebuilt from a
  checkout that supports it. It now loads in seconds. Both conformance rows
  are covered live, across TWO models because no single served model
  exercises both: `google_gemma-4-E4B-it-qat-q4_0-gguf` covers audio and
  reports the thinking block UNCOVERED (that model returned none),
  `unsloth_Qwen3.8-27B-UD-Q8_K_XL` covers the thinking block and reports
  audio UNCOVERED (no audio modality). The handoff's diagnosis recipe was
  itself confirmed: raising `observability_level` above `off` produced the
  llama-server log the failure message said was missing.
- **`/v1/models` no longer over-reports `vision`.** The capability is derived
  through `effective_loader_for_config` -- the same resolver `MLXProvider`
  uses -- so the advertised capability and the provider's image guard agree
  by construction. See the changelog for why the two-rules shape was the
  actual defect.

## v1.79.41 -- the conformance work reviewed

Five findings across .37-.40, each re-verified against running code before
being touched. Two are worth carrying forward.

**The nested `source` spelling still 422'd for the clients it was added for.**
`_flatten_source` gated on key PRESENCE, and `source_type` is Optional in the
published schema, so a client generated from `/openapi.json` sends explicit
nulls beside the nested object and was rejected on the exact spelling the docs
recommend. **The obvious one-line fix was not sufficient** -- the `setdefault`
calls underneath test presence too, so repairing only the gate moved the
failure from the discriminator to the data and would have looked fixed. That
is the durable part: when a bug is "this code tests presence where it means
value", the same mistake is usually spelled more than once in the same
function.

**A test exempted the defect it was written for.**
`TestStopReasonHasOneMapper` waved through any string literal, so
`stop_reason = "length"` -- OpenAI's vocabulary on the Messages wire, the
precise 1.79.39 bug -- passed green as long as it was spelled as a literal.
An exemption written for one legitimate case (an explicit abort end-state)
admitted the whole class. Literals are now checked for membership.

Also: dropping the Clone confirm in .37 removed the double-tap coalescing the
arm had been providing as a side effect (two taps, two conversations, two
racing selects), now a re-entry guard rather than a resurrected confirm; and
two doc drifts of the same kind the release was about.

## v1.79.38-.40 -- the Messages API became Messages-conformant

Started as "write a doc so another project's agent can integrate", became a
conformance pass, then a review of that pass.

**What was wrong.** `/v1/messages` is Anthropic Messages-SHAPED and was not
Messages-CONFORMANT for three payloads, each failing silently for a client
written against Anthropic's spec: media blocks were flat where Anthropic
nests under `source` (so an Anthropic SDK request was a 422), thinking blocks
and their deltas named the field `text` where Anthropic names it `thinking`
(a conformant reader found the block and no content), and `stop_reason`
carried the provider's OpenAI `finish_reason` vocabulary. A fourth item that
had been called a divergence -- the absent `[DONE]` sentinel -- turned out to
be heylook being right; Anthropic has none either.

**What shipped.** Both media spellings accepted and normalized in the schema,
with `source` a DECLARED field so `/openapi.json` advertises it; both thinking
spellings emitted (v3's `streaming.js` reads `text`); Anthropic's stop
vocabulary through ONE table shared by both Messages-grammar routes.
`/openapi.json` also stopped lying: `version` had been hardcoded at 1.20.0
for ~60 releases, `custom_openapi()` dropped every tag description by never
passing `tags=`, and the narrative described a one-provider server.

**The instructive failure.** The `stop_reason` passthrough existed in TWO
modules; fixing `/v1/messages` left `/v1/conversations/{id}/generate` --
v3's own chat wire -- emitting `"length"` for one commit, with 1700 tests
green throughout, because every test asserted one path's behaviour and none
asserted the paths AGREE. A second instructive one: v1.79.39 added an
`error` stop reason on an UNTRACED claim that api.py set it, and copied that
claim into four documents before a review traced it and found the member
unreachable. These are DIFFERENT failures and it is worth not collapsing
them: the first is a second copy nobody looked for, caught by asserting the
two paths AGREE rather than testing each; the second is a mechanism asserted
without reading the line that would have refuted it. One test and one habit,
not one lesson.

**Docs.** New `docs/api_integration.md` for external consumers (the
`heylook-provider` skill in the owner's marketplace is its consuming-side
twin); spec §4 records the contract change; CLAUDE.md carries the mechanisms.
The remaining deliberate differences from Anthropic are enumerated in
api_integration.md, and that list is hand-maintained and has been wrong once.

## v1.79.36 -- branch code review acted on

A review of the whole `frontend` branch (73 files vs main) returned 15
findings; each was re-verified against the code before being touched, and one
was WRONG (v3 does not retry a blocked MODEL_BUSY for the duration of the run
-- `MAX_BUSY_RETRIES` is 3). Twelve acted on; see the CHANGELOG for the list.

The two most serious were both in the "runs outlive their response" family
that has produced most of this branch's bugs: the idle-unload re-check could
SIGTERM a gguf generation, and a finished run could pop a NEWER generation's
claim, leaving that run unstoppable while the conversation reported idle.

**Both open items were ruled on by the owner (v1.79.37): remove.** The Clone
button's `armedConfirm` is gone (only LOSS gates; disclosure instead), and so
is `scripts/migrate_conversations.py` -- the "NEVER write migration code" rule
stands unamended. Consequence worth knowing: opening an OLDER conversation
store with newer code now drops its tables with no prompt, no backup and no
escape hatch. Intended for this deploy (solo, no conversation data to
preserve); revisit only if that posture changes.

`uv.lock` is committed and clean. It was regenerated from the current
pyproject rather than adopting another session's pending `uv lock --upgrade`,
so its diff was the two dependency removals and no version movement. Checked
first: no local paths, no git pins, no source overrides.

## v1.79.31-.34 -- engine coverage COMPLETE, and the lifecycle findings closed

`docs/project/plan_engine_coverage.md` is fully shipped (phase 0 at .28, 1-2 at
.31, 3-4 at .34). All three engine arms run green live and every arm now has a
cheap model. The deferred chat.js lifecycle findings are closed at .32/.33.

**The release standard (Phase 4) is now in `CLAUDE.md`:** before a release
touching provider, loader, template or lifecycle code, `tests/smoke/` runs
green on all three arms, and an uncovered arm -- or an uncovered Phase 3
mechanism -- is NAMED in the changelog rather than passed over.

**Standing UNCOVERED gap, by design not oversight:** thinking DEPTH on both MLX
arms. The only served MLX model advertising `reasoning_effort` is
`gpt-oss-120b-MXFP4-Q8-mlx`, so covering it costs a 120B load. It reports
uncovered on every run.

## v1.79.31 -- engine coverage phases 1-2

`GET /v1/admin/models` now carries `effective_loader` per row, DERIVED from the
config so it answers for UNLOADED models -- the provider attribute is null
unless the model is resident, which is the opposite of what a harness choosing
engine arms needs. `tests/helpers/engines.py` is one classifier shared by
`tests/smoke` and `tests/eval` (eval's `fetch_models` is gone), and both
harnesses now REPORT which engines a run spanned, with `UNCOVERED`
distinguishing "no model served" from "served, none run". Two cheap deferred
review findings went with it (the smoke contract's unguarded `got["params"]`,
render.mjs's positional Apply/Update selectors).

**What it found immediately:** this machine serves exactly ONE mlx-lm model,
`gpt-oss-120b-MXFP4-Q8-mlx`. `Qwen3.5-0.8B-MLX-8bit` -- which
`plan_engine_coverage.md` named as the cheap text arm -- is **mlx-vlm**
(model_type `qwen3_5`, which mlx-vlm registers). So the mlx-lm arm costs a 120B
load today and will not realistically get run. Cheapest fix, recorded in the
plan: pin `loader = "mlx-lm"` on a small model. Worth doing before Phase 3.

## The 2026-08-28 arc (v1.79.20-.30, twelve commits)

Started from a bug report -- "my presets disappear, and updating one changed
every other one". Root-caused, fixed, and then the fixes were reviewed three
times and the reviews found more than the original bug did.

**What was actually wrong.** One preset was overwritten, not many: the select
pre-filled the save-as name, so picking a preset to LOOK at it aimed Save at
it. Everything else the report described was display: the drawer's system-prompt
box shows the DOCUMENT's prompt whatever the select says, so every preset
looked identical. The lost prompt is gone (overwritten in place, no history);
the owner declined a restore from the copy that existed.

**What shipped.** Guards on the destructive direction (v1.79.20-.22), then the
comprehension gap underneath: a scope line on the sampler panel, "settings
only" in the dropdown, a drift line that names which half moved (.25); the
disclosure that a run survives you leaving, and Update / Save-as-new replacing
one overloaded control (.26); disabled-with-reason and per-model caveats (.27).
A `docs/frontend_v3_user_guide.md` was written along the way and is what
exposed the rough-edge list the later work closed.

**What the reviews found that mattered more.** `stopStream` marked every stop
as user-initiated before knowing whether the server stopped anything, so a
COMPLETE generation reported "Stopped." -- re-opening a regression the
2026-08-13 review had closed from the other side. Underneath it, `finishGenerate`
was awaiting the server's own `generating` flag and discarding it, then
inferring the same fact from three client-side proxies. That inference was the
common cause of every lifecycle bug in .26-.29 and is now deleted (.30).

**Two owner rules changed.**
- The red-then-green rule is REMOVED from the repo entirely, carve-outs
  included (.23). It had been rejected once in 2026-08-17 as an undocumented
  habit, then written INTO the repo as policy, where it justified itself on
  every read. `CLAUDE.md`'s test section records that it was removed so it does
  not grow back.
- Live coverage is now per ENGINE, not per provider: `tests/smoke/` (.28) plus
  `docs/project/plan_engine_coverage.md`. The extensive post-v1.62.3 arc shipped 60+ commits
covering the complete Chat Orchestration & Reliability phases (0 through 2),
Schema v7 media-by-reference and true single-message deletion, the Messages
API Phase 3b consumer migration (no v3 page speaks /v1/chat/completions
anymore), Q7 single-slot prompt-cache snapshot replacement (deleting radix),
v2 frontend deletion (v3 is the sole frontend, /v2 returns 404), port 8000
default, "show special tokens" wiring, iOS Safari scrolling/resume fixes,
and v1.79.7's frontend-v3 rendering & memory performance optimizations.

Two foundational contracts established in earlier arcs remain in force:
- **Only LOSS gates; cost is disclosed.** Choosing a model is choosing
  to pay for it -- gate only what destroys something (incompatible history
  media, overwriting prompts), disclose residency/costs inline.
- **The system prompt is an OVERRIDE BOX.** A preset owns a prompt and
  carries it; an empty prompt in a preset makes no claim and changes
  nothing.

NEXT, in order:
1. **Observability config & read surfaces** (P2, `observability.py` / v3 UI;
   TODO.md). Admin panel observability controls and a v3 `logs/*.jsonl` viewer
   page; consolidate `memory.py` legacy streams into the spine once verified.
   Promoted to the top: file logging is opt-in and OFF by default, so there is
   currently no way to look at what the spine records without leaving the app.
2. **Pre-warm load telemetry & lifespan ordering test** (P3, `server.py` / `api.py`;
   TODO.md). Startup `--model-id` load runs in server.py before the lifespan
   resolves `observability_level` from the DB, so telemetry is missing for
   pre-warmed loads. Needs settings resolution before pre-warm, and a contract
   test pinning that `log_startup_info()` runs strictly after
   `apply_runtime_settings()`. Pairs naturally with 1.
3. **mRoPE cache gate config override** (P3, `prompt_cache.py` / `models.toml`;
   TODO.md). The mRoPE cache gate keys on `_position_ids`/`_rope_deltas`
   attribute sniffing; add an explicit per-model `cache_reuse = true|false`
   config escape hatch in `models.toml` ahead of attribute sniffing.
   Live-verify extension reuse for quantized/rotating cache configs.
4. **Frontend post-cutover spec slimming & architecture docs cleanup** (P3;
   TODO.md). Slim `docs/frontend_v3_spec.md` down to §4 (the living API contract)
   + decision records now that v2 is deleted. Trim `config.md` and `mlx_provider.md`.
5. **J-Space visualizer next milestones** (P3, `jspace.js` / `jspace_api.py`;
   TODO.md). Live streaming analyze endpoint (SSE) and interactive
   steering/activation patching (porting `mlxui-core` op-semantics via forward-hooks).
6. **At-rest database encryption** (P3 / future state, `db.py`; TODO.md).
   Optional 1Password / `op read` integration for DuckDB file encryption.

CLOSED THIS SESSION (was items 1-2): engine coverage, all phases; and the three
deferred chat.js lifecycle review findings. One correction worth carrying: the
`setRemoteGenerating` / `releaseStream` pair (.33) had **no reachable symptom**
-- the abort unwind is microtasks and the switch awaits a GET, measured, not
assumed. The fixes remove the invariant's dependence on that ordering; they did
not fix an observed bug, and the render check guarding it goes green either way.
`ABANDON.TEARDOWN` was NOT removed: `ABANDON_RANK` keys on it and it now has a
call site (the delete whose stop never reached the server), so it is documented
as the weakest claim rather than a dead path.

UPDATE 2026-08-25 (v1.62.4-v1.79.7, 60+ commits across multiple milestones, on branch `frontend`):

- **SOLID -- Frontend-v3 rendering & memory performance optimizations (v1.79.7)**:
  - Token Explorer (`explore.js`): $O(N)$ streaming via incremental `DocumentFragment` appending and in-place `.tok--selected` toggling (eliminated $O(N^2)$ chip rebuilds, >98% fewer DOM ops).
  - J-Space stability (`jspace.js`): Replaced array spreading in `Math.min`/`Math.max` with single-pass iterative loop (preventing stack overflow on >65k element matrices) + fast cell hover/marking lookup.
  - HTML escaping (`markdown.js`): Zero-allocation compiled regex replacement replacing DOM-based `createElement('div')`.
  - Image/audio previews (`chat.js`): `URL.createObjectURL(file)` lifecycle with deterministic revocation replacing multi-megabyte base64 strings in thumbnail DOM.
  - Layout decoupling (`app.css`, `utils.js`): `@supports (field-sizing: content)` for modern textarea auto-sizing bypassing forced layout reads; canonical CSS design tokens.
  - Module preloading (`index.html`): `<link rel="modulepreload">` for core shared modules.
- **SOLID -- "Show special tokens" display pref wired (v1.79.6)**:
  - Opt-in `show_special_tokens` request field on `POST /v1/messages` and `POST /v1/conversations/{id}/generate`.
  - Drawer display panel wired on chat & notebook; generation-time toggle preserves declared specials (`special: true`) from the model.
  - Wire hygiene: assistant-stored declared specials stripped before replaying as prompt turns to prevent control-token injection; request-schema parity guard (`test_request_schema_parity.py`) pins wire consistency between OpenAI and Messages APIs.
- **SOLID -- Raw HTML preservation in replies (v1.79.5)**:
  - Escaped raw HTML in model responses at the renderer so tags (e.g. `<d>tag</d>` or `<b and c>`) render accurately rather than being stripped by DOMPurify. Guarded by 4 model-free render checks.
- **SOLID -- Tab resume store sync & page lifecycle hooks (v1.79.2-v1.79.4)**:
  - `page.js` lifecycle grew `ctx.onHide()` and `ctx.onResume()` registering `visibilitychange` + `pagehide`/`pageshow`.
  - Chat re-adopts store state (conversations, presets, params, rows) when returning from background/bfcache on iOS Safari; prompt editor debounced writes flush immediately on hide/unload with `keepalive`.
- **SOLID -- iOS Safari scroll anchoring fix (v1.79.1)**:
  - Root-caused iOS hidden/displaced row bug on simulator: `.message` rows' `content-visibility: auto` caused layout displacement on WebKit due to lack of `overflow-anchor`. Gated on `@supports (overflow-anchor: auto)`; message editor close re-aims scroll at row.
- **SOLID -- Default port 8000 & security token hardening (v1.79.0)**:
  - Default server port changed from 8080 to 8000 (`DEFAULT_PORT = 8000`); README rewritten with operational gotchas.
  - `HEYLOOK_ADMIN_TOKEN` security gating extended to `/v1/admin/config`.
- **SOLID -- mRoPE prompt-cache gate & test runner fixes (v1.78.0-v1.78.2)**:
  - mRoPE language models (Qwen3-VL, Qwen3.5) gated out of prompt-cache reuse inside `process_prompt_with_cache` to eliminate empty output on chained restores; spec-decode draft cache slicing defused.
  - Test runner order independence: session `sys.modules` MLX mock made conditional on absence of real MLX, module eviction bug fixed in `test_router_pinning.py`. Dependency floors bumped: `mlx>=0.32.1`, `mlx-vlm>=0.6.15`.
- **SOLID -- v2 frontend deleted (v1.77.0-v1.77.1)**:
  - `apps/heylook-frontend-v2/` deleted; `/v2` returns 404 (pinned by contract test); v3 is the sole frontend. Root `/` reports real `__version__`.
  - Dependency cleanup: removed unused `xxhash` and `PyTurboJPEG` (v1.76.0).
- **SOLID -- Q7 Single-slot prompt-cache snapshot replacement (v1.75.0)**:
  - Radix tree completely deleted. Replaced with per-model single-slot snapshot cache holding immutable `(state, meta_state)` arrays with native `trim_prompt_cache`.
  - Hybrid models (ArraysCache) refuse partial trims cleanly and re-prefill; zombie generation mutations quarantined.
- **SOLID -- Phase 3b Messages API consumer migration & model attribution (v1.74.0-v1.74.1)**:
  - Notebook and Explore migrated to `POST /v1/messages` (no v3 page speaks `/v1/chat/completions`).
  - `/v1/messages` extended with streaming `heylook_logprobs`, `message_stop.performance` timing/KV metrics, optional `max_tokens`.
  - Per-message model attribution chips (`.message-model-note`) rendered in mixed-model threads.
  - Disconnect-persistence live E2E check; `/v1/models` template probe cached (v1.74.1, 1650ms -> 1ms).
- **SOLID -- Media by reference (schema v7) & true single-message deletion (v1.73.0)**:
  - Base64 media moved to content-addressed per-conversation blob store (`/v1/conversations/{id}/media/{media_id}`); reads are text-sized; generate saga inlines bytes on demand; reference-counted GC on deletion.
  - `DELETE /v1/conversations/{id}/messages/{msg_id}` deletes a single message without truncating neighbor rows. `model_id` column added to messages table.
- **SOLID -- Drag-and-drop / paste attach & stream swap (v1.72.0-v1.72.3)**:
  - Unified staging pipeline for file picker, paste, and drag-and-drop with drop overlay and capability validation.
  - Stream end swaps saved rows synchronously from `heylook_saved` without blank frame or layout shift (v1.72.2).
  - `scripts/build_llama.py` repairs shallow clones and enforces binary rev verification (v1.72.3).
- **SOLID -- Reasoning effort knob & Discovery-as-registry (v1.68.0-v1.71.2)**:
  - `reasoning_effort` (`low|medium|high|xhigh`) thinking-depth knob supported per-request, preset, and model default; probed per-template on MLX and GGUF.
  - Discovery-as-registry (`model_registry.py`): models under `[scan].folders` served automatically without writing models.toml; admin routes moved off event loop; v3 scan config editor (`GET/PUT /v1/admin/models/scan-config`).
  - GGUF `chat_template_path` override (v1.68.0).
- **SOLID -- Chat Orchestration & Reliability Phases 0-2 (v1.64.0-v1.67.1)**:
  - Phase 0: Loud stream guards, unsaved-row honesty & retry/discard, saga reconciliation at stream end, editable thinking blocks.
  - Phase 1: Conversation-scoped generation backend (`POST/DELETE /v1/conversations/{id}/generate`), server-owned persistence (completion, abort, client disconnect via detached task), atomic truncation transactions (`replace_tail_with_message`).
  - Phase 2: v3 chat client cutover to `/generate` via `streamGenerate` in `streaming.js`; Stop = `DELETE .../generate`.
  - Canonical llama-server build enforcement with loud spawn warnings on overrides (v1.66.1).
  - 20 adversarial review findings resolved across two passes (v1.67.0-v1.67.1).
- **SOLID -- Plain uv manifest & build_llama.py (v1.62.6)**:
  - pyproject.toml simplified to a hand-maintained manifest of published releases; updater retired. `scripts/build_llama.py` manages canonical llama-server builds; `scripts/guard_stable_channel.sh` pre-commit guard prevents accidental pin commits.
  - `heylookllm import` merges by default, preserving hand-edits.

UPDATE 2026-08-11 late (v1.62.2-1.62.3, third session, all on main):

- **SOLID -- v3 assets no-cache (v1.62.2)**: the /v3 route sent no
  Cache-Control and starlette's FileResponse has no 304 path, so
  browsers used HEURISTIC freshness per file -- frequently-edited
  modules refetched while rarely-edited ones served stale for hours,
  silently mixing module versions ("presetForNewDoc is not a
  function"). Pinned by test_v3_assets_are_revalidated. This class is
  invisible to every suite we have: they load modules from disk, never
  through a browser cache.
- **SOLID -- system prompt: override box + draft persistence + chip
  (v1.62.3, reviewed b7ea7b6)**: a prompt typed before any conversation
  existed had no owner anywhere (page state only), so a reload ate it
  while the sampler params beside it survived via settings.js's
  localStorage -- the asymmetry behind "everything else loads just
  fine". It then fed real data loss: a Save with the box silently blank
  stored null over a good preset prompt. Draft now parks in
  localStorage until adopted; a promptless preset overrides nothing;
  `.chat__sysprompt-chip` states what is in force, empty case included.
  Provenance split into two predicates (matchesState = drift,
  equalsState = identity) so the override rule could not loosen the
  unstamped inference into a false claim.
- **Method note worth keeping**: four throwaway puppeteer repros against
  an isolated server (temp DB, NO model -- presets and conversations
  are pure DB, so this costs seconds) cleared the storage layer and the
  redraw before any code changed, and caught two of my own bad tests
  (a selector matching nothing, a premise that never reached the branch
  it claimed to test). The E2E-with-a-model run then found two more
  that static review could not.

UPDATE 2026-08-11 evening (v1.58.0-1.62.0, follow-on session, all on main):

- **SOLID -- models.toml comment preservation (v1.58.0)**: tomli_w stays
  authoritative for values; comments carried onto the fresh render ONLY
  while their anchor is unchanged (whole-model byte-identical through
  tomli_w normalization; a block above a [[models]] header also needs the
  FOLLOWING model unchanged). tomlkit is read-only extraction -- mutating
  a parsed AoT re-renders it as a malformed inline array (why the first
  attempt died); mechanism pinned in CLAUDE.md + toml_comments.py.
- **SOLID -- preset inheritance + edit box (v1.59.0, owner decisions)**:
  new conversations/notebooks START as the selected (or stamped) preset --
  prompt + params + applied_preset_id stamped at birth (starting-as IS an
  apply); create contracts gained applied_preset_id (spec §4). The
  "prompt not saved" report was the inert-select design colliding with
  the sampler-carries/prompt-doesn't asymmetry -- the save path never
  dropped anything. Chat's message-edit textarea now sizes to its message
  (rAF-timed grow, 60vh cap).
- **SOLID -- fit arc (v1.60.0, design §5 + ask #2)**: ram_report.py's
  arithmetic extracted to heylook_llm.ram_fit (script = renderer over the
  same structured report; --quiet contract unchanged); POST
  /v1/admin/models/{id}/fit with provider-derived hard_working_set
  (MLX-FAIL vs gguf-WARN is the point) + server-gated sysctl hint; v3
  Memory-fit section renders it verbatim (never client-computed), FAIL
  gates the row's Load button in place.
- **SOLID -- true continuation + Save & Continue (v1.61.0 + v1.61.1)**:
  continue_final_message on ChatRequest (auto = trailing assistant; true =
  any role, user-role MLX-only; false = never). The old convention only
  suppressed the generation prompt -- the turn rendered CLOSED and nothing
  continued. MLX passes continue_final_message through the template;
  continuation disarms the prefills_thinking parser assumption;
  llama-server continues natively but ECHOES the prefill (verified live) --
  stripped positionally. v3: Save & Continue on the message editor, both
  roles, streaming into the SAME row. v1.61.1 applied all 8 confirmed
  /code-review findings -- incl. two PRE-EXISTING Save & Regenerate races
  (missing s.stream guard; truncation anchored to a conv id captured
  before the first await) and typed in-band invalid_request_error frames
  on both streaming APIs (spec §4 now states the streaming caveat).
- **SOLID -- reload route + observed line (v1.62.0)**: POST
  /{id}/reload[?warm=true] = unload + load(+warm) as one server-owned op
  sharing /load's body; v3 "Reload now" uses it. Fit meter shows
  "Resident now" (measured after load) for loaded models -- §5's loop
  closer.
- **SOLID -- chat-reliability Phase 0 (v1.64.0, plan
  `plan_chat_orchestration.md`)**: loud refusals (stream + pendingSave
  latch + unsaved-row lock), unsaved-row Retry save/Discard, reconcile at
  every saga end, thinking-block editing. `e2e:render` grew 8->23 checks
  (stateful stub store, red-first, iPhone-emulation boot). The iOS
  hidden-row bug was reproduced on an iOS 26.5 simulator with a control and
  fixed in v1.79.1 (content-visibility gated on scroll-anchoring support;
  rows were displaced, never blank). File logging
  went opt-in in v1.63.0 (default `off`; llama-server log joined the switch).
- **SOLID -- conversation-scoped generate, backend half (v1.65.0, plan
  Phase 1)**: POST/DELETE /v1/conversations/{id}/generate -- server builds
  the request from the store, Messages SSE + `heylook_saved` authoritative
  rows, 409 arbitration, persistence on completion/abort/disconnect,
  truncation commits only with its replacement row (new atomic db ops).
  17 contract tests; spec §4. Phase 2 (v1.66.0) cut the client over:
  chat generates via /generate, Stop = DELETE, mirror adopts the store at
  stream end. render 23/23 through the cutover; live suite 45/45; gguf
  matrix 7/7 on DeepSeek V4 Flash. OPEN: Muse-Glimmer mis-templates at the
  engine level (TODO). CLOSED since (v1.74.0): disconnect-persist now has a
  live-suite check (reload mid-stream, detached task persists); notebook/
  explore migrated to /v1/messages (heylook_logprobs + message_stop timing
  extensions) -- no v3 page speaks /v1/chat/completions anymore.
- Also: worktree-per-session convention retired (owner call); editable
  install re-pointed at the primary; DuckDB store cleared by the owner
  (fresh state is intentional, not a bug).

UPDATE 2026-08-11 (v1.54.0-1.57.1, merged to main same day):

- **SOLID -- v3 per-model config editor (Wave 4 / 6b territory, arrived
  early)**: Models page Configure panel, schema-driven off
  `GET /v1/admin/model-options` -- the FIRST consumer of the v1.52 effect
  classes (until it, a misclassification was invisible, as v1.53's changelog
  predicted). Effect-class grouping, tri-state default selects, null =
  reset-to-default, armed reload, persistent reload-needed row marker,
  non-default summary chip. gguf plumbing (host/port/server_binary/
  startup_timeout_s) + mlx's derived `vision` deliberately not rendered.
  Design record: `internal/research/expert_offload_design_frontend.md`
  (this ships its schema-form half).
- **SOLID -- audit round (4 parallel review agents over yesterday's
  v1.52-1.53 work)**: found + fixed a HIGH semantic defect -- per_request
  defaults were load-time snapshots, so a PATCH on a loaded model reported
  no-reload-required while the process served the old value;
  `reload_config()` now refreshes them into loaded providers (pinned by
  `test_per_request_refresh.py`). Also: admin `config` = stored keys only
  (exclude_unset -- set vs default is now distinguishable on the wire);
  gguf gained the design-doc-approved-but-never-landed `-ctk`/`-ctv`,
  `-ncmoed`/`-cmoed`, `--spec-draft-n-min` (coverage vs pinned b10362
  otherwise clean); ModelUpdateRequest extra=forbid; chat stops an
  in-flight stream before a model switch re-labels the conversation.
- **SOLID -- /code-review high round (v1.56.0)**: 12 verified candidates,
  10 confirmed, all applied. The three load-bearing ones: reload-needed
  marker moved to SERVER truth (`stale_reload_fields` on admin responses,
  from router snapshot comparison -- client bookkeeping died on remount and
  mis-cleared on failed reloads); `_resolve_modalities` restores
  `__pydantic_fields_set__` (validator-assigned fields reported as stored,
  defeating exclude_unset); per_request refresh fixed for
  disabled-but-loaded models + guarded against re-import provider changes.
  Plus: hide policy as `ui:"hidden"` ON the fields (one source), arrays
  edit one-element-per-line (comma split corrupted `--tensor-split "3,1"`),
  save notes announced after mount.
- **SOLID -- chat model-switch arc (v1.57.0, design doc §15)**: G1 history
  media caps-gated at the wire with per-message drop disclosure (staged
  attachments still BLOCK -- deliberate asymmetry, commented both sites);
  G3 residency dots on the select + pre-switch warning (Cancel / Switch
  anyway; Send with unconfirmed target commits) + Load button (warm
  contract); G2 (v1.55.0) stop-stream before model_id changes hands.
  Still open from §15: G4 context estimate, G5 attribution (schema bump),
  F14 switch-lock during pending load.
- **SOLID -- update_deps writeback (audited KEEP WITH FIXES, fixes shipped
  v1.57.1)**: tomlkit round-trip comment-safe (verified live + now
  test-pinned); one guarded write helper refusing concurrent edits;
  rollback when the follow-up `uv lock` fails; first write-path tests.
  The primary's stray 7-package relock (incl. transformers 5.15.0) was
  committed as a NAMED act after the full suite passed under it.
  (Superseded v1.62.6: update_deps.py retired -- pyproject is a plain
  hand-maintained manifest, `scripts/build_llama.py` carries the llama.cpp
  build, `scripts/guard_stable_channel.sh` keeps git pins uncommitted, and
  `heylookllm import` is merge-preserving so hand edits survive reimport.)
- **Left**: see the HANDOFF block above + TODO.md
  "Config-editor / audit follow-ups". Final state: suite 1434 green,
  E2E pages 43/43 + chat 41/41.

UPDATE 2026-07-28 (v1.45.0-1.46.0 + full re-plan):

- **Gemma thinking-loop root-caused and fixed (v1.45.0)**: the cascade's
  thinking layer was keyed on a model-config flag nothing sets (dead code
  -- request-toggled thinking ran with zero repetition control), and the
  global floor was off-tune for gemma. NEW vendor layer reads each model's
  own generation_config.json (temp/top_p/top_k) above the floor; thinking
  overlay re-keyed on the effective switch and slimmed to loop control
  (presence_penalty). Template/tokenizer drift was RULED OUT against the
  latest upstream gemma-4 reference (byte-identical template).
  (The live re-verify + A/B this bullet originally left pending was
  DONE same day -- see the "Heretic gemma live verdict" bullet below.)
- **One shared cascade (v1.46.0)**: `samplers.resolve_effective_sampling`
  now serves BOTH providers; gguf adopted MLX semantics (request sampler
  suppresses default_sampler; unknown default_sampler log-skips). MLX
  `supports_thinking` config field removed (triple-shadowed by derived
  truth). Suite 1231 green, TDD red-first throughout.
- **Full RE-PLAN (plan_2026-07.md "Re-plan 2026-07-28")**: all remaining
  work regrouped into five WAVES -- 1 shrink (6a derive-at-load + RLM
  extract + batch collapse + radix simplification), 2 decompose (api.py +
  mlx_provider), 3 Messages migration, 4 grow (6b admin surface + eval
  page + perf-page analytics), 5 perf gated. New standing complexity
  guardrails (delete>decompose>migrate>grow; derived truth over
  materialized copies; one implementation per mechanism). Phase 6 split
  6a/6b; Phase 2 item 3 retired as superseded.
- **Wave 1 item 1 (6a) EXECUTED same day (v1.47.0-1.48.0)**: thin
  models.toml entries everywhere (CLI + admin import; all providers),
  modalities/template-source/cache-defaults derived at load
  (modality_detect.py, cache_defaults.py; stored values = explicit
  overrides), config_tui + --interactive + questionary retired, Phase 1
  item 8 discovered already-done. Suite 1243 green.
- **Heretic gemma live verdict (2026-07-28 PM)**: fix VALIDATED on
  healthy models (QAT passes vision bank 6/6 + the exact repro scenario
  on v1.48); the heretic itself is abliteration-DAMAGED on the
  vision+thinking path -- coherent for ~2 sentences then tail-token
  garbage collapse at every sampling incl. greedy, and its
  vision+thinking forwards trigger Metal command-buffer fault cascades
  that poison the process. RETIRED for vision+thinking (text-only OK).
  Details: session log 2026-07-28.
- **Wave 1 item 4 (radix simplification, Q7) EXECUTED 2026-08-18
  (v1.75.0)** -- single-slot snapshot cache, hybrid hole closed by
  refusal; see plan + mlx_provider.md §4.2. NEXT: item 2 (RLM extraction
  -- owner-gated, skipped for now per owner) -> item 3 (batch-internals
  collapse; NB its endpoint handler lives in api.py, which currently
  carries an unowned uncommitted diff -- resolve that first).**

UPDATE 2026-07-26 (PHASE 7, one day, v1.39.17 -> v1.44.2, 14 commits --
multi-provider is REAL again; full record: the session log + plan Phase 7 +
`internal/research/gguf_*_2026-07.md` dossiers):

- **SOLID -- provider seam (7a, v1.40.0)**: owned `GenerationChunk`
  (slotted, thinking channel, no error flag -- errors raise), BaseProvider
  capability surface (provider_name/template_info()), PROVIDER_CONFIG_CLASSES
  registry, dead mx.clear_cache gates fixed. OpenAPI byte-identical.
- **SOLID -- gguf/llama-server provider (7b, v1.41.0)**: subprocess-per-model
  (spawn/health/SIGTERM = router LRU), SSE->chunk adapter, sampler cascade.
  Eval bank 13/13 through the server on gemma-4-E4B gguf. Driver fleet in
  models.toml (E4B/12B/Qwen3.6s, modelzoo/gguf symlink).
- **SOLID -- spec decode (7c, v1.41.0-.42.0)**: both MTP stints live-verified
  on Metal (gemma sidecar drafter, qwen embedded); MEASURED model-dependent
  (+21% Qwen3.6 thinking @ 0.62 acceptance; NET LOSS on E4B) -> per-model
  opt-in policy; acceptance telemetry chunk->RequestEvent->trends->perf page.
- **SOLID -- audio input (7d, v1.42.0-.43.0)**: gguf-only `input_audio`
  parts + Messages AudioBlock + converters; MLX 400s loudly (was
  silent-drop); v3 capability-gated attach (fixed the never-gated image
  button) + audio chips/players; 2 audio eval tasks (bank 15/15 live).
- **SOLID -- gguf importer (7e detection, v1.44.0)**: scan builds gguf
  entries (mmproj-anywhere-in-name pairing, root mtp- drafter pairing,
  spec_type never auto-set), REFUSES HF assistant checkpoints, ignores
  imatrix. Real-fleet verified 6/8 dirs (2 assistants correctly skipped).
  Registry-substrate half (tomlkit merge, admin CRUD) stays with Phase 6.
- **Hardening (v1.44.1-.2)**: 5-reviewer /code-review (2 real bugs: dead
  gguf max_tokens overlay; Messages draft-telemetry omission) + 4-reviewer
  /simplify (presplit-merge + note_delta extractions, ATTACH_KINDS table,
  dead from_draft field dropped). Backend 1187 green; E2E chat 40/40.
- **Standing decisions recorded**: no gguf PIVOT (option purchase; posture
  doc validated, not amended); spec decode per-model; v3 gates modality UI
  off `capabilities` (modalities stays server-side description); trust
  order source > GGUF headers > model cards > docs pages (unsloth's MTP
  page had embedded/sidecar BACKWARDS).
- **Next per re-plan 2026-07-26**: Phase 2 (api.py decomposition) -- it
  grew again today and is formally recommitted ahead of Phase 3b.

UPDATE 2026-07-23 late (v1.39.12-.16, all eight handoff-memo findings closed;
memo + reasoning in `internal/handoff_findings_2026-07-23.md`):

- **Parser strip/holdback unified (v1.39.12).** One `StripSpecials` wrapper
  composed by `select_reasoning_parser`; the holdback is sized by the strip
  set and prefix-set based (declared specials are not all `<`-shaped). Also
  fixed harmony's dead final-flush partial strip. Design record:
  `docs/parser_strip_unification.md`.
- **The "post-abort immediate-EOS" was a PARSER BUG, not model behavior
  (v1.39.13).** Measured first: 0/64 empty replies across four history
  shapes, 0/6 vs 0/6 with real mid-stream disconnects -- aborts were never
  the variable. gemma-4 sometimes emits a spurious `<|channel>` mid-answer;
  with no newline terminating the header both channel parsers swallowed the
  rest of the turn as a channel name. Unrouted text now goes to content at
  end of turn. `tests/e2e/README.md`'s "empty-EOS is legal" principle is
  qualified accordingly: a RISING rate is a bug signal.
- **finish_reason=length (v1.39.14)** on both non-streaming paths, scraped
  once in `ChunkTelemetry.absorb` and latched. The eval bank's token-count
  workaround is retired to a fallback.
- **One shared prompt-section factory + notebook field-scoped prompt PUT
  (v1.39.15)**, plus E2E `serverGet`/`proveQuiet`. `raceState` deliberately
  not added (its two sites make different decisions).
- **Durable preset provenance + parser invariants as properties (v1.39.16).**
  `applied_preset_id` (schema v6) on conversations + notebooks -- explicit
  Apply/Update/Save-as-new stamps only, holds a preset id or null and never any prompt or
  model output; NOT stashed in `params` (that bag reaches the model).
  `TestParserInvariants` states the two properties today's bugs violated
  (chunking invariance; no silent loss) and checks them over randomised
  chunkings -- which retired both parked parser ideas (the `in_name`
  streaming bail-out and moving the streaming probe into the eval bank).
  Hover wrapped in `:where()` so selected state wins by source order.
- Bars now: **1115** unit/contract, **76/76** E2E (chat 40 + pages 36).
- OPERATIONAL: schema v6 means the next server start DROPS existing
  conversations and notebooks (owner approved starting fresh).

UPDATE 2026-07-23 (v1.39.1-.5, frontend v3 + E2E, all heylook code):

- **Chat system-prompt data loss FIXED (v1.39.1) -- SOLID, live-verified.**
  The editor committed only on `change` (blur); closing the settings drawer
  with Escape or a route change removed the focused textarea before `change`
  could fire, discarding the typed prompt from both state and the server (a
  preset saved in that window captured `system_prompt: null` and would later
  erase the conversation's prompt on apply). Editor now commits state on
  every keystroke (notebook parity) and debounces the PUT (400ms, flushed on
  blur) -- no close path can outrun the save. Same session: chat top-bar gear
  added (in-context opener for the shared drawer, alongside the existing
  sidebar-foot/bottom-nav gears), sysprompt section now always expanded
  (was collapsed-when-empty), editor grew 3 rows -> 9rem-min auto-growing,
  drawer widened 22rem -> 26rem.
- **v1.39.2-.4: the drawer-close data-loss class fixed at its root, not just
  for sysprompt.** v1.39.4 found the v1.39.1 fix was a point patch: ANY
  commit-on-`change` field (sampler number inputs included) loses an
  unblurred edit when `close()` removes it while focused. `close()` now
  blurs the focused field before clearing the drawer body -- one fix at the
  convergence point all close paths share, not per-field patches.
- **Preset apply made explicit + legible (v1.39.2).** Preset select is now
  INERT (no apply-on-`change`); a dedicated Apply button, armed-confirmed
  ("Replace prompt?") only when it would overwrite a differing non-empty
  prompt. A live drift line ("Matches current settings." / "Differs from
  current settings...") updates in place on prompt keystrokes and sampler
  edits, `role="status"` (v1.39.5) so the flip reaches screen readers.
- **Shared preset bar (v1.39.3) -- notebook is no longer chat-only.** The
  preset section was extracted to `apps/heylook-frontend-v3/js/preset-bar.js`
  (`createPresetBar` + a `getPrompt`/`setPrompt`/`onStatus` adapter); the
  notebook page now contributes the same bar ahead of its sysprompt section,
  identical grammar to chat. Decision: apply writes the notebook's system
  prompt too (a preset is a prompt+sampler bundle everywhere).
- **`tests/e2e/` full suite run live 2026-07-23 -- 65/66 on the full run,
  every check green across runs (chat re-run 30/30); all new
  preset/sysprompt/gear checks passed first try.** The one failure is a
  model-behavior flake, now documented (`tests/e2e/README.md`): post-abort,
  the model can emit EOS immediately on the next turn, so an empty reply
  saves nothing and the "new send completes normally" check flags -- not a
  UI bug (`finishStream` deliberately drops empty completions).
- Both prior "REMAINING" follow-ups from the 2026-07-09 preset entry below
  are now closed: notebook reuses the preset bar (this session), and the
  "panel drifted from selected preset" indicator shipped as the drift line
  (v1.39.2).

UPDATE 2026-07-20 (v1.34.59-.65 + dep bump 0bfe60b -- all heylook code; jlens-mlx
entries below are unchanged from 2026-07-13):

- **Gemma-4 thinking, end to end (v1.34.60-.63) -- SOLID + live-verified.** Root
  cause chain: the stop-token gate rejected gemma-4's canonical template because
  `_read_eos_tokens` only read `added_tokens_decoder` (tokenizer_config.json),
  missing the `<turn|>` terminator (id 106) that lives in tokenizer.json's
  `added_tokens` -- template_info emptied, thinking leaked inline as plain text
  with channel markers stripped (v1.34.62 fix: union both files, same
  dual-source rule as `_read_special_tokens`). New `GemmaChannelParser`
  (streaming state machine, `thought` channel -> `thinking` field) selected via
  template sniffing. VLM path additionally dropped `enable_thinking` on both
  legs (text-only + `prepare_vlm_inputs_parallel`) and generated past
  end-of-turn (raw HF tokenizers don't absorb `generation_config.json`'s eos
  list; mlx-lm's `stream_generate` auto-wraps with only the single
  `eos_token_id`) -- fixed by `extend_eos_from_generation_config` at load +
  `run_generation` wrapping raw tokenizers itself (`ensure_gen_tokenizer`) with
  the full resolved stop set (v1.34.63). Also removed the decode-path hygiene
  patch (`tokenizer_hygiene.py`, v1.34.5x) that defaulted
  `skip_special_tokens=True` and stripped channel markers before the parser
  could see them. Live-verified on gemma-4-31b: vision + text, think on/off,
  1/2/large images -- the previously reported multi-image + thinking gibberish
  did not reproduce.
- **Qwen3.5 thinking fixed + hardcoded vocab ids removed (v1.34.64).** Its
  template pre-fills `<think>\n` into the generation prompt, so output started
  mid-block with no opening tag and the parser routed everything to content --
  new `prefills_thinking` detection + `initial_thinking` parser state. The
  token-level thinking mode keyed on hardcoded Qwen3 ids (151667/151668),
  silently failing to split for any other `<think>`-family vocabulary; the
  parser is text-based only now (same as the harmony/gemma channel parsers).
- **Thinking capability auto-detected from the template (v1.34.60).** A chat
  template that references `enable_thinking` (Qwen3 `<think>` blocks, gemma-4
  thought channels) now reports the `thinking` capability on `/v1/models`
  without a manual `models.toml` flag -- the v3 checkbox/composer icon appear
  for these models automatically. Messages API's hardcoded `<think>`-only
  parser also replaced with `select_reasoning_parser` (streaming and
  non-streaming), matching chat/completions.
- **Model-agnostic vision token budget (v1.34.64) -- SOLID, live-verified,
  ahead of the Q8 spike.** `vision_tokens` request field + per-model
  models.toml default + v3 drawer control (cap-gated on `vision`), mapped by
  duck-typing the loaded processor: gemma-4 discrete buckets (snap to
  70/140/280/560/1120), qwen2/3-VL continuous pixel budget, unknown families
  degrade to processor defaults. Vision-feature cache key carries the budget.
  Verified on gemma-4-31b + Qwen3.5-27B (pixel patches 630/2520/10080 at
  70/280/1120). Closes the TODO.md research item (collapsed there into this
  entry). The Q8 ACCURACY question (does a bigger budget improve detail QA?)
  is still open -- the new eval harness's vision tasks are the vehicle.
- **v3 composer polish (v1.34.60-.61).** Multi-image attach capped at 8 with
  an aria-live announcement when exceeded + per-image "Remove image N" labels
  (the attach strip/picker/paste/N-block store shipped earlier, v1.34.20 --
  this closes the cap + a11y gaps). Attach button and a new thinking-toggle
  button are now icon buttons (`.btn--icon`, 40px touch floor, same cap-gate +
  true/unset semantics as the drawer checkbox, kept in sync via
  `onSettingsChange`); toggle state is styled off `aria-pressed`, not a class
  (pattern recorded in `apps/heylook-frontend-v3/DESIGN.md` §7).
- **`mlx_cache_limit_gb` operational setting (v1.34.59).** Opt-in cap on MLX's
  buffer cache via `/v1/admin/config` (bounds idle RSS -- the allocator never
  returns freed buffers to the OS -- at the cost of realloc on the next spike;
  clearing restores MLX's own default). Plus a real bug fix: `DELETE
  /v1/admin/config/{key}` now re-applies immediately instead of only taking
  effect after a restart.
- **`tests/eval/` -- opt-in LLM behavior-eval harness (v1.34.65), NEW.** 13
  tasks / 7 programmatic judges (`color_mention`, `marker_leak`, `repetition`,
  `token_budget_exhausted`, `exact_word_count`, `non_empty_non_gibberish`,
  `substring_present`), generalized from that day's live-verification scripts.
  Covers the four bug classes fixed today: thinking split/leak, stop
  discipline, vision single/multi-image correctness, vision_tokens budgets.
  Needs an already-running server (never spawns one); not wired into
  `/test-suite` -- opt-in via `uv run python tests/eval/run.py`. Direction for
  an API surface + a v3 eval page: TODO.md.
- **Pyright noise triage (v1.34.65).** Real fixes: deprecated
  `datetime.utcnow`, untyped `= None` defaults, a latent bug (batch responses
  could hand pydantic `model=None` -> runtime 500, now coalesced), a float
  re-binding bug in `_format_bytes`, duck-typed route discovery. Systemic: all
  96 positional Pydantic `Field(default, ...)` calls converted to explicit
  `default=` keyword form (this pyright build only recognizes the keyword
  form -- the source of every false "arguments missing" constructor
  complaint). `pyrightconfig.json` already existed from a prior session; this
  pass didn't need a new one, just fixed the real signal it was surfacing.
- **Dependency bump (0bfe60b): mlx-lm a790972->15b522f, mlx-vlm
  8e2638b->c9e27b08 (0.6.5->0.6.6, 53 commits).** Consumed surface verified
  signature-stable (22 contract tests green); transformers stays pinned to
  5.5.4 via the deliberate `override-dependencies` (mlx-vlm HEAD declares
  `>=5.14`; decoupled on purpose, contract tests are the gate, not the
  version floor). Verified: full suite 1085 green + a live eval-bank A/B (old
  vs new pin) on gemma-4-31b + Qwen3.5-27B, no regressions (two chronic
  trivial-prompt degeneration flaps exist on BOTH pins; one budget-exhaustion
  flap is harness calibration, not model drift).
- **MTP research probe (no server code shipped; `internal/research/mtp_probe/`,
  local-only).** First live measurement of the gemma-4 MTP assistant head
  (open mlx-lm PR #1276's model class, vendored standalone) drafting for the
  daily-driver MoE: 50.8% greedy acceptance with thinking on (CoT preamble
  flatters it), 36.2% with thinking off (still net-positive, ~1.30x decode
  ceiling projected at ~5-10% drafter overhead). Found and documented a PR
  #1276 gap (`AssistantAttention` doesn't do per-layer-type KV head selection,
  mis-shapes SDPA on this pair's `num_global_key_value_heads=2`); commenting on
  the PR with the data is an optional owner action, tracked in TODO.md. Feeds
  the existing "Gemma-4 MTP self-speculation" TODO item; doctrine still
  applies (greedy-only, fingerprint gate invalid for any batched-verify run).

No obsoletions found in this file from today's work -- all additive. The
Phase 4 item 3 "enable_thinking tri-state (auto/on/off)" REMAINING note below
is still accurate and unaffected: today's composer icon shipped the same
binary true/unset semantics as the existing drawer checkbox, not a tri-state.

UPDATE 2026-07-13 (jlens-mlx: abliteration diff finding + explainer, sibling repo):
- **The abliteration diff landed -- and it's the finding the project chases.** Stock lens
  (`out/band-n14-stock`, `identity_ok: true`) fit clean overnight on the SAME corpus as the
  abliterated `band-n14-fixed`; `diff_lenses.py` run BOTH substrate directions.
- **Result (robust, substrate-independent -- both directions agree layer-for-layer):** the
  abliterated transport surfaces safety/refusal vocab MORE in the mid-late band (L32-42:
  `Safety`/`unsafe`/`unethical`/`dangerous`/`Cannot`/`violations`, CJK `安全风险`/`违反`, Russian
  `безопасность`), and SUPPRESSES geography (China/Europe) + retrieval verbs. The tallest raw-l2 bars
  are the illegible early band (`*`-junk); the SIGNAL is the coherent mid-late safety cluster (read
  the shape, not the peak -- same lesson as the legibility metric).
- **Why MORE, not less (the headline) -- CORRECTED 2026-07-13:** abliteration edits the TRANSPORT,
  not the readout. Heretic (confirmed by reading its source) does directional ablation:
  it orthogonalizes the residual-WRITING matrices (every layer's `attn.o_proj`/`mlp.down_proj`)
  against `r = mean(harmful) - mean(harmless)` -- tail blocks INSIDE the fitted Jacobian, not
  outside it. `model.norm` is untouched (bit-identical), so the diff is a PURE TRANSPORT
  difference. **Interpretation RETRACTED 2026-07-13 (second correction, same day):** the diff is
  NOT a content-conditional "disposition preserved" reading -- a per-prompt re-run
  (`scripts/per_prompt_diff.py`, `out/per_prompt_diff.txt`) showed a benign weeknight-recipe
  prompt lights up the same L32-42 safety band just as strongly (mean l2 596 vs 524-571 for the
  safety-adjacent prompts) and surfaces the same refusal vocab (Nothing/Impossible/cannot/unsafe).
  The effect is PROMPT-INDEPENDENT -- no benign floor, no evidence of a content-conditional
  internal state. The diff recovers abliteration's STATIC WEIGHT-EDIT FINGERPRINT: a refusal
  direction in vocabulary space, readable on any input because the weight edit is always present,
  localized to WHERE the edit lives (L33/L36, matching the weight footprint).
- **Cross-validated by an independent weight-footprint analysis** (`scripts/abliteration_footprint.py`,
  jlens sibling repo; `out/abliteration_footprint.txt`): dequantizing both 8-bit builds shows the
  vision tower bit-identical (LM-only edit) and the edit ~6x concentrated in the residual-writing
  matrices vs input matrices (at the quant floor); the per-layer weight-delta peak (L33/L36)
  co-localizes with the transport-diff safety cluster (L32-42) -- two independent measurements
  agreeing on WHERE abliteration lives.
- **Artifacts (jlens research repo):** raw diffs in `out/` (`diff_ablit_vs_stock.txt` + `…_hereticsub.txt`);
  tracked write-up + visual explainer at `docs/abliteration_diff.md` + `docs/abliteration_diff_explainer.html`.
  Both original caveats RESOLVED: benign floor FALSIFIED on the old pair then REVERSED 2026-07-14 on a clean
  matched pair (magnitude prompt-independent, content prompt-conditional, floor HOLDS); quant-converter match
  CLOSED 2026-07-13 -- self-converted the base (mlx-vlm 0.6.5) vs the mlx-community base is uniform
  ~0.004 drift, no tent, o_proj/down_proj at the floor (~8x below the abliteration signal,
  structureless) -- converter asymmetry cannot manufacture the finding
  (`out/converter_drift_base_vs_mlxcommunity.txt`).
- **Own controlled abliteration + clean matched pair (2026-07-13 PM).** Owner ran Heretic on their
  4090 to abliterate the base (Trial 144: 41/100 refusals from 89 baseline, KL 0.0282, tent centers
  o_proj 45.6 / down_proj 57.9). Self-converted BOTH base and `heretic-ours` to 8-bit MLX with
  mlx-vlm 0.6.5 (VLM-aware; keeps the vision tower) -> `modelzoo/Qwen/Qwen3.5-27B-{8bit,heretic-8bit}-ours`.
  Clean footprint (`out/footprint_ourpair_trial144.txt`): untouched matrices EXACTLY 0.0 (same base+
  converter), only o_proj/down_proj/linear_attn.out_proj change (zero leakage). TWO results: (a) the
  footprint method is CALIBRATED against ground truth -- it recovers the known tent centers within ~2
  layers; (b) the ablation locus is RECIPE-DEPENDENT -- Trial 144 peaks DEEP (L42-59) vs coder3101's
  SHALLOW (L33/36), so "L33/36" was coder3101-specific, not a model property. **Overnight: two band
  lens fits running detached** (`jlens out/run_matched_pair_fits.sh`) -> by AM, diff+footprint on the
  clean pair; prediction = safety cluster deeper (~L42-59), still prompt-independent.

UPDATE 2026-07-12 PM (jlens-mlx: first clean own-fit lens + the memory saga, sibling repo):
- **First clean-corpus full-band own-fit lens produced.** `out/band-n14-fixed`: 11 items, band
  16-47, `identity_ok: true`, ~4.25h, zero SIGKILLs. Qualitative readout (the honest test) on the
  Eiffel probe: L40-42 surface meaningful tokens (Paris/city/France) -- the lens works where it
  matters; L45-47 collapse to degenerate ` __`/` ____`, but that's the MODEL's own degeneracy
  (abliterated+quant), not a fit fault.
- **The legibility metric MISLEADS -- same failure as the old fidelity gate.** It ranked the
  degenerate J_45/46/47 HIGHEST (0.91-0.93) and the meaningful J_40 at 0.85, because the degenerate
  readouts agree with the model's own degenerate output. Reproduced with a clean corpus AND the new
  disposition-aware metric -> it's a metric problem, not a corpus one. Judge readouts qualitatively;
  a real disposition-aware metric is still open.
- **Memory model CORRECTED: peak scales with FITTED POSITIONS, not sequence length** (~63GB base +
  ~2.1GB/position; validated live). The caching-allocator SIGKILLs (transition exit-137) were fixed
  for real with `mx.clear_cache()` between items (jlens `e56fad6`); item 10's drop via
  `JLENS_MAX_FIT_SEQ` was an over-drop on the wrong (sequence-slope) model -- it would have fit.
- **Abliteration diff set up as the overnight run.** A stock-model lens (`Qwen3.5-27B-8bit-mlx`) is
  fitting on the SAME corpus (`out/band-n14-stock`, same token sequences) so only the model varies;
  by morning, `diff_lenses.py` (ablit vs stock) yields the transport-geometry difference abliteration
  introduced. Also this session: capture parity verified BIT-EXACT (`36d859b`), fit metrics store +
  dashboard finished, both fit-math branches code-reviewed clean.

UPDATE 2026-07-12 (jlens-mlx corpus incident + upstream GDN PR eval, sibling repo):
- **Corpus incident found + fixed.** The `band-n12`/`band-n12b` own-fits on the served abliterated
  Qwen3.5-27B were degenerate: mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects
  `enable_thinking=True`, so every on-policy completion collapsed into shared CoT-preamble
  boilerplate (62% of all fitted positions, 71% of on-policy). `band-n12b` was stopped at 9/11
  items (checkpoint kept); both fits' results are discarded — the fitter math itself (chain-vs-
  direct, kernel parity) is unaffected. Fixed in jlens-mlx (`238826e`/`951dd76`/`232b98b`):
  explicit `enable_thinking` (default False, matching heylook's own served default), role-aware
  off-policy spans, a 16-token sink floor, and a diversity gate that hard-fails a corpus this
  degenerate. Refit on the fixed corpus is next.
- **Upstream mlx-lm GDN differentiability PRs validated.** PRs #1389 and #1217 (both add
  differentiable gated-delta ops) are numerically correct (3-way gradient agreement, rel ≤2.6e-7)
  and remove the GDN kernel's `T≤128` cap that's been capping fit corpus length — 27B QLoRA
  ~145-150 tok/s / 38-39GB on either PR vs ~50 tok/s / 117.5GB on main, no inference regression.
  **Decision: do not fork mlx-lm for the fit path** — jlens-mlx's outer-layer design keeps fit-side
  numerics identical to the SHA the server serves; the fork stays eval-only. Full detail + the
  serving-relevant PR triage (#1486/#1456, #1515/#1532, #1526, #1077, #997):
  `docs/jspace_integration_plan.md` § "Observations & watch-items", 2026-07-12 subsection.
- No heylook (this repo) code changed today — pure jlens-mlx + docs session. TODO.md updated with
  the follow-up items.

UPDATE 2026-07-11 (jlens-mlx fitting + DRY settings drawer):
- **Fitting pipeline matured** (jlens-mlx sibling repo): exact reverse-mode CHAIN fitter
  (verified == direct, cos 1.0; the default), cotangent dim-batching (2.4x), a corpus builder
  (weighted strata + on-policy generation + role/think position masks), a sequence-length cap
  (drop-not-truncate), per-item checkpoint/resume + JLENS_FINALIZE, and `decode_corpus` (stores a
  readable corpus by default). Own-fits on the served abliterated Qwen3.5-27B: band-5L done;
  `band-n12b` (band layers 16-47, N=12, cap 128 = all on-kernel) running as of this date.
- **Perf reality (designer+verifier pass):** the chain fit is ~44 min/item, a full band ~7-8h, and
  there is NO config-level 2-3x. `chunk_size` is a dead knob; the trap is the GDN kernel **MAX_T=128
  cliff** -- items >128 tokens fall to the slow differentiable ops fallback (~10x slower + memory
  blowup, OOM-adjacent at 143GB/192). Added a guardrail (warn + kernel-eligibility metadata in the
  sidecar); rule of thumb: cap corpora <=128 for the served qwen. Real speedup = seq-tile the GDN
  scan (scoped, deferred -- exact since it's a recurrence).
- **Fidelity gate misleads -> legibility metric:** the final-logit-agreement gate ranked a degenerate
  near-target layer ABOVE a meaningful mid-band one (band-5L: ' __' junk beat ' Paris'). New
  disposition-aware `verify.legibility_report` ranks band layers by real-content-vs-junk, wired into
  the fit output + sidecar (15 tests).
- **Abliteration study tooled + ready:** control = `mlx-community/Qwen3.5-27B-8bit` (downloaded, same
  base+quant, differs only in abliteration; heretic KL 0.065 from base). `scripts/diff_lenses.py`
  drives `verify.diff`. Fit both bands, then diff.
- **Fit/apply capture parity is ASSERTED, not verified** (fit captures cache-less; apply uses a fresh
  cache the hybrid qwen3_5 requires) -> a cheap go-forward numerical check. Does NOT invalidate current
  lenses (identity KL~0 is consistent). The old "capture.py must be byte-identical" invariant was false,
  corrected.
- **DRY settings drawer (v3 Phase 2 / plan Phase 4):** the chat settings UI extracted into an app-shell
  **global slide-over drawer** shared by all 6 pages (sampling / global display prefs / per-page extras
  taxonomy). Edge cases preserved (focus guard, preset fingerprint diff, stale-textarea sysprompt),
  browser-verified, code-reviewed. `show_special_tokens` display pref gated (`wired:false`) until a
  render surface honors it (DESIGN.md §6).
- **Docs:** an end-to-end research report (`internal/research/jspace_jlens_end_to_end.md`) written, then
  two-agent critiqued + corrected; `docs/jspace_integration_plan.md` Part 2 extended (Neuronpedia
  refinements, `coderef/mlxui-core` prior-art for activation patching, the fit/apply parity item).

UPDATE 2026-07-10 (j-space + jlens-mlx): the j-space **apply** feature
(Jacobian-lens workspace readout) shipped in the ~v1.34.31-.35 range (per
CHANGELOG) -- `/v1/jspace/{models,analyze}`, the v3 `jspace` page, V1/V2
apply-parity (cos 1.0), the V4 hallucination router, gen-gate coordination, a
thread-stream crash fix. Lens **fitting** was EXTRACTED to a new sibling repo
**`jlens-mlx`** (this server only APPLIES -- same lean-scheduler pattern as Q6's
`rlm-heylook`). Two GREEN milestones there: apply-path parity + a baseline fitter
(Anthropic direct-VJP design -- norm outside J, no chain, no closed-form seed)
cross-checked vs Anthropic's torch `jlens` (J cos 1.0). Go-forward:
`docs/jspace_integration_plan.md` Part 2. Known issue: the served
`Qwen3.5-27B-abliterated` lens is `hf_model_name=""` / likely fit on STOCK Qwen --
treat its readouts as provisional until we own-fit.

UPDATE 2026-07-09 (v1.34.22): per-conversation system prompt editing +
saved presets shipped in v3. Backend: new `presets` table in the DuckDB
store (db.py), added ADDITIVELY via `CREATE TABLE IF NOT EXISTS` --
deliberately no `_SCHEMA_VERSION` bump (a version mismatch drops and
recreates every table, which would nuke existing data for an additive
change). Name uniqueness enforced in code on the store's single serialized
writer (`PresetNameTaken` -> HTTP 409), same rationale as the earlier
dropped messages FK. Presets are deliberately EXCLUDED from
`clear_all_data`/`POST /v1/data/clear` (config, not data) -- a test pins
this. New router `preset_api.py`: `/v1/presets` GET/POST,
`/v1/presets/{id}` PUT/DELETE, tag "Presets" (server's 8th API router).
409 name collision, 400 bad/empty fields, 404 unknown id. These are
UI-authored bundles (`{name, system_prompt, params}`) expanded
CLIENT-side into explicit request fields when applied -- deliberately
distinct from the server-side TOML sampler registry (now `samplers.py`
/ `ChatRequest.sampler`, renamed from "presets" 2026-07-20); no wire relationship. Frontend v3: chat settings
panel gained a per-conversation system-prompt editor (details/textarea,
PUTs `system_prompt` to the conversation on blur; a prompt typed before
the first send rides along on the implicit conversation create) and a
preset bar (select applies a preset = COPIES params into the sampler
panel + system prompt onto the conversation, LM Studio semantics, no
live binding; name input + Save creates or overwrites by name; armed
Del). `settings.js` gained `snapshotSettings()`/`applySettings()` and a
`lead` option on `buildSettingsPanel`. Tests: +16 store unit
(`tests/unit/test_preset_store.py`), +9 HTTP-level router tests
(`tests/unit/test_preset_api.py` -- the repo's first HTTP-level unit test
for a storage router, minimal FastAPI app + httpx ASGITransport); suites
880 green. E2E chat suite +3 checks (system-prompt persist, preset
save/apply round-trip, armed delete): 55/55 live. Follow-ups at the time:
notebook page could reuse the preset bar (the `lead` hook makes it cheap;
chat-first was deliberate); a "panel drifted from selected preset"
indicator was deferred. BOTH CLOSED 2026-07-23 (v1.39.2-.3, see the update
above): notebook now contributes the shared preset bar, and the drift line
is the drifted-panel indicator.

Phase: **PHASE 1 COMPLETE + E2E rebuilt (51 checks live-green) + mlx 0.32.0
upgrade DONE** (v1.34.11): v0.32.0 ships the real CompilerCache-teardown fix
(mlx#3619/PR#3628) -- proven by a discriminating A/B repro (tuple-output
compiled fn on a worker thread: SIGTRAP on 0.31.2, clean on 0.32.0); both
venvs upgraded; suites green (839 backend + 65 optloop-lib + E2E; the
cadence guard has a documented cold-shader false positive after mlx bumps,
see tests/e2e/README.md). Spec-decode TEXT baseline + first experiment
DONE (v1.34.12, mlx 0.32.0): classic 1B->27B draft decode is NET-NEGATIVE on
the bandwidth-bound bf16 target (composite 0.91 at num_draft=2, 0.96 at 4;
short-context turns +10% at nd=4 but long_context -40%) -- confirms the
"verification-based decoding, not classic draft" thesis. Full numbers +
verdict in docs/optimization_log.md; harness validated itself. TEXT was
photo-independent so it did NOT wait on photos.
VLM VISION BASELINES DONE 2026-07-07 (v1.34.16). The mrope "blocker" was just
the stale Mar-15 mlx-vlm fork -- pulling the owner's synced forks (mlx-vlm
0.6.5 #1529, mlx-lm 0.31.3 #1431; uv sync clean, mlx stayed 0.32.0) runs
gemma-4 dense/MoE AND Qwen3-VL clean through the manual vision path. No
wrap_language_model port needed (that v1.34.13 TODO is moot). Owner's synced
forks + downloaded models (gemma-4 8bit dense/MoE + assistant drafter),
per-model baselines (v1.34.14), model-name scrub (v1.34.15),
3 models safe-merged into models.toml (append; 13 tuned entries intact).
Baselines: dense gemma-4-31b-it-8bit = 15.3 gen_tps / 1592ms vision / 33.3GB;
MoE gemma-4-26b-a4b-it-8bit = 48.1 gen_tps / 524ms vision / 27.3GB. MoE ~3x
faster (bandwidth-bound dense vs dispatch-bound MoE = distinct optimization
profiles). Full numbers in docs/optimization_log.md.
NEXT (optloop lane): the MTP experiment -- MoE + gemma-4-26B-A4B-it-assistant-
bf16 drafter via mlx-vlm's draft_kind="mtp" (model exposes speculative_draft_
hidden / speculative_logits_from_hidden). Needs the bench's decode routed
through mlx-vlm's generate (not mlx-lm stream_generate) -- own focused pass;
the shot at the verification-based-decoding win classic 1B-draft (net-negative)
missed. Minor: bench_config vlm still points at Qwen3-VL (v1.34.13); switch to
a gemma-4 default with the MTP work. Branch: `main` (v1.34.25). The E2E/optloop
lane has wrapped; the version counter has since advanced through the v3 lane
(DuckDB+images v1.34.20-.21, presets v1.34.22-.24, legacy-app deletion +
drift-guard retirement v1.34.25). The MTP experiment above is the one item still
open in the optloop lane.

## Phase 1 results (this session, all TDD, each its own commit)

- **Item 1 (v1.34.1) SOLID + LIVE-VERIFIED**: streaming delivery
  unquantized (asyncio.wait on the chunk future). Live A/B on the MoE
  gemma-4-26B-A4B: 88.3 chunks/s client-observed, 11.1ms median gap
  (old ceiling: ~10/s, ~100ms). 31B dense gemma measures ~10.4 tok/s
  NATIVE -- it genuinely decodes at the old ceiling, don't use it to
  demo this fix.
- **Item 2 (v1.34.1) SOLID + LIVE-VERIFIED**: recorded tok/s = native
  mlx-lm generation_tps (perf_collector.headline_tps; fallback excludes
  queue wait); TTFT excludes queue_wait (kept as own field);
  /v1/messages prompt_tps formula fixed; trends + resource-snapshot
  averages success-only (live: a failed request left the trend at 50.0
  avg over 2 successes instead of dragging it to 33). RequestEvent
  gains prompt_tps -> flows to request_events.jsonl.
- **Item 3 (v1.34.1) SOLID**: close-timed-out executors quarantined
  (strong ref forever), never dropped-to-GC.
- **Item 4 (v1.34.2) SOLID**: reasoning parser per-request from
  _template_info (shared instance raced concurrent streams); strip
  pattern lru_cached; provider no longer builds a load-time parser.
- **Item 5 (v1.34.3) SOLID**: router TOCTOU (placeholder slot reserved
  under cache_lock; concurrent different-model loads can't over-commit)
  + idle unload checks generation_queue_stats (active+waiting) under
  the same lock as the pop.
- **Item 6 (v1.34.2) SOLID**: embedding pad_token -> eos_token at load.
- **Item 7 (v1.34.5) SOLID**: tests/contract/test_mlxvlm_surface.py, 22
  tests pinning the consumed mlx-lm/mlx-vlm surface, each naming its
  consumption site; anti-contamination guard vs the contract conftest's
  session mocks.
- **Item 8 (v1.34.4) SOLID**: already_configured matches resolved paths
  too (symlink-safe); re-import = PUT semantics.
- **Item 9 DRAFTED, filing = owner action**: ready-to-file upstream MLX
  issue at internal/backend/upstream_mlx_compilercache_issue.md (stacks
  re-verified from both .ips reports; minimal-repro attempt documented
  negative). Public post under owner's name -- review and paste.

Suite: 762 unit + 72 contract green. Two broken model configs noticed
during live verification (pre-existing, NOT from this work):
Ministral-3-3B-Base-2512 (mlx-vlm tokenizer_utils crash:
PixtralImageProcessorPil has no .vocab) and
gemma-4-E4B-...-int8-affine (missing k_proj/v_proj weights at load).
Worth pruning or re-quantizing.

This file is the handoff. Statuses are graded honestly:
**SOLID** = built + verified end-to-end · **HALF-BAKED** = works but a known gap remains ·
**UNCERTAIN** = done but verification was partial · **STUB/NOT DONE** = planned, not built.

ECOSYSTEM POSTURE (read before perf/provider work):
`docs/architecture/ecosystem_strategy.md` -- Python MLX stack is
the right rail but maintenance-mode upstream (SHA-pin, check the open-PR
backlog before workarounds, expect spec-decode etc. via sidecars like
dflash-mlx, never an mlx-lm release; Apple's frontier ships in mlx-swift-lm
-- feeds the Swift tripwire watch). Invariant to protect: run_generation()
single chokepoint + the provider seam in config.py. Distilled into the
plan's Direction section too.

THE PLAN for what's next: `docs/project/plan_2026-07.md` -- phased
(0: owner decisions, 1: correctness debt, 2: consolidation, 3: v2
retirement, 4: v3 hardening incl. rebuilding E2E, 5: perf). Start there.

Daily detail: `internal/log/log_2026-07-05.md` (v3 build) and `log_2026-07-06.md`
(crash fix, defaults audit, this cleanup). Older Slice-1 status that used to live
here: see git history of this file (all landed long ago).

## Evening additions (2026-07-06, after the doc above was written)

All in the plan with full rationale; one-liners here so nothing is missed:

- **Phase 0: every decision made except Q3 (E2E timing -- decide during
  re-plan).** Q4 Messages-first + namespaced extensions (consumer inventory
  done: shrug-prompter has a batch-parse bug to fix client-side); Q5 DuckDB
  app-state + JSONL analytics; Q6 RLM -> sibling repo `rlm-heylook`
  (exists, empty; client+research shape recommended); Q7 radix ->
  single-slot; images server-owned w/ ACCURACY-primary spike; batch
  collapses onto native upstream generators.
- **Swift rewrite: NO** -- decision + 4 tripwires in plan Direction;
  BaseProvider contract is the sidecar hedge.
- **mlx-vlm dependency strategy adopted** -- contract tests are Phase 1
  item 6 (transcribe the 07-06 drift audit); per-model shrink; upstream
  postmortems; contingency doc.
- **Measurement reality check (plan Phase 5, READ before any perf work):**
  server telemetry cannot detect a ~20% streaming regression (100ms poll
  quantization, queue-wait conflation, error-polluted means, native mlx-lm
  tps never read). Phase 1 streaming fix is now ALSO a measurement
  prerequisite; new honest-headline-metrics item; spec-decode baseline
  protocol pinned.
- **STATUS CHANGE -- optloop: app-level DELETED (v1.34.0).** Its benches
  never imported heylook_llm; zero cycles ever ran. optloop-lib is the only
  bench (new CLAUDE.md there; models.toml resolution ported in, 65 tests).
  Remaining tail: real-photo + long-context workloads BEFORE first
  baseline; thin HTTP serving-path bench AFTER Phase 1 telemetry fixes.
  NOTE: root pyproject pins UPSTREAM mlx-lm/mlx-vlm, not the local forks --
  fork-side wins don't reach the server until upstreamed/repointed.

## The plan (as executed 2026-07-05/06)

1. Rewrite frontend v2 as v3 per `docs/frontend_v3_spec.md` (the complete build
   contract -- §4 is the authoritative backend API contract, §8 the build order).
2. Fix whatever the build shook loose in the backend (§9 co-evolution ladder).
3. Audit the months-old model import/config/loading system.
4. Clean up the test suite + verify assumptions against current mlx/mlx-lm/mlx-vlm.

## 1. Frontend v3 (`apps/heylook-frontend-v3/`, served at /v3; /v2 deleted in v1.77.0)

- **SOLID -- chat**: conversations CRUD, streaming w/ thinking blocks,
  edit/regenerate/delete via single-message delete & server-side generate,
  media by reference (schema v7), drag-and-drop & paste staging, model attribution,
  status telemetry line, mobile drawer, tab resume store re-adoption, row reuse
  with in-place reconcile (content-visibility jumping eliminated).
- **SOLID -- shared layer**: `js/page.js` createPage lifecycle (hide/resume hooks),
  hash router, table-generated `api.js`, `streaming.js` (typed SSE, `streamMessages`
  and `streamGenerate` sharing `streamTypedSSE`), settings panel & global drawer
  (null = backend-cascade contract, `show_special_tokens` wired).
- **SOLID -- notebook, models, perf, explore**: notebook & explore migrated to
  `POST /v1/messages` (Phase 3b); models page has discovered vs config distinction
  + editable scan watch folders; perf page has no-polling perf; token explorer
  has $O(N)$ streaming fragment optimizations.
- **SOLID -- visual design & layout**: impeccable audit + polish across all pages;
  iOS Safari scroll anchoring fix (`@supports (overflow-anchor: auto)`);
  modern `@supports (field-sizing: content)` textarea auto-sizing decoupling
  forced layout reads; canonical CSS design tokens.
- **SOLID -- cutover (v1.77.0)**: retiring v2 / promoting v3 is COMPLETE.
  `apps/heylook-frontend-v2/` deleted; `/v2` returns 404 (pinned by contract test);
  v3 at `/v3` is the sole frontend.
- **SOLID -- E2E & render suites**: `tests/e2e/` (puppeteer-core + system Chrome,
  chat 46/46, pages 43/43 live-green) + model-free `bun run e2e:render` (36/36 checks).
- **HALF-BAKED -- j-space visualizer track (v1.34.36-.37, v1.79.7)**: click-to-pin
  readout (strip rows + heatmap cells, Esc/arrow-key walk, same-top-token echo,
  onset marker), layer-range slider + aggregation detail panel, provisional lens
  badge, matrix performance optimizations (v1.79.7 single-pass iterative min/max,
  fast cell lookup). Live streaming (SSE) and interactive interventions remain open.
- **STUB -- batch page**: dropped from v3 scope on purpose (spec §6); the
  backend endpoint remains.

## 2. Backend changes v3 depends on (all committed, v1.31.1-v1.32.0)

- **SOLID -- crash fixes** (each root-caused from evidence, deterministic
  repro before/after, regression-tested):
  - radix snapshots materialized on the generating thread (v1.31.1) --
    "no Stream(gpu,N)" on prefix reuse; A/B proven 4/4 crash -> 4/4 clean.
  - persistent pinned-executor pool (v1.31.2) -- MLX threads are NEVER
    destroyed; per-request thread teardown aborted the whole process via
    MLX's thread-local CompilerCache destructor (SIGTRAP, two matching
    .ips crash reports). See tests/unit/test_streaming_executor_pool.py.
- **SOLID -- SSE error contract** (v1.31.1): generation failures are
  `data: {"error":{...,"code":"generation_failed"}}` + `[DONE]` (OpenAI),
  `event: error` (Messages), HTTP 500 (non-streaming) -- never assistant
  content. Contract-tested. v3 handles it; **v2/legacy do NOT** (empty
  response on failure -- accepted for retiring UIs).
- **SOLID -- error altitude (v1.33.0)**: provider now RAISES typed
  GenerationFailed / InvalidGenerationRequest (400) instead of yielding an
  is_error sentinel; batch and RLM fail loudly through their existing
  handlers instead of concatenating error text. Streaming wire contract
  unchanged (v3 needed no edits). Contract-tested; 774 passed.
- **SOLID -- request defaults** (v1.32.0): global sampler floor 0.1/512 ->
  0.7/4096 (`GLOBAL_SAMPLER_FLOOR`); imports stamp default_preset
  "balanced" (was deprecated "moderate"). UI-visible: bare chat requests no
  longer near-greedy or truncated at 512 tokens. Unit-tested; live-verified
  standard cache; the long-generation live check timed out (slow 32B model)
  so 4096-cap behavior is unit-verified only.
- **SOLID -- import defaults** (v1.31.3): KV quant is RAM-relative (>35% of
  unified memory), max_kv_size NEVER defaulted (it silently truncates
  context). models.toml migrated: everything <67GB -> standard cache; the
  122GB/155GB giants keep 8-bit KV uncapped; max_loaded_models=2,
  idle_unload_seconds=0 for this 192GB box.
- **SOLID -- config strictness** (v1.32.0): extra="forbid" (typos fail at
  load), kv_bits Literal[2,4,8], kv_group_size Literal[32,64,128],
  rotating-requires-max_kv_size validator, max_queue_depth is a real field.
  `quantized_kv_start` REMOVED (was dead config) -- strip it from any old
  models.toml. Importer size_gb now from safetensors bytes (name regex was
  returning params-count as GB).
- **UNCERTAIN -- radix gate for non-standard caches** (v1.32.0): lookup+store
  now bypassed unless cache_type=standard with no max_kv_size (fixes a
  documented silent-wrong-output risk). Unit-tested via mocks; NOT yet
  exercised live against a quantized-cache giant (122B/235B) -- behavior is
  "radix off for them", which is safe but means no prefix reuse there.

## 3. Model import/config audit -- findings fixed vs deferred

Fixed: everything in section 2 above. Deferred (documented, not built):
- typed GenerationFailed refactor (TODO.md, batch/RLM behavior change);
- re-import is skip-not-update and `already_configured` matches by id only;
- update-on-reimport flow;
- upstream MLX bug report (CompilerCache TLS destructor drops Python refs
  without the GIL -- .ips reports in the macOS DiagnosticReports folder).

## 4. Test-suite + library-drift cleanup -- DONE (v1.32.1)

- **SOLID -- library drift**: every load-bearing mlx/mlx-lm/mlx-vlm
  assumption verified against the INSTALLED libraries (0.31.2/0.31.3/0.6.3,
  transformers 5.5.4): no broken sites. Fixed 3x mx.metal.device_info
  deprecation; deleted 2 dead transformers patches + the dead VLM
  strict-load fallback.
- **SOLID -- test consolidation**: -26 tests without coverage loss (drifted
  fake provider file deleted, tautological classes removed, duplicates
  folded, uninterruptible timeout payload fixed). 760 passing (713 unit).
  tests/README.md scrubbed of false "pre-existing failures" claims.
- **KNOWN GAPS left on purpose** (noted in-place in the test files): real
  coverage for the prefill-convention logic (mlx_provider) and
  _extract_from_layer index math (hidden_states) needs small refactors to
  expose testable helpers -- the deleted tests only asserted their own math.
- One unexplained single-run flake earlier on 2026-07-06 (1/778, never
  reproduced; name not captured). Watch for recurrence.

## Invariants that bite (learned the hard way)

- tests/unit green with 0 unexpected skips (Metal-gated skips OK); any
  failure is a regression.
- NEVER destroy a thread that ran MLX work (process abort); executors are
  leased from _PinnedExecutorPool, never shut down.
- Radix snapshots must be mx.eval'd before insert; radix only for
  cache_type=standard without max_kv_size.
- Verification of /v3 in a browser: puppeteer-core + system Chrome;
  claude-in-chrome blocks localhost by policy.

## Blockers

None.
