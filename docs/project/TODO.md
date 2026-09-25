# Persistent TODOs

Cross-session task backlog organized by priority.

*Last reviewed: 2026-09-25, a full triage (through v2.0.154); the "Next
session" section below added at v2.0.167. Every section
was classed done, obsolete, open or mixed with evidence
(internal/claude/todo_triage.md); the done and obsolete ones moved verbatim
to internal/archive/todo_closed_2026-09-25.md. Mixed sections stay whole: their
checked-off items are history kept beside the open ones.*

## Next session: open owner calls and queued work (2026-09-25, mrpurple handoff)

The three-phase plan is done (v2.0.150 - v2.0.167; CURRENT.md handoff). What
is left, each with the recommendation the owner was given:

- [ ] **Raw output view (owner asked 2026-09-25; recommendation given, awaiting
  go).** Input side is DONE today on both engines: the prompt preview (composer
  eye, editor "Preview prompt") shows the engine's own render with the model's
  own markers highlighted (v2.0.163). Response side:
  - MLX: fully buildable. Store the engine's text before the reasoning split
    (every marker the model emitted) as a raw copy per reply, and append the
    stop token that ended it (never decoded; the engine knows its id). A
    per-message **Raw** toggle renders it with `highlightSpecials`. Replies
    from before the change have no raw copy.
  - gguf: llama-server hides control tokens and splits reasoning before heylook
    sees the text. Recommended: the toggle shows disabled with that reason.
    The one real option is per-request `reasoning_format: "none"` (reasoning
    inline with `<think>` as text; control tokens still hidden) with heylook's
    parser doing the split, which replaces llama-server's parser on the main
    gguf path: a separate owner decision. `--special` at spawn is not
    recommended (every request and llama-server's own parser see control
    tokens). `__verbose` needs server verbosity > 9; `return_tokens` reaches
    only the non-chat completion JSON (llama.cpp server-schema.cpp:34).
- [ ] **Remove the static gzip cache (awaiting owner yes).** `frontend_static`
  keeps each gzipped asset in memory; recompressing all of them costs the
  server a few ms per cold page load, and the only test is a call-count pin
  (a hit is byte-identical to a miss). Recommended: remove the cache, keep
  gzip, delete `test_the_gzip_cache_survives_a_multi_asset_page_load`.
- [ ] **The 3 live-only kept tests** (second pass, v2.0.165):
  `TestVisionFeatureCachePatterns::test_our_call_site_hands_over_the_cache`
  can be deleted now (vlm_parity_probe reports `feature_cache_hit`, green at
  v2.0.166). `test_extension_sampler_fields_reach_the_provider` (presence
  penalty) and `test_a_deliberate_unload_still_sweeps_the_engine` each need a
  live replacement in `tests/smoke/` first (a fixed-seed A/B with and without
  the penalty; Metal memory falls after an explicit unload), then delete.
- [ ] **The 49 borderline tests** (`internal/claude/prune/second_pass.md`
  "Borderline"). Recommended, mechanical (tests only, no GPU): delete the
  tautological half of each (e.g. asserting a monkeypatch it just installed)
  and keep the observable half. Owner calls recommended as KEEP: the ~5 spec
  pins of deliberate owner defaults (`max_loaded_models == 1`), the 2
  "derive, never hand-copy" lints, the MODEL_BUSY AST lint (2 tests), and the
  destructor-branch check. Delete the 2 test-helper-fidelity tests with their
  helper.
- [ ] **Tools / function calling:** 422 until a client needs them (owner call).
- [ ] **Notebook images:** later (owner call); audio waits for a model that
  supports it.

## From the 2026-09-24 improvement loop (merged as v2.0.121)

Record: `internal/claude/improve/archive/runs-2026-09-24/` (report.html has the evidence).

- [x] **Done in v2.0.135** (kwargs for models without `encode_image()`, the old branch elsewhere, content keys; parity probe with a warm-features case and a perf A/B, record in `internal/claude/vision_cache/`). **qwen3_5 vision features: owner said yes (2026-09-24), done the reworked way below, in a fresh session.** heylook's vision feature
  cache runs only for models with `encode_image()`; qwen3_5 has none, so every
  turn of an image conversation re-runs the vision tower. Passing the cache as
  mlx-vlm's `vision_cache`/`_image_key` kwargs (as mlx-vlm's server does)
  fixes it: commit 9573911 on ref `improve/2026-09-24-vision-cache`. It was
  reverted because the image-adding turn on Qwen3.5-0.8B breached the loop's
  speed tolerance; the 27B showed no cost. Do NOT cherry-pick it as is: it
  deletes heylook's `encode_image()` branch, and lfm2_vl, mimo_v2,
  minimax_m3_vl, molmo, molmo2 and sam3 have `encode_image()` but do not read
  `vision_cache`, so they would lose feature caching. Re-apply as: the
  kwargs where the model reads them, the old branch elsewhere, and a content
  key for http(s) image URLs (local paths are refused since v2.0.128; the URL key serves stale features when the
  file changes). Then `scripts/vlm_parity_probe.py` and an image warm==cold.
- [x] **mlx-vlm pin moved to upstream main** `ac737ef3` (v0.7.3, v2.0.124; owner:
  "the current latest commit"), which includes #2328. Suite green; vision
  parity ok and chain probe matching on Qwen3.5-0.8B. Smoke green on all three arms on it (v2.0.133 record).
- [ ] **Move the pin again past Blaizzy/mlx-vlm#2356** once it merges (restored
  qwen3_5 decode). Usual suite, chain probe and smoke.
- [x] **Security: CORS, admin argv, RLM** (v2.0.123, owner call): the CORS
  wildcard is gone, admin writes refuse `config.FILE_ONLY_FIELDS` (422), and
  RLM (with its `sandbox: false` request field) is removed.
- [x] **The partial inference API key is removed** (v2.0.127, owner: the
  server is LAN-only and the key was never set). `HEYLOOK_API_KEY` gated
  messages, model load and request cancellation but not the conversation,
  notebook, preset or generate routers, so it looked like protection without
  being it. If the server ever leaves the trusted LAN, a real gate must cover
  every router at once (one app-level dependency, not per-router) and ship
  with the Host check below.
- [x] **MLX image sources: local file paths refused** (v2.0.128, owner:
  `utils.load_image` accepts only `data:image/...` and `http(s)://`; llama-server
  already refuses `file://` without `--media-path`, which heylook never passes).
  Web links stay (Anthropic's `url` image source); keying the vision feature
  cache by content rather than by URL comes with the qwen3_5 re-apply above.
- [x] **Host check built (v2.0.137)**: `host_check.py`; IP addresses, localhost, the
  machine's own names and heylook.toml's `allowed_hosts` pass. **Owner: if a client
  reaches the server by a LAN DNS or VPN name, list it in `allowed_hosts`.** Was: No Host check, so DNS rebinding
  can reach the API through a browser already inside the LAN (LAN-only does
  not cover this one). The allowed hosts come from local config (heylook.toml),
  never from a tracked file.
- [ ] **Upstream PR to separate APC captures from store size** and let a
  caller name boundaries; it would delete heylook's local capture rule
  (`vlm_engine.install_capture_policy`). Drafted in `internal/claude/improve/ledger.md`, under "Upstream drafts (NOT posted)".
  Kept here, not filed (owner, 2026-09-24): file it only after #2356 lands
  and the owner says go, since filing sends it off this machine.
- [x] **Hook: refuse `git add -A`/`-u`/`.`** and `git commit -a`
  (`scripts/hooks/git_add_guard.py`, v2.0.125, owner-approved).
- [x] **Model pinning deleted** (v2.0.126, owner yes): RLM was its only caller.

## Runtime visibility + one behaviour across engines (2026-09-23) — APPROVED

Plan: [`plan_runtime_visibility.md`](./plan_runtime_visibility.md). Evidence:
[`../testing/gguf_runtime_audit_2026-09-23.md`](../testing/gguf_runtime_audit_2026-09-23.md).

Its W0 IS the entry below (registry sidecars), and it runs in parallel rather
than first. Order, revised 2026-09-23 after the measurements (the plan's
Sequencing section carries the reasons):
- W8 non-causal image guard + W9 keep-alive (both small; W9 measured);
- the eval-bank port (done v2.0.71);
- W13 one engine contract, then W5 cache/spec reporting as its first member;
- W10 MLX caching: first a spike ending in a named outcome (A1 mlx-vlm drives
  vision / A2 remove mlx-lm / B1 own checkpoints / B2 upstream hooks), then
  build (reopened; the biggest daily win);
- W2+W3 thinking detection + templates, with W7 thinking budget;
- W4 image geometry;
- W0 from Phase 0, then W1 load panel (both shipped; W1 v2.0.136, and the
  observed flash-attention auto confirmed on a live gguf load in v2.0.144);
- W6 only if W5 shows budget skips;
- W11 upstream llama.cpp PRs, optional.
- W12 profiling (decode overhead + image preprocessing): DONE 2026-09-25,
  outcome "both small, stay Python" (plan W12; data in internal/claude/w12/).

## gguf Continue: does llama-server's prefill render drop a generation prefix? (2026-09-23)

v2.0.85 fixed MLX: gemma-4 with thinking off opens an empty thought channel
in its generation prompt that its history render omits, and Continue then
degraded (fixed-seed A/B record in `internal/claude/w2/`). llama-server
continues an assistant prefill with its own render, so the same question
stands there for gemma-4 gguf: compare `/apply-template` for the history plus
generation prompt against the continuation render, then a seeded A/B on the
output if they differ. Belongs with W2 (thinking and templates).

## Unit-suite pruning pass, AFTER W5 + W10 (2026-09-23)

Owner direction: the tier that finds real bugs here is live checks that
assert on engine behaviour and telemetry. Unit tests earn their keep on pure
invariants (schema, parser properties, config validation, a guard tested
through its route) and not on mocked engine behaviour.

Once W5's live cache checks and W10 exist:
- classify each unit test as a pure invariant (keep), mock-driven (candidate
  to delete now that live coverage exists) or covering removed code (delete);
- let the owner weigh the candidates.

No mutation ritual; the owner rejected it. Start from
`docs/testing/audit_2026-09-08_backend_suite.md`.

## Eval-gate reminder: bring it back or not (2026-09-23)

The eval bank speaks `/v1/messages` again (v2.0.71, an adapter in `run.py`;
`/eval-ab` unblocked). Still open: whether the eval-gate reminder retired with
hookify comes back as a native PostToolUse hook (logic in `scripts/hooks/`).

## Retire per-model entries from models.toml (2026-09-08)

Phases 0-4 of [`plan_registry_sidecars.md`](./plan_registry_sidecars.md) are
done (v2.0.107-115): models.toml holds no per-model entries, a model's own
settings live in its folder's `model.heylook.toml`, admin edits write that
file, only the daily server writes, and discovery pairs drafters wherever
they ship. Read the plan before touching `model_registry.py`,
`model_importer.py`, `model_service.py` or the admin config editor.

- [x] **DeepSeek live checks** (2026-09-24): Vision Q8 and Q4 draft with the
  neighbouring folder's MXFP4 dspark on text and image; ggml-org Vision and
  0731 draft with their own. With too little free memory, Q4 dropped its
  drafter at spawn as designed ("short by N GiB"). Record:
  `internal/log/log_2026-09-24.md`.
- [x] **Spec decode in the engine report** (`engine.speculative`, v2.0.120): whether a
  drafter is available, whether it is in force, and why, with provenance, so
  "available, not in use" shows on the models page.
- [x] **models.toml becomes `heylook.toml`** (v2.0.122) with the DuckDB settings
  (`observability_level`, `observability_retention_days`,
  `mlx_cache_limit_gb`) folded in (owner decision 2026-09-23). Rename the
  references in `.worktreeinclude` and `docs/loops.md` in the same change.
- [x] **Retire the `enabled` field and `watch_hf_cache`** (v2.0.118, v2.0.119; the periodic rescan and `/discovered` went with them) (owner: presence in
  a scan folder is enabled).
- [ ] **Qwen3.8-Flash-Next's `MTP/` drafter**: a split-out head (no
  `token_embd.weight`) that the llama.cpp build cannot load. Its
  `model.heylook.toml` unsets it; drop that `unset` once a build loads it.

## Loop setup for the improvement-loops plugin (2026-09-24)

`docs/loops.md` is the profile the plugin's `/improve` and `/optimize` read as
part of AGENTS.md.

- [ ] **Fill Measurement with `/design-scoreboard`** (user-invoked): primary
  scenarios, guardrails and counter-checks for the reuse and
  time-to-first-token goal; then `/claim-audit` on `docs/loops.md`.
- [ ] **Delete the two heylook loop prompts** in
  `docs/prompts/optimizer_and_improvement_loops/` once everything in them
  lives in `docs/loops.md` (after `/design-scoreboard`). The generic prompts
  and template went in v2.0.116: the plugin owns them, and the copies here had
  already fallen behind it.
- [ ] **Promote the loop's scenario runner** from
  `internal/claude/improve/harness/` into `scripts/` once it has tracked real
  use over several runs; `scripts/perf_ab.py` is the provider-level half.
- [x] **Pre-plugin run record archived** (2026-09-24, owner and mropt): moved to
  `internal/claude/improve/archive/runs-2026-09-24/`, so it no longer blocks tidying.

## e2e: audit for clicks or seeds before an async list lands (2026-09-23)

Races of this shape surfaced at v2.0.73 (the panel seed before capabilities
landed; the danger-zone click during the model-list layout shift). Audit
only the checks the next workstream touching the chat or models page
touches, not the whole suite.

## Frontend coverage gaps found mapping chat.js (2026-09-06)

Surfaced by a full structural map of `frontend/js/pages/chat.js` (3078 lines).
None is a known bug; each is a path no automated check reaches, which is what
a future refactor would fly blind through. Listed most-worth-doing first.

- [ ] **Audio attachments have no end-to-end check at all** (P2). `addFiles`
  and `ATTACH_KINDS` are ONE shared funnel for images and audio, and every
  image path through it is covered (cap at 8, paste, drop, refusal, resize,
  round-trip) while no audio path is -- so the factory is half-tested and
  reads as fully tested. The staging half is client-side and stubbable in
  `render.mjs` (cap, chips in `renderAttachStrip`, `buildContentBlocks`);
  the send/render half needs a gguf model, which `tests/e2e/suites/chat.mjs`
  already has an arm for. Asymmetric coverage of a shared factory is exactly
  the shape that let paste ship image-only while looking done.
- [ ] **Partly covered since v2.0.136**: `e2e:render`'s load-panel check clicks
  `chat__load-btn` on a cold gguf model and asserts the reload body. Still
  unchecked: the resident Reload branch, `warm_error` and the
  `context_running` status line. **The Load / Reload button path is untested** (P3). `refreshLoadBtn`,
  `loadModelNow`, the gguf reload-with-`ctx_size` branch, and the
  `warm_error` / `context_running` status lines: no check references
  `chat__load-btn`. This is the one place the chat page can spend a model
  load, and the ctx_size branch writes config through the server.
- [ ] **Sidebar rename is untested** (P3), including the guard in
  `refreshAfterResume` that leaves the list alone while a rename input is
  open. That guard exists because a WebKit rename commits against the
  conversation object it started on and the list swapping underneath orphans
  it -- a real bug, with nothing pinning the fix.
- [ ] **Per-message model attribution is untested** (P3): the
  `message-model-note` element and the `mixedModels` branch of `msgSignature`.
- [ ] **`ABANDON_RANK` is unfalsifiable as written** (P4). Its own comment
  says nothing reads `DELETE` today, so no observable behaviour distinguishes
  the ranks and no check can pin them. Either give it an observable or drop
  the rank.

Two structural notes from the same pass, both already acted on: the two
misleading section dividers were renamed (v2.0.5), and the superseded-stream
guard in `finishGenerate` was fixed with a check that was shown red first.
The map's conclusion on splitting the file: don't. The streaming seam alone
needs 11 state fields and ~15 functions crossing the boundary, and would
separate the scroll-follow rule from `paintStream`, which enforces it.

## Frontend/backend state boundary (2026-09-08) -- DONE except the device pass

From an audit of what the frontend keeps client-side versus what the server
stores. Two rules sorted every finding, and the audit found both broken in
opposite directions:

1. **A client-local value may decide what you SEE. It may never decide what the
   server STORES.** The twin of the rule `params` already has.
2. **Per-browser state belongs per-browser; per-document state belongs on the
   document.** Place-keeping was stored nowhere at all.

Net effect: `settings.js` now touches no browser storage, and the app is down to
two browser-local keys (the parked system-prompt draft and which conversation
this browser is looking at), both behind one `heylook.` namespace and one
wrapper in `utils.js`. Backend suite green; `bun run e2e:render` green.

- [x] **The chat-template textarea was live before it had a baseline -- DONE**
  (`194cabf`, found and fixed in a parallel session). It could PUT a typed
  fragment as a model's entire chat template after a failed load. Its comment
  covers a case worth knowing: a REJECTED save deliberately leaves the box
  enabled, because that path never re-renders and the repair belongs in the box
  you typed into.
- [x] **`show_special_tokens` REMOVED -- DONE v2.0.38.** Owner call: remove
  rather than fix, until there is a plan to do it right. It was a per-BROWSER
  display pref that decided what the conversation store PERSISTED, so the same
  conversation continued from two devices accumulated rows of two kinds with
  nothing recording which. It was the only `DISPLAY_META` entry, so the whole
  display-pref layer went with it (store, wire helper, drawer panel, both pages'
  declarations, the `heylook-v3-display` key). A client still sending it on
  `/v1/messages` gets a 422 naming the removal, guarded on `MessageCreateRequest`
  and tested THROUGH the route; the generate route simply does not declare it.
  `_strip_history_specials` STAYS and is not redundant -- `content` is
  user-updatable, so an edited assistant row can still carry a control token
  that must not re-enter a prompt, which is how its two tests now seed.
  **2026-09-25, owner: the NEED is seeing the markers, and the prompt preview
  serves it; the store-unstripped design below was set aside, not built.**
  Checked before building it: a stop token ends generation without being
  decoded (vlm_engine), structural markers (`<think>`, channels) are consumed
  by the routing parser, and llama-server emits no specials without a
  server-wide `--special` that would change every request. So a stored
  unstripped reply would show almost nothing. v2.0.163 instead makes the
  preview's highlighting exact: the route returns `markers`, the model's own
  added tokens (special or not) that occur in the rendered prompt, read from
  its tokenizer files (gguf: the header's control + user-defined tokens).
  Still open, only if the preview proves not enough: a per-message RAW
  output view on MLX (what the model produced before the split) -- that does
  need a stored raw copy. The `strip_specials=False` seam and its tests stay
  for it.
  **The earlier design, kept for the record:** store always
  UNSTRIPPED and strip at READ (`GET /v1/conversations/{id}` takes the flag,
  `specials_stripper` does the work), which makes the choice genuinely
  render-time and applies it to replies that already exist -- the wart the
  removed pref's own help text had to admit. Three traps, each verified while
  scoping this:
  (a) the strip SET costs a parse of the model's `tokenizer.json`, which is
  large, so it needs a cached probe beside `capabilities.py`'s two -- with its
  OWN stamp, because `_TEMPLATE_SOURCE_FILES` omits `tokenizer.json` while the
  strip set reads both files; keying it on `_template_stamp` would be the named
  subset-of-a-list defect in the one function that has already shipped a cache
  bug twice (uncached v1.71.0, stale-keyed until v2.0.22).
  (b) `model_id` on a row is nullable AND sometimes co-written: `_persist_result`
  stamps a FRESH assistant row only, and a continuation keeps the anchor's stamp
  because the merged row had two authors. So: strip per row against its own
  model's set, and do NOT strip where the id is absent.
  (c) under-stripping is the safe direction (it shows a marker that should have
  been hidden, which §6's default already accepts); stripping against a union of
  the conversation's models would OVER-strip and delete real text.
  The parser-level `strip_specials=False` knob is kept deliberately for this --
  it is the seam that design needs. `tests/unit/test_reasoning_parser.py`
  says so, and says to delete the knob with the design if it is abandoned.
- [x] **Composer text is per-conversation -- DONE.** `s.composerDrafts`, a Map
  for the life of the mount, swapped in `selectConversation` beside
  `clearPendingAttachments`. The defect fixed is a WRONG ACTION, not lost work:
  text typed in one conversation followed you to another and Send put it there,
  while its attachments did not follow. Not stored -- re-typable text is not
  worth a key that needs collecting on delete. `refreshAfterResume` cannot
  disturb it by construction: it never moves `activeId` and never touches the
  textarea.
- [x] **The sampler bag is no longer persisted -- DONE.** `settings.js` holds it
  in memory. The panel is a VIEW of a document, overwritten by
  `hydrateDocParams` on every select, so the stored copy's only durable job was
  seeding the next new document -- already lost across a reload the moment
  `conversations[0]` hydrated. The cross-page carry (chat and notebook share the
  module cache within a session) SURVIVES by decision; `documentScopeNote` now
  states it rather than the code hiding it. Scoping the cache per page was
  priced and declined.
- [x] **Active conversation id restored -- DONE.** The one place adding a
  browser-local key is right. Degrades silently to the newest conversation when
  the id is missing, unreadable or gone -- which on a phone is the COMMON path,
  since iOS evicts script-writable storage.
- [x] **`mergeKnown` range-checks values -- DONE.** Against the min/max
  `PARAM_META` already declares, for every source: `presets.params` and a
  document's `params` hydrate the same panel through the same function, so
  de-persisting localStorage removed one source, not the class. An out-of-range
  value becomes null (the cascade), which the panel shows as its placeholder,
  rather than riding to the wire and returning a 422 that names the field but
  not where the value was stored.
- [x] **`heylook-v3-scan-paths` deleted; keys renamed -- DONE.** A one-off scan
  is by definition the thing you are not doing again; a folder worth re-scanning
  belongs in the watch list, which is server config and reaches every browser.
  Remaining keys live under one `heylook.` prefix behind `lsRead`/`lsWrite` in
  `utils.js`, and `sweepRetiredStorage()` clears the retired names at boot --
  stale keys are not clearable by hand on a phone, which decided it.
- [x] **The e2e harness seeds DOCUMENTS, not localStorage -- DONE.** It capped
  generation length by seeding `heylook-v3-settings` before boot. That is gone,
  and the replacement is not a workaround: the document was ALREADY the
  authoritative half (`chat.mjs`'s "the DOCUMENT's params win" check exists
  because someone found that out), so `ctx.open()` and `newFreshConversation`
  PUT params directly and a failed seed is now an HTTP error instead of a
  silently ignored cache write. `ctx.readSettings()` reads the document too,
  which is the stronger assertion. `settings: null` means reload-and-seed-
  nothing, distinct from `{}` which would erase.
- [x] **`render.mjs` resets browser storage per boot -- DONE, and it caught a
  real one.** Every boot shares one browser profile, so the new
  remembered-conversation key leaked across them and landed later boots on an
  earlier boot's conversation. It failed two checks that never mention storage
  ("no thinking control on a thinking-capable model"; a composer reading Send
  for a generating conversation) because both were looking at the wrong
  conversation. Confirmed by control: green with the restore disabled, those
  two red with it enabled, green again with the per-boot sweep. The sweep is a `heylook.` prefix
  match in `evaluateOnNewDocument`, so no future key needs remembering there.

- [ ] **Verify the batch on the iPhone 17 Pro** (P2). THE ONE THING NOT DONE --
  it needs the physical device. `bun run e2e:render` is green and real WebKit at
  phone size is available in the simulator, but neither can raise the software
  keyboard (see the entry below). Three of the changes above have device-only
  risk. The composer Map is the sharp one: swapping textarea content plus
  `autoGrow` while the keyboard is up is exactly where iOS keeps the LAYOUT
  viewport and shrinks only the VISUAL one, and restoring text into a focused
  field can move the caret and scroll. The active-conversation restore needs its
  evicted-storage path walked rather than assumed. And the models page's
  unsaved-template warning rides `createUnloadGuard`, which binds `beforeunload`
  ALONE while the codebase binds both spellings everywhere else (`ctx.onHide`)
  precisely because one is not enough here -- so whether that dialog fires on the
  phone at all is unknown and worth one check while the device is attached.
  Do it in one sitting with the iOS entry below.

## iOS keyboard check: RUN, keyboard still uncovered (2026-09-05, updated 2026-09-06)

- [ ] **Run the keyboard half on a real device** (P2). The harness itself is
  no longer unrun: `tests/e2e/ios-sim.mjs` first ran 2026-09-06 and its own
  header carries the result and the method. What it established is a NEGATIVE
  worth keeping -- the Simulator will not raise the software keyboard under
  `safaridriver` (WebDriver click leaves activeElement on BODY;
  ConnectHardwareKeyboard=false changes nothing; Element Send Keys DOES focus
  the field and the visual viewport still never shrinks). Focus works, the
  keyboard does not exist there. The four keyboard checks are therefore gated
  on the ENVIRONMENT, not on the outcome, so they skip loudly rather than
  passing vacuously, and `IOS_REAL_DEVICE=1` runs them for real. What still
  runs in the simulator is real WebKit at phone size, which the Chrome 390px
  checks only emulate. LEFT TO DO: attach the iPhone 17 Pro over Web Inspector
  and run with `IOS_REAL_DEVICE=1`; the "report" check prints whether the
  bottom nav is still on screen with the keyboard up, which decides whether
  hiding the nav on composer focus (log_2026-09-05) is worth building. The
  simulator lists 15 Pro devices, so safe areas differ by a few points from
  the real target anyway. This is the same device pass the state-boundary
  section above depends on -- do them in one sitting.

## Observability follow-ups (2026-08-19, from the startup-record review)

- [x] **Pre-warm load telemetry is dropped -- FIXED v2.0.166** (P3): the pre-warm now runs from the lifespan after settings (`ModelRouter.prewarm_startup_model`); it was worse than stated, the constructor ran before the memory manager existed, so the load was recorded at no level at all. a `--model-id` startup
  load runs in server.py BEFORE the lifespan resolves `observability_level`
  from the DB, so with telemetry enabled the most expensive load of the run
  is missing from events.jsonl/model_events.jsonl while every later load is
  recorded. Same pre-configure class as the fixed startup-record bug; fix =
  resolve settings (or replay the load event) before/after the pre-warm.
- [x] **No test pins the lifespan ordering -- DONE v2.0.166** (`tests/contract/test_startup_order.py`: settings, then the startup record, then the pre-warm, through the real lifespan) (P3): log_startup_info must run
  AFTER apply_runtime_settings in api.py's lifespan; both unit tests call it
  directly with the level already set, so a refactor moving it back beside
  MemoryManager construction regresses silently. Needs a contract test that
  seeds the settings DB before app startup.

## A checkout's bundled chat template is a VERSION (2026-09-20)

Found while tracing "the system prompt is being ignored". Not urgent -- the
one affected model has a local override -- but the failure mode is silent and
will recur with the next download.

mlx-vlm wraps system content in BLOCK form for every `LIST_WITH_*` format, so
a template whose system branch is string-only renders a Python repr of the
prompt into the model's context. `google_gemma-4-31B-it-mlx-mxfp8` ships such
a template (`{{ messages[0]['content'] | trim }}`); the other seven gemma-4
dirs here ship the canonical Google one (published 2026-07-09) which branches
on `is string` / `is sequence`. Same base model, same family, different
template version -- so "gemma-4 works" is a claim about a CHECKOUT, not a
model. It hits text turns too, not just vision: an mlx-vlm-routed model
renders through `vlm_apply_chat_template` on both paths.

Worked around with `chat_template.heylook.jinja` in that model's folder
(untracked, no config write, revert = delete the file).

- [ ] Consider a LOAD-TIME check: render a probe message with block-form
  system content and warn if the output contains `{'type':` . Cheap, catches
  the whole class, and needs no per-model knowledge. The alternative --
  noticing that a model has quietly been reading a Python repr of its own
  system prompt -- is not something the current surfaces would ever show.
- [ ] Decide whether `render_prompt`'s preview should surface it instead (it
  would have made this visible immediately, since the preview shows the
  rendered prompt).

## MLX vision prefill: what v2.0.55 left open (2026-09-21)

The vision path now prefills the way mlx-vlm's own loop does and lets mlx-lm
sample the first token (`docs/architecture/mlx_provider.md`, `VLMVisionStrategy`).
Verified by token parity against `mlx_vlm.generate.ar.generate_step`
(`scripts/vlm_parity_probe.py`) on qwen3_5 and qwen3_vl. Open:

- **PARKED (owner, 2026-09-25): gemma models on MLX are set aside until new
  weights arrive; every MLX gemma except diffusiongemma was removed from the
  model folder.** The two gemma-4 vision items below wait for that.
- [ ] **Measured 2026-09-25, still open**: `scripts/vlm_parity_probe.py` on
  `gemma-4-26B-A4B-it-heretic-4bit` DIVERGES from mlx-vlm's own loop in every
  case at an upstream top-1/top-2 margin of 0.5 (four bf16 quanta, not a
  tie), identically on v2.0.140 and with v2.0.141's cache route, with the
  prompt matching mlx-vlm's. So it predates the caching change; warm features
  equal the cold run. Records: `internal/claude/vision_cache/parity_gemma4_*`.
  **gemma-4 vision is UNVERIFIED on the new path** (owner call: "an outlier,
  another time"). The path is generic, so gemma-4 image requests go through it
  and their output changed with nothing having checked it. Run the parity probe
  on `gemma-4-26b-a4b-it-8bit-mlx`. Two things make gemma-4 the interesting
  family rather than a formality: upstream REFUSES to chunk its prefill when
  images are present (bidirectional vision blocks), so it takes the single-call
  branch the qwen runs never exercise, including the `_PER_TOKEN_PREFILL_KWARGS`
  slice of `mm_token_type_ids`; and see the next item.
- [ ] **The pre-v2.0.55 gemma-4 vision path ran NON-CAUSAL attention over the
  prompt, and nobody has measured what that cost.** heylook always passed an
  int32 ones `mask` into the full-VLM forward; gemma-4's language model uses a
  caller's mask INSTEAD of building its own causal + sliding-window +
  bidirectional masks (`mlx_vlm/models/gemma4/language.py`, the
  `if mask is None` branch), and at the op level that ones mask equals no mask.
  Read from code and checked on the attention op alone -- never on a loaded
  model. It is fixed by construction now (no mask reaches the language model),
  but "gemma-4 vision quality before v2.0.55" is an open question, and any
  earlier gemma-4 vision observation in `internal/research/` was taken under
  it. The same exposure applied to gemma3, pixtral and llava_next; qwen was
  never affected (its mask does not reach attention). A parity-probe run of
  the OLD commit on gemma-4 would size it: expect a real divergence, not the
  one-quantum near-ties the qwen baseline showed.
- [x] **Penalty scope differed by path** -- CLOSED v2.0.60, and not the way
  this item first proposed. It was filed as "processors do not see the prompt
  on the vision path", with seeding the prompt as the fix. Reading mlx-lm
  showed the scope was an accident on EVERY path (whole prompt on a cold text
  request, the uncached suffix on a prompt-cache hit, reply-only on vision), so
  "make vision match text" would have spread the variant that penalises
  end-of-turn tokens and a system prompt's vocabulary. Owner chose generated
  tokens only, everywhere (`generation_core.generated_only`).
- [x] `mx.reset_peak_memory()` ran after the vision prefill -- CLOSED v2.0.60;
  the strategy resets before its prefill and `run_generation` no longer resets
  again for a pre-filled cache.
- [ ] **gguf penalties still count the tail of the prompt, and cannot be made
  not to by request.** llama.cpp's server accepts every prompt token into the
  sampler and penalises over its recent-token window (read in `coderef/`, not
  measured). So `presence_penalty` is not the same knob on the two engines.
  Documented in `docs/api_integration.md`; nothing to build unless upstream
  grows a generated-only switch.

## E2E chat suite: `send streams an assistant reply that persists` fails on a warm repeat run (2026-09-08)

- [ ] **The suite's first generating check times out.** (P2) The failure line is
  `never entered streaming timed out after 15000ms` -- kept on one line here
  because that string is what someone hitting this will grep for. Reproducible
  on the second and later runs against an already-warm server. It passed on the first run of the session against the
  same server and the same commit, so the trigger is server/warmth state and
  not code.

  NOT from the v2.0.38 work: a control with `tests/e2e/suites/chat.mjs`
  reverted to HEAD fails identically (48/49), which is what rules the session's
  own harness and skip-conversion changes out. Two other explanations were
  checked and ruled out too, so do not re-spend the time:

  - **Not a model load.** `/v1/admin/models` reported the model `loaded: true`
    at the time of failure, and `models.toml` configures no idle-unload
    (`idle_unload_seconds` / `unload_after_idle_seconds` are unset;
    `max_loaded_models = 1` is the only related key). There is no reload for
    the 15s budget to be short for.
  - **Not a stale assertion string** -- the failure class this repo hits most.
    The check waits for the composer button to read exactly `'Stop'`
    (`tests/e2e/suites/chat.mjs`, via `sendBtnLabel`), and
    `frontend/js/pages/chat.js:3007` sets exactly that for a local stream. The
    string is current.

  WHAT IS LEFT, and the lead worth taking first: the check polls a TRANSIENT
  state. `waitFor` runs every 100ms for 15s, and the composer's `'Stop'` label
  exists only while the run is in flight -- so a generation short enough to
  start and finish between two polls is never observed, and the suite caps
  generations at `E2E_MAX_TOKENS` (24 by default) against a model measured here
  at ~99 tok/s. Warm repeat runs replay an identical prompt, so a warm prompt
  cache makes each run faster than the last, which fits "passed cold, fails
  warm" without proving it. Confirm before acting: log the observed label
  sequence, or assert on something non-transient (the assistant bubble
  appearing, or the generate POST) instead.

  Worth knowing before "just assert on something else": the label is not an
  arbitrary handle. `frontend/js/pages/chat.js:2160` documents it as the only
  user-visible thing separating the composer's two REST states, written from
  one speller precisely because a drifting title once made "Stop" mean two
  different things with nothing on screen distinguishing them. The check is
  aimed at a real contract; the fragility is that it samples a transient
  edge of it rather than that it reads a label.

## Sidecar-template follow-ups (2026-08-30 code review, v1.79.43)

Acted-on findings are in the changelog. These three were judged real and left
open; each names what would settle it.

- [x] **Latent false positive in the v2.0.139 check -- FIXED v2.0.164** (P3, review 2026-09-25): on MLX
  `loaded_chat_template` is read back from the processor, and transformers' processor
  takes a legacy `chat_template.json` over `chat_template.jinja`, while heylook's
  ladder ranks jinja first. A VL folder whose two bodies differ would show
  `chat_template` stale forever (a reload cannot clear it) and `/reload` would never
  take its plain-load shortcut. The template editor's `stale` has had the same gap.
  Not live: the two local folders with both files have identical bodies. The real
  question under it is which body MLX actually renders with; settle that, then make
  the ladder and the loaded value agree.
  Settled: the VISION path renders the processor's copy (mlx-vlm's
  get_chat_template picks the processor first), so it was the .json body. Auto
  now puts a `chat_template.jinja` winner on a processor holding a different body
  (`template_info._jinja_outranks_processor`), except when the jinja has no media
  handling and the processor's copy does; a tokenizer_config winner still leaves
  the processor alone (it may be the text-only template). The loaded value, the
  ladder, thinking detection and `stale` now agree.
- [x] **Fixed v2.0.139**: `stale_reload_fields` carries `chat_template` when the file a
  respawn would use differs from the loaded template (the editor's own comparison). Was:
  **A sidecar swap is invisible to `stale_reload_fields`** (P2): that
  field is the repo's own "saved value differs from what the running process
  has" truth, and it derives from models.toml. A `chat_template.jinja`
  created, edited or deleted next to a loaded model's weights changes what a
  respawn would use while models.toml is untouched, so the Models page shows
  the model fully in sync. The spawn log says what you GOT; nothing says what
  is running has gone stale. Done when a template file's mtime/presence feeds
  the staleness answer, or the limitation is recorded as accepted with a
  reason.
- [ ] **Re-measure the Qwen3.8-27B template facts** (P3): CLAUDE.md records,
  as live-measured, that ggml-org embeds the official template at 8952 bytes
  and unsloth a patched 9993, and that the difference decides whether two
  leading system messages render or 500. That entry's sidecar is 9708 bytes --
  a THIRD template -- so as of v1.79.43 the entry no longer runs the template
  that prose describes. Done when the two-leading-system-messages probe is
  re-run against the sidecar and CLAUDE.md either confirms or splits the
  claim. Note this compounds the `reasoning_effort` re-check above: both are
  template-dependent and the default flip moved the variable underneath them.
- [x] **Fixed 2026-09-25**: the drift test's model lives in a directory the module
  creates (`MODEL`), so no shared `/tmp` file can reach its argv. Was:
  **`_build_args` reaches the filesystem now** (P3): it is documented as
  pure and exercised by `tests/unit/test_gguf_argv_matches_metadata.py` with
  paths that do not exist. Sidecar discovery probes the model file's parent,
  and that test uses `model_path="/tmp/model.gguf"` -- so a
  world-writable `/tmp/chat_template.jinja` would make every parametrized
  case emit a spurious `--chat-template-file`. Nothing creates that file
  today. Done when the drift test builds under a tmp_path it owns, or
  `_build_args` takes the template as a required argument so it cannot probe.
- [ ] **"Fails open" is overstated for the vision capability** (P3): the
  guarantee is that only POSITIVE non-support drops `vision`, but
  `mlx_vlm_supports` returns a clean False when mlx-vlm is merely absent or
  broken -- so an install condition reads as a model property and every
  `loader=auto` VLM loses the capability at once, with one INFO line per
  model_type as the only signal. Sharpened by the uncommitted uv.lock already
  bumping mlx-vlm, which makes `uv lock --upgrade` a thing that can change
  which models advertise vision. Done when an import failure is
  distinguished from an unregistered model_type, or the docs stop claiming
  the stronger guarantee.

## Chat reliability (2026-08-13, plan: `plan_chat_orchestration.md`)

- [x] **Stream-end blank frame -- DONE v1.72.2**: post-stream adoption now
  assigns from `heylook_saved`'s rows synchronously; the wholesale GET is
  the no-rows fallback only; list DOM structure is renderMessages-only.
  Guarded by the render-suite swap check (shown red against pre-fix tree).
- [x] **Media by reference -- DONE v1.73.0 (schema v7)**: base64 relocates
  to a content-addressed per-conversation blob store at every message
  write; rows carry url sources (`/{id}/media/{media_id}`, immutable +
  cacheable); the generate saga inlines blob bytes at wire build; GC on
  last-reference delete (retention-safe direction). Conversation reads are
  text-sized now.
- [x] **Delete means delete -- DONE v1.73.0**: single-row
  `DELETE /{id}/messages/{msg_id}`; the v3 button calls it; positions keep
  gaps. `?after` truncation stays API-only. Render check pins
  neighbor-survival (shown red against the truncation-era client).
- [ ] **Intermittent Metal fault cascade / load-hang under load-unload
  churn** (P2, observed 2026-08-18 during Q7 verification): three
  incidents in one afternoon on a hot machine -- a mid-generation wedge, a
  0.8B weight-load hang after a MoE unload, and a
  kIOGPUCommandBufferCallbackErrorSubmissionsIgnored cascade that 500'd
  every later MLX request. The cascade REPRODUCED ON STASHED BASELINE
  code, so it predates the v1.75.0 cache rewrite; morning runs of the
  same bank were clean. Related: the streaming_utils quarantine
  warnings. Needs its own investigation (candidate mitigations: detect
  the fault signature and refuse-with-restart-hint; py-spy needs sudo).
- [ ] **`prefill_step_size` / `vision_tokens` are unmeasured** (P3, raised
  2026-09-20): both are `per_request`, so trying either costs no reload, and
  both are the plausible levers on a PREFILL-BOUND workload -- a long fixed
  system prompt with a short answer, which is what the Qwen-Image prompt
  encoders are. Nothing here has measured either at any value. `vision_tokens`
  applies to the image-to-image encoder only; the text-to-image one takes no
  image. Measure with the usual controls: match
  prompt length, cache state, sampling and seed across arms, never at temp 0,
  and keep the numbers out of tracked docs.

  Do NOT reach for the prompt cache instead. On these models (`model_type:
  qwen3_5`) it is blocked twice over: `_mrope_reuse_safe` gates the family
  explicitly, and underneath that the slot stores the last generation, so a
  new request is a TRIM -- which qwen3_5 refuses, because its `make_cache`
  returns `ArraysCache` for the linear GDN layers and that inherits
  `is_trimmable() -> False`. Lifting the gate alone buys nothing. Owner call
  2026-09-20: leave prompt caching alone.

- [ ] **`max_loaded_models`: keep it or force 1?** (P3, raised 2026-09-20,
  priced and NOT done): the question was whether to remove the field so the
  server only ever holds one model. Removal deletes less than it looks --
  eviction is fully LIVE at 1 (a second model's request calls
  `_evict_lru_model`; pinning, which could refuse it, was removed in v2.0.126), so
  what actually goes is the `AppConfig` field, the importer writes, the `>1`
  branch and a test helper default. Eviction gets more frequent, not less.

  The argument for KEEPING it is a real local workload: the two Qwen-Image
  prompt encoders are ~18 GB each, ~36 GB for the pair on a 192 GB box, and a
  consumer alternating text-to-image and image-to-image pays an 18 GB
  evict-and-reload per switch at `max_loaded_models = 1`. Current
  recommendation: keep the field, default 1; if anything, warn at load when a
  `>1` combination exceeds the Metal working set, the way the fit panel does.
  The stale `= 2` that prompted this was a dead default in `schema/system.py`,
  deleted wholesale in v2.0.48.

- [ ] **mRoPE cache gate: fail-open + no config escape** (P3, review
  finding 2026-08-18): the gate keys on two private upstream attribute
  names; a rename fails OPEN (reuse re-enabled on a broken family) and
  there is no per-model `cache_reuse` config override to gate manually.
  Follow-up: a models.toml field (effect-classified) honored ahead of the
  attribute sniff.
- [ ] **Prompt-cache reuse for quantized/rotating CONFIGS** (P3, deliberate
  non-widening in v1.75.0): the config-level gate (cache_type standard, no
  max_kv_size) predates Q7 and was kept verbatim. Under the snapshot+
  native-trim design, EXTENSION reuse would be sound for those configs
  too (nothing is sliced); widening needs its own live verification on a
  kv-quantized model, not a ride-along.
- [ ] **retrySave drops media blocks** (P3, pre-existing): the unsaved-row
  Retry save re-POSTs `msg.content` (flattened text), so a media message's
  blocks would not survive it. Unreachable today -- generation refuses
  while an unsaved row exists and unsaved rows are only minted by
  generation flows -- but the day an unsaved row can carry media, Retry
  save must POST `content_blocks`.
- [x] **Per-message model attribution UI -- DONE v1.74.0**: a muted
  per-row label (`.message-model-note`) rendered only while the thread
  MIXES models; rides msgSignature so rows rebuild exactly when the label
  appears. Data since v1.73.0 (fresh-row commits stamp; continuations keep
  the anchor's stamp).
- [x] **/v1/models is ~1.7s on a 29-model registry -- FIXED v1.74.1**:
  the reasoning_effort template probe shipped uncached (its sibling was
  lru_cached) and re-read every MLX model's template files per call.
  Measured 1650ms -> 1ms live. Cache-property tests pin BOTH probes
  (shown red against the uncached version).

- [x] **Phase 0 hardening -- DONE v1.64.0**: loud stream-guards (+ a
  `pendingSave` latch closing the Stop-then-act window between stream
  release and the partial save landing), unsaved-row "Retry save"/"Discard"
  honesty + destructive-op-and-send lock, reconcile-on-saga-end re-fetch
  (unsaved rows survive adoption), thinking-block editing in the message
  editor. All guarded by `e2e:render` (23 checks, each new one shown red
  against a pre-fix tree first), incl. an iPhone-emulation boot (viewport +
  touch + hover:none via CDP) for touch reachability.
- [x] **Phase 0.5 iOS hidden-row bug -- DONE v1.79.1**: reproduced on an
  iOS 26.5 simulator (Safari 26.5) with a control: with `content-visibility:
  auto` the thread opened 1046px above its end and the saved row landed
  mostly under the composer; with it off, the landing was exact. The row
  was never blank -- it was displaced: skipped rows report their 3rem
  estimate until WebKit's lazy relevance check lays them out, and WebKit
  has no scroll anchoring (`overflow-anchor` unsupported) to absorb the
  shift. Fix: the optimization is gated on `@supports (overflow-anchor:
  auto)`; editor close re-aims at its row. Instrument that saw it (no
  suite can): a same-origin injecting proxy + `simctl openurl` +
  screenshot loop, recipe in the 2026-08-20 session log. v1.79.2 added
  chat's resume sync (`refreshAfterResume`: visibilitychange/pageshow
  re-adopt the store) + the prompt editor's flush-on-hide; NOTEBOOK still
  lacks the resume refetch (it gets the flush via the shared factory) --
  port it when notebook is next touched. Open follow-up:
  the keyboard-dismissal drift after Save is only re-aimed, not prevented
  (viewport `interactive-widget` is not in Safari 26's notes).
- [x] **Phase 1 backend -- DONE v1.65.0**: `POST/DELETE
  /v1/conversations/{id}/generate` (append/regenerate/continue, Messages
  SSE + `heylook_saved` final event, per-conversation 409 arbitration,
  server-owned persistence incl. disconnect, truncation commits only with
  its replacement row). 17 contract tests; spec §4 updated.
- [x] **Phase 2 client cutover -- DONE v1.66.0**: chat generates via
  `/generate` (streamGenerate parser in streaming.js; Stop = DELETE;
  teardown = fetch abort + server disconnect-persist). Verified: render
  23/23 unchanged through the cutover, live chat suite 45/45 (three checks
  rewritten off the old wire), gguf matrix 7/7 on DeepSeek V4 Flash.
- [x] **Disconnect-persistence test -- DONE v1.74.0**: live chat suite
  reloads the page mid-stream (no Stop; the beforeunload dialog is
  accepted) and asserts the detached server task persisted the partial,
  server-side by id and client-side after reload. Green live 2026-08-18.
- [x] **Muse-Glimmer 30B -- RESOLVED 2026-08-13 (the parse was never
  broken)**: the "all output in reasoning_content, content empty" report
  was an always-reasoning model given a 100-token budget -- it burned it
  all in the harmony ANALYSIS channel and never reached the final channel.
  With a real budget, llama.cpp b10353+ parses it perfectly (the model
  card confirms; we're on b10416). Entry corrected: `supports_thinking =
  true` (it reasons every turn; template ignores enable_thinking), the
  since-withdrawn `--reasoning-format none` workaround REMOVED, and
  `--chat-template-file` points at the fixed HF template downloaded beside
  the weights (this GGUF predates the card's template fix -- 7167 vs 9992
  chars; the fix normalizes "Reasoning effort"->"Reasoning strength" in
  system prompts; drop the flag after re-downloading the GGUF). Vendor
  sampling temp 1.0 / top-p 0.95 / top-k 64; never stop on `<|eom|>`.
  Generate matrix 7/7 incl. thinking persistence and analysis-only abort
  partials. LESSON, again: starved budgets mislead about CORRECTNESS, not
  just perf -- an empty content with a fat reasoning_content means "ran
  out mid-think", not "parser broken". Alternative if upstream
  stalls: heylook's own harmony-channels parser via a gguf template_info --
  but that cuts against the "never re-parse another engine's split"
  invariant and only makes sense with format=none (engine not splitting),
  design it deliberately if at all.
- [ ] **At-rest encryption for the conversation store** (future state,
  owner ask 2026-08-13): conversations/messages (prompt inputs/outputs)
  sit plaintext in the DuckDB file. VERIFIED on the installed duckdb
  1.5.5: native encryption works -- `ATTACH 'file' (ENCRYPTION_KEY ...)`
  round-trips and a key-less open is refused. Vehicle when picked up
  (owner direction 2026-08-13): connect-then-ATTACH in db.py, key held
  ONLY in 1Password and fetched at server start via `op read` (the owner's
  gemini-bridge pattern in fb-claude-skills) -- the Touch ID popup at
  launch is the human-in-the-loop: with the server down, ANY attempt to
  read the key (Claude included) fires a popup the owner can deny, which
  the silent-same-user Keychain cannot offer. One fingerprint per server
  start; launchd auto-start becomes impossible (a boot service cannot
  fingerprint); a down 1Password agent blocks start, fail-loud. iOS Safari
  is UNAFFECTED -- the server holds the key and decrypts, the phone never
  touches 1Password. Per-ACCESS fingerprinting was considered and
  rejected: approvals render on the Mac (stalls every phone read) and the
  server must see plaintext to build prompts at all -- an LLM chat server
  is structurally a plaintext-processing machine while it runs. Turning
  encryption on is a fresh start or one-time manual copy per the
  no-migration policy.
  THREAT-MODEL BOUNDARY (do not oversell, discussed 2026-08-13): this
  protects the FILE at rest -- backups, cloud sync of the raw .duckdb,
  other user accounts, copied disk images (FileVault covers only the
  powered-off disk). It does NOT protect against any live process running
  as the owner: the key must be readable at server start, and the running
  server serves decrypted conversations over the loopback API, whose
  conversation routes carry no auth (the partial inference API key was
  removed in v2.0.127; the Host check, v2.0.137, refuses foreign host
  names, not local processes). "An agent with my shell can't read it" requires
  harness-side sandbox deny rules + enforced API auth, not encryption.

## Upstream-borrow follow-ups (vllm-metal scan + delta review, 2026-07-20)

From the coderef/vllm-metal optimization scan + mlx-lm/mlx-vlm delta review
(session log 2026-07-20). Verdicts validated against the plan's Phase 5
ordering and the sole-user/minimal-custom-code posture.

- [ ] **Gemma-4 MTP self-speculation** (P2, the live lead): mlx-lm PR **#1276**
  (open, "Add Gemma 4 assistant (MTP drafter) model class") is the adoption
  vehicle -- model class only, no generation wiring yet. PROBED LIVE 2026-07-20
  (`internal/research/mtp_probe/`): the zoo's 26B-A4B assistant head drafting
  for the daily MoE measures **50.8% greedy acceptance** (lower bound; scaled
  embeddings are the trained convention), drafter ~1ms vs ~11ms/step ->
  projected ~1.4x decode ceiling at n_predict=1. Matches vllm-metal's published
  draft-model wins. Build list + caveats in the probe README. Watch #1276;
  when it merges (or from the fork), build the verify loop provider-internal
  behind `run_generation`. Greedy-only; fingerprint gate invalid (doctrine).
  OWNER ACTION (optional, with data): comment on #1276 -- `AssistantAttention`
  uses `num_key_value_heads` for both layer types; the 26B assistant needs
  `num_global_key_value_heads=2` on full-attention layers or SDPA mis-shapes
  (their network test uses the E4B where counts coincide). Acceptance numbers
  are shareable.
- [x] **mlx-vlm pin bump -- DONE 2026-07-20** (0bfe60b): bumped mlx-lm
  a790972->15b522f (+1 commit, server XTC fix, N/A to us) and mlx-vlm
  8e2638b->c9e27b08 (0.6.5->0.6.6, 53 commits: gemma-4 bf16 dtype-leak fix,
  `prepare_inputs` mask preservation, qwen3-vl PIL video-frame normalization).
  Consumed surface confirmed signature-stable (22 contract tests green);
  transformers deliberately stays pinned at 5.5.4 via `override-dependencies`
  (mlx-vlm HEAD declares `>=5.14` -- decoupled on purpose, contract tests are
  the gate, not the floor). Verified: full suite 1085 green + a live eval-bank
  A/B (old vs new pin) on gemma-4-31b + Qwen3.5-27B, no regressions.
- [ ] **optloop experiments** (P3, both gated on prerequisites): (a) re-test
  classic draft spec-decode with a ~4B gemma draft (the closed negative result
  used a 1B draft; vllm-metal's 1.36-1.48x with a favorable pairing says the
  ratio matters -- needs a gemma-3-4b download first); (b) n-gram
  prompt-lookup prototype (mlx-lm has no equivalent; zero draft cost, wins on
  repetitive/structured output, greedy-only).
- [x] **Gemma-4 QAT q4_0 swap -- CONVERTED + A/B DONE 2026-07-20, swap is
  the owner's call**: own conversions from the owner-downloaded unquantized
  QAT checkpoints (own-convert beat adopting OptiQ), 4-bit affine group-32
  matching the q4_0 32-block lattice per docs/mlx_conversion_guide.md:
  `modelzoo/google/gemma-4-{26B-A4B,31B}-it-qat-4bit-g32-mlx` (16G / 19G,
  ~5.2 bits/weight avg, vision towers bf16 via auto-skip), registered with
  Google-recommended sampler defaults (temp 1.0, top_k 64, top_p 0.95 --
  also added to the two 8-bit dailies). VERDICT: on the fixed v1.38.1 stack
  the full eval bank is **13/13 for all four models** (both QAT vs both
  8-bit dailies) -- parity. Every deficit seen mid-A/B (synthetic-image
  refusals, two-image failures, 31B thinking-skip, CJK degeneration on long
  thinking) traced to the server, not the quants: the transformers-5
  stop-set regression (fixed v1.36.1) let generation run past `<turn|>` far
  out of distribution, compounded by the pre-refactor sampler cascade. RAM
  halves (26B: 16G vs ~28G; 31B: 19G vs ~33G). Caveat before making QAT the
  daily: the bank's images are synthetic; run a real-photo spot-check.
  The QAT assistant heads (downloaded, unconverted) pair with the MTP item.
- [x] **finish_reason reports "stop" on budget-exhausted responses -- DONE
  v1.39.14** (P3, found during the 07-20 QAT A/B). Both non-streaming
  builders hardcoded "stop"; mlx-lm's own reason is now scraped in
  `ChunkTelemetry.absorb` (the one-scrape rule) and latched, so a trailing
  empty chunk cannot erase it, and both the then-OpenAI route and
  `/v1/messages` reported it. The Messages converter already mapped length ->
  stop_reason -- it was never fed one. The eval bank's token-count
  workaround is retired (kept only as a fallback for servers that report no
  reason). Live-verified: max_tokens=12 -> "length", natural -> "stop".
- [ ] **Trim the two surviving architecture KEEPs** (P3, from the 07-20
  architecture audit; exact section lists in the session log): config.md
  (drop the field tables / TOML examples / schema dump / troubleshooting;
  keep the three design-reversal records + validation rationale) and
  mlx_provider.md (drop 1.4-1.6 key-file/function inventories + 2.1 + 5;
  keep all of section 4 -- the invariants). Philosophy: design records
  only, mirrors die.
- [ ] **frontend_v3_spec.md post-cutover slimming** (P3, gated on Phase 3
  v2 retirement): the spec is a historical BUILD contract; once v2 dies,
  slim it to section 4 (the living API contract) + the decision records,
  and let frontend_v3.md carry the map.
- [x] **Unify the parser strip/holdback layer -- DONE v1.39.12** (P2, from
  /simplify 2026-07-20): declared-specials stripping is now ONE
  implementation -- `reasoning_parser.StripSpecials`, composed by the
  factory over whichever routing parser it picked (and only when the model
  declares specials). Its rolling per-kind holdback is sized by the STRIP
  SET (longest tail that is still a proper prefix of some declared
  special), so the harmony/gemma boundary leak and PassThrough's total
  absence of holdback are both closed, and the holdback now covers
  non-`<`-shaped specials (Mistral's `[INST]` family) that the old
  `rfind("<")` scan could never hold. Fixed alongside: harmony's dead
  final-flush partial-strip (the abort-mid-token garbage flush, gemma's
  2026-07-20 fix ported to a shared `_strip_partial_token`), and the
  duplicated `_safe_prefix_len` bodies collapsed to one free function.
  Eight failing-first tests (harmony abort pair, cross-parser boundary
  straddles, non-`<` specials); suites green. Design record:
  docs/parser_strip_unification.md.
- [x] **E2E checks for the 2026-07-20 v3 features -- DONE v1.39.9** (all
  four landed in chat.mjs's capability/thinking/image section, live-green
  73/73): (1) attach 9 -> 8 thumbs + aria-live cap message; (2) thinking
  toggle aria-pressed round-trip + wire contract (enable_thinking absent
  when off -- never false -- true when on, asserted on captured request
  bodies); (3) `#set-vision_tokens` present with min 16 / max 16384 +
  localStorage round-trip, absent on a non-vision model (negative model
  discovered from /v1/models, never loaded); (4) thinking `<details>` block
  renders with non-empty body on a real thinking generation (branch-on-
  empty per the suite hardening principle). Plus an image-message
  round-trip (content_blocks persist + render + survive reload) beyond the
  original list. README descriptions updated.
- [ ] **LLM behavior-eval harness follow-ups** (P2, owner-directed 2026-07-20):
  `tests/eval/` (opt-in, seed bank generalized from the 2026-07-20 live
  verification scripts) is the base. Direction the owner wants explored:
  (a) eval as an API surface (trigger runs / read results server-side) and
  (b) an eval page in frontend v3 (run the bank against loaded models,
  compare results -- fits the introspection identity). Design with the
  Phase 6 admin surface + observability pages, not a bolt-on; results
  storage could ride the observability JSONL + DuckDB-over-files pattern.
  Also queue: run the eval bank as the optional gate for changes touching
  templates/parsers/stop-tokens/vision (the 4 bug classes it was built on).
- [x] **Eval-gate reminder hook + /eval-ab skill -- APPROVED + BUILT
  2026-07-20** (local `.claude/` config, not committable by design; the
  tracked piece is `scripts/dev_server.sh`, the isolated live-server harness
  both lean on). Scoping that was approved: owner's constraints: no over-testing,
  no auto-spawning heavy servers, no 3-hour runs on small changes, no
  one-model-one-prompt tunnel vision. Scoped design that satisfies them:
  (1) the hook is INERT TEXT ONLY (hookify context injection on edits to
  reasoning_parser/thinking_parser/template_info/stop_tokens/vision_budget/
  generation_core) -- it never runs anything; (2) it names the MINIMAL
  category for the touched subsystem via `run.py --tasks` (parsers/template
  -> thinking,stop; vision_budget/vlm -> vision; stop_tokens/generation_core
  -> stop) -- a smoke tier of 2-4 tasks x 1 fast model, ~1-2 min, not the
  13-task bank; (3) it only suggests running against an ALREADY-RUNNING
  server (dev-server skill's reuse-first rule) -- if none is up, the
  suggestion is "queue it", never "spawn one"; (4) full bank x multiple
  models stays explicit-ask-only (pin bumps, QAT A/B, releases). /eval-ab
  is the explicit-ask wrapper: baseline JSONL storage + diff + the
  rerun-flapped-tasks-before-concluding discipline from the 07-20 pin bump.
  Build neither until the owner approves this scoping.
- [x] **Pyright noise triage -- DONE 2026-07-20** (0e236e4): real fixes
  (deprecated `datetime.utcnow`, untyped `= None` defaults, a latent bug where
  batch responses could hand pydantic `model=None` -> runtime 500 now
  coalesced, a float re-binding bug in `_format_bytes`, duck-typed route
  discovery) plus a systemic sweep of all 96 positional Pydantic
  `Field(default, ...)` calls to explicit `default=` keyword form (this
  pyright build only recognizes the keyword form -- the source of most false
  "arguments missing" complaints). `pyrightconfig.json` already existed from
  an earlier session; this pass didn't need a new one. The
  `LogprobsCollector.add_token_and_get_delta` stale-reference suspicion was
  not confirmed as real during this pass -- re-check if it resurfaces.
- [x] **Vision token budget knob -- SHIPPED GENERALIZED 2026-07-20**
  (v1.34.64, ahead of the Q8 spike): model-agnostic `vision_tokens`
  (request + models.toml per-model default + v3 drawer control) mapped by
  duck-typing the processor (gemma buckets / qwen pixel budget); cache key
  carries the budget; live-verified on gemma-4-31b + Qwen3.5-27B. The Q8
  ACCURACY question (does 1120 improve detail QA?) remains open -- the
  eval harness vision tasks are the vehicle.
- [ ] **Optional v3 nano-feature: per-attachment estimated token cost in the
  composer** (P3, leftover from the shipped vision-budget item above): the
  resize math is ~15 lines client-side (gemma bucket snap / qwen pixel
  formula, mirroring the server's own duck-typed mapping). Not required --
  the budget itself is already a first-class request/config/UI knob.
- [x] **Gemma-4 canonical template refresh (2026-07-09 "less laziness" fix)**
  -- DONE 2026-07-20: verified upstream state (template-ONLY fix -- no IT
  weight re-uploads; commits 07-15 "null handling, reasoning preservation,
  turn-tag balance"); owner refreshed 26B/31B jinja, E4B fetched + installed
  from upstream (E-series variant of the canonical). mlx-community 8-bit
  conversions are stale (07-05) -- local chat_template.jinja files are the
  fix, auto-picked-up at load (verified: transformers prefers the jinja
  file; no embedded template in these checkpoints).
- [x] **`mlx_cache_limit_gb` operational setting** -- DONE 2026-07-20
  (v1.34.59): opt-in MLX buffer-cache cap via /v1/admin/config (bounds idle
  RSS at the cost of realloc on the next spike; MLX default restored on
  reset). Plus fix: DELETE on config keys now re-applies immediately.
  Borrowed shape from vllm-metal's measured-overhead cache cap; ours is a
  manual knob, their auto-measurement is overkill for one box.

## Presets/system-prompt follow-ups (from the v1.34.22-.24 review passes)

- [x] **Cap-gated `enable_thinking` can be pinned invisibly -- FIXED
  v1.39.10**: `samplerParams(caps)` drops capability-gated keys
  (enable_thinking, vision_tokens) at request-build time when the current
  model lacks the cap; chat/notebook/explore each pass their page's caps.
  The cache deliberately KEEPS the value (switch back to a capable model
  and the control + value return) -- only the wire is filtered. E2E: the
  cap-filter wire check asserts on an intercepted+aborted request to a
  never-loaded negative model.
- [ ] **Presets vs TOML registry: dual-source by name** (P3, design decision):
  user presets are client-expanded only; `ChatRequest.sampler` resolves just
  the bundled TOML registry. If saved presets should ever work by name from
  the raw API/CLI, make the registry dual-source (TOML + DB rows, name-unique
  across both) instead of growing a second wire path. Deliberately NOT done
  in v1.34.22 -- client-side copy semantics (LM Studio) was the owner's ask.
- [x] **Notebook page preset bar** -- DONE v1.39.3: the bar was extracted to
  the shared `preset-bar.js` (createPresetBar + getPrompt/setPrompt/onStatus
  adapter) and notebook contributes it ahead of its sysprompt section.
  Decision: apply DOES write the notebook's system prompt (a preset is a
  prompt+sampler bundle everywhere; the armed confirm guards the overwrite).
- [x] **"Panel drifted from selected preset" indicator** -- DONE v1.39.2: live
  drift line under the preset select ("Matches/Differs from current settings"),
  updated in place on prompt keystrokes + sampler edits; apply became an
  explicit button (selection is inert), armed-confirmed only when it would
  replace a differing non-empty prompt.
- [ ] **Unknown `params` keys stripped on apply-then-save** (P3, edge): a
  preset authored via the API with keys outside `PARAM_META` loses them if
  the UI applies then re-saves it (panel state is the source of truth).
  Fine for UI-authored presets; document or merge-through if API authoring
  becomes real.
- [x] **v3 frontend doc** (P2, docs): DONE 2026-07-09 -- the v3 map (what's
  done/left + backend<->v3 coupling + remaining backend work) now lives
  git-tracked at `docs/frontend_v3.md` (renamed + promoted out of
  `internal/frontend/v3.md`); the 5 stale React-frontend docs were archived to
  `internal/frontend/archive/`. CLAUDE.md Orient-first + architecture paragraph
  point at it and at the plan as the roadmap.
- [x] **DRY settings drawer** (P2, plan Phase 4): DONE 2026-07-11 -- the chat settings UI
  extracted into an app-shell **global slide-over drawer** (`js/settings-drawer.js`) shared
  by all 6 pages; `registerSettings(contribution)` with the sampling / global display prefs /
  per-page extras taxonomy (DESIGN.md §6). jspace's toggles + samplers-greyed, notebook/chat
  sysprompt as sections, presets on chat. Edge cases preserved, browser-verified, code-reviewed.
- [x] **Impeccable design-quality pass (plan Phase 4 item 2)**: DONE 2026-07-11 -- ran
  `/impeccable audit` + `polish` across all 6 pages + app shell + settings drawer. Slop-detector
  clean; technical score 17/20. Fixed the mobile + a11y cluster: **delete/rename were unreachable
  on iPhone** (hover-gated, no `@media(hover:none)` fallback -- a genuine touch bug); status lines
  now `role=status` / error surfaces `role=alert` (honest states reach screen readers); sampler +
  display inputs get `<label for>`/id; drawer became a real modal (seals `#app` via `inert`);
  explore chips carry a `title` (non-color access, DESIGN.md §2); `aria-current` on active nav;
  `--radius-sm` undefined-token nit fixed. **Mobile settings gear moved FAB -> trailing `⚙` item
  in `#bottom-nav`** (owner-chosen: a FAB collided with chat's Send button). Verified at an
  iPhone-17-Pro viewport with touch-media emulation (19/19 checks) + full E2E (zero regressions vs
  baseline). Rules recorded in DESIGN.md §7. Deferred P3s (single-user pragmatic floor): conv/
  notebook list items not keyboard-focusable (`<div>`, not button); jspace layer-range slider is
  pointer-only.
- [x] **E2E suite stale vs the settings-drawer refactor** (P2): DONE 2026-07-11. The suite predated
  `42a1769` (drawer unification) + the jspace route -- it clicked a `.chat__bar` "Settings" button,
  poked on-page `.chat__sysprompt`/`#jspace-heatmap`, and asserted 5 routes, so 14 checks failed on
  BOTH HEAD and the design branch (never regressions). Added `openDrawer`/`closeDrawer` helpers
  (`lib/dom.mjs`) that drive settings/presets/sysprompt/jspace-toggles through the shared drawer, and
  fixed the route count (6). The drawer is modal (inert #app + covering backdrop), so the helpers are
  transition-aware: reset a leaked-open drawer, fire the gear via `evaluate` (a real click hits the
  fading backdrop), and wait for the slide-in + the backdrop's *delayed* hide to settle -- documented
  in the E2E README's new "Settings drawer" gotcha. Also hardened two pre-existing latent races
  (assistant-reply-persists polled instead of racing finishStream's save; post-abort waits for the
  conv list). Result: 62/63 green; the sole miss is the load-sensitive streaming-cadence perf check
  (Mac Studio throttled after many back-to-back 26B spawns -- passes idle, README notes it).
- [x] **Wire `show_special_tokens` render consumers -- DONE v1.79.6** (commits
  `a7a3b3f`, `447cb32`, `b489ab1`): `show_special_tokens` display pref wired
  in v3 settings drawer on chat & notebook; new opt-in request field on
  `POST /v1/messages` and `POST /v1/conversations/{id}/generate` bypassing
  the declared-specials filter; assistant-stored declared specials stripped
  before replay to prevent control-token prompt injection. Display panel
  gated to pages declaring support (`displayPrefs`). Request-schema parity
  guard (`tests/unit/test_request_schema_parity.py`) pins wire consistency.
- [ ] **jspace viz: chat-turn default + special tokens + prefill/token-walker** (P2/P3, 2026-07-11):
  see `docs/jspace_integration_plan.md` Part 2 (2026-07-11 refinements). Flip analyze to chat-turn-
  default (verify the "format-dominated onset" claim -- likely a provisional-lens artifact), show
  special tokens, add prefill/edit-the-assistant + per-token selection (the `coderef/mlxui-core`
  possibility-horizon walker collapses prefill+selection into one primitive). Activation patching
  (steer/swap/ablate) = port `mlxui-core`'s op-semantics via forward-hooks, NOT its per-arch subclassing.

## Chat-template resolution follow-ups (from the v1.34.38 review, 2026-07-10)

v1.34.38 made template resolution a registry concern (server import detects jinja,
chat_template.json fallback, auto-install-when-missing, actionable errors). The
review (10 verified findings) split into a quick hardening batch and design items.

- [x] **Quick hardening batch**: DONE v1.34.40 (same day) -- shared
  `detect_chat_template_source()` helper used by both import paths (+ expanduser,
  fixing tilde-path detection); `"auto"` no longer force-installs and
  `"chat_template_json"` became an accepted explicit source; the missing-template
  error is decided from tokenizer state (not transformers' error prose), respects
  wrapper-level python templates (`has_chat_template`), and covers all three apply
  sites (chat, batch, hidden-states); the load warning consumes install's return
  and no longer false-alarms on `chat_template_type` models (this also closed the
  "wrapper-level templates false-alarm the warning" design item).

**Design items** (need a decision, not just a patch):

- [ ] **List-form `chat_template` silently dropped** (P3): HF's legacy named-template
  list (`[{"name","template"}]`, still read AND written by transformers 5.5.4; real
  repos ship it, e.g. command-r-plus conversions) is treated as no-template by
  `_read_embedded_template` -- empty template for harmony/thinking detection and a
  false "lacks a chat_template field" warning under explicit source. Decide: pick
  the "default" entry, or keep string-only and log the list case explicitly.
- [ ] **chat_template.json fallback can change response shape** (P4, note-only?):
  a json-only model whose template carries <think> markers now selects the thinking
  parser (split `thinking` field) where it previously streamed inline -- arguably
  correct (the processor already applied that template) and no local model hits it
  (all json-shipping folders also have jinja), but it's an undocumented behavior
  change; consider a CHANGELOG amendment when touching this area next.

## Model registry: modalities/loader follow-ups (from the v1.34.43 split, 2026-07-11)

v1.34.43 split the overloaded `vision: bool` into `modalities` (description,
`model_importer.detect_modalities`) + `loader` (routing,
`providers/common/loader_routing.py`); `is_vlm`/`MLXProvider.effective_loader`
derive from it. Design + decision recorded in `plan_2026-07.md` Phase 6
("Refinement 2026-07-11"). Shipped, simplified (v1.34.43+), and audited against a
19-model modelzoo sweep. Three deferred items, none urgent, all Phase-6-coupled:

- [ ] **Registry entry `kind` (chat vs draft/MTP vs embedding)** (P2, needs a
  field + a decision): `provider` (mlx|mlx_embedding) doesn't capture that some
  `provider="mlx"` entries are NOT servable chat models -- e.g.
  `gemma-4-26B-A4B-it-assistant-bf16-mlx` is an MTP/draft head ("no chat template
  on purpose"), flagged only by a `draft` tag. Because it inherits gemma's
  `vision_config`/`audio_config`, `detect_modalities` OVER-CLAIMS
  `[text,vision,audio]` for it (routing stays correct via the positive-knowledge
  degrade -> mlx-lm; the over-claim is cosmetic, and only on re-import since the
  toml entry is hand-set `vision=false`). Fix: a `kind` field so UI /
  `/v1/models` capabilities / telemetry / `detect_modalities` don't treat a draft
  head as a chat model. This is the clearest driver for the Phase 6 "entry KIND
  is under-modeled" note.
- [ ] **Manual `loader` override isn't durable until the Phase 6 tomlkit merge**
  (P2, coupled): `loader` is honored at load, but a re-scan REGENERATES
  `models.toml` and wipes a hand-set value (Option 2 by design -- reserve `auto`
  now, durable editing when the non-clobbering merge lands). Sharp edge until
  then; don't advertise the manual override in the UI before it's durable.
- [ ] **Remove the `vision` derived-mirror once readers migrate to `modalities`**
  (P3, cleanup): `vision` is kept as a validator-maintained mirror of
  `"vision" in modalities` for back-compat. Migrate readers to `modalities`
  (grep `config.vision` / `.get("vision")` / `config["vision"]` -- known:
  `capabilities.py::infer_model_capabilities`, `model_importer` entry-build,
  `loader_routing._modalities_of` raw-dict fallback), then drop the mirror + the
  bool. Do NOT do piecemeal -- it's a coordinated removal.
  (`model_service._raw_to_scanned` was migrated 2026-08-07 -- it read
  `config["vision"]`, which no entry builder writes any more, and so reported
  `vision:false` for every scanned model of both providers.)
- [ ] **Audit the other five v3 pages for the status-area shape** (P2, found
  2026-08-07): the models page wrote an error and then awaited an internal
  refetch that cleared it on success, so it had never once shown a load failure
  since the page was built (fixed v1.50.2 via a `keepStatus` flag). For chat,
  notebook, explore, perf and jspace: find any handler that writes an error and
  then awaits a refresh whose success path clears it. Unknown whether any share
  the shape -- this item is the check, not a claim that they do.
- [ ] **`test-audit` over the payload/cascade tests** (P2, found 2026-08-07):
  two green-but-blind escapes in one session, both assertions written from the
  perspective of the case the author had in mind. One actively PINNED the bug
  as correct (`"chat_template_kwargs" not in unset`). Specific question, not a
  general sweep: does any other test assert a key's ABSENCE where the sent
  VALUE is the contract? Grep `not in payload` / `not in body` across `tests/`
  and review each hit.
- [ ] **Derive gguf `modalities`/`supports_thinking` at LOAD, like MLX**
  (P2, found 2026-08-07, Wave-1 derive-at-load coupled): the gguf importer
  STORES both in the entry, so an entry written before v1.49.4/.6 under-reports
  forever. Measured on the local fleet: every gemma-4 GGUF's entry says
  `["chat","vision"]` while a fresh scan of the same files derives
  `["text","vision","audio"]` + `supports_thinking: true`. Since v3 gates all
  modality UI on capabilities, those models show no thinking toggle and no audio
  attach even though the server serves both. `gguf_metadata` already reads
  headers cheaply (stops at the last requested key) and `modality_detect` has the
  mtime-cache precedent, so a `GGUFModelConfig` validator mirroring
  `MLXModelConfig._resolve_modalities` is the shape. NOTE this does not fix
  EXISTING entries: a stored value is indistinguishable from a deliberate
  override, so those need a re-import (or the two keys deleted) either way --
  deriving is what stops it recurring.

See also the Phase 6 "per-model SIDECAR ARTIFACTS" note (draft model / j-space
lens / future LoRA managed as a group on the admin CRUD surface).

## Observability + config redesign (2026-07-11)

Full design + status: `internal/research/observability_and_config_redesign.md`
(local-only). Backend spine + config layer landed this session
(v1.34.44-.55).

**Done (backend):**
- [x] Config foundation: App-DB `settings` table + `/v1/admin/config` (env > DB >
  default, then made DB-authoritative -- no env override for operational settings).
- [x] Observability spine: `observability.py` `record_event` -> `logs/*.jsonl`
  (metrics + events tiers), level-gated (`observability_level`), file rotation,
  startup disclosure; `diag_event` delegates to it; per-request + model-lifecycle
  emission; `POST /v1/telemetry/events` for v3 client events.
- [x] `internal/log/` -> `logs/` reconcile; `observability_level=off` master kill
  switch over memory.py's legacy streams.
- [x] Aborted/stopped streaming requests now logged (`stop_reason=abort`).
- [x] Chat-template robustness: reject stop-less templates, validate vs the
  model's own eos tokens (v1.34.55).

**Backend TODO (mine):**
- [x] **Per-DOCUMENT sampler settings** DONE 2026-07-11 (v1.34.56-.58): `params`
  JSON column on BOTH `conversations` (v4) and `notebooks` (v5), shared
  `_encode_params`/`_decode_params`; threaded through create/update + API models +
  PUT allowed sets. Unifies "settings in browser vs server" -- sampler knobs join
  the system prompt on the server. Frontend below shares ONE binding.
- [ ] **memory.py stream CONSOLIDATION** (P2): spine now duplicates memory.py's
  `request_events`/`model_events` streams. Once live-verified, remove the dupes +
  retire the 3 legacy env toggles (`HEYLOOK_REQUEST_LOG_ENABLED` /
  `_MODEL_EVENT_LOG_ENABLED` / `_BASELINE_LOG_INTERVAL_SECONDS`); resource snapshot
  moves to the spine. Gated on full live verification.
- [ ] **Live-verify the spine end-to-end** (P2): confirm `provider=mlx` +
  `effective_loader` (text=mlx-lm / vision=mlx-vlm) + `stop_reason=abort` in
  `logs/metrics.jsonl` from real runs before removing memory.py streams.
- [ ] **`modalities` dim in `request_complete`** (P3): only
  `provider`/`effective_loader`/`is_vlm` captured; `modalities` needs
  `model_config` threaded to `_maybe_log_request_event`.
- [ ] **Never-stops health signal** (P3): flag models whose requests consistently
  hit `stop_reason=length`/`abort` (surfaces broken templates in the metrics).

**Config-editor / audit follow-ups (2026-08-11, sourced from the four-agent
audit; design context `internal/research/expert_offload_design_frontend.md`
+ `_backend.md`):**
- [x] **Chat model-switch hardening G1+G3 -- DONE v1.57.0** (G2 v1.55.0):
  caps-gated `toWireContent` with per-message drop disclosure (staged
  attachments still block -- asymmetry commented at both sites); residency
  dots + pre-switch warning (Cancel / Switch anyway; Send with the
  unconfirmed target commits) + Load button on the chat bar. NB the
  LOAD-COST half of that warning was removed in v1.62.3 (owner rule:
  only loss gates, cost is disclosed) -- only incompatible-media
  warnings gate a switch now; the cost is stated by the dots, the Load
  button, and a live pre-first-token status on Send. Still open
  from doc §15: G4 context estimate, G5 per-message attribution (waits for
  a `_SCHEMA_VERSION` bump), F14 switch-lock during a pending load.
- [x] **Fit meter -- DONE v1.60.0** (P2, the frontend design doc's stated
  heart): `heylook_llm.ram_fit` (extracted from `ram_report.py`, which now
  renders it), `POST /v1/admin/models/{id}/fit` with the provider-derived
  `hard_working_set` (MLX-FAIL/gguf-WARN), v3 Memory-fit section + Load
  gating on FAIL. Fit stays server-computed. The §5 "observed after load"
  line landed in v1.62.0.
- [x] **Server-owned `POST /v1/admin/models/{id}/reload?warm=true` -- DONE
  v1.62.0** (P2, ask #4): one route sharing load's exact body (warm contract
  can't fork); v3's "Reload now" points at it. The fit meter's §5 observed
  line (resident memory, measured after load) landed in the same version.
- [ ] **Continuation loose ends** (P3, from v1.61.0-1.61.1; the feature is
  complete and review-hardened, these are parity/depth): MessageCreateRequest
  has no explicit `continue_final_message` field (the auto trailing-assistant
  convention works on /v1/messages and block-form prefill is flattened, but
  there is no explicit control; the removed OpenAI route had one, and the
  generate route's `continue` mode is where the explicit spelling lives now);
  image-history
  continuation is a deliberate 400 (the vision strategy has no open-turn
  spelling yet); the eval bank's thinking/stop tasks have not been run over
  the template change (explicit-ask tier per testing-cost discipline --
  normal-path rendering is covered by the suite + live E2E).
- [x] **update_deps hardenings -- DONE v1.57.1** (P2, from the KEEP-WITH-FIXES verdict):
  re-read pyproject before the final write and abort/re-apply on
  concurrent change (the C++ build makes the window minutes long);
  roll back the pyproject write when the follow-up `uv lock` fails;
  unit tests for the write path (latest-channel sources = git+rev only;
  tomlkit round-trip comment-preservation).
- [ ] Small backend nits (P3): `extra_args` schema reports `default: null`
  (default_factory not serialized) and array fields carry no `items` type;
  top-level `description`/`tags` not clearable via null (asymmetric with
  config nulls); admin `/import` still log-skips invalid entries while the
  CLI importer refuses loudly; `$HEYLOOK_LLAMA_SERVER` pointing at an older
  build than the default-location one gets no staleness warning; consider
  `-t`/`--threads` field (design doc's trigger -- experts-on-CPU landed)
  and widening `n_gpu_layers` to accept `auto`/`all` (b10362 idiom);
  `num_draft_tokens`/`prefill_step_size` effect-class re-read now that
  per_request means "refreshed live into loaded providers".
- [ ] Small editor nits (P3): armed reload label could name the recorded
  cost (retain last warm_ms per model); F8 field-level 422 mapping onto the
  offending input; F6b warn when an `extra_args` token collides with a
  managed field's declared `arg`; per-group collapse on phone widths.

**Frontend TODO (v3):**
- [x] **Per-document settings UI** DONE 2026-07-11 (v1.34.57-.58): ONE shared
  `settings.bindDocumentParams`/`hydrateDocParams`; chat.js + notebook.js both use
  it (no branched copy). Sampler drawer binds to the active conversation/notebook's
  `params`, hydrates silently on select, debounce-PUTs on change, carries forward
  on create; localStorage demoted to new-chat seed. **Browser/E2E UX check still
  recommended** (v3 has no unit tests).
- [ ] **v3 observability CONFIG + VIEW pages** (P2, owner-required): admin panel
  edits `observability_level`/retention via `/v1/admin/config`; a read page
  surfaces `logs/*.jsonl` (recent events/errors + metric summaries via
  `read_json_auto`). `js/telemetry.js` client logger -> `POST /v1/telemetry/events`.
- [ ] **Default sampler temp** (P3, owner: 1.0): no code default exists today
  (`settings.js emptySettings()` -> null; the `1.2` seen was a saved localStorage
  value). If a code default is wanted, set it in the new-chat defaults (frontend)
  or as a `None`->1.0 backend fallback -- decide where.

**Docs / rot:**
- [x] `docs/observability_guide.md` -- rewritten 2026-07-20, then DELETED
  same day under the owner's docs philosophy (code-inferable; the streams are
  self-describing JSONL + /v1/admin/config is the knob).

## P2 - Medium Priority (scheduled)

### Build v1.20.0: Models Config TUI + CI Foundation

#### gguf follow-ups (post-Phase-7 loose ends, 2026-07-26)
- [ ] GGUF-metadata reading in the importer: auto-detect audio modality +
      thinking capability from the file's own metadata (today: audio via
      manual `modalities`, thinking via `supports_thinking` flag).
- [ ] gemma-4 12B MTP A/B on Metal (E4B measured net loss, Qwen3.6 +21%;
      12B unmeasured -- entry has the drafter paired, spec_type set:
      measure before trusting).
- [ ] E2E: no browser-suite coverage for the audio attach flow (needs a
      gguf E2E_MODEL; today's coverage = contract tests + live eval bank).
- [ ] llama-server binary lifecycle: coderef build is pinned by checkout,
      not recorded anywhere machine-readable -- consider stamping the
      build SHA into logs at provider load.
- [ ] Importer scan currently iterates each gguf dir 3x (pickers) --
      fold into one listing pass next time that file is open (flagged by
      /simplify efficiency review; scan-only cost, deliberately deferred).

### Optimization Plan Doc Refresh
- [x] Update `docs/mlx_optimization_plan.md` -- phase 5 updated for v1.18.0 pre-filled cache pattern
- [ ] Mark plan as historical or rewrite deferred items as standalone proposals

## P3 - Nice to Have (opportunistic)

### CI/CD Pipeline
- [ ] Add GitHub Actions workflow (`.github/workflows/test.yml`, `.github/workflows/lint.yml`)
- [ ] Automated testing on commit/push
- [ ] Coverage reporting

### Benchmark Script
- [x] Create `scripts/benchmark.py` (DONE -- HTTP benchmark measuring TTFT, TPS, memory across OpenAI and Messages APIs; its chat/completions arm is dead since v1.79.66 and needs dropping or porting)
- [x] Token throughput, TTFT, memory usage metrics (DONE in `scripts/benchmark.py`)
