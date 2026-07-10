# Current Work

Last updated: 2026-07-10 (j-space apply shipped ~v1.34.31-.35; lens fitting
extracted to the `jlens-mlx` sibling repo. Prior v1.34.25: presets v1.34.22-.24;
legacy React app deleted + OpenAPI drift guard retired)

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
`Qwen3.5-27B-heretic` lens is `hf_model_name=""` / likely fit on STOCK Qwen --
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
distinct from the server-side TOML sampler preset registry (`presets.py`
/ `ChatRequest.preset`); no wire relationship. Frontend v3: chat settings
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
save/apply round-trip, armed delete): 55/55 live. Follow-ups: notebook
page could reuse the preset bar (the `lead` hook makes it cheap; chat-first
was deliberate); a "panel drifted from selected preset" indicator was
deferred.

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
per-model baselines (v1.34.14), heretic scrubbed everywhere (v1.34.15),
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

## 1. Frontend v3 (`apps/heylook-frontend-v3/`, served at /v3; /v2 untouched)

- **SOLID -- chat**: conversations CRUD, streaming w/ thinking blocks,
  edit/regenerate/delete via position truncation, stop = partial saved,
  status telemetry line, mobile drawer. 25 browser E2E checks + a /simplify
  pass. The most-verified surface in the app.
- **SOLID -- shared layer**: `js/page.js` createPage lifecycle (read FIRST
  before touching any page), hash router, table-generated `api.js`,
  `streaming.js` (keepalive comments, reader.cancel, abort-as-completion,
  built-in 503 retry via `onRetryWait`, mid-stream `{"error":...}` payloads
  -> onError), settings panel (null = backend-cascade contract).
- **SOLID -- notebook, models, perf, explore**: built by delegated agents
  against the spec, reviewed, 27 E2E checks (autosave, generate-at-cursor,
  scan/import/load/unload, no-polling perf, logprob chips + keyboard nav,
  390px viewports).
- **UNCERTAIN -- visual design**: functionality verified, but the impeccable
  design gates (`audit`/`polish` per page) NEVER RAN (plugin skill wasn't
  invocable that session) and `DESIGN.md` was never seeded. Tokens/rationale
  live only in css/app.css comments. Looks fine in screenshots; iPhone-Safari
  was checked only via viewport emulation, not a real device.
- **NOT DONE -- cutover**: retiring v2 / promoting v3 is deliberately open
  until the owner has lived in /v3. Nothing blocks it.
- **DONE -- E2E in repo (v1.34.8)**: rebuilt under `tests/e2e/` (puppeteer-core
  + system Chrome; claude-in-chrome refuses localhost). 51/51 live-green vs the
  MoE gemma-4-26B-A4B: chat 24 + pages 27. Spawns its own server with an
  isolated HEYLOOK_DB_PATH so real data is untouched; each suite clears the temp
  DB, pages ends on the danger-zone clear. `cd tests/e2e && bun install` then
  `node run.mjs [chat|pages]`. Must run UNSANDBOXED (Chrome profile dir + Metal).
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
  without the GIL -- .ips reports in ~/Library/Logs/DiagnosticReports).

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
