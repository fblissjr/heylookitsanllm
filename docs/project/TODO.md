# Persistent TODOs

Cross-session task backlog organized by priority.

*Last reviewed: 2026-07-12*

## J-space / jlens-mlx (from jspace_integration_plan.md Part 2)

Fitting lives in the `jlens-mlx` sibling repo; this server applies. Apply feature +
baseline fitter are GREEN (see CURRENT.md 2026-07-10).

- [ ] **Refit the band lens on the fixed corpus** (P1, 2026-07-12): `band-n12`/`band-n12b` were
  degenerate (mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects
  `enable_thinking=True` -> every on-policy completion collapsed into shared CoT-preamble
  boilerplate, 62% of fitted positions). Both fits' results are DISCARDED (method stack unaffected).
  jlens-mlx now has explicit `enable_thinking` control (default False) + a diversity gate
  (`238826e`/`951dd76`/`232b98b`) -- refit band 16-47 on the fixed corpus is the next action.
  Supersedes the old "band-n12b running" status. See `docs/jspace_integration_plan.md` §
  "Observations & watch-items", 2026-07-12 subsection.
- [~] **Fidelity gate -> legibility metric** (P2, 2026-07-11): the KL/top-k identity tripwire ships;
  the naive final-logit-agreement gate MISLEADS on band layers (ranked degenerate ' __' above
  meaningful ' Paris'), so a disposition-aware `verify.legibility_report` (content-vs-junk ranking)
  is now the band-layer signal, wired into the fit + sidecar. Apply/regrade it onto the band refit
  once it's done on the fixed corpus (`regrade_lens.py` doesn't emit legibility yet -- ~2min add).
- [ ] **Fit/apply capture parity -- numerical check** (P1, gates the refit -- run it FIRST in the refit session; 2026-07-11, re-affirmed 2026-07-12 as the
  top open correctness IOU by an architecture review): fitting captures residuals cache-less; apply
  uses a fresh cache (the hybrid served qwen3_5 crashes cache-less). Both are causal-from-scratch so
  they SHOULD match, but it's asserted, never verified -- and it's the foundation of served-model
  lens correctness. Cheap check: capture `h_l` both ways on one input, assert allclose. (Does not
  invalidate current lenses; identity KL~0 is consistent.)
- [~] **Post the PR comments** (P2, 2026-07-12; harness port DONE -- `bench/upstream_pr_eval/` on the pushed jlens `upstream-pr-eval` branch, links filled in the drafts): the
  2026-07-12 upstream mlx-lm GDN differentiability eval (PRs #1389/#1217, both numerically correct,
  see the plan doc) was run ad hoc; port it to a jlens-mlx `upstream-pr-eval` branch, then the owner
  posts data-backed comments on #1217 (full dataset) and #1389 (the log-domain fp32 `dg`-gradient
  finding -- not a bug, cancels at the parameter leaves). Draft comments live in jlens's internal
  folder.
- [ ] **Audit GDN cache-slice captures for the #1077 `mx.contiguous()` pattern** (P3, 2026-07-12):
  upstream mlx-lm #1077 (merged) fixed a shared-buffer memory leak by adding `mx.contiguous()` on
  GDN cache slices. Check our code (jspace capture path + any other raw GDN cache-slice reads) for
  the same unguarded-shared-buffer pattern.
- [ ] **Watch #1217 merge before any mlx-lm pin bump** (P2, 2026-07-12): #1217 adds a `training=`
  kwarg passed unconditionally at every qwen3_5 call site upstream; jlens-mlx's `gdn_fit_patch`
  already absorbs unknown future kwargs (`951dd76`) so a bump won't TypeError mid-fit, but confirm
  before bumping the served-side pin too.
- [ ] **Consider #1515/#1532, #1486/#1456, #1526 on the next serving-side mlx-lm bump** (P3,
  2026-07-12): #1515+#1532 add anchor-stride prefix reuse for non-trimmable hybrid caches (large
  TTFT claims, relevant to qwen3_5 serving); #1486/#1456 fix hybrid ArraysCache trimmability for
  speculative decoding (issue #1446); #1526 fixes `max_kv_size` being silently dropped for models
  with their own `make_cache` (qwen3_5 still needs the analogous one-line fix upstream). None of
  these are fitting-path, all are serving-path -- triage on the next pin bump, not now.
- [ ] **Fit speedup: seq-tile the GDN scan** (P3, 2026-07-11): the chain fit is ~44min/item / a full
  band ~7-8h; a designer+verifier pass found NO config-level 2-3x (`chunk` is a dead knob). The real
  lever is the GDN kernel `MAX_T=128` cliff -- tile the recurrence across 128-tok blocks (EXACT) so
  long items stay on the fast kernel. Delicate; must re-pass `check_chain_vs_direct` (cos 1.0). A
  guardrail (warn + kernel-eligibility sidecar metadata) already stops the silent slow path.
- [ ] **Standing golden gate for `/v1/jspace/analyze`** (P3): freeze onset top-k + features,
  tie-aware calibrated epsilon, mutation-checked -- turns the one-time V1/V2 parity into a
  wired regression gate.
- [ ] **Visualizer track** (P3): gate cleared 2026-07-10 -- `apps/heylook-frontend-v3/DESIGN.md`
  seeded (OKLCH strength/chip system formalized; paradigm = matrix-first, Neuronpedia-style
  layer-range slider + aggregation sidebar as the growth path). SHIPPED so far
  (v1.34.36-.37): item 1 click-to-pin readout (strip rows + heatmap cells, Esc/arrow
  walk, echo highlight, onset marker), the per-cell top-N analyze extension
  (`heatmap_top_k` -- every cell pins its full readout now), item 2 layer-range
  slider + aggregation view, and a "provisional lens" badge off the sidecar
  provenance (`/v1/jspace/models` meta). Remaining, in order: live streaming
  (new SSE analyze endpoint) -> steer/swap/ablate interventions (last).
  **Fold into the streaming rework** (both live in the analyze grid loop it
  will rewrite; from the 2026-07-10 review): (a) unify the onset column's two
  numeric paths -- onset_strip uses float64 np.argsort, the heatmap's last
  column uses float32 argpartition, so near-tied logits can show different
  top-1 tokens for the same position (breaks the echo highlight); one shared
  per-position reduce fixes it. (b) batch the per-layer device-to-host syncs
  (~4 x band_layers sequential np.asarray evals under the gen gate) into one
  mx.eval, and memo tok.decode per request (~5k redundant single-id decodes).
- [ ] **Confirm coverage for the deleted `verify_endpoint.py` / `probe_thread.py`** (P3,
  updated 2026-07-12): they were git-rm'd from jlens (its `migrated_from_scratch/` is fully
  dissolved; recoverable from jlens git history). This repo's `tests/contract/test_jspace_api.py`
  + `tests/unit/test_jspace_analyze.py` likely cover the same ground -- diff the checks, recover
  from history only if a gap shows.
- [ ] **HF lens repo** (P3): publish OUR fitted lenses post-own-fit; gated -- don't
  republish the converted third-party lenses (Gemma ToU).
- [ ] **Stale docstrings** (P4): `tests/unit/test_jspace.py`, `test_jspace_features.py`,
  `src/heylook_llm/jspace/capture.py` still name `coderef/jspace_scratch/` (dissolved into
  `jlens-mlx/migrated_from_scratch/`) -- fix when next touching those files.

## Presets/system-prompt follow-ups (from the v1.34.22-.24 review passes)

- [ ] **Cap-gated `enable_thinking` can be pinned invisibly** (P2): the global
  settings cache keeps cap-gated keys when the current model can't display
  the control (`requiresCap` hides the checkbox but `samplerParams()` still
  emits the flag on every chat/notebook/explore request until Reset). Predates
  presets -- switching models always had this gap -- but preset apply is a
  one-click entry point into it. Needs a caps-aware settings design (drop or
  grey cap-gated keys when unsupported), not a preset-side patch.
- [ ] **Presets vs TOML registry: dual-source by name** (P3, design decision):
  user presets are client-expanded only; `ChatRequest.preset` resolves just
  the bundled TOML registry. If saved presets should ever work by name from
  the raw API/CLI, make the registry dual-source (TOML + DB rows, name-unique
  across both) instead of growing a second wire path. Deliberately NOT done
  in v1.34.22 -- client-side copy semantics (LM Studio) was the owner's ask.
- [ ] **Notebook page preset bar** (P3): chat-first was deliberate; the
  sections prepend cleanly onto notebook's settings panel. Notebook already
  has its own sysprompt input -- decide whether preset apply writes it.
- [ ] **"Panel drifted from selected preset" indicator** (P3): applying copies
  with no live binding; after edits the select still shows the preset name.
  LM Studio shows a modified-dot. Cosmetic, deferred.
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
- [ ] **Wire `show_special_tokens` render consumers** (P2, 2026-07-11): the display pref exists +
  is honesty-first `true`, but is gated (`wired:false`) out of the UI because NO surface reads it --
  the token-rendering paths still strip specials (DESIGN.md §6 "known violation"). Wire chat /
  notebook / token-explorer to read `getDisplayPref` + subscribe to `onDisplayChange` (unsubscribe on
  teardown), then flip `wired:true`. Edit surfaces must ALWAYS show raw tokens regardless (§6).
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
  `api.py::_infer_model_capabilities`, `model_importer` entry-build,
  `loader_routing._modalities_of` raw-dict fallback), then drop the mirror + the
  bool. Do NOT do piecemeal -- it's a coordinated removal.

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
- [ ] `docs/observability_guide.md` has a redesign banner + corrected paths but
  its body still describes the legacy `internal/log/` 4-stream layout -- full
  rewrite pending (P3).

## Recently Completed (Phase 2 -- 2026-03-13)

- [x] Remove STT provider (`mlx_stt_provider.py`, `stt_api.py`, parakeet-mlx dep)
- [x] Narrow provider type to `Literal["mlx", "mlx_embedding"]`
- [x] Rename `embedding_gemma.py` → `embedding_model.py`, `EmbeddingGemmaModel` → `EmbeddingModel`
- [x] Dynamic backbone loading via `load_backbone()` using `mlx_lm.utils._get_classes()`
- [x] Pydantic V2 migration (`@field_validator`, `@model_validator`)
- [x] Stop-token utility extracted to `providers/common/stop_tokens.py`
- [x] Fix transformers 5.x VLM processor loading (4 patches in `_apply_transformers_patches()`)
- [x] Fix `eos_token_ids` null safety in `mlx_batch_text.py`

## P0 - Critical (blocks other work)

None currently.

## Slice 1 (in flight)

- [x] S1.1 -- per-request peak memory + KV bytes (v1.28.0, `be0f15f`)
- [x] S1.2 -- three-stream observability (v1.28.0, `2f9b03d`/`f28d52d`/`3641cf0`)
- [x] S1.3 -- byte cap on VisionFeatureCache (v1.28.0, `312db4e`)
- [x] S1.4 -- provider.warmup() + prefill_step_size (v1.28.0, `31e59a2`/`915dab6`)
- [ ] S1.5 -- batched docs + cleanup (in progress: STT removal + test cleanup + cache_keys refactor done; docs-audit items landing next)
- [ ] S1.6 -- LAN hardening (Caddy reverse proxy + optional `HEYLOOK_ADMIN_TOKEN`)

## Slice 1 gated work

- [x] S1.2b -- preset + import redesign: SHIPPED (presets.py registry, default_preset cascade -- see "C1 of S1.2b" in presets.py docstring). Stale-gated entry closed 2026-07-06 during docs audit; the import-defaults follow-through landed in v1.31.3/v1.32.0.

## P1 - High Priority (do soon)

### TOML Migration Completion
- [x] Integrate `--interactive` flag into model_importer.py (DONE v1.18.1)
- [x] Wire up ConfigEditor in import workflow (DONE v1.18.1)
- [x] Move profiles to TOML files, rename profiles, dynamic discovery (DONE v1.19.0)
- [x] Fix `ModelProfile.apply()` precedence bug (DONE v1.19.0)
- [ ] Test with: `heylookllm import --folder ~/test --interactive` (manual)

### Dependency Cleanup
- [x] Remove `mlx` optional extra (duplicated core deps) (DONE v1.19.0)
- [x] Purge unused deps: torch, torchvision, opencv-python, scipy (DONE v1.19.0)
- [x] Move `datasets` to `analytics` extra, `rich` to `scripts` extra (DONE v1.19.0)

### Stale Code Removal
- [x] Delete stale integration test `test_performance_monitoring` (DONE v1.19.0)
- [x] Delete `/v1/performance` stub and `/v1/performance/profile/{time_range}` endpoint (DONE v1.19.0)

## P2 - Medium Priority (scheduled)

### Build v1.20.0: Models Config TUI + CI Foundation

#### Models Config Command (Phase 4)
- [ ] Create `src/heylook_llm/commands/models_config.py`
- [ ] Implement TUI menu for model management (ConfigEditor in `config_tui.py` exists, 344 lines -- needs CLI wrapper)
- [ ] Add subparser to `server.py`
- [ ] Integrate into server.py CLI

#### llama-server Provider (Phase 5)
- [ ] Create `src/heylook_llm/providers/llama_server_provider.py`
- [ ] Subprocess management for llama-server binary
- [ ] HTTP client with SSE streaming
- [ ] Router integration
  - Note: LlamaCppProvider was removed in v1.21.0. llama-server provider would re-add GGUF support via subprocess rather than embedded library.

### Optimization Plan Doc Refresh
- [x] Update `docs/mlx_optimization_plan.md` -- phase 5 updated for v1.18.0 pre-filled cache pattern
- [ ] Mark plan as historical or rewrite deferred items as standalone proposals

## P3 - Nice to Have (opportunistic)

### CI/CD Pipeline
- [ ] Add GitHub Actions workflow (`.github/workflows/test.yml`, `.github/workflows/lint.yml`)
- [ ] Automated testing on commit/push
- [ ] Coverage reporting

### Benchmark Script
- [ ] Create `scripts/benchmark.py` (standalone, useful before shape bucketing)
- [ ] Token throughput, TTFT, memory usage metrics

### Build v1.21.0: llama-server Provider + GGUF + Benchmark
- [ ] llama-server provider complete
- [ ] GGUF extra activation (uncomment in `pyproject.toml`, blocked on provider)
- [ ] Benchmark script validates cross-platform performance

## Deferred (blocked on upstream)

### MLX Engine Optimization (remaining)
- [ ] Shape bucketing for prefill (needs attention mask correctness verification)
- [ ] `mx.compile` on decode step (deprioritized -- marginal gains vs. complexity; revisit if mlx-lm adds native compile support)
- [ ] Automatic draft model selection (vocabulary compatibility checking)
- [x] Full vision path unification (DONE v1.18.0 -- pre-filled cache pattern, no `inputs_embeds` needed)
- [ ] Vision + speculative decoding (pre-filled cache incompatible with speculative prefill)
- [ ] Radix cache for vision (pre-filled cache bypasses radix tree)

- [x] Error-chunk altitude: DONE in v1.33.0 -- provider raises typed GenerationFailed/InvalidGenerationRequest; batch/RLM now fail loudly; non-streaming client errors return 400. See docs/architecture/postmortems + CHANGELOG 1.33.0.