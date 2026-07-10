# Persistent TODOs

Cross-session task backlog organized by priority.

*Last reviewed: 2026-07-10*

## J-space / jlens-mlx (from jspace_integration_plan.md Part 2)

Fitting lives in the `jlens-mlx` sibling repo; this server applies. Apply feature +
baseline fitter are GREEN (see CURRENT.md 2026-07-10).

- [ ] **Corpus recipe + first own-fit** (P2): stratified corpus (chat + reasoning +
  over-weighted safety + matched-benign + WikiText control), fit on-policy, first own-fit
  on the served `Qwen3.5-27B-abliterated` (Metal-gated; add the GDN speed accelerator only if
  the direct-VJP baseline is too slow through the 48 GDN layers).
- [ ] **Held-out fidelity gate + abliterated-vs-stock lens diff** (P2): per-layer KL/top-k vs
  true logits on held-out data (refuse to save below threshold); the stock-vs-abliterated diff
  is the first real finding.
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
- [ ] **Re-home `verify_endpoint.py` / `probe_thread.py`** (P3): now in
  `jlens-mlx/migrated_from_scratch/`; they test the running server + MLX thread semantics,
  so they belong back as real heylook `tests/`.
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