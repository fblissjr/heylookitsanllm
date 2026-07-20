# CLAUDE.md

<!-- Nav hub. Repo-specific only -- global conventions (uv, orjson, no-emoji,
conventional commits, TDD, path-privacy, docs) live in the user-level CLAUDE.md
and still apply. Don't duplicate them here. Last verified: 2026-07-09 -->

Personal MLX inference server on Apple Silicon: FastAPI backend + a vanilla-JS
frontend (v3, current, served at `/v3`) with two retiring React frontends (v2, legacy).

## Orient first

- **Roadmap** -- the master plan, phased 0-5 (§"v3 frontend guardrails" + Phase 4 = v3 hardening; Phase 3b = Messages-API migration): [docs/project/plan_2026-07.md](./docs/project/plan_2026-07.md).
- **Status + backlog**: [docs/project/CURRENT.md](./docs/project/CURRENT.md) (graded done/left narrative), [docs/project/TODO.md](./docs/project/TODO.md). Read before starting.
- **v3 frontend map** -- what's done/left + the backend<->v3 coupling: [docs/frontend_v3.md](./docs/frontend_v3.md) (git-tracked). Build contract: [docs/frontend_v3_spec.md](./docs/frontend_v3_spec.md) (§4 = API contract).
- Deep dives: **backend reference** is git-tracked in [docs/architecture/](./docs/architecture/) (config, mlx_provider, ecosystem_strategy -- design records + invariants only, the live surface is code + /openapi.json; see its [README](./docs/architecture/README.md)); crash **postmortems** (read before touching providers) are in [docs/architecture/postmortems/](./docs/architecture/postmortems/). Local-only in [internal/](./internal/): `log/`, `research/`, `thoughts/`, and stale subsystem notes (batch/logprobs/thinking) pending refresh. The old React-frontend docs are in `internal/frontend/archive/`.
- **Research / design**: Jacobian-lens ("j-space") interpretability feature -- how-to + tutorial [docs/jspace_guide.md](./docs/jspace_guide.md); build + verifier plan [docs/jspace_integration_plan.md](./docs/jspace_integration_plan.md) (Phase 5-ish; V1/V2 parity GREEN; go-forward plan = its Part 2). Lens **fitting** lives in the `jlens-mlx` sibling repo; this server only **applies** (the old `coderef/jspace_scratch/` spike was relocated there).
- Setup/commands [README.md](./README.md) · tests [tests/README.md](./tests/README.md). (The v3 API contract is spec §4; the live schema is at `/openapi.json` + `/docs`.)
- `internal/`, `models.toml`, `coderef/` are gitignored -- local-only, never committed.

## Architecture

**Backend `src/heylook_llm/`** -- Two providers (`Literal["mlx", "mlx_embedding"]`):
MLXProvider (text+vision), MLXEmbeddingProvider. Router keeps `max_loaded_models=1`
by default (LRU evict + pin + idle-unload via `idle_unload_seconds`/`unload_after_idle_seconds`);
config in `models.toml`. API routers (counts rot; the list is the point): messages, rlm, conversation, notebook,
preset, admin, admin_ops, scan_import, jspace (Jacobian-lens interpretability), config (operational
settings), telemetry (frontend ingestion). DuckDB store (`db.py`: conversations +
notebooks + presets + `settings`, single serialized writer thread, transactional ops;
`HEYLOOK_DB_PATH` override; dynamic field names gated by `_UPDATABLE_*_FIELDS`
frozensets; a `_SCHEMA_VERSION` bump DROPS all tables -- `settings`/`presets` are
additive + drop-safe, key->value/config, not in the drop list). DB/config POLICY (solo
deploy, no data to preserve): NEVER write migration code -- dropping, recreating,
or truncating any DuckDB store or config on a schema change is fine and preferred.
Use additive `CREATE TABLE IF NOT EXISTS` only when you just need to ADD a table
(cheap, non-destructive); for an actual schema CHANGE, bump `_SCHEMA_VERSION` /
drop / recreate. RLM (`rlm.py`): recursive inference with sandboxed REPL.
- [docs/architecture/](./docs/architecture/) (config, mlx_provider, ecosystem_strategy + postmortems -- design records and invariants only; live surface = code + /openapi.json) · [docs/rlm_guide.md](./docs/rlm_guide.md) · converting checkpoints to MLX: [docs/mlx_conversion_guide.md](./docs/mlx_conversion_guide.md)

**Frontend v3 `apps/heylook-frontend-v3/`** -- the current frontend: vanilla
JS, no build, served at `/v3`. 6 pages (chat, notebook, models, perf, explore, jspace);
chat streams over `/v1/chat/completions` today (Messages migration is plan
Phase 3b), takes image input + renders image content blocks out of the DuckDB
store, and has a per-conversation system-prompt editor + saved-preset bar.
Build contract: [docs/frontend_v3_spec.md](./docs/frontend_v3_spec.md) (§4 =
the authoritative backend API contract -- update it in the same commit as any
contract change); orientation + backend coupling: [docs/frontend_v3.md](./docs/frontend_v3.md).
Read `js/page.js` (createPage lifecycle) before touching any page. Design system +
the load-bearing a11y/mobile-parity rules new UI MUST honor -- touch-reveal
fallbacks (`@media (hover:none)`; hover-only affordances are unreachable on
iPhone), the settings drawer as a **modal** (seals `#app` with `inert`, closes on
`hashchange`), aria-live states, label association -- are [DESIGN.md](./apps/heylook-frontend-v3/DESIGN.md) §7.

**Frontend v2 `apps/heylook-frontend-v2/`** (React+Zustand+Vite) -- RETIRING
after v3 cutover (plan Q2/Phase 3); don't invest here. See its
[CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md). (The older legacy React app
`apps/heylook-frontend/` was deleted 2026-07-09 -- v3 has parity.)

**Optloop-lib `apps/optloop-lib/`** -- library-level bench for mlx-lm/mlx-vlm fork experiments (app-level optloop retired 2026-07-06: it bypassed the server code it claimed to measure). [docs/optloop_guide.md](./docs/optloop_guide.md) · its [CLAUDE.md](./apps/optloop-lib/CLAUDE.md). NB: root pyproject pins UPSTREAM mlx-lm/mlx-vlm, not its forks -- fork-side bench wins don't reach the server until upstreamed/repointed.

## MLX / library gotchas (the things you'll get wrong without knowing)

- All text+vision generation routes through `generation_core.run_generation()` -> `mlx_lm.generate.stream_generate`. Vision uses a pre-filled-cache pattern: the VLM forward pass fills the KV cache, then `run_generation()` continues.
- A VLM's forward returns a `LanguageModelOutput`, not raw logits, and caches `_position_ids`/`_rope_deltas` on its language model. Wrap it with `wrap_language_model()` (model_wrappers.py) before driving it with mlx-lm; position state is reset in `run_generation` via `_reset_vlm_positions()`.
- VLM prompt formatting: `mlx_vlm.prompt_utils.apply_chat_template`; inputs: `mlx_vlm.utils.prepare_inputs`. `prepare_vlm_inputs_parallel()` returns a 4-tuple `(images, formatted_prompt, has_images, image_urls)`.
- Vision feature cache (`providers/common/vision_feature_cache.py`): models with `encode_image()` accept `cached_image_features` to skip the vision tower; LRU keyed by image URL (pixel-hash fallback for base64).
- Load-library selection is `MLXProvider.effective_loader` (`providers/common/loader_routing.py`), derived from the config's `modalities` + `loader` fields -- NOT the raw `vision` bool, which is now a derived mirror of `"vision" in modalities`. `is_vlm = (effective_loader == "mlx-vlm")`. `loader="auto"` routes vision->mlx-vlm iff mlx-vlm registers the `model_type`, else mlx-lm (degrades only on POSITIVE non-support; an explicit `loader` forces the engine). Modality DESCRIPTION (`model_importer.detect_modalities`: config `*_config` blocks + `image_token_id`/`image_token_index`/`audio_token_id`...) is deliberately separate from this library-aware routing.
- Embedding backbone: `mlx_lm.utils._get_classes(config_dict)` (private API, takes a dict) -> extract `.model`. Gemma needs `sqrt(hidden_size)` embedding scaling (gated on `model_type.startswith("gemma")`).
- Radix prompt cache has byte-budget + segment-eviction invariants (documented in `providers/common/radix_cache.py`/`prompt_cache.py` docstrings). Hybrid models (KVCache+ArraysCache, e.g. Qwen3.5) have limited correctness -- ArraysCache can't trim to a prefix. See [docs/architecture/mlx_provider.md](./docs/architecture/mlx_provider.md) §4.2 (radix-eligibility gate).
- `mlx_lm.generate.GenerationResponse` is a non-slotted dataclass -- attach per-request metadata via `response.X = value` (`# type: ignore[attr-defined]`), read via `getattr`.
- `mx.set_wired_limit(...)` is set at startup, but the per-generation `wired_limit()` CM is still needed for stream sync. Call `mx.reset_peak_memory()` at `run_generation` start to scope `mx.get_peak_memory()` per request.
- Verify a library is actually broken before working around it.
- Perf numbers are honest as of v1.34.1: recorded tok/s = native mlx-lm `generation_tps`, TTFT/tok-s exclude queue-wait (own `queue_wait_ms` field), trends are success-only. Per-chunk scraping goes through `perf_collector.ChunkTelemetry.absorb()` -- add new chunk fields THERE, not at call sites (batch_processor.py still has 3 unconverted scrape sites, moot when Phase 2 collapses it).
- The FIFO generation gate is a PROCESS-GLOBAL singleton shared by all providers (`_get_generation_gate`); `generation_queue_stats()` reports gate-wide traffic, not per-model. Any "is this model busy" logic built on it is conservative across models. `unload()` waits for actives AND gate waiters (30s cap) -- the active counter decrements before `gate.release()`, so never gate teardown on actives alone.
- Any NEW code path running MLX forwards off the event loop (analysis endpoints etc.) must run on `streaming_utils._executor_pool` (a pinned, REUSED thread) inside `with mx.stream(generation_core._get_generation_stream())`, AND acquire the process-global gen gate + `router.pin_model()`. Starlette's `run_in_threadpool` has no thread-local MLX stream ("There is no Stream(cpu/gpu, 0)") and a dying MLX thread aborts the PROCESS. Verify on a real worker thread, not the main thread (see `jspace_api.py`).
- `mx.load` is lazy/mmap-backed -- `mx.eval()` the arrays at load time if they'll first be used on a different (worker) thread, else the first eval crashes on that thread's missing CPU stream.
- A model's `.layers` can be a fresh-slice `@property` (pipeline-parallel Qwen3.5/deepseek/glm4_moe: `pipeline_layers = self.layers[start:end]`), NOT the list the forward iterates -- to hook/mutate blocks use the underlying list on the inner decoder (`inner.layers`/`.h`), not `model.layers`.
- Live-verifying streaming/latency changes: the 31B dense gemma natively decodes ~10 tok/s (looks identical to the old delivery cap); use the MoE `gemma-4-26B-A4B` (~90 tok/s) as the discriminating model.
- Stop-token/eos resolution has the same dual-source trap as chat templates: a model's full eos set can be split across tokenizer_config.json's `added_tokens_decoder` and tokenizer.json's `added_tokens` (gemma-4's `<turn|>` terminator lives only in the latter) -- read both. Raw HF tokenizers also don't absorb `generation_config.json`'s eos list, and mlx-lm's `stream_generate` auto-wraps a raw tokenizer with only the single `eos_token_id` -- `run_generation` wraps it itself (`ensure_gen_tokenizer`) with the full resolved stop set, or a model generates past its own end-of-turn.
- Chat-template resolution: `providers/common/template_info.py` is the single source of truth (per-model `chat_template_source` in models.toml; auto = `chat_template.jinja` > embedded `tokenizer_config.json` > `chat_template.json`; explicit source force-installs on the tokenizer at load, auto only fills a missing one). Traps: mlx-lm `chat_template_type` python templates live on the TokenizerWrapper -- the inner tokenizer's `chat_template` attr stays None, so never gate "has a template" on that attr alone; HF's legacy list-form `chat_template` isn't parsed (string-only); import-time jinja detection is the shared `detect_chat_template_source()` (used by BOTH the CLI wizard and the `/v1/admin` import route -- don't re-inline it). An MTP/spec-decode head registered as a model legitimately has NO chat template -- the load warning is expected there.
- mlx-lm's `TokenizerWrapper.apply_chat_template` silently injects `enable_thinking=True` when the kwarg is ABSENT -- always pass an explicit bool. The kwarg is the cross-model thinking control: transformers forwards extra apply_chat_template kwargs as template variables (Qwen3 renders `<think>`, gemma-4 renders thought channels; others ignore it), and "template references enable_thinking" is the thinking-capability signal.
- Upstream posture (details: `docs/architecture/ecosystem_strategy.md`): mlx-lm is release-starved -- SHA-pin rather than wait for PyPI, check its open-PR backlog BEFORE writing any workaround, expect new capabilities via sidecar packages.

## Repo conventions (beyond the global ones)

- New endpoint or changed response model: module with `APIRouter(tags=["Name"])`, add the tag to `openapi_tags` + `app.include_router()` in `api.py`. (The OpenAPI drift guard -- `generated-api.ts` / `scripts/check_openapi_sync.sh` / the pre-commit block / `/openapi-regen` -- was retired 2026-07-09 with the legacy React app that consumed the generated TS types; v3 hand-writes `api.js`. The live schema stays at `/openapi.json` and `/docs`.)
- Removing a provider/feature: grep the repo, then check `config.py` (Literal+Union), `router.py`, `api.py`, README/ARCHITECTURE, `pyproject.toml` extras, frontend type unions, test fixtures.
- TWO preset systems, one term each -- don't conflate or re-alias: **sampler presets** = bundled TOMLs `src/heylook_llm/data/presets/` via `PresetRegistry` (`presets.py`), resolved in the provider cascade (`thinking` auto-applies for `enable_thinking` models; `vlm-*` encode mlx-vlm's ignored-param subset), reachable as `ChatRequest.preset` / models.toml `default_preset` / `/v1/admin/models/sampler-presets`, discoverable via `/v1/capabilities.sampler_presets`; roster is exact-pinned by `test_preset_registry.py` (a new preset must name its consumer). **User presets** = `/v1/presets` DuckDB prompt+sampler bundles (v3's preset bar). The import/admin term "profile" was retired 2026-07-20 (`--profile` survives only as a CLI alias for `--preset`) -- don't reintroduce it (collides with `/v1/performance/profile`). Details: `docs/architecture/config.md`.
- The security hook false-positives on `mx.eval` (MLX graph materializer, not Python's eval) -- prefer `mx.async_eval` or acknowledge it.
- Observability spine (`observability.py`): ONE ingestion path `record_event(type, *, tier, min_level, source, fields=<dict>)` -> level-gated JSONL under `logs/` (`metrics.jsonl` content-free/aggregatable; `events.jsonl` correlated, may carry BOUNDED error text -- type+message+cause-chain, still NEVER prompts/responses/token IDs). `fields` is an explicit dict (NOT `**kwargs`) so caller/client keys can't collide with the reserved kwargs. Best-effort: never raises (inference must not break). `diag_event` (diagnostic_logger.py) delegates here; `memory.py`'s legacy streams also write under `logs/` and are gated by the master off switch (see below).
- Observability CONTROL is a single knob: `observability_level` (off|minimal|standard|debug), an operational SETTING resolved **DB > default** (no env override -- env silently beating the admin UI is a footgun; env is bootstrap-only: `HEYLOOK_LOGS_DIR`, `HEYLOOK_DB_PATH`). `off` is the master kill switch (silences the spine AND memory.py's streams). Settings live in the App-DB `settings` table (`db.get_setting`/`set_setting`), contract in `settings.py` (`SettingsSchema` + `resolve_settings`), CRUD via `/v1/admin/config`; the level+retention are cached in-process (`observability.configure`), refreshed at startup + on PUT. Rotation is file-based (size + age, hourly on the tick). Content invariant is level-INDEPENDENT: `minimal` is NOT "content-free" (its events carry error text) -- only the metrics tier is guaranteed content-free.
- Route MemoryManager calls through `memory.safe_mm_call(...)` (no-op when None, swallows errors); use `sampler_summary_from_request` (memory.py) for "what was this configured with". Redesign status + the memory.py-stream consolidation follow-ups: `docs/project/TODO.md` + `internal/research/observability_and_config_redesign.md`.
- Pydantic `Field` defaults MUST use keyword form (`Field(default=None, ...)`, never `Field(None, ...)`) -- this pyright build only recognizes the keyword form; positional defaults make every model constructor flag false "arguments missing" errors (repo-wide sweep done 2026-07-20).
- Pydantic model + custom headers: `Response(content=model.model_dump_json(), media_type="application/json", headers=...)` (`JSONResponse` double-serializes). SSE post-generation telemetry (peak mem, cache bytes) goes in the usage chunk's `timing` (client needs `stream_options.include_usage=true`).
- Frontend v2 specifics (Pydantic `model_fields_set`, `<base href="/v2/">` SPA serving, sanitization): see its [CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md).
- Never commit runtime data: `*.db`, `*.jsonl`, `/data/*`, `apps/*/data/*` are gitignored; package data at `src/heylook_llm/data/` is intentionally NOT ignored. `modelzoo/` and `adapters/` (j-space lenses; `HEYLOOK_JSPACE_DIR` default `adapters/jspace/<model_id>/`) are git-tracked dirs with gitignored contents (only their `.gitkeep`).
- Commits fine without asking; never push unless told. Update `internal/log/log_YYYY-MM-DD.md` before ending a session.
- The roadmap/status/backlog (`plan_2026-07.md`, `CURRENT.md`, `TODO.md`) + the v3 map live git-tracked in `docs/project/` and `docs/` -- git IS their history, edit them directly. `internal/` is still unversioned (gitignored): it holds the local-only docs (research/, log/, thoughts/, scratch/, frontend/archive/). Before destructively rewriting a long-lived doc that remains under `internal/`, copy the old version to an `archive/` subdir first -- that copy IS the history.
- CLAUDE.md carries MECHANISMS (how things work, what bites); STATUS (what's done, counts, "until X lands") lives in `docs/project/CURRENT.md` + the plan. Status lines here rot into being actively wrong -- the perf-distrust note did exactly that within a day.

## Tests

- Run via `/test-suite` (backend + frontend in parallel). `tests/unit/` + `tests/contract/` are fully green (Metal-gated skips OK) -- any failure is a regression, investigate it. There is no pre-existing-failure allowlist. (No counts here on purpose: they rot; green-is-the-invariant doesn't.)
- **Behavioral eval bank** (`tests/eval/`, opt-in): 13 tasks covering thinking split/leak, stop discipline, vision correctness, vision-token budgets. Run for changes touching templates/parsers/stop-tokens/vision -- `uv run python tests/eval/run.py --server <url> --models <ids>` against a RUNNING server (never spawns one). Unit tests cannot certify these subsystems (the 07-20 turn-overrun + thinking-leak bugs passed 1000+ of them).
- **Browser E2E** (`tests/e2e/`, v1.34.8+): puppeteer-core + system Chrome (claude-in-chrome refuses localhost). Spawns its own server with an isolated `HEYLOOK_DB_PATH` (real data untouched); each suite clears its temp DB. NOT wired into `/test-suite` (Metal/GPU-gated + slow + spawns a server) -- opt-in: `cd tests/e2e && bun install`, then `node run.mjs [chat|pages]`, MUST run UNSANDBOXED. Carries a client-side streaming-cadence guard -- the ONLY automated check for the Phase 1 delivery fix (server telemetry can't see it); needs a fast `E2E_MODEL` (default MoE gemma-4-26B-A4B).
- NEVER apply an MLX `sys.modules` mock at module level with `.start()`; use `with patch.dict(...)` or the `mock_mlx` fixture. A module-level start leaks mocks across the whole session and fakes ~50 "Metal context" failures (the bug that produced the old allowlist).
- `test_mlx_provider.py` SEGFAULTS at GC teardown when run in near-ISOLATION (MLX `unload`/`__del__` flakiness) but passes clean in any multi-file batch / the full suite -- not a regression; run it batched, not alone.
- Provider unit tests build `MLXProvider` from RAW config dicts (bypassing `MLXModelConfig` validation; production passes `model_config.config.model_dump()`), so provider/loader code must tolerate un-normalized config (e.g. missing `modalities`) -- a back-compat branch that looks dead in the router path may be live only in tests.
- Backend: `uv run pytest tests/unit/ tests/contract/ -v`. `--timeout` is not installed. `settings.local.json` exempts `uv run pytest`/`uv sync`/`uv lock`/`bun install`/`bun run build` from the sandbox.
- Root venv: sync with `uv sync --all-extras` -- plain `uv sync` strips the performance/test extras (pyturbojpeg = the multipart JPEG decoder, uvloop, pytest plugins) and the loss is silent until an image request or test run fails.
- Separate venvs (cd first): batch-labeler (`uv sync --dev`), optloop-lib (`uv sync`).
- GPG signing needs the 1Password agent; if a commit fails on socket errors use `git -c commit.gpgsign=false commit` (`-c` before `commit`).
- Sandbox traps: `ENV=x uv run ...` does NOT match the uv exemption (env-var prefix changes the command match -> sandboxed, no Metal); sandboxed `curl` can't reach localhost (probe via `uv run python` + urllib); never launch the server piped to `head` (SIGPIPE wedges it -- redirect to a file). To verify schema-neutrality of a change (no committed OpenAPI artifact exists -- deliberate), export `app.openapi()` from a HEAD~1 worktree and byte-compare. Sandboxed `find` can silently return nothing traversing `modelzoo/` (files present per `ls`) -- enumerate model dirs with `ls` or `uv run python` glob/`os.walk` instead.
