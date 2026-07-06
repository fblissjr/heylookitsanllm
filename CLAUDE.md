# CLAUDE.md

<!-- Nav hub. Repo-specific only -- global conventions (uv, orjson, no-emoji,
conventional commits, TDD, path-privacy, docs) live in the user-level CLAUDE.md
and still apply. Don't duplicate them here. Last verified: 2026-07-06 -->

Personal MLX inference server on Apple Silicon: FastAPI backend + two frontends
(legacy React, vanilla-JS v2).

## Orient first

- WIP and backlog: [internal/session/CURRENT.md](./internal/session/CURRENT.md), [internal/session/TODO.md](./internal/session/TODO.md). Read before starting.
- Deep dives: [internal/](./internal/) -- `backend/`, `frontend/`, `bugs/` (postmortems; read before touching providers), `research/`, `log/`.
- Setup/commands [README.md](./README.md) · frontend API [docs/frontend_api_reference.md](./docs/frontend_api_reference.md) · tests [tests/README.md](./tests/README.md).
- `internal/`, `models.toml`, `coderef/` are gitignored -- local-only, never committed.

## Architecture

**Backend `src/heylook_llm/`** -- Two providers (`Literal["mlx", "mlx_embedding"]`):
MLXProvider (text+vision), MLXEmbeddingProvider. Router keeps `max_loaded_models=1`
by default (LRU evict + pin + idle-unload via `idle_unload_seconds`/`unload_after_idle_seconds`);
config in `models.toml`. 7 API routers: messages, rlm, conversation, notebook,
admin, admin_ops, scan_import. SQLite conversation store via aiosqlite (`db.py`,
`conversation_api.py`; `HEYLOOK_DB_PATH` override; dynamic field names gated by
`_UPDATABLE_MESSAGE_FIELDS`). RLM (`rlm.py`): recursive inference with sandboxed REPL.
- [internal/backend/](./internal/backend/) (architecture, api, router, config, providers/mlx) · [docs/rlm_guide.md](./docs/rlm_guide.md) · [docs/observability_guide.md](./docs/observability_guide.md)

**Frontend v2 `apps/heylook-frontend-v2/`** -- vanilla JS, no framework/bundler,
served at `/v2`. Pages: Chat, Batch, Models, Performance, Notebook, Token Explorer.
See its [CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md).

**Frontend legacy `apps/heylook-frontend/`** -- React+Zustand+Vite, 7 applets,
being replaced by v2. [ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md).

**Optloop-lib `apps/optloop-lib/`** -- library-level bench for mlx-lm/mlx-vlm fork experiments (app-level optloop retired 2026-07-06: it bypassed the server code it claimed to measure). [docs/optloop_guide.md](./docs/optloop_guide.md) · its [CLAUDE.md](./apps/optloop-lib/CLAUDE.md). NB: root pyproject pins UPSTREAM mlx-lm/mlx-vlm, not its forks -- fork-side bench wins don't reach the server until upstreamed/repointed.

## MLX / library gotchas (the things you'll get wrong without knowing)

- All text+vision generation routes through `generation_core.run_generation()` -> `mlx_lm.generate.stream_generate`. Vision uses a pre-filled-cache pattern: the VLM forward pass fills the KV cache, then `run_generation()` continues.
- A VLM's forward returns a `LanguageModelOutput`, not raw logits, and caches `_position_ids`/`_rope_deltas` on its language model. Wrap it with `wrap_language_model()` (model_wrappers.py) before driving it with mlx-lm; position state is reset in `run_generation` via `_reset_vlm_positions()`.
- VLM prompt formatting: `mlx_vlm.prompt_utils.apply_chat_template`; inputs: `mlx_vlm.utils.prepare_inputs`. `prepare_vlm_inputs_parallel()` returns a 4-tuple `(images, formatted_prompt, has_images, image_urls)`.
- Vision feature cache (`providers/common/vision_feature_cache.py`): models with `encode_image()` accept `cached_image_features` to skip the vision tower; LRU keyed by image URL (pixel-hash fallback for base64).
- Embedding backbone: `mlx_lm.utils._get_classes(config_dict)` (private API, takes a dict) -> extract `.model`. Gemma needs `sqrt(hidden_size)` embedding scaling (gated on `model_type.startswith("gemma")`).
- Radix prompt cache has byte-budget + segment-eviction invariants (see router.md). Hybrid models (KVCache+ArraysCache, e.g. Qwen3.5) have limited correctness -- ArraysCache can't trim to a prefix. See [internal/bugs/radix_cache_vlm_crash.md](./internal/bugs/radix_cache_vlm_crash.md), [vlm_vision_bug.md](./internal/bugs/vlm_vision_bug.md).
- `mlx_lm.generate.GenerationResponse` is a non-slotted dataclass -- attach per-request metadata via `response.X = value` (`# type: ignore[attr-defined]`), read via `getattr`.
- `mx.set_wired_limit(...)` is set at startup, but the per-generation `wired_limit()` CM is still needed for stream sync. Call `mx.reset_peak_memory()` at `run_generation` start to scope `mx.get_peak_memory()` per request.
- Verify a library is actually broken before working around it.
- Recorded perf numbers are untrustworthy until plan Phase 1 items 1-2 land: streaming tok/s is quantized to ~10/s by the 100ms poll loop (`streaming_utils.py`), headline tok/s+TTFT include queue-wait, and mlx-lm's native per-chunk `prompt_tps`/`generation_tps` are never read. Don't cite or optimize against them; see `internal/backend/plan_2026-07.md` Phase 5 "measurement reality check".

## Repo conventions (beyond the global ones)

- New endpoint or changed response model: module with `APIRouter(tags=["Name"])`, add the tag to `openapi_tags` + `app.include_router()` in `api.py`, then run `/openapi-regen` to refresh `generated-api.ts`. A pre-commit hook runs `scripts/check_openapi_sync.sh` (offline schema via `app.openapi()`) when a top-level `src/heylook_llm/*.py`, `schema/`, or the generated file is staged, and blocks on drift. Run manually: `cd apps/heylook-frontend && bun run check:api`. The check skips gracefully when uv/bun/MLX are unavailable, so it only blocks on detected drift; the hook lives in `.git/hooks/` (not committed) -- fresh clones rely on `check:api`/CI.
- Removing a provider/feature: grep the repo, then check `config.py` (Literal+Union), `router.py`, `api.py`, `generated-api.ts`, README/ARCHITECTURE, `pyproject.toml` extras, frontend type unions, test fixtures.
- The security hook false-positives on `mx.eval` (MLX graph materializer, not Python's eval) -- prefer `mx.async_eval` or acknowledge it.
- Observability invariant: log streams record numbers + metadata only, never prompts/responses/token IDs. Use `sampler_summary_from_request` (memory.py) for "what was this configured with"; route MemoryManager calls through `memory.safe_mm_call(...)` (no-op when None, swallows errors -- observability must never break inference).
- Pydantic model + custom headers: `Response(content=model.model_dump_json(), media_type="application/json", headers=...)` (`JSONResponse` double-serializes). SSE post-generation telemetry (peak mem, cache bytes) goes in the usage chunk's `timing` (client needs `stream_options.include_usage=true`).
- Frontend v2 specifics (Pydantic `model_fields_set`, `<base href="/v2/">` SPA serving, sanitization): see its [CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md).
- Never commit runtime data: `*.db`, `*.jsonl`, `/data/*`, `apps/*/data/*` are gitignored; package data at `src/heylook_llm/data/` is intentionally NOT ignored.
- Commits fine without asking; never push unless told. Update `internal/log/log_YYYY-MM-DD.md` before ending a session.
- `internal/` is unversioned (gitignored): before destructively rewriting a long-lived internal doc (plans, specs), copy the old version to an `archive/` subdir first -- that copy IS the history.

## Tests

- Run via `/test-suite` (backend + frontend in parallel). `tests/unit/` is fully green (713 passed as of 2026-07-06; Metal-gated skips OK) -- any failure is a regression, investigate it. There is no pre-existing-failure allowlist.
- NEVER apply an MLX `sys.modules` mock at module level with `.start()`; use `with patch.dict(...)` or the `mock_mlx` fixture. A module-level start leaks mocks across the whole session and fakes ~50 "Metal context" failures (the bug that produced the old allowlist).
- Backend: `uv run pytest tests/unit/ tests/contract/ -v`. `--timeout` is not installed. `settings.local.json` exempts `uv run pytest`/`uv sync`/`uv lock`/`bun install`/`bun run build` from the sandbox.
- Separate venvs (cd first): batch-labeler (`uv sync --dev`), optloop-lib (`uv sync`). Frontend legacy: `cd apps/heylook-frontend && bunx vitest run`.
- GPG signing needs the 1Password agent; if a commit fails on socket errors use `git -c commit.gpgsign=false commit` (`-c` before `commit`).
