# CLAUDE.md

<!-- Nav hub -- link out, don't duplicate. Last verified: 2026-04-09 -->

## Table of Contents

- [Get Up to Speed](#get-up-to-speed)
- [Active Work](#active-work)
- [Architecture](#architecture)
- [Change Tracking](#change-tracking)
- [Rules: Library APIs](#rules-library-apis)
- [Rules: Code Style](#rules-code-style)
- [Rules: Agent Behavior](#rules-agent-behavior)

## Get Up to Speed

FastAPI backend (MLX) + two frontends: React (legacy, 7 applets) and vanilla JS v2 (in progress).

- [README.md](./README.md) -- setup, install, commands
- [internal/](./internal/) -- architecture deep-dives (backend/, frontend/, bugs/, research/, reports/, session/, log/, scratch/)
- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- legacy React frontend architecture
- [apps/heylook-frontend-v2/](./apps/heylook-frontend-v2/) -- new vanilla JS frontend (no framework, Pretext for text layout)
- [docs/frontend_api_reference.md](./docs/frontend_api_reference.md) -- API reference for frontend (formerly FRONTEND_HANDOFF.md)
- [tests/README.md](./tests/README.md) -- testing guide and coverage matrix

## Active Work

Check before making changes:
- [internal/session/CURRENT.md](./internal/session/CURRENT.md) -- work in progress, quick resume
- [internal/session/TODO.md](./internal/session/TODO.md) -- cross-session task backlog

## Architecture

### Backend: `src/heylook_llm/`

Providers: MLXProvider (text+vision), MLXEmbeddingProvider (dynamic backbone via mlx-lm).
LRU cache hot-swaps up to 2 models with model pinning support for long-running batch jobs. Config in `models.toml`.
Provider type: `Literal["mlx", "mlx_embedding"]`.
`coderef/` contains reference forks of mlx-lm and mlx-vlm for comparing upstream patterns (gitignored).
Conversation storage: SQLite via aiosqlite (`db.py`), CRUD endpoints in `conversation_api.py`. DB auto-creates at `data/conversations.db` (override via `HEYLOOK_DB_PATH` env var). Linear message model with `UNIQUE(conversation_id, position)`.
Single shared aiosqlite connection with `timeout=10` -- serializes writes, fine for personal use. Dynamic SQL field names must go through `_UPDATABLE_MESSAGE_FIELDS` allowlist in `db.py`.
`POST /v1/data/clear` deletes all conversations, messages, and notebooks. Frontend exposes this on the Models page.
RLM endpoint (`rlm.py`): recursive inference scaffold with sandboxed Python REPL, uses providers directly (no HTTP round-trip). Supports compaction (history summarization), recursive depth (`rlm_query()` child RLMs), and `max_errors` threshold.

- [internal/backend/architecture.md](./internal/backend/architecture.md) -- system overview, provider pattern
- [internal/backend/providers/](./internal/backend/providers/) -- per-provider deep-dives (mlx.md)
- [internal/backend/api.md](./internal/backend/api.md) -- endpoint architecture, API reference
- [internal/backend/router.md](./internal/backend/router.md) -- routing and LRU cache
- [internal/backend/config.md](./internal/backend/config.md) -- configuration system
- [docs/rlm_guide.md](./docs/rlm_guide.md) -- RLM endpoint usage, request fields, examples
- [docs/observability_guide.md](./docs/observability_guide.md) -- three JSONL log streams, env vars, content invariant, monitoring + optimization recipes
- [docs/mlx_optimization_plan.md](./docs/mlx_optimization_plan.md) -- historical six-phase optimization plan (v1.13.0 through v1.17.0); design rationale for radix caching, speculative decoding, strategy unification

### Frontend v2: `apps/heylook-frontend-v2/`

Vanilla JS, no framework, no bundler, no node_modules.
Conversations stored server-side in SQLite (`/v1/conversations` API).
Served at `/v2` by the FastAPI backend. Hash-based routing. `marked` + `DOMPurify` vendored locally.
Pages: Chat, Batch, Models, Performance, Notebook, Token Explorer (all working).

- [apps/heylook-frontend-v2/CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md) -- file structure, page pattern, rules, gotchas

### Frontend (legacy): `apps/heylook-frontend/`

7 applets: Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook, Models.
React + Zustand + Vite. ~39 test files. Being replaced by v2.

- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- component hierarchy, state, persistence
- [internal/frontend/architecture.md](./internal/frontend/architecture.md) -- migration details and patterns

### Optimization Loops: `apps/optloop/`, `apps/optloop-lib/`

Autonomous inference tuning with dual text+VLM benchmarks, composite scoring, and config-driven thresholds.
`optloop` targets application code (`src/`); `optloop-lib` targets library forks (`repos/`).
Config: `bench_config.toml` in each directory.

- [docs/optloop_guide.md](./docs/optloop_guide.md) -- optloop user walkthrough, scoring, configuration
- [docs/optloop_advanced.md](./docs/optloop_advanced.md) -- bench activation gap, monkey patching, failure modes, FAQ
- [docs/optimization_log.md](./docs/optimization_log.md) -- cross-session knowledge base (baselines, findings)

### Observability: `src/heylook_llm/memory.py`

`MemoryManager` owns three disk-backed JSONL streams under `internal/log/` plus
one startup record. Content invariant: numeric + metadata only, never prompts
or responses. Env vars: `HEYLOOK_BASELINE_LOG_INTERVAL_SECONDS` (default 3600,
`0` disables), `HEYLOOK_REQUEST_LOG_ENABLED`, `HEYLOOK_MODEL_EVENT_LOG_ENABLED`.

- [docs/observability_guide.md](./docs/observability_guide.md) -- full rundown, env vars, content invariant, monitoring + optimization recipes (jq one-liners)

## Change Tracking

- [CHANGELOG.md](./CHANGELOG.md) -- user-facing release history (semver)
- `internal/log/` -- **ALWAYS UPDATE** before ending a session: write `log_YYYY-MM-DD.md` with what changed, why, bugs found, and what to look into next

## Rules: Library APIs

- All generation (text + vision) routes through `generation_core.run_generation()` calling `mlx_lm.generate.stream_generate`
- Vision requests use a pre-filled cache pattern: VLM forward fills KV cache, then `run_generation()` continues
- Embedding backbone loading uses `mlx_lm.utils._get_classes(config_dict)` (private API, takes a dict not keyword). Returns (CausalLM wrapper, ArgsClass) -- extract `.model` for the transformer body.
- Gemma models need `sqrt(hidden_size)` embedding scaling; other architectures do not. `EmbeddingModel` gates on `model_type.startswith("gemma")`.
- Use `mlx_vlm.prompt_utils.apply_chat_template` for VLM prompt formatting
- Use `mlx_vlm.utils.prepare_inputs` for VLM input tokenization (handles image grid dimensions per model)
- Verify a library is actually broken before implementing a workaround
- VLM models cache `_position_ids` and `_rope_deltas` on the LanguageModel instance -- must reset before each new request when using radix cache
- Radix cache snapshots contain full end-of-generation KV state; `restore_kv_from_snapshot(trim_to=matched_len)` trims KVCache layers to the prefix boundary
- Hybrid models (KVCache + ArraysCache like Qwen3.5) have limited radix cache correctness: ArraysCache can't be trimmed to a prefix. See [internal/bugs/radix_cache_vlm_crash.md](./internal/bugs/radix_cache_vlm_crash.md)
- See [internal/bugs/vlm_vision_bug.md](./internal/bugs/vlm_vision_bug.md)
- VLM vision feature caching: models with `encode_image()` support `cached_image_features` kwarg in their forward pass, bypassing the vision tower. `VisionFeatureCache` in `providers/common/vision_feature_cache.py` manages LRU cache keyed by image URL (with pixel-hash fallback for base64 sources).
- `prepare_vlm_inputs_parallel()` returns 4-tuple: `(images, formatted_prompt, has_images, image_urls)` -- tests must destructure all 4
- Server sets `mx.set_wired_limit(max_recommended_working_set_size)` at startup; per-generation `wired_limit()` CM is still needed for stream synchronization
- Radix cache tracks `snapshot_bytes` per node and `_total_bytes` per tree for byte-level budget enforcement (`--prompt-cache-bytes` CLI flag)
- Radix nodes have `segment_type` ("system"/"assistant") for priority-based eviction; system prompt KV evicted last
- Radix cache node replacement must subtract the replaced subtree's bytes and node count (`_subtree_bytes`, `_subtree_count`) -- otherwise byte budget drifts
- `mlx_lm.generate.GenerationResponse` is a non-slotted `@dataclass` -- attach per-request metadata via `response.X = value` (+ `# type: ignore[attr-defined]`), read on the API side via `getattr(chunk, 'X', default)`. Pattern: `cached_tokens`, `kv_cache_bytes`.
- `mx.get_peak_memory()` is monotonic process-wide; call `mx.reset_peak_memory()` at the start of `run_generation` to scope it per request.

## Rules: Code Style

- Pre-commit hook rejects files containing literal `/Users/<username>/` paths -- use generic placeholders in docs/examples
- Adding a new endpoint: create module with `APIRouter(tags=["Name"])`, add tag to `openapi_tags` list + `app.include_router()` in `api.py`
- When removing a provider/feature, grep the full repo then check: `config.py` (Literal + Union), `router.py`, `api.py`, `generated-api.ts`, `FEATURES.md`, `ARCHITECTURE.md`, `README.md`, `pyproject.toml` extras, frontend type unions, test fixtures
- No emojis, no hype language ("Enhanced", "Advanced", etc.)
- PEP-8 with 120-char lines, type hints throughout
- Import order: standard, third-party, local (blank line separated)
- Conventional commits: `feat(router):`, `fix(mlx):`, `chore(docs):`
- Use `uv` for all Python deps (never pip). Use `orjson` for JSON.
- Security hook false-positives on `mx.eval` (MLX graph materializer, not Python's). Use `mx.async_eval` where possible, or acknowledge the warning.
- Frontend v2: see [apps/heylook-frontend-v2/CLAUDE.md](./apps/heylook-frontend-v2/CLAUDE.md) for page patterns, sanitization rules, and gotchas
- Frontend v2: Pydantic update endpoints must use `model_fields_set` to distinguish "not sent" from "explicitly null" (see `conversation_api.py` pattern)
- Frontend v2: SPA sub-path serving needs `<base href="/v2/">` in HTML so relative paths resolve correctly. Static file handler must `resolve()` + `is_relative_to()` before serving.
- Returning a Pydantic model with custom headers: `Response(content=model.model_dump_json(), media_type="application/json", headers=...)`. `JSONResponse(content=model.model_dump())` double-serializes the whole tree.
- SSE response headers ship before the generator runs; put post-generation telemetry (peak memory, cache bytes) in the usage chunk's `timing` object -- client must pass `stream_options.include_usage=true` to receive it.
- Observability log streams (`internal/log/*.jsonl`) never record prompts, responses, or token IDs. Counts, sampler knobs, timings, and metadata only. `sampler_summary_from_request` in `memory.py` is the canonical extractor for sampler knobs -- anywhere you need a "what was this request configured with" dict, call it.
- MemoryManager calls from request/router paths go through `memory.safe_mm_call(mm, "method_name", ...)`, which is a no-op when `mm is None` and swallows exceptions to `logging.debug` (observability failures must never break inference).
- Never commit runtime data or logs. `*.db`, `*.jsonl`, `/data/*`, and `apps/*/data/*` are gitignored; `data/.gitkeep` and `internal/log/` structure stay. Package data at `src/heylook_llm/data/` (profiles, service templates) is excluded from the ignore on purpose.

## Rules: Agent Behavior

- Push back when a request breaks things, a better alternative exists, or the approach is known to fail
- Check `internal/` before modifying providers -- see [internal/bugs/](./internal/bugs/) for postmortems
- Commits are fine without asking; never push unless the user explicitly says to
- Check `internal/session/CURRENT.md` before starting work

## Running Tests

- Prefer `/test-suite` (skill at `.claude/skills/test-suite/`) -- runs backend + frontend in parallel and filters pre-existing failures. Saves constructing `--ignore` flags by hand.
- After schema changes (new response headers, Pydantic fields, endpoints), run `/openapi-regen` (skill at `.claude/skills/openapi-regen/`) to refresh `apps/heylook-frontend/src/types/generated-api.ts` from the live server's OpenAPI schema.
- `sandbox.excludedCommands` in `settings.local.json` exempts `uv run pytest:*`, `uv sync*`, `uv lock*`, `bun install*`, `bun run build*` -- those run outside the sandbox to avoid uv cache friction. Other `uv run` calls still respect the sandbox and may fall back to `dangerouslyDisableSandbox: true` if they hit cache-access errors.
- GPG signing (`commit.gpgsign`) requires 1Password agent running -- if `git commit` fails with socket errors, use `git -c commit.gpgsign=false commit` (the `-c` must come before `commit`)
- `--timeout` flag is not installed; pytest runs without it
- Backend: `uv run pytest tests/unit/ tests/contract/ -v`
- Conversation API: `uv run pytest tests/unit/test_conversation_api.py -v` (22 tests, in-memory SQLite)
- Notebook API: `uv run pytest tests/unit/test_notebook_api.py -v` (11 tests, in-memory SQLite)
- Frontend v2: no build step, no tests yet. Manual test at `http://localhost:8080/v2` when server is running (port from server.py, default 8080)
- Frontend (legacy): `cd apps/heylook-frontend && bunx vitest run` (must run from frontend dir, not repo root)
- Frontend (legacy) build: `cd apps/heylook-frontend && bun run build` (verify production build)
- Pre-existing failures: 5 router tests (YAML config vs TOML parser), 3 mlx_perf tests (removed mlx_batch_vision module) -- do not investigate
- MLX embedding/sampler tests fail in full suite (Metal context conflicts) but pass individually -- pre-existing, not a regression
- `tests/unit/test_conversation_api.py` + `test_notebook_api.py` use `pytest_asyncio` (dev dep; added in v1.21). `uv sync` on a fresh checkout installs it automatically.
- Batch labeler: `cd apps/batch-labeler && uv sync --dev && uv run pytest tests/ -v` (separate venv, must cd first)
- Optloop-lib: `cd apps/optloop-lib && uv sync && uv run pytest tests/ -v` (separate venv, 60 tests)
- `internal/` and `models.toml` are gitignored -- changes there are local-only, never committed
