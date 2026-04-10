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
- [internal/](./internal/) -- architecture deep-dives (backend/, frontend/, bugs/, research/)
- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- legacy React frontend architecture
- [apps/heylook-frontend-v2/](./apps/heylook-frontend-v2/) -- new vanilla JS frontend (no framework, Pretext for text layout)
- [docs/FRONTEND_HANDOFF.md](./docs/FRONTEND_HANDOFF.md) -- API reference for frontend
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
RLM endpoint (`rlm.py`): recursive inference scaffold with sandboxed Python REPL, uses providers directly (no HTTP round-trip). Supports compaction (history summarization), recursive depth (`rlm_query()` child RLMs), and `max_errors` threshold.

- [internal/backend/architecture.md](./internal/backend/architecture.md) -- system overview, provider pattern
- [internal/backend/providers/](./internal/backend/providers/) -- per-provider deep-dives (mlx.md)
- [internal/backend/api.md](./internal/backend/api.md) -- endpoint architecture, API reference
- [internal/backend/router.md](./internal/backend/router.md) -- routing and LRU cache
- [internal/backend/config.md](./internal/backend/config.md) -- configuration system
- [docs/rlm_guide.md](./docs/rlm_guide.md) -- RLM endpoint usage, request fields, examples

### Frontend v2: `apps/heylook-frontend-v2/`

Vanilla JS, no framework, no bundler, no node_modules. ~2,400 lines JS + ~800 lines CSS.
Conversations stored server-side in SQLite (`/v1/conversations` API).
Served at `/v2` by the FastAPI backend. Hash-based routing. `marked` + `DOMPurify` loaded from CDN.
Pages: Chat, Batch, Models, Performance, Notebook (all working). Token Explorer (planned).

Key patterns:
- Each page exports `mount(el)` returning `{ teardown() }`. Module state reset via `freshState()` on mount.
- `teardown()` must null `state`, stop streams/intervals, and clean up handlers on persistent DOM (sidebar, nav).
- Streaming display uses RAF throttling (one render per frame, not per token).
- Static file serving at `/v2` uses `resolve()` + `is_relative_to()` for path traversal prevention.
- All `innerHTML` writes go through DOMPurify via `renderMarkdown()`. No raw user content in innerHTML.
- Shared helpers (`createEl`, `statCard`) live in `js/utils.js`. New pages import from there.
- Route handler DB access uses `get_db()` from `db.py` (shared, not per-module copies).

### Frontend (legacy): `apps/heylook-frontend/`

7 applets: Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook, Models.
React + Zustand + Vite. 874 tests across 38 files. Being replaced by v2.

- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- component hierarchy, state, persistence
- [internal/frontend/architecture.md](./internal/frontend/architecture.md) -- migration details and patterns

### Optimization Loops: `apps/optloop/`, `apps/optloop-lib/`

Autonomous inference tuning with dual text+VLM benchmarks, composite scoring, and config-driven thresholds.
`optloop` targets application code (`src/`); `optloop-lib` targets library forks (`repos/`).
Config: `bench_config.toml` in each directory.

- [docs/optloop_guide.md](./docs/optloop_guide.md) -- optloop user walkthrough, scoring, configuration
- [docs/optloop_advanced.md](./docs/optloop_advanced.md) -- bench activation gap, monkey patching, failure modes, FAQ
- [docs/optimization_log.md](./docs/optimization_log.md) -- cross-session knowledge base (baselines, findings)

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
- Frontend v2: `marked.use()` not `marked.setOptions()` (removed in marked v5+). `renderMarkdown()` already sanitizes via DOMPurify -- never double-wrap with `sanitize()`.
- Frontend v2: Pydantic update endpoints must use `model_fields_set` to distinguish "not sent" from "explicitly null" (see `conversation_api.py` pattern)
- Frontend v2: SPA sub-path serving needs `<base href="/v2/">` in HTML so relative paths resolve correctly. Static file handler must `resolve()` + `is_relative_to()` before serving.
- Frontend v2: Polling pages use recursive `setTimeout` (not `setInterval`) to prevent overlapping requests when backend is slow

## Rules: Agent Behavior

- Push back when a request breaks things, a better alternative exists, or the approach is known to fail
- Check `internal/` before modifying providers -- see [internal/bugs/](./internal/bugs/) for postmortems
- Commits are fine without asking; never push unless the user explicitly says to
- Check `internal/session/CURRENT.md` before starting work

## Running Tests

- Sandbox mode blocks uv cache access -- `additionalWritePaths` in `settings.local.json` covers `~/.cache/uv` and `.venv/`; if uv cache errors still occur, run with sandbox disabled
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
- Batch labeler: `cd apps/batch-labeler && uv sync --dev && uv run pytest tests/ -v` (separate venv, must cd first)
- Optloop-lib: `cd apps/optloop-lib && uv sync && uv run pytest tests/ -v` (separate venv, 60 tests)
- `internal/` and `models.toml` are gitignored -- changes there are local-only, never committed
