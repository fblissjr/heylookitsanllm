# CLAUDE.md

<!-- Nav hub -- link out, don't duplicate. Last verified: 2026-03-13 -->

## Table of Contents

- [Get Up to Speed](#get-up-to-speed)
- [Active Work](#active-work)
- [Architecture](#architecture)
- [Change Tracking](#change-tracking)
- [Rules: Library APIs](#rules-library-apis)
- [Rules: Code Style](#rules-code-style)
- [Rules: Agent Behavior](#rules-agent-behavior)

## Get Up to Speed

FastAPI backend (MLX) + React frontend (7 applets, 874 tests).

- [README.md](./README.md) -- setup, install, commands
- [internal/](./internal/) -- architecture deep-dives (backend/, frontend/, bugs/, research/)
- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- frontend architecture
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

- [internal/backend/architecture.md](./internal/backend/architecture.md) -- system overview, provider pattern
- [internal/backend/providers/](./internal/backend/providers/) -- per-provider deep-dives (mlx.md)
- [internal/backend/api.md](./internal/backend/api.md) -- endpoint architecture, API reference
- [internal/backend/router.md](./internal/backend/router.md) -- routing and LRU cache
- [internal/backend/config.md](./internal/backend/config.md) -- configuration system

### Frontend: `apps/heylook-frontend/`

7 applets: Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook, Models.
React + Zustand + Vite. 874 tests across 38 files.

- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- component hierarchy, state, persistence
- [internal/frontend/architecture.md](./internal/frontend/architecture.md) -- migration details and patterns

## Change Tracking

- [CHANGELOG.md](./CHANGELOG.md) -- user-facing release history (semver)
- `internal/log/` -- **ALWAYS UPDATE** after every iteration (`log_YYYY-MM-DD.md`)

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

## Rules: Code Style

- When removing a provider/feature, grep the full repo then check: `config.py` (Literal + Union), `router.py`, `api.py`, `generated-api.ts`, `FEATURES.md`, `ARCHITECTURE.md`, `README.md`, `pyproject.toml` extras, frontend type unions, test fixtures
- No emojis, no hype language ("Enhanced", "Advanced", etc.)
- PEP-8 with 120-char lines, type hints throughout
- Import order: standard, third-party, local (blank line separated)
- Conventional commits: `feat(router):`, `fix(mlx):`, `chore(docs):`
- Use `uv` for all Python deps (never pip). Use `orjson` for JSON.

## Rules: Agent Behavior

- Push back when a request breaks things, a better alternative exists, or the approach is known to fail
- Check `internal/` before modifying providers -- see [internal/bugs/](./internal/bugs/) for postmortems
- Commits are fine without asking; never push unless the user explicitly says to
- Check `internal/session/CURRENT.md` before starting work

## Running Tests

- Backend: `uv run pytest tests/unit/ tests/contract/ -v`
- Frontend: `cd apps/heylook-frontend && bunx vitest run` (must run from frontend dir, not repo root)
- Frontend build: `cd apps/heylook-frontend && bun run build` (verify production build)
- Pre-existing failures: 5 router tests (YAML config vs TOML parser), 3 mlx_perf tests (removed mlx_batch_vision module) -- do not investigate
- MLX embedding/sampler tests fail in full suite (Metal context conflicts) but pass individually -- pre-existing, not a regression
- Batch labeler: `cd apps/batch-labeler && uv sync --dev && uv run pytest tests/ -v` (separate venv, must cd first)
- Optloop-lib: `cd apps/optloop-lib && uv sync && uv run pytest tests/ -v` (separate venv, 60 tests)
- `internal/` and `models.toml` are gitignored -- changes there are local-only, never committed
