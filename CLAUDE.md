# CLAUDE.md

<!-- Nav hub -- link out, don't duplicate. Last verified: 2026-02-23 -->

## Table of Contents

- [Get Up to Speed](#get-up-to-speed)
- [Active Work](#active-work)
- [Architecture](#architecture)
- [Change Tracking](#change-tracking)
- [Rules: Library APIs](#rules-library-apis)
- [Rules: Code Style](#rules-code-style)
- [Rules: Agent Behavior](#rules-agent-behavior)

## Get Up to Speed

FastAPI backend (MLX, llama.cpp) + React frontend (6 applets, 711 tests).

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

Providers: MLXProvider (text+vision), LlamaCppProvider (GGUF, legacy), MLXSTTProvider (parakeet-mlx STT).
LRU cache hot-swaps up to 2 models. Config in `models.toml`.

- [internal/backend/architecture.md](./internal/backend/architecture.md) -- system overview, provider pattern
- [internal/backend/providers/](./internal/backend/providers/) -- per-provider deep-dives (mlx.md, llama_cpp.md)
- [internal/backend/api.md](./internal/backend/api.md) -- endpoint architecture
- [internal/backend/router.md](./internal/backend/router.md) -- routing and LRU cache
- [internal/backend/config.md](./internal/backend/config.md) -- configuration system

### Frontend: `apps/heylook-frontend/`

6 applets: Chat, Batch, Token Explorer, Model Comparison, Performance, Notebook.
React + Zustand + Vite. 711 tests across 31 files.

- [apps/heylook-frontend/ARCHITECTURE.md](./apps/heylook-frontend/ARCHITECTURE.md) -- component hierarchy, state, persistence
- [internal/frontend/architecture.md](./internal/frontend/architecture.md) -- migration details and patterns

### API

OpenAI-compatible + Anthropic Messages-inspired endpoints. Live Swagger at `/docs`.

- [internal/backend/api.md](./internal/backend/api.md) -- endpoint design
- [docs/FRONTEND_HANDOFF.md](./docs/FRONTEND_HANDOFF.md) -- complete API reference

## Change Tracking

- [CHANGELOG.md](./CHANGELOG.md) -- public, user-facing release history (semver)
- `internal/log/` -- detailed daily development logs (naming: `log_YYYY-MM-DD.md`)

CHANGELOG.md is the summary. `internal/log/` is the raw record. When completing work, update both: add a CHANGELOG entry for anything user-visible, and log implementation details in the daily log.

## Rules: Library APIs

- All generation (text + vision) routes through `generation_core.run_generation()` calling `mlx_lm.generate.stream_generate`
- Vision requests use a pre-filled cache pattern: VLM forward fills KV cache, then `run_generation()` continues
- Use `mlx_vlm.prompt_utils.apply_chat_template` for VLM prompt formatting
- Use `mlx_vlm.utils.prepare_inputs` for VLM input tokenization (handles image grid dimensions per model)
- Verify a library is actually broken before implementing a workaround
- See [internal/bugs/vlm_vision_bug.md](./internal/bugs/vlm_vision_bug.md)

## Rules: Code Style

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

## Finishing a Plan, Task, Bugfix, or Feature
- **Ask yourself what you would do differently:** Now that you've finished and done the analysis, how would you do it the right way? What would you do differently if you could start over with more time now that you have new insights and could reflect more? What, if anything, do you think is a poor short term fix? If you wouldn't change anything, that's totally cool as well.
