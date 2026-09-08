# Documentation index

Last updated: 2026-09-08

Git-tracked docs for heylookitsanllm. Working notes, daily logs, strategy notes,
and research live local-only in `internal/` (gitignored) and are not part of this
tree. The project's nav hub for agents is the root [CLAUDE.md](../CLAUDE.md).

## Architecture reference
These two overlap in subject and differ in kind. Read the wiki to learn how a
subsystem works; read `architecture/` for why a specific decision was made and
what must not be broken. Neither is the live API surface -- that is the code
plus `/openapi.json`.

- [wiki/](./wiki/README.md) -- **engineering wiki**: how the system works end to
  end (architecture, backend, frontend, providers, the llama-server build/spawn
  deep dive, performance). Explanatory and self-contained; carries no throughput
  or model-quality figures (see its principle 6).
- [architecture/](./architecture/) -- backend design records + invariants (config
  history, provider mechanisms, MLX ecosystem posture, crash postmortems).
  Narrower and decision-shaped. Start at its [README](./architecture/README.md).

## Test audits
Dated evidence records, not living documents: what was measured, on what commit,
and by what method. Each is a snapshot -- re-derive before trusting a figure.
They exist because a green suite proves only what its checks can actually fail
on, and both of these found checks that could not.

- [testing/audit_2026-09-08_render_suite.md](./testing/audit_2026-09-08_render_suite.md)
  -- mutation audit of `tests/e2e/render.mjs` via `E2E_V3_ROOT`; carries the
  repeatable method
- [testing/audit_2026-09-08_backend_suite.md](./testing/audit_2026-09-08_backend_suite.md)
  -- runtime + mutation audit of `tests/unit/` + `tests/contract/`; claims are
  marked verified-here vs reported

## Project -- roadmap / status / backlog
- [project/plan_2026-07.md](./project/plan_2026-07.md) -- the phased roadmap (0-7; Phase 0 is the
  decisions block, and the 2026-07-28 re-plan re-cut Phases 1-5 into Waves 1-5)
- [project/CURRENT.md](./project/CURRENT.md) -- graded done/left status
- [project/TODO.md](./project/TODO.md) -- backlog
- [project/plan_consolidation.md](./project/plan_consolidation.md) -- getting
  back to a releasable state after a day of audit work and a parallel session:
  the tree, the smoke standard the repo already sets, the changelog, and the
  parallel-session protocol. Explicitly ends rather than becoming a programme.

## Research / design
- [frontend_v3_user_guide.md](./frontend_v3_user_guide.md) -- how the UI behaves, for the person USING it
  (the state model behind presets vs ad-hoc settings, the generation lifecycle, editing; ends with the
  known rough edges it exposed)
- [frontend_v3.md](./frontend_v3.md) -- orientation + backend coupling map
- [frontend_v3_spec.md](./frontend_v3_spec.md) -- build contract (§4 = the API contract)

## Guides
- [api_integration.md](./api_integration.md) -- wiring an EXTERNAL app to this server
  (which wire to pick, capability discovery, media block spellings, SSE, errors, and the
  deliberate differences from Anthropic's Messages spec). A scoped view of
  `frontend_v3_spec.md` §4, which stays authoritative for the contract
- [rlm_guide.md](./rlm_guide.md) / [rlm_advanced.md](./rlm_advanced.md) -- recursive inference (RLM)
- [optimization_log.md](./optimization_log.md) -- cross-session performance findings
- [optloop_guide.md](./optloop_guide.md) -- optloop-lib benchmark harness
- [mlx_optimization_plan.md](./mlx_optimization_plan.md) -- MLX engine optimization plan (historical)

## Not current

- [archive/](./archive/) -- superseded docs kept for history, marked as such in
  its own README ("should not be used for current development"). Worth knowing
  the directory exists: it holds a 620-line `CLIENT_INTEGRATION_GUIDE.md` that
  predates [api_integration.md](./api_integration.md) and still names retired
  model ids, so a search for integration docs can land on it with no in-file
  signal that it is history.

