# Documentation index

Last updated: 2026-07-09 (jspace plan added)

Git-tracked docs for heylookitsanllm. Working notes, daily logs, strategy notes,
and research live local-only in `internal/` (gitignored) and are not part of this
tree. The project's nav hub for agents is the root [CLAUDE.md](../CLAUDE.md).

## Architecture reference
[architecture/](./architecture/) -- backend design records + invariants (config
history, provider mechanisms, MLX ecosystem posture, crash postmortems).
Start at its [README](./architecture/README.md).

## Project -- roadmap / status / backlog
- [project/plan_2026-07.md](./project/plan_2026-07.md) -- the phased roadmap (0-5)
- [project/CURRENT.md](./project/CURRENT.md) -- graded done/left status
- [project/TODO.md](./project/TODO.md) -- backlog

## Research / design
- [jspace_guide.md](./jspace_guide.md) -- Jacobian-lens ("j-space") interpretability feature:
  how-it-works + end-to-end tutorial (install a lens, `/v1/jspace/analyze`, the v3 J-Space page).
- [jspace_integration_plan.md](./jspace_integration_plan.md) -- the j-space build + verifier plan
  (design rationale, phases, parity results). Lens **fitting** + the Phase-1 spike harness moved to the `jlens-mlx` sibling repo (2026-07-10).

## Frontend (v3)
- [frontend_v3_user_guide.md](./frontend_v3_user_guide.md) -- how the UI behaves, for the person USING it
  (the state model behind presets vs ad-hoc settings, the generation lifecycle, editing; ends with the
  known rough edges it exposed)
- [frontend_v3.md](./frontend_v3.md) -- orientation + backend coupling map
- [frontend_v3_spec.md](./frontend_v3_spec.md) -- build contract (§4 = the API contract)

## Guides
- [api_integration.md](./api_integration.md) -- wiring an EXTERNAL app to this server
  (which wire to pick, capability discovery, the flat image block, SSE, errors). A scoped
  view of `frontend_v3_spec.md` §4, which stays authoritative for the contract
- [rlm_guide.md](./rlm_guide.md) / [rlm_advanced.md](./rlm_advanced.md) -- recursive inference (RLM)
- [optimization_log.md](./optimization_log.md) -- cross-session performance findings
- [optloop_guide.md](./optloop_guide.md) -- optloop-lib benchmark harness
- [mlx_optimization_plan.md](./mlx_optimization_plan.md) -- MLX engine optimization plan (historical)
