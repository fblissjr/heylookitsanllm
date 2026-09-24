# Test Suite

last updated: 2026-09-23

## Overview

Backend tests for the heylookitsanllm API server. v3 frontend coverage is the browser E2E harness in `tests/e2e/` (opt-in, unsandboxed).

## Organization

```
tests/
  unit/              # Fast isolated tests, no server required
  unit/mlx_perf/     # MLX performance correctness tests
  contract/          # API contract tests with TestClient (no server required)
  smoke/             # Live smoke against a running server, per engine arm (opt-in)
  eval/              # Behavioural eval bank against a running server (opt-in)
  e2e/               # Browser E2E (opt-in, unsandboxed)
  helpers/           # Shared mocking utilities (mlx_mock.py)
  fixtures/          # Shared test data
  input/             # Test input files (images, audio)
  conftest.py        # Root fixtures (chat requests, temp config)
  README.md          # This file
```

What a file covers is in its own module docstring; this README does not keep
a per-file list (the last one drifted into listing files that no longer
existed). Before concluding the suite covers something, read the dated audits
in `docs/testing/`.

## Running Tests

```bash
# The suite (unit + contract) -- pytest.ini's testpaths, so a bare run is the
# same thing
uv run pytest -v

# Unit + contract (no server needed)
uv run pytest tests/unit/ tests/contract/ -v

# Unit only
uv run pytest tests/unit/ -v

# Contract only
uv run pytest tests/contract/ -v
```

**Invariant:** the suite is fully green. Any failure is a regression -- investigate it. There is no pre-existing-failure allowlist.

## Contract Tests (`tests/contract/`)

API contract tests using FastAPI TestClient with mocked router/service. No real models or server needed.

## Testing Guidelines

- Use pytest for new tests
- Create fixtures for shared test data in `conftest.py`
- Test error cases, not just happy path
- Mock external dependencies in unit tests (see `helpers/mlx_mock.py`)
- Descriptive names: `test_router_evicts_lru_model_when_cache_full`
- Contract tests use `TestClient` -- no server needed, fast iteration

## Related

- `tests/smoke/` -- live smoke per engine arm; never spawns a server. See `tests/smoke/README.md`.
- `tests/e2e/` -- v3 browser E2E (puppeteer-core + system Chrome; opt-in, unsandboxed)
- `tests/eval/` -- LLM behavior-eval harness (chat-template/thinking-parser/stop-token/
  vision-budget regressions; needs a running server, opt-in, not wired into `/test-suite`).
  See `tests/eval/README.md`.
- `docs/frontend_v3_spec.md` §4 -- the backend API contract
