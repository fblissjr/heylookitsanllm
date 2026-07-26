---
name: test-suite
description: Run the backend test suites and report only regressions
user-invocable: true
disable-model-invocation: true
---

# Test Suite Runner

Run the backend suites and report regressions. (There is no frontend unit
suite anymore: the legacy React app that carried one was deleted 2026-07-09;
v3 is no-build vanilla JS whose automated check is the OPT-IN browser E2E in
`tests/e2e/` -- `bun run e2e[:chat|:pages]`, unsandboxed, spawns a server;
never run it as part of this skill.)

## Steps

1. Run the backend suite (from repo root):

   ```bash
   uv run pytest tests/unit/ tests/contract/ -v --tb=short 2>&1
   ```

2. Parse results. Green is the invariant -- there is NO
   pre-existing-failure allowlist and no expected count (counts rot; see
   CLAUDE.md Tests). Any failure or error is a regression. Metal-gated
   skips are OK. Known non-regression: `test_mlx_provider.py` can SEGFAULT
   at GC teardown when run in near-isolation -- run it batched, not alone.

3. Report format:
   - Total passed / failed / errors
   - List ALL failures and errors -- every one is a regression
   - Re-run any failure individually for a clean traceback:
     ```bash
     uv run pytest <failing_test_file>::<test_name> -v --tb=short
     ```
