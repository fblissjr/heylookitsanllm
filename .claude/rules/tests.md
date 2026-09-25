---
paths:
  - "tests/**"
---

# Test harnesses

The test discipline and which check to run for which change are in AGENTS.md's Tests section. This file holds the harness mechanics.

## Backend suite

- Invocation order is not load-bearing on Apple hardware. `--timeout` is not installed.
- Never apply an MLX `sys.modules` mock at module level with `.start()`; use `with patch.dict(...)` or the `mock_mlx` fixture.
- The `helpers.mlx_mock` tree must cover every dotted module path a heylook module imports at module level, and only those. Mocking a path that product code probes optionally makes the absent-dependency branch untestable.
- `test_mlx_provider.py` segfaults at GC teardown when run in near-isolation; run it batched, not alone. That is not a regression.
- A `Fatal Python error: gilstate_tss_set` (exit 134, printed after the pass count) is a separate teardown crash, not the MLX-GC class: it needs the MagicMock MLX tree and remains a residual on the mocked path only.
- Real-MLX failures poison later real-MLX tests in the same process: chase the first such failure, not the count.
- Provider unit tests build `MLXProvider` from raw config dicts, so provider and loader code must tolerate un-normalized config (e.g. missing `modalities`). A back-compat branch that looks dead in the router path may be live in tests.
- A `git archive` export is not isolation. The package is installed editable, so `import heylook_llm` resolves to this repo's `src/` from any cwd. Put `PYTHONPATH=<export>/src` ahead of the venv python and print `heylook_llm.__file__` before trusting the run. Make the probe's setup fail loudly.

## Live harnesses

- Eval bank (`tests/eval/`, opt-in): thinking split/leak, stop discipline, vision and audio correctness, against a running server (it never spawns one). It runs `stream=False`, so chunk-boundary behaviour belongs to `TestParserInvariants`. It reports how many tasks ran on no model; check it.
- Live smoke (`tests/smoke/`, opt-in, never spawns a server). Arms are not providers (mlx-text, mlx-vision, gguf: one mlx-vlm engine, two MLX paths); a missing arm reports as uncovered, never green. `--contract-only` loads nothing. Point it at an isolated server (`scripts/dev_server.sh`); it writes presets and conversations. The taxonomy is `tests/helpers/engines.py`, shared with `tests/eval/run.py` and the e2e harness; it reads `engine.runtime` off the admin row and splits MLX on the served `vision` capability (the same resolver as `is_vlm`).
- `scripts/dev_server.sh`'s RAM pre-flight sizes via `scripts/ram_report.py` using the router's own `discover()`/`merge_discovered()`. An unsizeable model exits 2 with a reason; a zero-size report is never a pass.

## Browser E2E (`tests/e2e/`)

- puppeteer-core with system Chrome (claude-in-chrome refuses localhost). It spawns its own server with an isolated `HEYLOOK_DB_PATH`, and each suite clears its temp DB. It refuses to start if anything listens on `E2E_PORT`; keep that guard, and probe by connecting, never by binding. Load and warm readiness is the server-owned `POST /v1/models/{id}/load?warm=true`; never hand-roll poll/warm logic in a harness.
- It must run unsandboxed and is not part of the backend run. Its client-side streaming-cadence guard is the only automated check for the streaming delivery fix and needs a fast `E2E_MODEL` (default `Qwen3.5-0.8B-MLX-8bit`, which also has the vision and thinking the capability checks need).
- `bun run e2e:render` is model-free and server-free and is not part of `bun run e2e`: it guards that the chat message list is reconciled, not rebuilt. `E2E_V3_ROOT` points it at a copy of the frontend.
- A check that reaches a legal early exit before its assertion calls `skip()` (harness.mjs) and is tallied as skipped, never as a pass.
- `bun run e2e:ios` (`ios-sim.mjs`) drives real Mobile Safari in the iOS Simulator via `safaridriver` against an already-running server; Chrome emulation cannot see iOS keyboard behaviour. Its run status lives in `docs/project/TODO.md` and the file's header.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "Tests".
