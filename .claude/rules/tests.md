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

- No dependencies: `lib/cdp.mjs` starts system Chrome headless and speaks the DevTools protocol with node's own `fetch` and `WebSocket` (v2.0.281; puppeteer-core had been removed in v2.0.240, which left every browser suite unable to launch). A check is written against two primitives, `page.js(fn, ...args)` and `page.cdp(method, params)`. Do not grow a selector API, element handles or request interception onto the driver: that is a second puppeteer to maintain. A response is held or failed from inside the page (`sendAndHold` in `run.mjs`).
- `node run.mjs` starts its own server through `lib/server.mjs` with an isolated `HEYLOOK_DB_PATH` and the access gate isolated. It refuses to start if anything listens on `E2E_PORT`; keep that guard, and probe by connecting, never by binding. Load and warm readiness is the server-owned `POST /v1/models/{id}/load?warm=true`; never hand-roll poll/warm logic in a harness.
- It must run unsandboxed and is not part of the backend run. It needs a fast `E2E_MODEL` with vision (default `Qwen3.5-0.8B-MLX-8bit`).
- Wait for a condition, never for time: `selectConv` waits for the conversation's body request to finish, because the row highlights a request before the rows arrive. Chrome is not WebKit: a check can assert that a CSS rule is in force, never what Safari does with it.
- The puppeteer-shaped suites (`render.mjs`, `suites/`, `paint_cost.mjs`) are gone; `tests/e2e/README.md` says what they covered and what was ported. There is currently no server-free chat suite, no streaming-cadence guard and no paint-cost instrument.
- A check that reaches a legal early exit before its assertion calls `skip()` (harness.mjs) and is tallied as skipped, never as a pass.
- `bun run e2e:ios` (`ios-sim.mjs`) drives real Mobile Safari in the iOS Simulator via `safaridriver` against an already-running server; Chrome emulation cannot see iOS keyboard behaviour. Its run status lives in `docs/project/TODO.md` and the file's header.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "Tests".
