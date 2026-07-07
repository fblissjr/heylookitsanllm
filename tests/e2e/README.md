# v3 frontend E2E harness

Last updated: 2026-07-07

Browser end-to-end tests for the `/v3` frontend. Drives a **real** running server
with a real model through **system Chrome** (puppeteer-core). `claude-in-chrome`
refuses `localhost`, so puppeteer is the only path to exercise `/v3` against the
backend.

## Safety

The suites CREATE and CLEAR conversations and notebooks. By default `run.mjs`
spawns its OWN `heylookllm` with an **isolated** `HEYLOOK_DB_PATH` (a temp file),
so real data is never touched. The temp DB and server log are deleted on exit.

Driving an already-running server is possible but shares that server's DB and is
therefore refused unless you opt in — see `E2E_BASE_URL` below.

## Prerequisites

- `bun install` (from this directory) — installs `puppeteer-core`.
- Google Chrome at the default macOS path, or `E2E_CHROME=/path/to/chrome`.
- A working `uv run heylookllm` (Metal-gated: run on the Mac, not in a sandbox).
- The model in `E2E_MODEL` must exist in `models.toml`.

## Run

```bash
cd tests/e2e
bun install          # first time only
bun run e2e          # both suites (chat + pages)
bun run e2e:chat     # chat suite only
bun run e2e:pages    # pages suite only
```

Exit code is non-zero if any check fails.

Known false positive: after an mlx version bump, the FIRST run pays Metal
shader JIT compilation and the streaming-cadence guard can read low (seen:
25.3/s vs the 30/s floor on the 0.32.0 upgrade day; warm re-run passed at
full speed). If ONLY the cadence guard fails right after an mlx/model
change, re-run before diagnosing a delivery regression.

## Config (env vars)

| var                   | default                              | meaning |
|-----------------------|--------------------------------------|---------|
| `E2E_MODEL`           | `gemma-4-26B-A4B-it-heretic-4bit`    | model to preload + drive. Use a fast one — the MoE decodes ~90 tok/s; the 31B dense gemma decodes ~10 tok/s and makes streaming look broken. |
| `E2E_PORT`            | `8080`                               | server port for the spawned instance |
| `E2E_MAX_TOKENS`      | `24`                                 | per-generation cap, seeded via `localStorage` so runs stay fast/deterministic |
| `E2E_CHROME`          | `/Applications/Google Chrome.app/…`  | Chrome binary path |
| `E2E_HEADFUL`         | (unset)                              | set to any value to watch the browser (debugging) |
| `E2E_BASE_URL`        | (unset)                              | drive an ALREADY-RUNNING server instead of spawning one. **Shares its DB.** Requires `E2E_ALLOW_SHARED_DB=1`. |

## Layout

- `run.mjs` — orchestrator: spawn server (isolated DB) → launch Chrome → run
  suites → tear down → exit non-zero on failure.
- `lib/server.mjs` — spawn/readiness/stop for `heylookllm`. Readiness = model
  listed AND a warm generation completes.
- `lib/browser.mjs` — Chrome launch + per-suite page context (localStorage
  settings seed, hash-router navigation, page-error capture).
- `lib/harness.mjs` — `Suite`/`check`/`assert`/`waitFor`, summary printer.
- `lib/dom.mjs` — shared DOM helpers (`clickByText`, `armedClick` two-tap
  confirm, overflow check).
- `suites/chat.mjs` — ~25 checks: streaming, a client-side streaming-cadence
  regression guard (see below), edit/regenerate/delete truncation,
  stop=partial-saved, post-abort health, settings + seed, conversation CRUD,
  390px mobile.
- `suites/pages.mjs` — ~27 checks: notebook autosave + generate-at-cursor tail
  preservation, explore logprob chips + keyboard nav, perf no-polling proof +
  ranges, models list/load/unload + HF scan + danger-zone clear.

## Notes / gotchas

- Generation length is capped by seeding `localStorage['heylook-v3-settings']`
  with `max_tokens` BEFORE the app boots (settings.js reads localStorage once at
  module import, so changes need a fresh load — the `ctx.open()` helper reloads).
- The stop-mid-stream checks reopen with `max_tokens: 400` so there is time to
  click Stop before the generation finishes.
- The danger-zone clear check runs LAST in the pages suite; it wipes the
  (isolated) DB.
- The **streaming-cadence guard** (`streaming delivery is not poll-quantized`)
  measures client-observed inter-chunk gaps via an in-page fetch — the Phase 1
  delivery fix is invisible to server-side telemetry, so only a client timing
  the stream can catch a regression back to the ~100ms poll ceiling. It asserts
  median gap < 50ms and > 30 tok/s, which needs a FAST model. The default MoE
  passes with margin (~11ms, ~88/s); a natively-slow dense model (e.g. the 31B
  gemma at ~10 tok/s) would false-fail — override `E2E_MODEL` only with a fast
  model, or expect this one check to flag.
