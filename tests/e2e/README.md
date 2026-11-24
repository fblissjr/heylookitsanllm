# Frontend browser checks

Last updated: 2026-11-24

Checks that drive the real frontend in a real browser. There is nothing to
install: the driver is `lib/cdp.mjs`, which starts the system's Chrome headless
and speaks the Chrome DevTools Protocol with node's own `child_process`,
`fetch` and `WebSocket`.

```bash
cd tests/e2e
node run.mjs          # or: bun run e2e      the browser checks
node --test           # the pure JS unit tests (*.test.mjs), no browser
node ios-sim.mjs      # or: bun run e2e:ios  real Mobile Safari in the iOS Simulator
```

Exit code is non-zero if any check fails.

## Safety

`run.mjs` creates conversations and sends messages. It never touches real data:
it starts its OWN `heylookllm` through `lib/server.mjs`, with a temporary
database, the access gate isolated from the operator's key, and a loopback
bind on `E2E_PORT` (not the daily server's port). It refuses to start if
anything already listens there, and it stops the server and Chrome when it
finishes. The temp directory is removed on a green run and kept, with its
path printed, on a failure.

## Prerequisites

- Google Chrome at the default macOS path, or `E2E_CHROME=/path/to/chrome`.
- A working `uv run heylookllm`.
- The model in `E2E_MODEL` must be served (`GET /v1/models` lists the ids).
- An UNSANDBOXED shell. The run needs Metal, a local port and Chrome; inside
  the sandbox it dies before the first check.

## Writing a check

The driver has two primitives, and a check is written against them:

- `page.js(fn, ...args)` runs a function in the page and returns its JSON
  value, awaiting a promise. The function is serialized, so it cannot close
  over node-side variables: pass what it needs as arguments.
- `page.cdp(method, params)` is one protocol call, for anything the driver
  does not wrap.

On top of those: `page.waitFor(fn, what)`, `page.goto`, `page.reload`,
`page.press(key)` (a real, trusted key event), `page.emulate({ width, height,
touch })`, and two logs kept for the life of the page, `page.errors` (uncaught
exceptions and `console.error`) and `page.requests` (`{ method, url, done }`).

There is deliberately no selector API, no element handles and no request
interception. The old suites were written against puppeteer's page object, and
when that dependency was removed (v2.0.240) nothing could launch for three
weeks. A check here reads and drives the DOM with page-side JavaScript, which
needs no wrapper to keep working.

The rules the checks follow:

- Ask the server what landed. A check ends by reading the API, not by trusting
  what is painted.
- Wait for a condition, never for time. `page.waitFor` for the page;
  `route` in `run.mjs` waits for the router's `data-mounted` mark on `#main`,
  because a page's skeleton is in the DOM before its setup has resolved;
  `selectConv` waits for the conversation's body request to finish, because
  the row highlights a request before the rows arrive. The one exception is
  proving that nothing more happened, which has no condition.
- State your preconditions. A check that needs a touch device, a shrunken
  root font or a reply still streaming asserts that it has one, and calls
  `skip()` when a race it staged did not happen, so it cannot pass against
  nothing.
- A check sets up its own state (its conversation, its open panel). One that
  leans on what the previous check left behind fails or passes for the wrong
  reason when that one does.
- Stage a race, do not hope for one. `sendAndHold` in `run.mjs` holds a
  request inside the page until the check releases or fails it.
- One property over a table of cases beats one check per case.
- A check that reaches a legal early exit before its assertion calls `skip()`
  (`lib/harness.mjs`) and is tallied as skipped, never as a pass.

## What Chrome cannot tell you

Chrome is not WebKit. These checks hold DOM, state, request and CSS-rule
behaviour. They say nothing about Safari's scroll anchoring, the iOS software
keyboard, focus zoom or `field-sizing`. A check can assert that a CSS rule is
in force (the touch check reads computed `overflow-anchor` and font sizes);
it cannot assert what Safari does with it. That needs `ios-sim.mjs`, and for
anything involving the keyboard, a phone.

## iOS Simulator check (`ios-sim.mjs`): runs, the keyboard needs a device

`ios-sim.mjs` drives real Mobile Safari in the iOS Simulator through Apple's
own `/usr/bin/safaridriver`, with raw `fetch()` to the driver and no
dependency. It first ran on 2026-09-06. What it established is a negative: the
Simulator will not raise the software keyboard under `safaridriver`, by three
separate routes (the file's header records them). Focus lands; the keyboard
does not exist there. The keyboard checks are gated on the ENVIRONMENT rather
than on the outcome, so they skip loudly instead of passing vacuously, and
`IOS_REAL_DEVICE=1` runs them against an attached phone. What the Simulator
still gives is real WebKit at phone size.

Why Chrome cannot stand in: with the keyboard open, iOS Safari keeps the
LAYOUT viewport at full height and shrinks only the VISUAL viewport, then
scrolls the page to the focused field. Chrome shrinks the layout viewport
itself. The `100dvh` shell and the composer at its foot follow the layout
viewport, so an emulated Chrome answers a different question.

One-time setup it cannot do for you: `safaridriver --enable` on the Mac, and
in the Simulator, Settings > Safari > Advanced > Remote Automation = ON. The
session error says so if either is missing. Point it at a running server with
`IOS_SIM_BASE`; it only reads the page and types into the composer, and never
sends.

## What was removed, and where it is

v2.0.281 removed the puppeteer-shaped suites: `render.mjs`, the old `run.mjs`,
`suites/chat.mjs`, `suites/pages.mjs`, `embeddings.mjs`, `paint_cost.mjs`,
`lib/browser.mjs`, `lib/dom.mjs` and `lib/server-state.mjs`. They are in git
history at commit `310c7ac4` (`git show 310c7ac4:tests/e2e/render.mjs`).

Ported into `run.mjs`: the streaming-markdown property check, the raw-HTML and
URL-scheme checks on `renderMarkdown`, and the checks written for v2.0.279.

Not ported, and worth porting first when a change touches their area:

- The server-free chat checks in `render.mjs`, which ran the real page against
  a stub store and an SSE drip (`serveV3`, `makeStubStore`, `streamDrip`). The
  stub store is a plain object and the drip is a plain node server; only their
  wiring used puppeteer. Most of that file's checks were about the message
  list being reconciled rather than rebuilt, scroll-follow across a mid-stream
  viewport change, and what an open editor survives.
- `paint_cost.mjs`, the instrument that read Chrome's layout and script
  counters across one streamed reply (`docs/project/plan_dark_mode.md`,
  Phase 5). It depended on the same stub store and drip.
- The model-driven suites: streaming cadence, edit, regenerate and delete
  truncation, stop keeping the partial, presets, the Models and Perf pages,
  the notebook, and the multi-engine arms.

## Config (env vars)

| var | default | meaning |
|---|---|---|
| `E2E_MODEL` | `Qwen3.5-0.8B-MLX-8bit` | model to load and drive (must be served). It needs vision for the image check, and must decode fast. |
| `E2E_PORT` | `1264` | port for the server this run starts. Deliberately not the daily server's port. |
| `E2E_CHROME` | `/Applications/Google Chrome.app/…` | Chrome binary path |
| `E2E_HEADFUL` | (unset) | set to any value to watch the browser |
| `IOS_SIM_BASE` | `http://127.0.0.1:8000` | `ios-sim.mjs` only: the running server to load the frontend from |
| `IOS_SIM_DEVICE` | `iPhone 15 Pro` | `ios-sim.mjs` only: simulator device name; `IOS_SIM_UDID` bypasses the lookup |
| `IOS_SIM_PORT` | `4445` | `ios-sim.mjs` only: safaridriver port |
| `IOS_SIM_SHOTS` | `<os tmpdir>/heylook-ios-sim` | `ios-sim.mjs` only: where screenshots go |
| `IOS_SIM_KEEP` | (unset) | `ios-sim.mjs` only: leave the simulator booted afterwards |

## Layout

- `run.mjs`: the browser checks. Starts the isolated server, starts Chrome,
  runs the checks in order, tears both down.
- `lib/cdp.mjs`: the driver. Chrome launch on a free port, the `Page` object
  and its two primitives.
- `lib/server.mjs`: start, readiness and stop for an isolated `heylookllm`.
  Readiness is the model listed, then the server-owned
  `POST /v1/models/{id}/load?warm=true`, the same contract as
  `scripts/dev_server.sh`.
- `lib/harness.mjs`: `Suite`, `assert`, `waitFor`, `skip` and the summary
  printer.
- `ios-sim.mjs`: the Simulator check above.
- `embed-kit.test.mjs`, `preset-io.test.mjs`: pure JS unit tests under
  `node --test`.
