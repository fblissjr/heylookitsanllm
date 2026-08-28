# Live smoke test

last updated: 2026-08-28

The half of the v3 story the browser suite cannot see.

`tests/e2e/render.mjs` drives the real `/v3` page against a **stubbed** `/v1`.
It proves the client behaves and says nothing about whether the server does.
Everything the client's preset and lifecycle work rests on — a preset store
that refuses a duplicate name, params that round-trip onto a conversation, a
run that detaches and finishes after the reader walks away — is invisible to
it. This is the other half.

Opt-in, like `tests/eval/`. **Never spawns a server.**

```bash
# contract only -- no model loads, seconds
uv run python tests/smoke/run.py --server http://127.0.0.1:8991 --contract-only

# every engine that has a model served
uv run python tests/smoke/run.py --server http://127.0.0.1:8991

# one engine, with the model pinned
uv run python tests/smoke/run.py --server http://127.0.0.1:8991 \
    --arm gguf --model gguf=google_gemma-4-E4B-it-qat-q4_0-gguf
```

Run it against an **isolated** server (`scripts/dev_server.sh start …` gives
you one with its own `HEYLOOK_DB_PATH`). It creates and deletes presets and
conversations; it cleans up in a `finally`, but a crash mid-run would leave
`smoke-<timestamp>` rows in whatever store it was pointed at.

## Why arms are ENGINES, not providers

The provider `Literal` has three values. They are not three engines:

```
provider "mlx"   ──┬──▶ mlx-lm    (text)     two SEPARATE upstream repos,
                   └──▶ mlx-vlm   (vision)   separate release trains
provider "gguf"  ─────▶ llama-server subprocess (one engine, one local binary)
```

Which MLX library actually decodes is `MLXProvider.effective_loader`, derived
from the model's modalities — **not** the provider field. So "we covered mlx"
is a claim about a config value, not about code: a text model and a vision
model on the same provider run through different libraries, with different
release cadences and different failure modes. This harness treats them as
separate arms and reports a missing arm as **uncovered**, never as green.

`mlx_embedding` is deliberately out of scope (owner call, 2026-08-28).

## What each arm checks

Per engine: load + warm, a generation that streams and persists, **a run that
survives the reader disconnecting**, and a stop that leaves the conversation
idle. The vision arm additionally asserts the model reports the `vision`
capability.

The walk-away check is the reason this file exists. A generation outlives the
response that started it: dropping the connection ends the *subscription*
while the run detaches, finishes, and commits the whole answer. The client
discloses that as of v1.79.26 — and no stubbed suite can tell whether the
claim is true.

## Fixtures

The vision arm sends a generated 64×64 PNG built with stdlib `zlib`/`struct`.

It used to be a 1×1 PNG, and the arm failed on its first ever run: gemma's
aspect-ratio-preserving resize hands PIL a degenerate `(1,1,1)` array and PIL
refuses it, so the request died in the image processor before the model saw
anything. A smoke test should go red when the **engine** is broken, not when
a fixture is degenerate. If you shrink this fixture, that is what you are
trading away.

## What it is not

Not a behavior eval. It asserts that each engine *runs and persists*, never
that an answer is good — vision correctness, thinking splits and stop
discipline belong to `tests/eval/`. Not part of `/test-suite` either: it needs
a running server, real weights and Metal.
