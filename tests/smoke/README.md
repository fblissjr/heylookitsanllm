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

Which MLX library actually decodes is `effective_loader`, derived from the
model's modalities and its `loader` hint — **not** the provider field. So "we
covered mlx" is a claim about a config value, not about code: a text model and
a vision model on the same provider run through different libraries, with
different release cadences and different failure modes. This harness treats
them as separate arms and reports a missing arm as **uncovered**, never as
green.

`mlx_embedding` is deliberately out of scope (owner call, 2026-08-28); it
classifies as *excluded*, which prints differently from *unclassified* on
purpose — "we chose not to" and "we could not tell" are different answers.

The classification itself is `tests/helpers/engines.py`, shared with
`tests/eval/run.py` (two copies of a taxonomy is one drifting copy). It reads
`effective_loader` off `GET /v1/admin/models`, which the server answers for
UNLOADED models too (v1.79.31) — a field sourced from a live provider would be
null for exactly the models an arm has yet to choose from. Against a server too
old to serve it, the engine is inferred from the vision capability and the model
is reported as *engine identity NOT confirmed* rather than claimed.

## What each arm checks

Per engine: load + warm, a generation that streams and persists, **a run that
survives the reader disconnecting**, and a stop that leaves the conversation
idle. The vision arm additionally asserts the model reports the `vision`
capability.

Plus the **same-feature-two-mechanisms** rows (plan Phase 3) — one feature with
two implementations split by engine, which is where reasoning from the provider
Literal hides the second one:

| feature | mlx-lm / mlx-vlm | gguf |
|---|---|---|
| audio input | tower stripped at load — must fail **loudly** with a 400 | supported |
| thinking capability | probes the template FILE | rides `supports_thinking` from GGUF metadata |
| thinking depth | `apply_chat_template` kwargs | `chat_template_kwargs` |
| chat-template source | `chat_template_source` | `chat_template_path`, taken at SPAWN |

The audio row goes to `POST /v1/messages` **non-streaming**, and both halves of
that matter. The conversation generate endpoint drops media the current model
cannot take at the wire (the mid-conversation model-switch rule), so the
provider guard is unreachable there BY DESIGN — a check aimed at it would pass
without ever running it. And a provider's request guards fire at the first
`next()`, which on the streaming path is after the headers flush, so the
refusal arrives as a 200 carrying an in-band error event; only the
non-streaming form is a status code.

The chat-template row is model-free and runs in the contract half: the settable
field set is derived from the provider config classes, so
`/v1/admin/model-options` is where a mechanism leaking to the wrong provider
would show. Its failure mode is silence — setting the wrong key does nothing,
and the publisher's embedded template goes on picking your prompt format.

Where a row's precondition is unmet it reports **uncovered**, never green. The
standing one: thinking depth on both MLX arms, because the only served MLX
model advertising `reasoning_effort` is a 120B.

The gguf audio row's precondition is the `audio` capability, which is DERIVED
from the entry's `modalities` -- not read out of the GGUF. So an entry that
lost `"audio"` from its modalities would turn the supported half into a silent
skip rather than a failure. That is the correct behaviour (an unmet
precondition is uncovered, not red), but it means the difference between
"uncovered because no audio model is served" and "uncovered because someone
edited an entry" is invisible from the output. The gemma-4 gguf entries are the
ones that carry it; if that row starts reporting uncovered, check `modalities`
before concluding anything about the engine.

## Which model each arm runs

Resident first, otherwise the SMALLEST by measured weight size
(`POST /v1/admin/models/{id}/fit` sizes from file stats -- no load; ~1s for a
30-model registry). Selection used to sort by `len(id)`, which is not a cost
signal: an unnarrowed run on this machine picked a 120B and two 27Bs, which
made the release standard something nobody would run. `--model ARM=ID` is
still the way to be sure, and a model whose size cannot be determined sorts
LAST -- an unknown size must not win a contest about smallness.

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
