# Plan: engine coverage

last updated: 2026-08-28 (ALL PHASES SHIPPED -- 0 at v1.79.28, 1-2 at v1.79.31, 3-4 at v1.79.34)

## Context

The provider `Literal` has three values. They are not three engines:

```
provider "mlx"   ──┬──▶ mlx-lm    (text)     two SEPARATE upstream repos,
                   └──▶ mlx-vlm   (vision)   separate release trains, separate
                                             maintainers, separate breakage
provider "gguf"  ─────▶ llama-server subprocess (one engine, one local binary)
```

Which MLX library actually decodes is `MLXProvider.effective_loader`
(`providers/common/loader_routing.py`), derived from the model's modalities —
**not** from the provider field. So "we covered mlx" is a claim about a config
value, not about code.

This matters because the repo already pins upstream posture on the two moving
separately: mlx-lm is release-starved and gets SHA-pinned, mlx-vlm has its own
cadence, and llama.cpp is built locally from a release tag by
`scripts/build_llama.py`. Three independent things break independently.

**The problem is not that coverage is absent — it is that coverage is
incidental.** Unit and contract tests cover routing and config with MLX mocked
or a `FakeProvider`; they never run an engine. The two harnesses that *do* run
engines both take `--models` from a human:

- `tests/eval/run.py` filters tasks by model capability, so a run given only
  text models silently exercises no vision path and still prints green.
- `tests/e2e/` (chat suite) drives one `E2E_MODEL`.

Neither reports **which engines were exercised**. A green run is not a
coverage claim, and nothing in the repo says it isn't.

### Non-goals

- **`mlx_embedding` is out of scope** (owner call, 2026-08-28).
- Behaviour and quality — thinking splits, stop discipline, vision correctness
  — stay with `tests/eval/`. This plan is about *which engines ran at all*.
- Performance. No numbers, per the repo's standing rule.

## The invariant to establish

> A live harness reports the engines it exercised. An engine with no model is
> **uncovered**, and uncovered is never reported as green.

Everything below serves that one sentence.

---

## Phase 0 — a live smoke test with engine arms (DONE, v1.79.28)

`tests/smoke/` — opt-in, never spawns a server, `--contract-only` runs in
seconds. Arms are engines: `mlx-lm`, `mlx-vlm`, `gguf`. A missing arm prints as
skipped under a banner saying a skipped engine is uncovered, not green.

Per arm: load + warm, a generation that streams and persists, **a run that
survives the reader disconnecting**, and a stop that leaves the conversation
idle. Verified green on all three arms against a real server.

Two things it already earned:

- The walk-away check is the first real confirmation of the detached-run
  behaviour the client learned to disclose in v1.79.26. No stubbed suite can
  see it.
- The vision arm failed on its first run — a 1×1 PNG fixture dies in gemma's
  resize before the model is reached. A degenerate fixture tests the
  preprocessor, not the engine.

## Phase 1 — one engine classifier, not two (DONE, v1.79.31)

`tests/helpers/engines.py` — `classify(server) -> Coverage`, imported by both
`tests/smoke/run.py` and `tests/eval/run.py`. It lives in `tests/helpers/`
rather than the `tests/lib/` this plan first named: `helpers/` already IS the
shared-test-code import path (`mlx_mock`, `sse`), and standing up a second
parallel directory for the same purpose is the copy this plan objects to,
one level up. Both harnesses run as scripts, so each inserts `tests/` on
`sys.path` itself.

It was unblocked by a server change, not a test change. **`GET /v1/admin/models`
now serves `effective_loader`** per row (`mlx-lm` | `mlx-vlm`, null for every
non-mlx provider), DERIVED from the config rather than read off a loaded
provider — `MLXProvider.effective_loader` is null for every model that is not
resident, which is exactly the set a harness has to choose its arms from. So
`classify` asks the server instead of guessing, and the `unconfirmable` bucket
that Phase 0 had to report for every auto-routed vision model is now empty.
The bucket stays: a server too old to serve the field still classifies, and
still says which answers were inferred.

`Coverage` also carries capabilities and residency, so `tests/eval/run.py`'s
`fetch_models` is gone rather than duplicated.

**What it found the first time it ran** (2026-08-28, 30 served models): this
machine serves exactly ONE mlx-lm model, `gpt-oss-120b-MXFP4-Q8-mlx`. The cheap
text arm this plan assumed — `Qwen3.5-0.8B-MLX-8bit` — is `mlx-vlm`: Qwen3.5
declares vision and mlx-vlm registers `qwen3_5`, so it POSITIVELY routes there.
Every sentence below that names it as the mlx-lm model was wrong, and nothing
could have said so before, which is the point. Running the mlx-lm arm cheaply
now needs either a small text-only MLX model or an explicit
`loader = "mlx-lm"` pinned on a small one — the explicit-loader rule is
honoured by the field, so a pinned model classifies as mlx-lm.

Verified live: for all 17 served mlx models, the value on the wire equals what
`MLXProvider.__init__` computes for the same id. Same rule, same two inputs —
the check is that the two CALL SITES feed it the same thing.

### What the phase called for

`tests/smoke/run.py` has `classify()`; `tests/eval/run.py` has
`fetch_models()`. They already answer overlapping questions from the same two
endpoints. That is the repo's own named defect class — a hand-copied second
copy that drifts — and it will drift the moment a fourth engine appears or the
vision capability is spelled differently.

**Do:** extract one helper both import — `model_id -> engine`, resolving
`gguf` from `/v1/admin/models` and splitting `mlx` on the vision capability.
It belongs beside the harnesses (a shared `tests/lib/`), **not** in
`src/heylook_llm/` — the server has no reason to carry a test taxonomy.

**Check:** point it at a server and assert every served model classifies, or is
explicitly excluded. An unclassifiable model is a coverage hole with no name.

## Phase 2 — coverage is reported, not assumed (DONE, v1.79.31)

Both harnesses print an engine-coverage paragraph naming, per arm, `covered` or
`UNCOVERED` — and `UNCOVERED` distinguishes "no model of this engine is served"
from "models are served and this run touched none of them", which is the
quieter and more dangerous of the two.

- `tests/smoke/run.py` prints it BEFORE the arms run, not after: a summary you
  read after ten minutes of loads has already let you assume. Its exit rule was
  already in place from Phase 0 (exit 2 on an absent arm unless narrowed).
- `tests/eval/run.py` prints it after the summary, plus the line it most needed:
  **how many tasks ran on NO model**, by category. The bank's
  `required_capabilities <= model_caps` filter was silent, so a text-only
  `--models` list ran zero vision tasks under a full-width green.

The exit rule reduces differently for eval, deliberately: `--models` is
required there and auto-picking models for a 13-task bank is the heavyweight
auto-testing this repo declines, so **every eval run is narrowed** and coverage
never changes its exit code. Narrowing is a decision; what Phase 2 buys there
is that the decision's consequence is stated rather than implied.

### What the phase called for

**Do:**

- `tests/eval/run.py` prints an engine-coverage summary: which engines the
  `--models` list actually spanned, and which were absent.
- Both harnesses exit non-zero on an absent engine **unless the run was
  explicitly narrowed** (`--arm`, or an explicit `--models`). Narrowing is a
  decision; silence is not.

**Why it is not just cosmetic:** the eval bank's task filter (`required_capabilities
<= model_caps`) means a text-only model list produces a green run with every
vision task quietly skipped. That is the exact shape this plan exists to end.

## Phase 3 — the same-feature-two-mechanisms invariants (DONE, v1.79.34)

All four rows shipped. Three run per arm in `tests/smoke/run.py`
(`audio_checks`, `thinking_checks`); the fourth is model-free and lives in the
contract half, so `--contract-only` covers it in seconds.

**Audio** is the row that earned itself. Two things had to be got right:

- It is sent to `POST /v1/messages` **non-streaming**, not to the conversation
  generate endpoint. The conversation path drops media the current model cannot
  take AT THE WIRE with a per-message disclosure (the mid-conversation
  model-switch rule), so the provider's guard is unreachable there by design and
  a check aimed at it would have passed without ever running the guard.
  Near-unreachable rather than unreachable: the drop keys on the DERIVED
  capability, so a hand-written `capabilities = ["audio"]` override on an MLX
  entry routes the part through and does hit the 400.
- Non-streaming because the provider's request guards fire at the first
  `next()`, which on the streaming path is *after* the headers flush — the same
  refusal then arrives as a 200 carrying an in-band `invalid_request_error`
  event. Only the non-streaming form turns it into a status code.

Verified live on all three arms: MLX refuses with a 400 whose text names gguf
as the way to run audio; the gguf arm accepts it.

**Thinking capability and depth** check that a model ADVERTISING the capability
ACCEPTS the corresponding request — the pairing is the invariant, because a
capability nothing honours is a control the UI shows and the model ignores, and
on gguf a value the template rejects is a raised jinja exception that
llama-server returns as a 500. Depth sends `medium`, the one value in both
published sets (Qwen3.8 takes xhigh|medium|low, harmony low|medium|high), which
keeps the check about the mechanism rather than one model's vocabulary.

**Chat-template source** is model-free: `chat_template_source` must be offered
for mlx and not gguf, `chat_template_path` for gguf and not mlx. The failure
mode is silence — setting the wrong one for a provider does nothing at all, and
the publisher's embedded template goes on picking your prompt format. The
settable set is derived from the provider config classes, so
`/v1/admin/model-options` is exactly where a leak would show.

**Standing UNCOVERED gap:** thinking DEPTH on both MLX arms. The only served
MLX model advertising `reasoning_effort` is `gpt-oss-120b-MXFP4-Q8-mlx`, so
covering it costs a 120B load. Reported as uncovered on every run rather than
quietly skipped — which is the rule this plan exists to enforce, applied to
itself.

### What the phase called for



The highest-value additions, because these are places where one *feature* has
two *implementations* and reasoning from the provider Literal hides the second.
Each becomes a smoke check on the arms it applies to:

| Feature | mlx-lm / mlx-vlm | gguf |
|---|---|---|
| **Audio input** | tower stripped at load; must fail **loudly** with a 400 | supported |
| **Thinking capability** | probes the template file (`capabilities.template_supports_thinking`) | rides `supports_thinking` from GGUF metadata |
| **Thinking depth** | `apply_chat_template` kwargs | `chat_template_kwargs` |
| **Chat template source** | `chat_template_source` (MLX-only) | `chat_template_path` at spawn — the publisher picks your format |

**Do:** add a check per row, on the arms it applies to. The audio one is the
most valuable and the cheapest: send an `input_audio` part to an MLX arm and
assert a **400**, not a silent drop. A silent drop there is a wrong answer the
user cannot see.

**Watch for:** these need per-model preconditions (a thinking-capable model, an
audio-capable gguf). Where the precondition is unmet, report **uncovered** —
same rule as a missing arm. Do not weaken a check to make it runnable.

## Phase 4 — say what "covered" means for a release (DONE, v1.79.34)

Written into `CLAUDE.md`'s test section: before a release touching provider,
loader, template or lifecycle code, `tests/smoke/` runs green on all three
arms, and an uncovered arm — or an uncovered Phase 3 mechanism — is named in
the changelog rather than passed over.

### What the phase called for



Write the standard down in `CLAUDE.md`'s test section: before a release that
touches provider, loader, template, or lifecycle code, `tests/smoke/` runs
green on all three arms, and an uncovered arm is called out in the changelog
rather than passed over.

Deliberately a documented standard, not a CI gate — there is no CI here, the
hardware is one machine, and a gate nobody can run is worse than a rule
somebody follows.

---

## Sequencing and cost

Phase 1 is small and unblocks the reporting in Phase 2. Phase 3 is the
substance and can be done a row at a time — each row is independently useful.
Phase 4 is a paragraph.

All four phases are done. Every arm now has a cheap model:

| arm | model | note |
|---|---|---|
| mlx-lm | `Qwen3.5-0.8B-MLX-8bit-textonly` | added in models.toml — the SAME weights as the entry below with `loader = "mlx-lm"` |
| mlx-vlm | `Qwen3.5-0.8B-MLX-8bit` | routes to mlx-vlm: it declares vision and mlx-vlm registers `qwen3_5` |
| gguf | `google_gemma-4-E4B-it-qat-q4_0-gguf` | also the audio-capable half of the audio row |

The mlx-lm entry is the fix this plan recommended for its own finding. Two
entries sharing one `model_path` is deliberate and is NOT the accidental
duplicate `CLAUDE.md` warns about: discovery cannot create it (`merge_discovered`
filters discovered-vs-explicit, never explicit-vs-explicit), and the entry
carries a comment saying so. `heylookllm import` dedups on resolved path, so a
reimport is the one thing likely to collapse it back — check after importing.

## The failure mode this plan is guarding against

Not "an engine breaks". It is: **an engine breaks, and the suite is green,
because that engine had no model in the list and nothing said so.** Every
phase above is aimed at making that state impossible to reach quietly.
