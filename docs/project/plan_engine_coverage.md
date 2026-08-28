# Plan: engine coverage

last updated: 2026-08-28

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

## Phase 1 — one engine classifier, not two

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

## Phase 2 — coverage is reported, not assumed

**Do:**

- `tests/eval/run.py` prints an engine-coverage summary: which engines the
  `--models` list actually spanned, and which were absent.
- Both harnesses exit non-zero on an absent engine **unless the run was
  explicitly narrowed** (`--arm`, or an explicit `--models`). Narrowing is a
  decision; silence is not.

**Why it is not just cosmetic:** the eval bank's task filter (`required_capabilities
<= model_caps`) means a text-only model list produces a green run with every
vision task quietly skipped. That is the exact shape this plan exists to end.

## Phase 3 — the same-feature-two-mechanisms invariants

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

## Phase 4 — say what "covered" means for a release

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

Phases 1 and 2 need no model loads. Phase 3 needs one model per arm, and the
existing small ones suffice (`Qwen3.5-0.8B-MLX-8bit`,
`google_gemma_4-E4B-it-bf16-mlx`, `google_gemma-4-E4B-it-qat-q4_0-gguf`) —
keeping the whole thing runnable in minutes, which is the only version of it
that will actually get run.

## The failure mode this plan is guarding against

Not "an engine breaks". It is: **an engine breaks, and the suite is green,
because that engine had no model in the list and nothing said so.** Every
phase above is aimed at making that state impossible to reach quietly.
