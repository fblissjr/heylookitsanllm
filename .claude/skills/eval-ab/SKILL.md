---
name: eval-ab
description: A/B the behavioral eval bank across a change (pin bump, quant swap, release) with baseline storage and flap-vs-regression discipline
disable-model-invocation: true
---

# eval-ab

**BLOCKED until the bank is ported (2026-09-23).** `tests/eval/run.py` still
posts to `/v1/chat/completions`, which was removed in v1.79.66, so every task
comes back "request failed" on both sides of an A/B. The port to
`/v1/messages` is pending in docs/project/TODO.md.

Until it lands:
- say so rather than running the bank;
- use `tests/smoke/run.py --server <url>` for a live wire check per engine arm.

The flow below applies once the bank speaks the Messages wire again.

Explicit-ask wrapper around `tests/eval/run.py` for comparing eval-bank
results across a change: dependency pin bumps, quant swaps, or pre-release
checks. This is the FULL-bank tier. For small code changes, use the scoped
check the eval-gate hook describes instead. Do not let this grow into hours
of testing: the whole bank on the one or two models actually under
comparison, nothing more.

## Flow

1. **Server**: reuse-first via `bash scripts/dev_server.sh status`; only spawn
   (via the /dev-server skill) if nothing suitable is running. Both sides of
   an A/B MUST run against the same server config.
2. **Baseline (side A)**: run the bank and store it labeled:
   ```bash
   uv run python tests/eval/run.py --server <url> --models <ids-under-test> \
     --out internal/eval_baselines/<YYYY-MM-DD>_<label-A>.jsonl
   ```
   (`internal/` is gitignored -- baselines are local data, never committed.)
3. **Apply the change** (bump the pin, swap the model, etc.), restart/reload
   as needed.
4. **Side B**: same command, `<label-B>` out path. Same models, same tasks.
5. **Diff**: compare per-task pass/fail between the two JSONLs (task name +
   model -> passed). Report: newly failing, newly passing, unchanged.
6. **Flap discipline -- mandatory before calling a regression**: for any
   newly-failing task, rerun JUST its category (`--tasks <category>`) 2-3
   times on side B. Sampling flaps (e.g. the 07-20 `stop_discipline_long_form`
   flap: tight token budget + sampling variance) pass on rerun; real
   regressions fail consistently. Only consistent failures are findings.
   Thinking length in particular varies several-fold run to run at a fixed
   setting (docs/testing/gguf_runtime_audit_2026-09-23.md §6), so a task
   judged on it needs repeats before any verdict.

## Reporting

State per model: n passed / n failed, the confirmed regressions (task, what
the judge saw), and flaps you dismissed with rerun counts. If both sides are
clean, say so in one line -- no table needed.
