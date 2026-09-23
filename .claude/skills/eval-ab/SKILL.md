---
name: eval-ab
description: A/B the behavioral eval bank across a change (pin bump, quant swap, release) with baseline storage and flap-vs-regression discipline
disable-model-invocation: true
---

# eval-ab

The bank speaks `/v1/messages` (ported v2.0.71). A result JSONL from before
that port recorded every task as "request failed" and is not a baseline.

Explicit-ask wrapper around `tests/eval/run.py` for comparing eval-bank
results across a change: dependency pin bumps, quant swaps, or pre-release
checks. This is the FULL-bank tier. For small code changes, run one category
(`--tasks <category>`) on one fast model against an already-running server
instead. Do not let this grow into hours
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
