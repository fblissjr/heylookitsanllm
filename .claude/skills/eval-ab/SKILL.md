---
name: eval-ab
description: A/B the behavioral eval bank across a change (pin bump, quant swap, release) with baseline storage and flap-vs-regression discipline
disable-model-invocation: true
---

# eval-ab

Explicit-ask wrapper around `tests/eval/run.py` for comparing eval-bank
results across a change: dependency pin bumps, quant swaps (e.g. QAT 4-bit vs
8-bit dailies), or pre-release checks. This is the FULL-bank tier -- for small
code changes use the smoke tier the eval-gate hook describes instead (1
category x 1 model). Do not let this grow into hours of testing: default to
the 13-task bank x the 1-2 models actually under comparison, nothing more.

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
   flap: tight 150-token budget + sampling variance) pass on rerun; real
   regressions fail consistently. Only consistent failures are findings.

## Reporting

State per model: n passed / n failed, the confirmed regressions (task, what
the judge saw), and flaps you dismissed with rerun counts. If both sides are
clean, say so in one line -- no table needed.
