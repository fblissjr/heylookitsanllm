# CLAUDE.md -- optloop-lib

Last updated: 2026-07-06

Library-level inference benchmark harness for mlx-lm / mlx-vlm fork
experiments. This is the repo's ONLY optimization bench: the app-level
`apps/optloop` was retired 2026-07-06 because its benchmarks called
mlx-lm directly while chartered to optimize server code -- a change to the
server scored 1.0 either way. Here, direct library calls are the CORRECT
measurement: the thing under test IS the library (editable-install forks
in `repos/`).

## Orient

- `program.md` -- benchmark protocol and (optional) agent-driven loop.
- `docs/optimization_log.md` (repo root) -- cross-session knowledge base:
  baselines, ceilings, what worked/failed. READ IT FIRST; update at
  session end. Key prior findings: M2 Ultra ceilings put Gemma-3-27B bf16
  at 79% and Qwen3.5-27B mxfp8 at 72% of bandwidth-bound max; Python
  overhead is <0.1% of per-token time -- real gains require reducing GPU
  work, not Python tuning.
- Serving-path (HTTP) benchmarking is explicitly OUT of scope here -- see
  the measurement section of `internal/backend/plan_2026-07.md`.

## The venv gotcha (this bites)

Own venv, own pyproject. Always `cd apps/optloop-lib` first; run
everything with `uv run`. The editable installs resolve mlx-lm/mlx-vlm to
`repos/` forks -- do NOT touch the forks' pyproject/setup files (they
track upstream), and never commit fork changes to heylookitsanllm.

## Run

```bash
cd apps/optloop-lib
uv run scripts/bench_text.py --reset-baseline   # once, before experiments
uv run scripts/bench_text.py                    # compare vs baseline
uv run scripts/bench_vlm.py                     # VLM variant
uv run scripts/bench_analysis.py                # summarize cycles
uv run pytest tests/ -q                         # harness unit tests (65)
```

Model paths resolve CLI > bench_config.toml id > the server's root
`models.toml` (local path, no re-download) > HF download.

## Honesty constraints (do not weaken)

- Greedy decode (temp=0) + SHA-256 token-ID fingerprint vs baseline:
  any output change auto-rejects. This freezes BEHAVIOR; it does NOT
  certify the baseline output was good -- there is no ground-truth quality
  metric. Don't present composite scores as quality claims.
- Hard constraints and per-prompt regression checks are code-enforced in
  `scripts/bench_common.py`; the keep/discard call on mixed results is
  agent judgment -- log the reasoning.
- Never modify `scripts/`, `bench_config.toml`, or fork test suites as
  part of an optimization.

## Known gaps (as of 2026-07-06)

- No optimization cycle has ever completed end-to-end; the first real run
  is the speculative-decoding baseline (plan, Phase 5). Treat that run as
  also validating the harness.
- Vision workloads are synthetic PIL renders; TODO: add real photographs
  and one genuinely long-context prompt (both change fingerprint
  baselines -- do it BEFORE establishing baselines, not after).
