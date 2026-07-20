# LLM behavior-eval harness

last updated: 2026-07-20

An OPTIONAL eval pass for code changes that can alter LLM *behavior* --
chat template changes, parser/stop-token changes, vision pipeline changes,
`vision_tokens` budget plumbing. Generalizes two ad-hoc debugging scripts
(`full_matrix.py`, `repro_multiimage.py`) that used to get rewritten from
scratch every time someone needed to re-check this stuff.

**NOT part of `/test-suite`.** It needs a real, already-running server with a
real model doing real inference (Metal/GPU-gated, slow). This tool NEVER
spawns a server itself -- point `--server` at an already-running `heylookllm`
instance. Models load on-demand via the server's own router; there is no
separate preload step.

This is the behavior-regression analogue of the browser E2E harness in
`tests/e2e/` (also opt-in, also not wired into `/test-suite`) -- same idea,
different failure mode: E2E watches the UI, this watches what the model
actually says.

## Run

```bash
uv run python tests/eval/run.py --list-tasks    # no server needed, dry look at the bank

uv run python tests/eval/run.py \
    --server http://localhost:8080 \
    --models gemma-4-31b-it-8bit-mlx,Qwen3.5-27B-8bit-mlx

uv run python tests/eval/run.py --server http://localhost:8080 \
    --models gemma-4-31b-it-8bit-mlx --tasks vision,thinking
```

Exit code is non-zero if any task failed (or if a `--models` entry wasn't
found on the server) -- gate a local pre-release check on it manually, it is
not meant to run in CI.

## When to run this

- After chat-template changes (`providers/common/template_info.py`, model
  `chat_template_source` changes).
- After stop-token or sampler changes (`providers/common/stop_tokens.py`,
  `generation_core.py`).
- After vision pipeline / `vision_tokens` budget changes (vision feature
  cache, VLM prompt formatting).
- Before a release that touches `generation_core.py` or the `providers/`
  tree.

## Task bank + capability gating

`run.py` calls `GET {server}/v1/models` once, builds a `model -> capability
set` map, and for each requested model runs only the tasks whose
`required_capabilities` are a subset of that model's capabilities (a task
with no required capabilities always runs). A model missing from the
`/v1/models` response is skipped with a warning, not a crash.

## Adding a task

A task is an `EvalTask` in `tasks.py`: a name, a category (`vision` /
`thinking` / `stop` / `text` -- matches `--tasks`), the capabilities it
needs, a `build_request()` closure returning a `/v1/chat/completions` body
(minus `model`/`stream`, which `run.py` fills in), and a `judge()` closure
that takes `{content, thinking, completion_tokens, max_tokens}` and returns
a `judges.Verdict`. Prefer composing the existing property-checks in
`judges.py` (`color_mention`, `marker_leak`, `repetition`,
`token_budget_exhausted`, `exact_word_count`, `non_empty_non_gibberish`,
`substring_present`) via `combine_verdicts()` over writing a new one --
judges must test properties (colors mentioned, no leak markers, terminated
cleanly), never a specific model version's exact phrasing.

## Results

One JSONL line per `(model, task)` pair, written to `--out` (default
`tests/eval/results.jsonl`): `model`, `task`, `category`, `passed`,
`evidence`, `elapsed_ms`, `timestamp`, and on failure an `error` field with
the exception text.
