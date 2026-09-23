---
name: eval-gate
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: src/heylook_llm/(reasoning_parser|thinking_parser|vlm_inputs|chat_template_files|providers/common/(template_info|stop_tokens|generation_core))\.py$
---

You just edited a file whose behaviour unit tests cannot certify: thinking
split, stop discipline, or which chat template renders. The 07-20
turn-overrun and thinking-leak bugs passed 1000+ unit tests. Only a live model
checks these.

This reminder runs NOTHING, and you should not over-test. For trivial edits
(comments, log strings, type annotations), ignore it entirely.

**Status of the instruments (2026-09-23):**

- `tests/eval/` (the behavioural bank) STILL POSTS TO `/v1/chat/completions`,
  which was removed in v1.79.66. Every task fails as "request failed" until it
  is ported to `/v1/messages`; that port is pending in docs/project/TODO.md.
  Do not treat its red as a regression, or its absence as green.
- `tests/smoke/` speaks the live wire and names each engine arm (mlx-lm,
  mlx-vlm, gguf). It is the check that runs today:
  `uv run python tests/smoke/run.py --server <url>`
  (`--contract-only` loads nothing).
- A template edit on a CHAT model can also break multi-turn prompt caching
  without changing any output. After one, render turn 1 with the generation
  prompt and turn 2 as history, and confirm turn 1 is a prefix of turn 2.
  docs/testing/gguf_runtime_audit_2026-09-23.md §5 has the case that cost a
  model its cache for weeks.

**Server rule:** run only against an already-running server. Check with
`bash scripts/dev_server.sh status`, the /dev-server skill's reuse-first rule.
If none is up, do NOT spawn one just for this. Note the pending check in your
wrap-up so it runs with the next live session.
