---
name: eval-gate
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: src/heylook_llm/(reasoning_parser|thinking_parser|providers/common/(template_info|stop_tokens|vision_budget|generation_core))\.py$
---

You just edited a BEHAVIORAL-EVAL-GATED file. Unit tests provably cannot
certify this subsystem (the 07-20 turn-overrun and thinking-leak bugs passed
1000+ of them); the check is `tests/eval/` against a live server.

This reminder runs NOTHING and you should not over-test. Scope by what you
touched -- the smoke tier is 1 category x 1 fast model (~1-2 min), not the
full bank:

- `reasoning_parser.py` / `thinking_parser.py` / `template_info.py` -> `--tasks thinking,stop`
- `vision_budget.py` -> `--tasks vision`
- `stop_tokens.py` / `generation_core.py` -> `--tasks stop`

```bash
uv run python tests/eval/run.py --server <url> --models <one-fast-model-id> --tasks <mapped-categories>
```

Server rule: run ONLY against an already-running server (check with
`bash scripts/dev_server.sh status` -- the /dev-server skill's reuse-first
rule). If no server is up, do NOT spawn one just for this -- note the pending
smoke check in your wrap-up so it runs with the next live session.

Full bank x multiple models is explicit-ask-only (pin bumps, quant A/B,
releases) -- that is the /eval-ab skill, not this reminder. For trivial edits
(comments, log strings, type annotations) ignore this entirely.
