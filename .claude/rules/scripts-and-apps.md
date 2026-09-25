---
paths:
  - "scripts/**"
  - "apps/**"
---

# scripts/ and apps/

- rich (batch-labeler, scripts): square brackets in dynamic text are markup and vanish silently. Wrap model output and prompts in `Text(...)` or `rich.markup.escape()`.
- Still targeting the removed OpenAI route, pending port (owner: small potatoes): `scripts/benchmark.py`'s OpenAI arms. `apps/batch-labeler` speaks `/v1/messages` since v2.0.154; `tests/contract/test_batch_labeler_wire.py` checks its payload and parser against the server's schema classes, so a server wire change that breaks it fails the main suite.
- Separate venvs, `cd` first: batch-labeler (`uv sync --dev`), optloop-lib (`uv sync`).
- `apps/optloop-lib/` is a library-level bench for mlx-vlm fork experiments ([docs/optloop_guide.md](../../docs/optloop_guide.md) and its own CLAUDE.md). The server does not depend on mlx-lm, so mlx-lm results there are server-irrelevant, and mlx-vlm fork wins reach the server only when upstreamed or repointed.
- Build flags for llama-server and their rationale: `scripts/README.md`.
