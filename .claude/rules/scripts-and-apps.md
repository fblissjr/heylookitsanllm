---
paths:
  - "scripts/**"
  - "apps/**"
---

# scripts/ and apps/

- rich (batch-labeler, scripts): square brackets in dynamic text are markup and vanish silently. Wrap model output and prompts in `Text(...)` or `rich.markup.escape()`.
- Still targeting the removed OpenAI route, pending port (owner: small potatoes): `apps/batch-labeler` and `scripts/benchmark.py`'s OpenAI arms.
- Separate venvs, `cd` first: batch-labeler (`uv sync --dev`), optloop-lib (`uv sync`).
- `apps/optloop-lib/` is a library-level bench for mlx-vlm fork experiments ([docs/optloop_guide.md](../../docs/optloop_guide.md) and its own CLAUDE.md). The server does not depend on mlx-lm, so mlx-lm results there are server-irrelevant, and mlx-vlm fork wins reach the server only when upstreamed or repointed.
- Build flags for llama-server and their rationale: `scripts/README.md`.
