# scripts/

Last updated: 2026-07-24

Standalone developer/ops scripts. All are run with `uv run` (PEP 723 headers
provision their own deps where noted, so they don't add anything to the project
environment). Flat by design -- six files don't need subdirectories, and several
are referenced by path from `CLAUDE.md`, `docs/`, and `tests/`, so moving them
would just create churn.

| Script | What it does | Run |
| --- | --- | --- |
| `update_deps.py` | **Dependency updater.** Bumps git-sourced packages (mlx-lm, mlx-vlm) to their latest commit and writes the resolved SHA back as a pinned `rev` in `[tool.uv.sources]`, then relocks -- plain uv has no command that pins HEAD into pyproject. Also does `--release` PyPI bumps with an optional `--pin` floor-raise. Has `--dry-run`. | `uv run scripts/update_deps.py [pkgs...] [--release] [--pin] [--branch B] [--dry-run]` |
| `dev_server.sh` | Spawns an **isolated** heylookllm server for live verification (temp DB, RAM pre-flight, server-owned load+warm readiness). Same warm contract as `tests/e2e`. The `dev-server` skill wraps this. | `scripts/dev_server.sh start\|stop\|status [--port N] [--model ID]` |
| `benchmark.py` | HTTP benchmark against a **running** server: TTFT, generation TPS, memory; OpenAI + Messages endpoints, streaming and not. (Uses `rich`, in the dev group.) | `uv run scripts/benchmark.py [--url ...] [--model ...]` |
| `export_openapi.py` | Exports the OpenAPI spec from the FastAPI app **without** running the server. | `uv run python scripts/export_openapi.py [--format yaml] [-o PATH] [--stats]` |
| `jspace_convert_lens.py` | Converts a Jacobian-lens `.pt` into an mx-safetensors lens for the j-space feature. Self-contained deps via its PEP 723 header (torch, safetensors, jlens). | `uv run scripts/jspace_convert_lens.py ...` |
| `syntax_check.py` | Fast `ast.parse` sweep over the source tree -- a cheap pre-flight for syntax errors. | `uv run python scripts/syntax_check.py` |

## Updating dependencies (common workflows)

```bash
# Bump mlx-lm + mlx-vlm to latest commit, re-pin the SHAs in pyproject + relock
uv run scripts/update_deps.py

# Preview only
uv run scripts/update_deps.py --dry-run

# Track a non-default branch
uv run scripts/update_deps.py mlx-lm --branch main

# Latest RELEASED transformers, and raise its >= floor to what uv resolved
uv run scripts/update_deps.py transformers --release --pin
```

Why this exists instead of a one-liner: uv can pin a git commit in `uv.lock`
(`uv lock --upgrade-package <pkg>`) but leaves the source **floating** in
`pyproject.toml`. This project's policy is to always keep an explicit `rev`
pinned in both, and `update_deps.py` is the only thing that does that. It
replaced the old `update-packages.sh` (removed 2026-07-24), which only
refreshed the lock.
