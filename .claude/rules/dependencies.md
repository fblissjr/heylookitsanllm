---
paths:
  - "pyproject.toml"
  - "uv.lock"
  - "scripts/guard_stable_channel.sh"
---

# Dependencies

- pyproject.toml is a hand-maintained manifest. Root venv: plain `uv sync` (no extras; dev tooling is the default `dev` group; `uv sync --no-dev` for runtime only). Updates are plain uv: `uv lock --upgrade[-package X]` + `uv sync`.
- The MLX engine is a committed git pin (owner decision): `[tool.uv.sources]` pins mlx-vlm to an exact upstream SHA with `rev =`, never `branch =`. mlx-lm is not a dependency. `mlx` and `mlx-metal` stay on PyPI releases in `pyproject.toml` / `uv.lock`; custom `ml-explore/mlx` source builds (a pinned SHA plus optional PRs) live outside the repo under `~/.heylook/mlx` via `scripts/build_mlx.py` (owner decision 2026-11-19, mirroring `llama.cpp` under `~/.heylook/llama.cpp`). By default `--overlay` unpacks into the canonical `~/.heylook/mlx/overlay`, which `src/heylook_llm/mlx_overlay.py` (`server.py` + `scripts/mlx_pipeline_worker.py`) picks up automatically with zero config, while standing down on ABI mismatch or whenever `.venv` PyPI `mlx` is strictly newer; `$HEYLOOK_MLX_OVERLAY=0` disables it or points at an alternate overlay, and explicit `--install` still needs `uv run --no-sync` or `.venv/bin/python` (plain `uv run` re-syncs `.venv` back to `uv.lock`).
- Moving the pin: edit the rev, `uv lock && uv sync` (changing the rev is what moves it; `--upgrade-package` does not, and an exact rev is invariant under `uv lock --upgrade`), suite green, new SHA in the commit message. A pin move alone needs no CHANGELOG entry or version bump (owner, 2026-09-30). `scripts/guard_stable_channel.sh` allows exact 40-hex `rev` git pins (`#<40-hex>` in `uv.lock`) without an override, while still blocking floating refs (`branch`, `tag`, unpinned `git`), non-40-hex revs, local path sources, and `latest` overrides unless `HEYLOOK_ALLOW_CHANNEL_COMMIT=1` is set. A cloner's `uv sync` needs git and GitHub reachable.
- Check mlx-vlm's open-PR backlog before writing any workaround, and verify it is actually broken first.

Why and history: [sharp_edges.md#engine-pins-and-dependencies](../../docs/architecture/sharp_edges.md#engine-pins-and-dependencies).
