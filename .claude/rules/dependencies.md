---
paths:
  - "pyproject.toml"
  - "uv.lock"
  - "scripts/guard_stable_channel.sh"
---

# Dependencies

- pyproject.toml is a hand-maintained manifest. Root venv: plain `uv sync` (no extras; dev tooling is the default `dev` group; `uv sync --no-dev` for runtime only). Updates are plain uv: `uv lock --upgrade[-package X]` + `uv sync`.
- The MLX engine is a committed git pin (owner decision): `[tool.uv.sources]` pins mlx-vlm to an exact upstream SHA with `rev =`, never `branch =`. mlx-lm is not a dependency. `mlx` itself stays a PyPI release.
- Moving the pin: edit the rev, `uv lock --upgrade-package <name> && uv sync`, suite green, new SHA named in CHANGELOG. `scripts/guard_stable_channel.sh` blocks a git pin by default; commit one deliberately with `HEYLOOK_ALLOW_CHANNEL_COMMIT=1`. A cloner's `uv sync` needs git and GitHub reachable.
- Check mlx-vlm's open-PR backlog before writing any workaround, and verify it is actually broken first.

Why and history: [sharp_edges.md#engine-pins-and-dependencies](../../docs/architecture/sharp_edges.md#engine-pins-and-dependencies).
