---
paths:
  - "pyproject.toml"
  - "uv.lock"
  - "scripts/guard_stable_channel.sh"
---

# Dependencies

- pyproject.toml is a hand-maintained manifest. Root venv: plain `uv sync` (no extras; dev tooling is the default `dev` group; `uv sync --no-dev` for runtime only). Updates are plain uv: `uv lock --upgrade[-package X]` + `uv sync`.
- The MLX engine is a committed git pin (owner decision): `[tool.uv.sources]` pins mlx-vlm to an exact upstream SHA with `rev =`, never `branch =`. mlx-lm is not a dependency. `mlx` itself stays a PyPI release.
- Moving the pin: edit the rev, `uv lock && uv sync` (changing the rev is what moves it; `--upgrade-package` does not, and an exact rev is invariant under `uv lock --upgrade`), suite green, new SHA in the commit message. A pin move alone needs no CHANGELOG entry or version bump (owner, 2026-09-30). `scripts/guard_stable_channel.sh` allows exact 40-hex `rev` git pins (`#<40-hex>` in `uv.lock`) without an override, while still blocking floating refs (`branch`, `tag`, unpinned `git`), non-40-hex revs, local path sources, and `latest` overrides unless `HEYLOOK_ALLOW_CHANNEL_COMMIT=1` is set. A cloner's `uv sync` needs git and GitHub reachable.
- Check mlx-vlm's open-PR backlog before writing any workaround, and verify it is actually broken first.

Why and history: [sharp_edges.md#engine-pins-and-dependencies](../../docs/architecture/sharp_edges.md#engine-pins-and-dependencies).
