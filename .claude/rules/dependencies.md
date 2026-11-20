---
paths:
  - "pyproject.toml"
  - "uv.lock"
  - "scripts/guard_stable_channel.sh"
---

# Dependencies

- pyproject.toml is a hand-maintained manifest. Root venv: plain `uv sync` (no extras; dev tooling is the default `dev` group; `uv sync --no-dev` for runtime only). Updates are plain uv: `uv lock --upgrade[-package X]` + `uv sync`.
- Public `pyproject.toml` and `uv.lock` stay on clean PyPI releases (`mlx`, `mlx-metal`, `mlx-vlm` >= 0.8.0; zero external git SHAs committed to public history; mlx-lm is not a dependency). Local bleeding-edge upstream tracking lives outside the repo under `~/.heylook/` (owner decision 2026-11-20, unifying all three engines with `~/.heylook/llama.cpp`):
  - `llama.cpp`: `scripts/build_llama.py` -> `~/.heylook/llama.cpp/`
  - `mlx` (C++/Metal): `scripts/build_mlx.py` -> `~/.heylook/mlx/overlay`
  - `mlx-vlm` and any other pure-Python upstream repo (e.g. `transformers`): `scripts/sync_overlay.py <pkg>` -> `~/.heylook/overlays/<pkg>` (relocatable via `$HEYLOOK_OVERLAYS_DIR`).
- `src/heylook_llm/mlx_overlay.py` (`activate_all_overlays()`, wired into `server.py`, `scripts/mlx_pipeline_worker.py`, `scripts/vlm_parity_probe.py`, and `scripts/perf_ab.py`) auto-discovers `~/.heylook/mlx/overlay` and `~/.heylook/overlays/<pkg>` before imports with zero config, standing down on ABI mismatch (compiled `mlx`) or whenever `.venv` PyPI is strictly newer; `$HEYLOOK_OVERLAYS=0` disables all local overlays, `$HEYLOOK_MLX_OVERLAY` / `$HEYLOOK_<PKG>_OVERLAY` override or disable one package.
- `scripts/guard_stable_channel.sh` blocks external `git` sources, local path sources, and `latest` overrides in staged `pyproject.toml` / `uv.lock` (allowing self `editable = "."`) unless `HEYLOOK_ALLOW_CHANNEL_COMMIT=1` is set.
- Check mlx-vlm's open-PR backlog before writing any workaround, and verify it is actually broken first.

Why and history: [sharp_edges.md#engine-pins-and-dependencies](../../docs/architecture/sharp_edges.md#engine-pins-and-dependencies).
