# MLX >= 0.32.5 Upgrade & Gemma 4 Native Sliding-Window Attention Plan

Status: **All five upstream PRs merged on `ml-explore/mlx` `main`; `#4808` closed as non-bug; source build enabled via `scripts/build_mlx.py` (owner 2026-11-19)** (`uv.lock` stays at PyPI `mlx == 0.32.4` until `0.32.5` is released on PyPI; Phase 0/2 runs isolated via `scripts/build_mlx.py --overlay`).
Related backlog: [TODO.md](./TODO.md) Track 1.3 & Gemma 4 parked items (`#2b`, `#3a`).

---

## 1. Release Channel & Scope

- **PyPI pin + out-of-repo source builds**: per [dependencies.md](../../.claude/rules/dependencies.md), `pyproject.toml` and `uv.lock` stay on PyPI `mlx` / `mlx-metal` until upstream tags and publishes `0.32.5+`, while `scripts/build_mlx.py` builds pinned `ml-explore/mlx` revs (plus candidate upstream PRs) outside the repo into `~/.heylook/mlx` (`--overlay` for `PYTHONPATH` isolation or opt-in `--install`).
- **No blind retirement of owned Gemma 4 kernels**: upgrade kernels must be measured and checked numerically against existing owned paths (`providers/common/gemma4_vision.py`, `providers/common/gemma4_runtime.py`) before replacing any carried optimization.

---

## 2. Upstream Kernel Inventory & Hardware / Model Applicability

### 2.1 `mlx#4734` (merged `2026-10-23`): Native `window_size` in `mx.fast.scaled_dot_product_attention` (Primary Gemma 4 Win)

- **Current bottleneck**: [`gemma4_vision.gemma4_make_masks`](../../src/heylook_llm/providers/common/gemma4_vision.py) (and upstream `mlx_vlm/models/gemma4/language.py`) materializes full 4D boolean sliding masks whenever `N > 1` crosses `window_size` (`1024`) -- covering both chunked prefill past the window and every `N > 1` MTP verification step past `window_size` across all sliding layers (`50` on `gemma-4 31B`, `25` on `gemma-4 26B-A4B`).
- **Upstream API**: `mx.fast.scaled_dot_product_attention(q, k, v, *, scale, window_size=-1, mask=None, sinks=None, ...)`, enforcing `dist = abs(q_idx - k_idx) < window_size` inside Metal `sdpa_vector` and `steel_attention` / `steel_attention_nax` kernels without a materialized 4D boolean mask.
- **Integration design**:
  1. Feature-detect `window_size` support cleanly in `gemma4_vision.mlx_sdpa_supports_window_size()` (inspecting signature first and falling back to a one-time CPU probe for nanobind extensions, cached by `(id(fn), bool)` so mocks/overlays never read stale state).
  2. In `gemma4_make_masks` + `gemma4_runtime._bind_attention_subclass` (unquantized single-row `BatchRotatingKVCache` or `RotatingKVCache(keep=0)`, `not use_bidirectional_vision`, `not exact_speculative_verify`), return `mask="causal"` across all `N > 1` steps -- including past `window_size` and after rotation -- and call `mx.fast.scaled_dot_product_attention(..., mask="causal", window_size=window_size)` once `cache.update_and_fetch` has unrolled keys into temporal order via `_update_concat`.
  3. Keep 4D masks whenever `mlx_sdpa_supports_window_size()` is False, `use_bidirectional_vision` is active, `exact_speculative_verify` is enabled, `cache` is quantized, or the batch is padded / multi-row.

### 2.2 `mlx#4719` (merged `2026-10-19`): `float16` `head_dim=256` `TK=1` SDPA Vector Kernel on M1/M2

- Adds a `TK=1` fallback route in Metal SDPA for `float16` `head_dim=256` on M1/M2 where larger threadgroups hit device limits (`Qwen3.8 27B`, M1/M2 Ultra). Relevant primarily to `float16` inputs; verify M2 Ultra parity alongside `#4723`.

### 2.3 `mlx#4723` (merged `2026-10-25`): `sdpa_split` for Multi-Token Speculative Verification (`#4808` Closed Non-Bug)

- **Upstream Metal dispatch conditions** (`scaled_dot_product_attention.cpp` `use_split`):
  - `head_dim in {64, 72, 80, 96, 128, 192, 256}`
  - `q_len <= 8` (`2..8` verify tokens)
  - `33 <= q_len * (n_q_heads / n_kv_heads) <= 128` stacked GQA rows
  - `k_len >= 128`, `!has_arr_mask`, `!has_sinks`, `window_size <= 0`, `!output_logsumexp`.
- **Model applicability**:
  - Does **not** fire on `gemma-4` global layers (`head_dim=512` is outside `{64..256}`) or `gemma-4` sliding layers (`gqa <= 2`, so `q_len * gqa <= 8..10 <= 32`).
  - **Does** fire on `Qwen3.8 27B` (`head_dim=256`, `gqa=6`, `q_len >= 6`, long `k_len >= 128`) and other GQA families with `head_dim <= 256` under speculative decoding.
- **Resolution of `ml-explore/mlx#4808` (closed `2026-11-04`)**:
  - Upstream confirmed `sdpa_split` itself has no numerical regression: the `mlx-c` repro read a non-contiguous strided Metal output buffer directly as row-contiguous; calling `mlx_contiguous` restored float64 parity on both M2 Pro and M5 Max/Ultra. Still keep targeted parity checks across `chain_probe.py` and tiny tiny-model contract verification.

### 2.4 `mlx#4787` (merged `2026-11-02`) & Open Ultra PRs (`#4890`, `#4826`, `#4893`)

- **`#4787` (`affine qmv_mma` for 3-16 rows)**: `use_qmv_mma` skips Ultra dies (`d.get_architecture().back() == 'd'`).
- **Open upstream Metal PRs under evaluation via `scripts/build_mlx.py --pr`**:
  - **`ml-explore/mlx#4890`** (open): *[Metal] Use affine qmv_mma on M1 and M2 Ultra from 4 rows* -- directly targets M2 Ultra speculative verification (`N >= 4`) across all quantized linear projections (`q/k/v/o`, `gate/up/down`, LM head); benchmark on M2 Ultra with and without `gemma4_runtime._hoisted_8bit_qmv_kernel` before changing Ultra defaults.
  - **`ml-explore/mlx#4826`** (open): *[Metal] Use 64x64 tiles for large affine QMM* -- targets dense prefill (`gemma-4 31B`).
  - **`ml-explore/mlx#4893`** (open): *Use fused SDPA for causal head_dim 256 with 512 to 1023 query rows* -- targets `Qwen3.8 27B` mid-length prefill chunks.

### 2.5 `mlx#4737` (merged `2026-10-23`): Distributed Ring Cleanup Race Fix

- Fixes pending send/recv list races during distributed ring teardown, hardening `providers/common/mlx_pipeline.py` multi-Mac teardown and `unload()` paths.

---

## 3. Phased Rollout & Verification Gates

### Phase 0 (Optional Pre-PyPI Smoke in Disposable Source Venv)
1. Build `ml-explore/mlx` in an isolated temporary venv outside `heylook` (leaving `pyproject.toml` and `uv.lock` clean).
2. Run the `#4808` float64 SDPA numerical check across `sdpa_split` shapes.
3. Run `scripts/chain_probe.py` on `gemma-4 31B`, `gemma-4 26B-A4B`, and `Qwen3.8 27B`, plus `scripts/vlm_parity_probe.py` on `Qwen3.5 0.8B` and `gemma-4 26B-A4B`.

### Phase 1 (PyPI `mlx >= 0.32.5` Dependency Bump)
1. Precondition: `mlx >= 0.32.5` is published on PyPI and Phase 0 `#4808` check passes cleanly (or upstream includes the fix).
2. Update `mlx` / `mlx-metal` floor in `pyproject.toml` and refresh `uv.lock` (`uv sync`).
3. Verification gates:
   - `uv run pytest tests/unit/ tests/contract/ -q`
   - Standalone `#4808` multi-token SDPA accuracy check (keep as regression guard)
   - `scripts/chain_probe.py` (`gemma-4 31B`, `gemma-4 26B-A4B`, `Qwen3.8 27B`)
   - `scripts/vlm_parity_probe.py` (`Qwen3.5 0.8B`, `gemma-4 26B-A4B`)
   - Three-arm live `tests/smoke/run.py` + `tests/smoke/embeddings.py`
   - Clean idle `scripts/perf_ab.py` comparing pre-bump vs `0.32.5+` on speculative and undrafted workloads (`internal/claude/perf/`).

### Phase 2 (Native Sliding-Window SDPA Fast Path for Unrotated Gemma 4 Caches)
1. Check if the pinned `mlx-vlm` release already passes `window_size=` into `mx.fast.scaled_dot_product_attention`; if not, wire the capability-gated `window_size=self.window_size` path into `gemma4_vision.gemma4_make_masks` (and `gemma4_runtime.py` sliding attention wrapper if needed) strictly gated on `not c.rotated` and `not use_bidirectional_vision`.
2. Verification gates:
   - New contract parity test in `tests/contract/test_mlxvlm_surface.py` comparing native `window_size` output vs reference 4D sliding mask across `offset < window_size`, crossing `window_size`, and MTP `N > 1` verify steps.
   - `scripts/chain_probe.py` (`gemma-4 31B`, `gemma-4 26B-A4B`) and `scripts/vlm_parity_probe.py` (`gemma-4 26B-A4B`).
   - Clean idle `scripts/perf_ab.py` prefill + MTP verify comparison (`internal/claude/perf/`).

### Phase 3 (Re-evaluate Parked Gemma 4 Tuning)
1. Re-measure the parked `draft_max_context` break-even crossover on `gemma-4 31B` and `gemma-4 26B-A4B` under `mlx >= 0.32.5` (noting global `head_dim=512` layers still leave `sdpa_vector` on `N > 1`).
2. Update `TODO.md`, `docs/architecture/sharp_edges.md`, and `docs/wiki/performance_optimizations.md` based on clean `perf_ab.py` verdicts (`internal/` records only; no raw numbers in tracked docs).
