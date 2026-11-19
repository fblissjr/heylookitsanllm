# MLX >= 0.32.5 Upgrade & Gemma 4 Native Sliding-Window Attention Plan

Status: **Blocked on upstream PyPI release** (locked at `mlx == 0.32.4` in `uv.lock` as of 2026-11-19; `0.32.5` exists on `ml-explore/mlx` `main` only).
Related backlog: [TODO.md](./TODO.md) Track 1.3 & Gemma 4 parked items (`#2b`, `#3a`).

---

## 1. Release Channel & Scope

- **PyPI release only**: per [dependencies.md](../../.claude/rules/dependencies.md), `mlx` and `mlx-metal` stay on PyPI releases (never a `git` source pin in `pyproject.toml`). Pre-PyPI validation runs only in a disposable source-built venv so the repo lockfile stays untouched.
- **No blind retirement of owned Gemma 4 kernels**: upgrade kernels must be measured and checked numerically against existing owned paths (`providers/common/gemma4_vision.py`, `providers/common/gemma4_runtime.py`) before replacing any carried optimization.

---

## 2. Upstream Kernel Inventory & Hardware / Model Applicability

### 2.1 `mlx#4734`: Native `window_size` in `mx.fast.scaled_dot_product_attention` (Primary Gemma 4 Win)

- **Current bottleneck**: [`gemma4_vision.gemma4_make_masks`](../../src/heylook_llm/providers/common/gemma4_vision.py) (and upstream `mlx_vlm/models/gemma4/language.py`) materializes full 4D boolean sliding masks whenever `N > 1` crosses `window_size` (`1024`) -- covering both chunked prefill past the window and every `N > 1` MTP verification step past `window_size` across all sliding layers (`50` on `gemma-4 31B`, `25` on `gemma-4 26B-A4B`).
- **Upstream API**: `mx.fast.scaled_dot_product_attention(q, k, v, *, scale, window_size=-1, mask=None, sinks=None, ...)`, enforcing `dist = abs(q_idx - k_idx) < window_size` inside Metal `sdpa_vector` and `steel_attention` / `steel_attention_nax` kernels without a materialized 4D boolean mask.
- **Integration design**:
  1. Feature-detect `window_size` support once at load/first call (`"window_size" in inspect.signature(mx.fast.scaled_dot_product_attention).parameters` or cached helper probe).
  2. When active on unpadded single-row `BatchRotatingKVCache` sliding layers (`not use_bidirectional_vision` and `c.max_size <= self.window_size`) **and `not c.rotated`** (keys are strictly in temporal order): pass `mask="causal"` with `window_size=self.window_size` instead of calling `create_attention_mask(..., return_array=True)`.
  3. **Keep 4D masks whenever `c.rotated` is True** (or when `use_bidirectional_vision` is active, or on padded/multi-row batches): once `BatchRotatingKVCache` rotates into a ring buffer, physical key indices wrap mod `max_size`, so positional `window_size` inside Metal SDPA does not match logical causal distance. Preserve exact `Gemma4ExactSpeculativeVerifier` behavior across all layers.

### 2.2 `mlx#4719`: `float16` `head_dim=256` `TK=1` SDPA Vector Kernel on M1/M2

- Adds a `TK=1` fallback route in Metal SDPA for `float16` `head_dim=256` on M1/M2 where larger threadgroups hit device limits (`Qwen3.8 27B`, M1/M2 Ultra). Relevant primarily to `float16` inputs; verify M2 Ultra parity alongside `#4723`.

### 2.3 `mlx#4723`: `sdpa_split` for Multi-Token Speculative Verification + Hard `#4808` Gate

- **Upstream Metal dispatch conditions** (`scaled_dot_product_attention.cpp` `use_split`):
  - `head_dim in {64, 72, 80, 96, 128, 192, 256}`
  - `q_len <= 8` (`2..8` verify tokens)
  - `33 <= q_len * (n_q_heads / n_kv_heads) <= 128` stacked GQA rows
  - `k_len >= 128`, `!has_arr_mask`, `!has_sinks`, `window_size <= 0`, `!output_logsumexp`.
- **Model applicability**:
  - Does **not** fire on `gemma-4` global layers (`head_dim=512` is outside `{64..256}`) or `gemma-4` sliding layers (`gqa <= 2`, so `q_len * gqa <= 8..10 <= 32`).
  - **Does** fire on `Qwen3.8 27B` (`head_dim=256`, `gqa=6`, `q_len >= 6`, long `k_len >= 128`) and other GQA families with `head_dim <= 256` under speculative decoding.
- **Hard numerical release blocker (`ml-explore/mlx#4808`)**:
  - Upstream reported a large Metal precision regression in `sdpa_split` (`2..8` query tokens, `33..128` stacked rows, `>= 128` keys) relative to a float64 CPU reference.
  - **Gate**: Before accepting any `mlx >= 0.32.5` bump into `pyproject.toml` / `uv.lock`, run a targeted contract/verification test checking `mx.fast.scaled_dot_product_attention` under the exact `#4808` trigger shapes (`q_len in {2, 4, 6, 8}`, `gqa in {6, 8, 16}`, `k_len in {128, 1024, 4096}`, `head_dim in {128, 256}`, `bfloat16` & `float16`) against a float64 reference. Do not bump `mlx` while `#4808` reproduces unless upstream ships the fix in `0.32.5`.

### 2.4 `mlx#4787`: `affine qmv_mma` for 3-16 Rows (Keep `_hoisted_8bit_qmv_kernel` on Ultra)

- Upstream `use_qmv_mma` in `mlx/backend/metal/quantized.cpp` explicitly returns `false` on Ultra dies (`d.get_architecture().back() == 'd'`), keeping vector kernels instead of `qmv_mma`.
- **Action**: Keep `gemma4_runtime._hoisted_8bit_qmv_kernel` (`uint2`-hoisted 8-bit loads for wide-vocab tied LM heads and fused `gate_proj + up_proj` MLPs) enabled on Ultra Macs after upgrading to `mlx >= 0.32.5`. Re-benchmark on non-Ultra Macs (M-Pro / M-Max / M4+ NAX) only if non-Ultra targets become primary.

### 2.5 `mlx#4737`: Distributed Ring Cleanup Race Fix

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
