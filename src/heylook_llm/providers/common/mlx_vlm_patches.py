"""Local carries of heylook's own open mlx-vlm PRs, applied per generator.

The pin stays on an upstream SHA; a fix we have filed but upstream has not
merged is carried here as a wrapper instead of a fork. Each patch:

- names its PR and keeps the PR's own condition, so the carried behaviour is
  the reviewed diff, not a local variant;
- wraps the upstream method and falls through to it for every other case;
- binds to one request's ``BatchGenerator`` (``vlm_engine.generate`` calls
  ``apply_to`` right after building it), never to the class, so nothing is
  left patched in the process;
- applies only while the wrapped signature is the one it was written
  against, and logs and stands down otherwise.

Retirement is enforced, not remembered: tests/contract/test_mlxvlm_surface.py
runs the UNPATCHED upstream restore path and fails once it already behaves
like the patch, naming the patch to delete. Moving the pin past the PR is the
moment that test goes red.
"""

import inspect
import logging
import os

# Blaizzy/mlx-vlm#2356: a restored single row gets the cache a cold row would.
# mlx-vlm gives a cold single row the model's own caches (PromptProcessingBatch)
# unless it drafts, quantizes KV or the model has no make_cache, but sends a
# restored one through APCCoordinator.merge_rows, which builds BatchKVCache
# even for one row. qwen3_5's single-row shortcut then extract()s and
# re-merge()s the full KV of every full-attention layer on every decode step,
# so a restored request decodes slower than the same request run cold. The PR
# (as updated 2026-09-30) routes that row at the restore site in
# generate/ar.py; this carry reaches the same outcome at merge_rows, the one
# call the restore site makes, gated per generator by the same rule.
PATCH_2356 = "mlx-vlm#2356"
MERGE_ROWS_PARAMS = ("picks", "prefix_lens", "kv_quant_config")  # bound method

# Carry of mlx-lm's 8-row packed Gated DeltaNet Metal kernel
# (_make_gated_delta_packed_kernel) for scalar-gated Dk=128, Dv % 8 == 0
# multi-token recurrence. Upstream mlx-vlm landed this in
# mlx_vlm/models/qwen3_5/gated_delta.py (d13bd3cc, T > 1); this carry covers
# the shared mlx_vlm/models/gated_delta.py module (qwen3_next, kimi_linear,
# lfm3) until it is synced there too. Retires once mlx_vlm.models.gated_delta
# defines _gated_delta_kernel_packed.
PATCH_GDN_PACKED = "mlx-vlm#gdn-packed"
GDN_KERNEL_PARAMS = ("q", "k", "v", "g", "beta", "state", "mask", "state_steps")

_PACKED_KERNEL_SOURCE = r"""
    constexpr int lanes_per_row = 4;
    constexpr int rows_per_simdgroup = 32 / lanes_per_row;
    constexpr int values_per_lane = Dk / lanes_per_row;
    constexpr int partials_per_lane = values_per_lane / 4;

    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);

    auto lane = thread_index_in_simdgroup;
    auto row_in_simdgroup = lane / lanes_per_row;
    auto lane_in_row = lane & (lanes_per_row - 1);
    auto row_group = thread_position_in_grid.y;
    auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
    auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
    y += (b_idx * T * Hv + hv_idx) * Dv;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

    float state[values_per_lane];
    for (int i = 0; i < values_per_lane; ++i) {
      state[i] = static_cast<float>(i_state[i]);
    }

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float gt = static_cast<float>(g_[hv_idx]);

      float part[partials_per_lane];
      for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          state[e] = state[e] * gt;
          acc += state[e] * static_cast<float>(k_[e]);
        }
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      auto delta =
          (static_cast<float>(v_[dv_idx]) - kv_mem) *
          static_cast<float>(beta_[hv_idx]);

      for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          state[e] = state[e] + static_cast<float>(k_[e]) * delta;
          acc += state[e] * static_cast<float>(q_[e]);
        }
        part[pb] = acc;
      }
      float out =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      out += simd_shuffle_xor(out, 1);
      out += simd_shuffle_xor(out, 2);
      if (lane_in_row == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }

      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }

    for (int i = 0; i < values_per_lane; ++i) {
      o_state[i] = static_cast<StT>(state[i]);
    }
"""


def install_gated_delta_packed_kernel() -> bool:
    """Install the 8-row packed Gated DeltaNet Metal kernel on ``mlx_vlm.models.gated_delta``
    when the pinned upstream SHA does not yet carry ``_gated_delta_kernel_packed``."""
    try:
        import mlx.core as mx
        from mlx_vlm.models import gated_delta as gd
    except ImportError:
        return False

    current = getattr(gd, "gated_delta_kernel", None)
    if current is None:
        return False
    if getattr(current, "_heylook_patch", None) == PATCH_GDN_PACKED:
        return True
    # If upstream already defines _gated_delta_kernel_packed natively, stand down.
    if hasattr(gd, "_gated_delta_kernel_packed"):
        return False
    if tuple(inspect.signature(current).parameters) != GDN_KERNEL_PARAMS:
        logging.warning(
            "%s carry not applied: gated_delta_kernel signature changed",
            PATCH_GDN_PACKED,
        )
        return False
    if not mx.metal.is_available():
        return False

    packed_kernel = mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_PACKED_KERNEL_SOURCE,
    )
    upstream = current

    def gated_delta_kernel(
        q,
        k,
        v,
        g,
        beta,
        state,
        mask=None,
        state_steps=None,
    ):
        B, T, Hk, Dk = k.shape
        Hv, Dv = v.shape[2:]
        if (
            os.environ.get("MLX_GDN_PACKED", "1") != "0"
            and mask is None
            and state_steps is None
            and g.ndim == 3
            and Dk == 128
            and Dv % 8 == 0
            and g.dtype == mx.float32
            and state.dtype == mx.float32
        ):
            return packed_kernel(
                inputs=[q, k, v, g, beta, state, T],
                template=[
                    ("InT", q.dtype),
                    ("StT", state.dtype),
                    ("Dk", Dk),
                    ("Dv", Dv),
                    ("Hk", Hk),
                    ("Hv", Hv),
                ],
                grid=(32, Dv // 8, B * Hv),
                threadgroup=(32, 2, 1),
                output_shapes=[(B, T, Hv, Dv), state.shape],
                output_dtypes=[q.dtype, state.dtype],
            )
        return upstream(q, k, v, g, beta, state, mask=mask, state_steps=state_steps)

    gated_delta_kernel.__doc__ = upstream.__doc__
    setattr(gated_delta_kernel, "_heylook_patch", PATCH_GDN_PACKED)
    setattr(gated_delta_kernel, "_heylook_upstream", upstream)
    gd.gated_delta_kernel = gated_delta_kernel
    return True



def plain_single_row_generator(bg) -> bool:
    """#2356's rule from the generator's side: no drafting, no KV
    quantization, and a model with its own ``make_cache``. A generator with
    a drafter keeps batch caches on restore, as it does cold."""
    drafting = (getattr(bg, "draft_model", None) is not None
                and getattr(bg, "draft_kind", None) is not None)
    return (not drafting and getattr(bg, "kv_bits", None) is None
            and hasattr(getattr(bg, "model", None), "make_cache"))


def single_restored_row(picks, kv_quant_config) -> bool:
    """One row, restored, with its own warm cache (a checkpoint hit), no KV
    quantization. Block-mode hits carry no warm cache and still merge."""
    return (len(picks) == 1 and picks[0] is not None
            and kv_quant_config is None
            and picks[0].get("warm_cache") is not None)


def _patch_merge_rows(coord) -> bool:
    upstream = coord.merge_rows
    if getattr(upstream, "_heylook_patch", None) == PATCH_2356:
        return True
    if tuple(inspect.signature(upstream).parameters) != MERGE_ROWS_PARAMS:
        logging.warning(
            "mlx-vlm#2356 carry not applied: APCCoordinator.merge_rows "
            "signature changed; restored qwen3_5 requests may decode slower "
            "than cold ones")
        return False

    def merge_rows(picks, prefix_lens, *, kv_quant_config=None):
        # An upstream rename the signature check cannot see (the pick's keys,
        # the manager's lock or stats) falls through to upstream: slower,
        # never a failed request. Nothing is mutated before the increment.
        try:
            if single_restored_row(picks, kv_quant_config):
                with coord.manager.lock:
                    coord.manager.stats.restored_tokens += prefix_lens[0]
                return picks[0]["warm_cache"], prefix_lens[0]
        except (AttributeError, KeyError, TypeError) as exc:
            logging.warning("mlx-vlm#2356 carry fell through to upstream: %r", exc)
        return upstream(picks, prefix_lens, kv_quant_config=kv_quant_config)

    merge_rows.__doc__ = upstream.__doc__
    setattr(merge_rows, "_heylook_patch", PATCH_2356)
    setattr(merge_rows, "_heylook_upstream", upstream)
    coord.merge_rows = merge_rows
    return True


def apply_to(bg) -> bool:
    """Install every carried patch on this generator. True when #2356's
    carry is in force for it; False for a generator the rule leaves on
    batch caches, one without a prefix cache, or a stood-down carry."""
    install_gated_delta_packed_kernel()
    coord = getattr(bg, "apc", None)
    if coord is None or not plain_single_row_generator(bg):
        return False
    return _patch_merge_rows(coord)
