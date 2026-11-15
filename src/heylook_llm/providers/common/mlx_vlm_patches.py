"""Heylook-owned runtime customizations over mlx-vlm, applied per generator.

Heylook owns these optimizations locally alongside upstream pins rather than
waiting on upstream merges or maintaining a fork. Each wrapper:

- checks the target method or symbol signature before binding, logging and
  standing down cleanly if an upstream bump changes the surface or adopts an
  equivalent native symbol;
- wraps upstream and falls through to it for every unmatched shape or condition;
- binds per-request onto one ``BatchGenerator`` (via ``apply_to``) or
  idempotently installs stateless Metal kernels where noted.
"""

import inspect
import logging
import os

# Blaizzy/mlx-vlm#2356: a restored single row gets the cache a cold row would.
# mlx-vlm gives a cold single row the model's own caches (PromptProcessingBatch)
# unless it drafts, quantizes KV or the model has no make_cache, but sends a
# restored one through the coordinator's row merge, which builds BatchKVCache
# even for one row. Even after upstream 99b7f7c9 restricted the first
# _extract_row_cache + merge block to h.shape[1] > 1, qwen3_5's second
# _is_single_row_batch_cache(fa_cache) block runs on every decode token and
# evaluates cache_left_padding.max().item() and query_left_padding.max().item()
# synchronously on the CPU, so a restored request still decodes slower than a
# cold one without this patch (re-verified on 918763c6, record in
# internal/claude/perf/mlx_vlm_patches_ab_2026-10-30.md; and on 003311ec,
# internal/claude/perf/ab_Qwen3.5-0.8B-MLX-8bit_20261104-112909.json). The PR routes that
# row at the restore site in generate/ar.py; this carry reaches the same
# outcome at the one coordinator call the restore site makes, gated per
# generator by the same rule. That call is RESTORE_MERGE: upstream #2646
# (fb3b4623) moved the site from merge_rows to _merge_and_release_cache_rows,
# which still builds BatchKVCache for one row, and a carry left on merge_rows
# installed cleanly and never ran. TestCarriedPatches reads the site's source
# for this name so a move cannot go quiet again.
PATCH_2356 = "mlx-vlm#2356"
RESTORE_MERGE = "_merge_and_release_cache_rows"
MERGE_ROWS_PARAMS = ("picks", "prefix_lens", "kv_quant_config")  # bound method

# Packed 8-row Gated DeltaNet Metal kernel (_make_gated_delta_packed_kernel)
# for scalar-gated Dk=128, Dv % 8 == 0 multi-token recurrence in shared
# mlx_vlm/models/gated_delta.py (qwen3_next, kimi_linear, lfm3). Stands down
# if mlx_vlm.models.gated_delta defines _gated_delta_kernel_packed natively.
PATCH_GDN_PACKED = "mlx-vlm#gdn-packed"
GDN_KERNEL_PARAMS = ("q", "k", "v", "g", "beta", "state", "mask", "state_steps")

PATCH_MTP_B1_WALK = "heylook#mtp-b1-walk"
PATCH_MTP_LEFT_PADDING = "heylook#mtp-b1-left-padding"
MTP_WALK_PARAMS = (
    "lm",
    "target_hidden",
    "draft_tokens",
    "sampler",
    "budgets",
    "row_ids",
    "base_positions",
)
MTP_LEFT_PADDING_PARAMS = ("prompt_cache",)
MTP_QMV_MAX_T = 8

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
    upstream = getattr(coord, RESTORE_MERGE, None)
    if getattr(upstream, "_heylook_patch", None) == PATCH_2356:
        return True
    if (upstream is None
            or tuple(inspect.signature(upstream).parameters) != MERGE_ROWS_PARAMS):
        logging.warning(
            "mlx-vlm#2356 carry not applied: APCCoordinator.%s is gone or its "
            "signature changed; restored qwen3_5 requests may decode slower "
            "than cold ones", RESTORE_MERGE)
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
    setattr(coord, RESTORE_MERGE, merge_rows)
    return True


def _install_mtp_walk_fast_path(mtp, mx) -> bool:
    current = getattr(mtp, "_speculative_walk_batch_deferred_greedy", None)
    if current is None:
        return False
    if getattr(current, "_heylook_patch", None) == PATCH_MTP_B1_WALK:
        return True
    if tuple(inspect.signature(current).parameters) != MTP_WALK_PARAMS:
        logging.warning(
            "%s stood down: _speculative_walk_batch_deferred_greedy signature changed",
            PATCH_MTP_B1_WALK,
        )
        return False
    if not (
        callable(getattr(mtp, "_mtp_logits_from_hidden", None))
        and callable(getattr(mtp, "_sampler_supports_positioned_target", None))
        and hasattr(mtp, "generation_stream")
    ):
        return False

    upstream = current

    def _heylook_mtp_walk_batch(
        lm,
        target_hidden,
        draft_tokens,
        sampler,
        budgets,
        row_ids=None,
        base_positions=None,
    ):
        if (
            getattr(lm, "_heylook_gemma4_runtime", False)
            and getattr(draft_tokens, "ndim", 0) == 2
            and draft_tokens.shape[0] == 1
            and getattr(target_hidden, "ndim", 0) == 3
            and target_hidden.shape[0] == 1
        ):
            n_draft = int(draft_tokens.shape[1])
            total_t = n_draft + 1
            if 2 <= total_t <= MTP_QMV_MAX_T and target_hidden.shape[1] >= total_t:
                positioned = bool(mtp._sampler_supports_positioned_target(sampler))
                sample_target = (
                    getattr(sampler, "sample_target", None) if positioned else None
                )
                if not positioned or (
                    callable(sample_target)
                    and row_ids is not None
                    and base_positions is not None
                ):
                    budget = int(budgets[0])
                    draft_list = [int(tok) for tok in draft_tokens[0].tolist()]
                    with mx.stream(mtp.generation_stream):
                        logits = mtp._mtp_logits_from_hidden(
                            lm, target_hidden[:, :total_t, :]
                        )
                        if logits.ndim == 3 and logits.shape[0] == 1:
                            logits = logits[0]
                        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                        if sample_target is not None:
                            positions = [
                                int(base_positions[0]) + pos for pos in range(total_t)
                            ]
                            target_tokens = sample_target(
                                logprobs,
                                row_ids=[int(row_ids[0])] * total_t,
                                positions=positions,
                            )
                        else:
                            target_tokens = sampler(logprobs)
                    mx.eval(target_tokens)
                    target_list = [int(tok) for tok in target_tokens.reshape(-1).tolist()]
                    accepted = 0
                    new_tokens = []
                    for pos, token in enumerate(target_list[:total_t]):
                        if pos < n_draft and token == draft_list[pos]:
                            accepted += 1
                            if len(new_tokens) < budget:
                                new_tokens.append(token)
                            continue
                        if len(new_tokens) < budget:
                            new_tokens.append(token)
                        break
                    return [accepted], [new_tokens]
        return upstream(
            lm,
            target_hidden,
            draft_tokens,
            sampler,
            budgets,
            row_ids=row_ids,
            base_positions=base_positions,
        )

    _heylook_mtp_walk_batch.__doc__ = upstream.__doc__
    setattr(_heylook_mtp_walk_batch, "_heylook_patch", PATCH_MTP_B1_WALK)
    setattr(_heylook_mtp_walk_batch, "_heylook_upstream", upstream)
    mtp._speculative_walk_batch_deferred_greedy = _heylook_mtp_walk_batch
    return True


def _install_mtp_left_padding_fast_path(mtp) -> bool:
    current = getattr(mtp, "_batch_cache_left_padding", None)
    if current is None:
        return False
    if getattr(current, "_heylook_patch", None) == PATCH_MTP_LEFT_PADDING:
        return True
    if tuple(inspect.signature(current).parameters) != MTP_LEFT_PADDING_PARAMS:
        logging.warning(
            "%s stood down: _batch_cache_left_padding signature changed",
            PATCH_MTP_LEFT_PADDING,
        )
        return False

    from .gemma4_vision import _is_unpadded_single_row_batch_cache

    upstream = current

    def _heylook_batch_cache_left_padding(prompt_cache):
        found_lp = False
        for cache_entry in prompt_cache:
            if getattr(cache_entry, "left_padding", None) is not None:
                found_lp = True
                if not _is_unpadded_single_row_batch_cache(cache_entry):
                    return upstream(prompt_cache)
        return None if found_lp else upstream(prompt_cache)

    _heylook_batch_cache_left_padding.__doc__ = upstream.__doc__
    setattr(_heylook_batch_cache_left_padding, "_heylook_patch", PATCH_MTP_LEFT_PADDING)
    setattr(_heylook_batch_cache_left_padding, "_heylook_upstream", upstream)
    mtp._batch_cache_left_padding = _heylook_batch_cache_left_padding
    return True


def install_mtp_single_row_fast_paths() -> bool:
    """Install single-row B=1 MTP walk and left-padding fast paths on ``mlx_vlm.speculative.mtp``."""
    try:
        import mlx.core as mx
        from mlx_vlm.speculative import mtp
    except ImportError:
        return False

    if not isinstance(getattr(mx, "array", None), type):
        return False
    walk_ok = _install_mtp_walk_fast_path(mtp, mx)
    pad_ok = _install_mtp_left_padding_fast_path(mtp)
    return bool(walk_ok and pad_ok)


def apply_to(bg) -> bool:
    """Install every heylook-owned generator/runtime optimization. Returns True when
    #2356 is in force on this generator; False for batched/drafting generators or
    generators without a prefix cache."""
    install_gated_delta_packed_kernel()
    install_mtp_single_row_fast_paths()
    coord = getattr(bg, "apc", None)
    if coord is None or not plain_single_row_generator(bg):
        return False
    return _patch_merge_rows(coord)
