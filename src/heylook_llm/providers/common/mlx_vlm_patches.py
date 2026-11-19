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
PATCH_RESTORE_BATCH = "heylook#apc-b1-batch-restore"
RESTORE_MERGE = "_merge_and_release_cache_rows"
MERGE_ROWS_PARAMS = ("picks", "prefix_lens", "kv_quant_config")  # bound method

PATCH_STORE_CHECKPOINT = "heylook#apc-trim-rotating-store"
STORE_CHECKPOINT = "store_checkpoint"
STORE_CHECKPOINT_PARAMS = ("token_ids", "prompt_cache", "extra_hash", "batch_idx")

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
_MTP_DRAFTER_CLASSES: dict[type, type] = {}


def _method_accepts_greedy(cls: type, method_name: str) -> bool:
    fn = getattr(cls, method_name, None)
    if not callable(fn):
        return False
    try:
        return "greedy" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _with_effective_greedy(
    args: tuple,
    kwargs: dict,
    sampler,
    greedy_pos_index: int,
    accepts_greedy: bool,
):
    has_pos = len(args) > greedy_pos_index
    arg_greedy = bool(args[greedy_pos_index]) if has_pos else bool(kwargs.get("greedy", False))
    eff_greedy = bool(arg_greedy or getattr(sampler, "_heylook_greedy", False))
    if eff_greedy:
        if has_pos:
            args = (*args[:greedy_pos_index], True, *args[greedy_pos_index + 1 :])
        elif accepts_greedy:
            kwargs["greedy"] = True
    return eff_greedy, args, kwargs


def _run_with_q_capture(sampler, fn, *args, **kwargs):
    if isinstance(getattr(sampler, "_heylook_captured_q", None), list):
        return fn(*args, **kwargs), None
    buf = []
    sampler._heylook_captured_q = buf
    try:
        out = fn(*args, **kwargs)
    finally:
        sampler._heylook_captured_q = None
    return out, buf


def bind_mtp_drafter(drafter) -> bool:
    """Bind cross-family non-greedy distribution capture onto an MTP drafter.

    Gemma 4 assistant drafters are handled by ``gemma4_runtime.bind_gemma4_drafter``;
    this dynamic per-instance subclass handles every other ``mtp`` drafter (such
    as Qwen 3.5/3.8, ERNIE, GLM, Inkling) so single-row speculative rounds run
    Leviathan rejection sampling via ``sampler._heylook_pending_draft``.
    Survives ``copy.copy(draft)`` in ``MTPBatch.__init__``.
    """
    if drafter is None or "unittest.mock" in type(drafter).__module__:
        return False
    if ".gemma4_assistant." in type(drafter).__module__:
        return False
    if getattr(drafter, "_heylook_mtp_drafter", False):
        return True

    base_cls = type(drafter)
    draft_fn = getattr(base_cls, "draft_block", None)
    reset_fn = getattr(base_cls, "reset", None)
    if not (callable(draft_fn) and callable(reset_fn)):
        return False
    try:
        params = tuple(inspect.signature(draft_fn).parameters)
    except (TypeError, ValueError):
        return False
    if params[:6] != (
        "self",
        "last_bonus",
        "hidden",
        "cache",
        "block_size",
        "sampler",
    ):
        logging.warning(
            "mtp drafter distribution capture stood down: unexpected draft_block signature"
        )
        return False

    sub_cls = _MTP_DRAFTER_CLASSES.get(base_cls)
    if sub_cls is None:
        has_prefill = callable(getattr(base_cls, "prefill_from_target_hidden", None))
        has_accept = callable(getattr(base_cls, "accept_verified_tokens", None))
        has_accept_batch = callable(getattr(base_cls, "accept_verified_tokens_batch", None))
        has_abort = callable(getattr(base_cls, "abort_draft_round", None))

        prefill_greedy = _method_accepts_greedy(base_cls, "prefill_from_target_hidden")
        accept_greedy = _method_accepts_greedy(base_cls, "accept_verified_tokens")
        accept_batch_greedy = _method_accepts_greedy(base_cls, "accept_verified_tokens_batch")
        draft_greedy = _method_accepts_greedy(base_cls, "draft_block")

        attrs: dict = {}

        def reset(self, *args, **kwargs):
            self._heylook_seed_q = None
            return base_cls.reset(self, *args, **kwargs)

        attrs["reset"] = reset

        if has_abort:

            def abort_draft_round(self, *args, **kwargs):
                self._heylook_seed_q = None
                return base_cls.abort_draft_round(self, *args, **kwargs)

            attrs["abort_draft_round"] = abort_draft_round

        if has_prefill:

            def prefill_from_target_hidden(
                self, input_ids, hidden, bonus_token, sampler, *args, **kwargs
            ):
                eff_greedy, args, kwargs = _with_effective_greedy(
                    args, kwargs, sampler, 1, prefill_greedy
                )
                if (
                    not eff_greedy
                    and getattr(hidden, "shape", (0,))[0] == 1
                    and callable(getattr(sampler, "_heylook_spec_probs", None))
                ):
                    out, buf = _run_with_q_capture(
                        sampler,
                        base_cls.prefill_from_target_hidden,
                        self,
                        input_ids,
                        hidden,
                        bonus_token,
                        sampler,
                        *args,
                        **kwargs,
                    )
                    if buf is not None:
                        self._heylook_seed_q = buf[-1] if len(buf) == 1 else None
                    return out
                if not isinstance(getattr(sampler, "_heylook_captured_q", None), list):
                    self._heylook_seed_q = None
                return base_cls.prefill_from_target_hidden(
                    self, input_ids, hidden, bonus_token, sampler, *args, **kwargs
                )

            attrs["prefill_from_target_hidden"] = prefill_from_target_hidden

        if has_accept:

            def accept_verified_tokens(
                self, verify_hidden, draft_tokens, accepted, new_tokens, sampler, *args, **kwargs
            ):
                eff_greedy, args, kwargs = _with_effective_greedy(
                    args, kwargs, sampler, 1, accept_greedy
                )
                if (
                    not eff_greedy
                    and getattr(verify_hidden, "shape", (0,))[0] == 1
                    and callable(getattr(sampler, "_heylook_spec_probs", None))
                ):
                    out, buf = _run_with_q_capture(
                        sampler,
                        base_cls.accept_verified_tokens,
                        self,
                        verify_hidden,
                        draft_tokens,
                        accepted,
                        new_tokens,
                        sampler,
                        *args,
                        **kwargs,
                    )
                    if buf is not None:
                        self._heylook_seed_q = buf[-1] if len(buf) == 1 else None
                    return out
                if not isinstance(getattr(sampler, "_heylook_captured_q", None), list):
                    self._heylook_seed_q = None
                return base_cls.accept_verified_tokens(
                    self, verify_hidden, draft_tokens, accepted, new_tokens, sampler, *args, **kwargs
                )

            attrs["accept_verified_tokens"] = accept_verified_tokens

        if has_accept_batch:

            def accept_verified_tokens_batch(
                self, verify_hidden, draft_tokens, accepted, new_tokens, sampler, *args, **kwargs
            ):
                eff_greedy, args, kwargs = _with_effective_greedy(
                    args, kwargs, sampler, 1, accept_batch_greedy
                )
                if (
                    not eff_greedy
                    and len(accepted) == 1
                    and callable(getattr(sampler, "_heylook_spec_probs", None))
                ):
                    out, buf = _run_with_q_capture(
                        sampler,
                        base_cls.accept_verified_tokens_batch,
                        self,
                        verify_hidden,
                        draft_tokens,
                        accepted,
                        new_tokens,
                        sampler,
                        *args,
                        **kwargs,
                    )
                    if buf is not None:
                        self._heylook_seed_q = buf[-1] if len(buf) == 1 else None
                    return out
                if not isinstance(getattr(sampler, "_heylook_captured_q", None), list):
                    self._heylook_seed_q = None
                return base_cls.accept_verified_tokens_batch(
                    self, verify_hidden, draft_tokens, accepted, new_tokens, sampler, *args, **kwargs
                )

            attrs["accept_verified_tokens_batch"] = accept_verified_tokens_batch

        def draft_block(
            self, last_bonus, hidden, cache, block_size, sampler, *args, **kwargs
        ):
            import mlx.core as mx

            if hasattr(sampler, "_heylook_pending_draft"):
                sampler._heylook_pending_draft = None
            eff_greedy, args, kwargs = _with_effective_greedy(
                args, kwargs, sampler, 1, draft_greedy
            )
            spec_probs = (
                None if eff_greedy else getattr(sampler, "_heylook_spec_probs", None)
            )
            had_seed = (
                getattr(self, "_seed_token", None) is not None
                and getattr(self, "_seed_hidden", None) is not None
            )
            seed_q = getattr(self, "_heylook_seed_q", None)
            self._heylook_seed_q = None
            if (
                not callable(spec_probs)
                or getattr(hidden, "shape", (0,))[0] != 1
                or int(block_size) <= 1
            ):
                return base_cls.draft_block(
                    self, last_bonus, hidden, cache, block_size, sampler, *args, **kwargs
                )

            out_toks, buf = _run_with_q_capture(
                sampler,
                base_cls.draft_block,
                self,
                last_bonus,
                hidden,
                cache,
                block_size,
                sampler,
                *args,
                **kwargs,
            )
            if buf is None:
                return out_toks
            if had_seed:
                q_rows = ([seed_q] + buf) if seed_q is not None else []
            else:
                q_rows = buf
            if (
                getattr(out_toks, "ndim", 0) == 2
                and int(out_toks.shape[0]) == 1
                and len(q_rows) == int(out_toks.shape[1])
                and int(out_toks.shape[1]) > 0
            ):
                q_mat = mx.concatenate(q_rows, axis=0)
                mx.async_eval(q_mat)
                sampler._heylook_pending_draft = (out_toks, q_mat)
            return out_toks

        attrs["draft_block"] = draft_block
        sub_cls = type(base_cls.__name__, (base_cls,), attrs)
        sub_cls.__qualname__ = base_cls.__qualname__
        sub_cls.__module__ = base_cls.__module__
        _MTP_DRAFTER_CLASSES[base_cls] = sub_cls

    drafter.__class__ = sub_cls
    drafter._heylook_seed_q = None
    drafter._heylook_mtp_drafter = True
    return True


def _apply_mtp_penalties(mx, logits_2d, draft_list: list[int], mtp_penalties):
    if not (isinstance(mtp_penalties, tuple) and len(mtp_penalties) == 2):
        return logits_2d
    raw_procs, gen_history = mtp_penalties
    if not raw_procs:
        return logits_2d
    rows = []
    for pos in range(int(logits_2d.shape[0])):
        hist = gen_history + draft_list[:pos]
        row = logits_2d[pos : pos + 1]
        if hist:
            tok_arr = mx.array(hist, dtype=mx.int32)
            for proc in raw_procs:
                row = proc(tok_arr, row)
        rows.append(row)
    return mx.concatenate(rows, axis=0)

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


def _drafted_single_row_generator(bg) -> bool:
    """Drafted checkpoint generators keep batch caches on restore, matching cold."""
    drafting = (getattr(bg, "draft_model", None) is not None
                and getattr(bg, "draft_kind", None) is not None)
    return (drafting and getattr(bg, "kv_bits", None) is None
            and hasattr(getattr(bg, "model", None), "make_cache"))


def single_restored_row(picks, kv_quant_config) -> bool:
    """One row, restored, with its own warm cache (a checkpoint hit), no KV
    quantization. Block-mode hits carry no warm cache and still merge."""
    return (len(picks) == 1 and picks[0] is not None
            and kv_quant_config is None
            and picks[0].get("warm_cache") is not None)


def _single_row_batch_caches(warm_cache):
    """Wrap single-row APC snapshots into Batch* caches with one batched sync.

    Upstream ``_merge_and_release_cache_rows`` re-allocates ``mx.zeros`` slabs,
    copies every layer already cloned by ``lookup_exact_cache``, drops
    pre-reserved ``KVCache`` capacity, and calls ``mx.eval`` once per layer.
    Returns ``None`` on any unsupported layout without mutating ``warm_cache``.
    """
    if not warm_cache:
        return None
    import mlx.core as mx
    from mlx_vlm.models.cache import (
        ArraysCache,
        BatchKVCache,
        BatchRotatingKVCache,
        KVCache,
        RotatingKVCache,
    )

    out = []
    eval_targets = []
    for c in warm_cache:
        t = type(c)
        if t is RotatingKVCache:
            max_size = int(getattr(c, "max_size", 0) or 0)
            keep = int(getattr(c, "keep", 0) or 0)
            if max_size <= 0 or keep != 0:
                return None
            length = int(c.size())
            bc = BatchRotatingKVCache(max_size, [0])
            if c.keys is not None and c.values is not None and length > 0:
                if (
                    int(c.keys.shape[0]) != 1
                    or int(c.values.shape[0]) != 1
                    or int(c.keys.shape[2]) < length
                    or int(c.values.shape[2]) < length
                ):
                    return None
                if int(c._idx) == int(c.keys.shape[2]) and int(c.keys.shape[2]) == length:
                    k, v = c.keys, c.values
                else:
                    k = c._temporal_order(c.keys)[..., -length:, :]
                    v = c._temporal_order(c.values)[..., -length:, :]
                    eval_targets.extend((k, v))
                bc.keys = k
                bc.values = v
                bc.offset = mx.array([int(c.offset)])
                bc._idx = length
                bc._offset = length
            eval_targets.extend((bc.left_padding, bc.offset))
            out.append(bc)
        elif t is KVCache:
            length = int(c.size())
            bc = BatchKVCache([0])
            if c.keys is not None and c.values is not None and length > 0:
                if (
                    int(c.keys.shape[0]) != 1
                    or int(c.values.shape[0]) != 1
                    or int(c.keys.shape[2]) < length
                    or int(c.values.shape[2]) < length
                ):
                    return None
                bc.keys = c.keys
                bc.values = c.values
                bc.offset = mx.array([length])
                bc._idx = length
            eval_targets.extend((bc.left_padding, bc.offset))
            out.append(bc)
        elif t is ArraysCache:
            if (
                c.batch_size != 1
                or getattr(c, "is_speculating", False)
                or getattr(c, "_left_padding", None) is not None
                or getattr(c, "_lengths", None) is not None
            ):
                return None
            for arr in c.cache:
                if arr is not None and (
                    not isinstance(arr, mx.array)
                    or arr.ndim < 1
                    or int(arr.shape[0]) != 1
                ):
                    return None
            ac = ArraysCache(len(c.cache))
            if c.empty():
                ac.left_padding = mx.array([0])
                eval_targets.append(ac.left_padding)
            else:
                ac.cache = list(c.cache)
            out.append(ac)
        else:
            return None

    if eval_targets:
        mx.eval(eval_targets)
    return out


def _patch_batch_restore(coord) -> bool:
    upstream = getattr(coord, RESTORE_MERGE, None)
    if getattr(upstream, "_heylook_patch", None) in (PATCH_2356, PATCH_RESTORE_BATCH):
        return True
    if (
        upstream is None
        or tuple(inspect.signature(upstream).parameters) != MERGE_ROWS_PARAMS
    ):
        logging.warning(
            "%s carry not applied: APCCoordinator.%s is gone or its signature changed",
            PATCH_RESTORE_BATCH,
            RESTORE_MERGE,
        )
        return False

    def merge_rows(picks, prefix_lens, *, kv_quant_config=None):
        if single_restored_row(picks, kv_quant_config) and len(prefix_lens) == 1:
            memory = getattr(coord.manager, "memory_plan", None)
            if memory is not None and len(prefix_lens) == len(getattr(memory, "lengths", ())):
                if not coord.prepare_prefill(
                    memory.lengths,
                    prefill_step_size=memory.chunk_size,
                    prefix_lengths=prefix_lens,
                    extra_reserve_bytes=memory.extra_reserve_bytes,
                ):
                    from mlx_vlm.apc import APCAdmissionError

                    raise APCAdmissionError("APC batch merge deferred by memory admission")
            try:
                warm_cache = picks[0]["warm_cache"]
                caches = _single_row_batch_caches(warm_cache)
                if caches is not None:
                    if isinstance(warm_cache, list):
                        for i in range(len(warm_cache)):
                            warm_cache[i] = None
                    with coord.manager.lock:
                        coord.manager.stats.restored_tokens += prefix_lens[0]
                    return caches, prefix_lens[0]
            except (AttributeError, KeyError, TypeError, IndexError, ValueError) as exc:
                logging.warning("%s fell through to upstream: %r", PATCH_RESTORE_BATCH, exc)
        return upstream(picks, prefix_lens, kv_quant_config=kv_quant_config)

    merge_rows.__doc__ = upstream.__doc__
    setattr(merge_rows, "_heylook_patch", PATCH_RESTORE_BATCH)
    setattr(merge_rows, "_heylook_upstream", upstream)
    setattr(coord, RESTORE_MERGE, merge_rows)
    return True


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


def _single_row_trimmed_views(prompt_cache, batch_idx):
    """Build single-row cache views with rotating layers trimmed to ``max_size``.

    Avoids both the redundant intermediate copy in upstream batch extraction
    (``extract_prompt_cache_from_batch`` + ``store_exact_cache``) and serial
    per-layer host syncs, while dropping chunk-boundary overshoot past
    ``max_size`` that ``BatchRotatingKVCache.merge`` never reads back.
    Returns ``None`` on any unsupported layout so the caller falls through to
    upstream.
    """
    if not prompt_cache or batch_idx not in (None, 0):
        return None
    import mlx.core as mx
    from mlx_vlm.models.cache import (
        ArraysCache,
        BatchKVCache,
        BatchRotatingKVCache,
        KVCache,
        RotatingKVCache,
    )

    meta_arrays = []
    for c in prompt_cache:
        t = type(c)
        if t is RotatingKVCache:
            max_size = int(getattr(c, "max_size", 0) or 0)
            keep = int(getattr(c, "keep", 0) or 0)
            if max_size <= 0 or keep < 0 or keep >= max_size:
                return None
            if c.keys is not None and int(c.keys.shape[0]) != 1:
                return None
        elif t is KVCache:
            if c.keys is not None and int(c.keys.shape[0]) != 1:
                return None
        elif t is BatchRotatingKVCache:
            max_size = int(getattr(c, "max_size", 0) or 0)
            if (
                max_size <= 0
                or getattr(c, "keep", 0)
                or not c.is_single_row()
                or getattr(c, "_lengths", None) is not None
                or getattr(c, "rotated", False)
            ):
                return None
            if c.keys is not None:
                if (
                    getattr(c.left_padding, "shape", None) != (1,)
                    or getattr(c.offset, "shape", None) != (1,)
                ):
                    return None
                if isinstance(c.left_padding, mx.array):
                    meta_arrays.append(c.left_padding)
                if isinstance(c.offset, mx.array):
                    meta_arrays.append(c.offset)
        elif t is BatchKVCache:
            if (
                int(getattr(c.offset, "size", 0)) != 1
                or getattr(c, "_right_padding", None) is not None
            ):
                return None
            if c.keys is not None:
                if getattr(c.left_padding, "shape", None) != (1,):
                    return None
                if isinstance(c.left_padding, mx.array):
                    meta_arrays.append(c.left_padding)
        elif t is ArraysCache:
            if c.batch_size != 1 or getattr(c, "is_speculating", False):
                return None
        else:
            return None

    if meta_arrays:
        mx.eval(meta_arrays)

    views = []
    for c in prompt_cache:
        t = type(c)
        if t is RotatingKVCache:
            if c.keys is None or c.values is None:
                views.append(c)
                continue
            ordered_k = c._temporal_order(c.keys)
            ordered_v = c._temporal_order(c.values)
            max_size = int(c.max_size)
            ordered_len = int(ordered_k.shape[2])
            if ordered_len > max_size:
                keep = int(c.keep)
                v = RotatingKVCache(max_size=max_size, keep=keep)
                if keep == 0:
                    v.keys = ordered_k[..., -max_size:, :]
                    v.values = ordered_v[..., -max_size:, :]
                else:
                    trim_size = ordered_len - max_size
                    v.keys = c._trim(trim_size, ordered_k)
                    v.values = c._trim(trim_size, ordered_v)
                v.offset = int(c.offset)
                v._idx = int(v.keys.shape[2])
                views.append(v)
            else:
                views.append(c)
        elif t is KVCache:
            views.append(c)
        elif t is BatchRotatingKVCache:
            max_size = int(c.max_size)
            v = RotatingKVCache(max_size=max_size)
            if c.keys is not None and c.values is not None:
                padding = max(0, int(c.left_padding[0].item()))
                offset = int(c.offset[0].item())
                end = int(c._idx)
                if end > int(c.keys.shape[2]) or padding > end:
                    return None
                start = max(padding, end - max_size)
                v.keys = c.keys[0:1, :, start:end, :]
                v.values = c.values[0:1, :, start:end, :]
                v.offset = offset
                v._idx = end - start
            views.append(v)
        elif t is BatchKVCache:
            v = KVCache()
            if c.keys is not None and c.values is not None:
                padding = int(c.left_padding[0].item())
                end = int(c._idx)
                if padding < 0 or end > int(c.keys.shape[2]) or padding > end:
                    return None
                v.keys = c.keys[0:1, :, padding:end, :]
                v.values = c.values[0:1, :, padding:end, :]
                v.offset = end - padding
            views.append(v)
        elif t is ArraysCache:
            v = ArraysCache(len(c.cache))
            v.cache = [arr[0:1] if arr is not None else None for arr in c.cache]
            views.append(v)
    return views


def _patch_store_checkpoint(coord) -> bool:
    upstream = getattr(coord, STORE_CHECKPOINT, None)
    if getattr(upstream, "_heylook_patch", None) == PATCH_STORE_CHECKPOINT:
        return True
    if (
        upstream is None
        or tuple(inspect.signature(upstream).parameters) != STORE_CHECKPOINT_PARAMS
    ):
        logging.warning(
            "%s carry not applied: APCCoordinator.%s is gone or its signature changed",
            PATCH_STORE_CHECKPOINT,
            STORE_CHECKPOINT,
        )
        return False

    def store_checkpoint(
        token_ids,
        prompt_cache,
        *,
        extra_hash: int = 0,
        batch_idx=None,
    ) -> bool:
        if not coord.enabled or not coord.is_checkpoint:
            return False
        if getattr(coord.manager, "disk", None) is not None:
            return upstream(
                token_ids, prompt_cache, extra_hash=extra_hash, batch_idx=batch_idx
            )
        try:
            views = _single_row_trimmed_views(prompt_cache, batch_idx)
            if views is not None:
                return coord.manager.store_exact_cache(
                    token_ids, views, extra_hash=extra_hash
                )
        except (AttributeError, KeyError, TypeError, IndexError, ValueError) as exc:
            logging.warning("%s fell through to upstream: %r", PATCH_STORE_CHECKPOINT, exc)
        return upstream(
            token_ids, prompt_cache, extra_hash=extra_hash, batch_idx=batch_idx
        )

    store_checkpoint.__doc__ = upstream.__doc__
    setattr(store_checkpoint, "_heylook_patch", PATCH_STORE_CHECKPOINT)
    setattr(store_checkpoint, "_heylook_upstream", upstream)
    setattr(coord, STORE_CHECKPOINT, store_checkpoint)
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
        pending = getattr(sampler, "_heylook_pending_draft", None)
        if hasattr(sampler, "_heylook_pending_draft"):
            sampler._heylook_pending_draft = None

        if (
            getattr(draft_tokens, "ndim", 0) == 2
            and draft_tokens.shape[0] == 1
            and getattr(target_hidden, "ndim", 0) == 3
            and target_hidden.shape[0] == 1
        ):
            n_draft = int(draft_tokens.shape[1])
            total_t = n_draft + 1
            if total_t >= 2 and target_hidden.shape[1] >= total_t:
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
                    mtp_penalties = getattr(sampler, "_heylook_mtp_penalties", None)
                    spec_probs = (
                        None
                        if sample_target is not None
                        else getattr(sampler, "_heylook_spec_probs", None)
                    )
                    if (
                        callable(spec_probs)
                        and isinstance(pending, tuple)
                        and len(pending) == 2
                        and getattr(pending[0], "shape", None) == draft_tokens.shape
                        and getattr(pending[1], "ndim", 0) == 2
                        and int(pending[1].shape[0]) == n_draft
                    ):
                        _, q_probs = pending
                        draft_1d = draft_tokens[0].astype(mx.int32)
                        with mx.stream(mtp.generation_stream):
                            logits = mtp._mtp_logits_from_hidden(
                                lm, target_hidden[:, :total_t, :]
                            )
                            if logits.ndim == 3 and logits.shape[0] == 1:
                                logits = logits[0]
                            logits = _apply_mtp_penalties(
                                mx, logits, draft_list, mtp_penalties
                            )
                            logprobs = logits - mx.logsumexp(
                                logits, axis=-1, keepdims=True
                            )
                            p_probs, _ = spec_probs(logprobs, already_logprobs=True)
                            if q_probs.shape[1] == p_probs.shape[1]:
                                p_sel = mx.take_along_axis(
                                    p_probs[:n_draft], draft_1d[:, None], axis=-1
                                ).squeeze(-1)
                                q_sel = mx.take_along_axis(
                                    q_probs, draft_1d[:, None], axis=-1
                                ).squeeze(-1)
                                u = mx.random.uniform(shape=(n_draft,))
                                rej = (u * q_sel) >= p_sel
                                rej_padded = mx.concatenate(
                                    [rej, mx.array([True])]
                                )
                                acc_idx = mx.argmax(rej_padded.astype(mx.int32))
                                q_padded = mx.concatenate(
                                    [
                                        q_probs,
                                        mx.zeros(
                                            (1, p_probs.shape[1]), dtype=p_probs.dtype
                                        ),
                                    ],
                                    axis=0,
                                )
                                p_row = p_probs[acc_idx]
                                q_row = q_padded[acc_idx]
                                res = mx.maximum(p_row - q_row, 0.0)
                                res_sum = mx.sum(res)
                                chosen = mx.where(res_sum > 1e-12, res, p_row)
                                log_dist = mx.where(
                                    chosen > 0, mx.log(chosen), -mx.inf
                                )
                                bonus_tok = mx.random.categorical(log_dist)
                                run_leviathan = True
                            else:
                                run_leviathan = False
                        if run_leviathan:
                            mx.eval(acc_idx, bonus_tok)
                            accepted = int(acc_idx.item())
                            bonus = int(bonus_tok.item())
                            new_tokens = draft_list[:accepted]
                            if len(new_tokens) < budget:
                                new_tokens.append(bonus)
                            return [accepted], [new_tokens[:budget]]

                    non_greedy = (
                        sample_target is None
                        and (
                            getattr(sampler, "_heylook_greedy", None) is False
                            or (
                                getattr(sampler, "_heylook_greedy", None) is not True
                                and "temp"
                                in getattr(
                                    getattr(sampler, "__code__", None),
                                    "co_freevars",
                                    (),
                                )
                            )
                        )
                    )
                    if non_greedy:
                        with mx.stream(mtp.generation_stream):
                            logits = mtp._mtp_logits_from_hidden(
                                lm, target_hidden[:, :total_t, :]
                            )
                            if logits.ndim == 3 and logits.shape[0] == 1:
                                logits = logits[0]
                            logits = _apply_mtp_penalties(
                                mx, logits, draft_list, mtp_penalties
                            )
                            logprobs = logits - mx.logsumexp(
                                logits, axis=-1, keepdims=True
                            )
                        mx.eval(logprobs)
                        accepted = 0
                        new_tokens = []
                        for pos in range(total_t):
                            with mx.stream(mtp.generation_stream):
                                target_token = sampler(logprobs[pos : pos + 1])
                            mx.eval(target_token)
                            token = int(target_token.reshape(-1).item())
                            if pos < n_draft and token == draft_list[pos]:
                                accepted += 1
                                if len(new_tokens) < budget:
                                    new_tokens.append(token)
                                continue
                            if len(new_tokens) < budget:
                                new_tokens.append(token)
                            break
                        return [accepted], [new_tokens]

                    with mx.stream(mtp.generation_stream):
                        logits = mtp._mtp_logits_from_hidden(
                            lm, target_hidden[:, :total_t, :]
                        )
                        if logits.ndim == 3 and logits.shape[0] == 1:
                            logits = logits[0]
                        logits = _apply_mtp_penalties(
                            mx, logits, draft_list, mtp_penalties
                        )
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
    if coord is None:
        return False
    is_ckpt = bool(getattr(coord, "enabled", False) and getattr(coord, "is_checkpoint", False))
    if is_ckpt:
        _patch_store_checkpoint(coord)
    if plain_single_row_generator(bg):
        return _patch_merge_rows(coord)
    if is_ckpt and _drafted_single_row_generator(bg):
        _patch_batch_restore(coord)
    return False
