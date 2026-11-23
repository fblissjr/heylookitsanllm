"""Heylook-owned runtime optimizations for Qwen 3.5 / 3.8 models and vision towers.

Speeds up dense and MoE Qwen 3.5/3.8 checkpoints (with or without MTP) without
maintaining a fork of mlx-vlm:
- Packed 8-row ``gated_delta_with_states`` Metal kernel: mirrors upstream's
  packed 8-value-row Dk=128 kernel while emitting bounded speculative state
  snapshots ``(B, StateT, Hv, Dv, Dk)`` for verify transactions.
- Fast B=1 speculative verify path: replaces upstream's ``T``-step SDPA
  concat loop with a single batched ``mask="causal"`` SDPA call (`sdpa_split`
  on Metal) and dispatches 8-bit affine projections adaptively per ``T`` via
  ``native_8bit_qmv_beats_hoisted_for_t`` (`_hoisted_8bit_qmv_kernel` when it
  beats native ``quantized_matmul``, native ``qmv_mma`` once ``T`` crosses the
  hardware threshold).
- Vision tower fast path (`qwen3_vl` / `qwen3_5`): keeps grid interpolation on
  GPU instead of 8 host ``.tolist()`` round-trips per call, precomputes rotary
  ``cos``/``sin`` and ``cu_seqlens`` splits once outside the 27-block loop, and
  avoids ``mx.split``/``mx.concatenate`` on single-image inputs.
"""

from functools import lru_cache
import inspect
import logging
from typing import Any, Optional

from .gemma4_runtime import (
    MAX_VERIFY_T,
    MIN_VERIFY_T,
    _has_additive_bias,
    gemma4_qmv_fused,
    gemma4_qmv_multi,
    native_8bit_qmv_beats_hoisted_for_t,
)

PATCH_GDN_PACKED_STATES = "heylook#qwen3_5-gdn-packed-states"
_PATCH_VERIFY = "heylook#qwen3_5-verify"
_PATCH_LOGITS = "heylook#qwen3_5-logits"
_PATCH_ARGMAX = "heylook#qwen3_5-argmax"

GDN_WITH_STATES_PARAMS = (
    "q",
    "k",
    "v",
    "a",
    "b",
    "A_log",
    "dt_bias",
    "state",
    "mask",
    "use_kernel",
    "state_steps",
)

_QWEN_LM_MODULES = (
    ".qwen3_5.",
    ".qwen3_5_moe.",
    ".qwen3_5_text.",
    ".qwen3_5_moe_text.",
)
_QWEN_VISION_MODULES = (
    ".qwen3_5.vision",
    ".qwen3_vl.vision",
    ".qwen3_5_moe.vision",
)

_VISION_CLASSES: dict[type, type] = {}
_PACKED_GDN_WITH_STATES_KERNEL = None
_PACKED_GDN_WITH_STATES_CHECKED = False


def _is_mock(obj: Any) -> bool:
    return obj is None or "unittest.mock" in type(obj).__module__


def _make_gated_delta_packed_with_states_kernel():
    """Packed 8-row scalar-gate Dk=128 Metal kernel retaining ``StateT`` snapshots."""
    global _PACKED_GDN_WITH_STATES_KERNEL, _PACKED_GDN_WITH_STATES_CHECKED
    if _PACKED_GDN_WITH_STATES_CHECKED:
        return _PACKED_GDN_WITH_STATES_KERNEL
    _PACKED_GDN_WITH_STATES_CHECKED = True

    import mlx.core as mx

    if not mx.metal.is_available():
        return None

    source = r"""
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

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto states_ = states + (((b_idx * StateT) * Hv + hv_idx) * Dv + dv_idx) * Dk +
            lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

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

          if (t < StateT) {
            for (int i = 0; i < values_per_lane; ++i) {
              states_[i] = static_cast<StT>(state[i]);
            }
            states_ += Hv * Dv * Dk;
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
    _PACKED_GDN_WITH_STATES_KERNEL = mx.fast.metal_kernel(
        name="qwen3_5_gated_delta_with_states_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out", "states"],
        source=source,
    )
    return _PACKED_GDN_WITH_STATES_KERNEL


def install_qwen3_5_packed_gdn_with_states() -> bool:
    """Install packed 8-row ``gated_delta_update_with_states`` on ``qwen3_5.gated_delta``."""
    try:
        import mlx.core as mx
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
    except ImportError:
        return False

    if hasattr(gd, "_gated_delta_with_states_kernel_packed"):
        return True

    upstream = getattr(gd, "gated_delta_update_with_states", None)
    if getattr(upstream, "_heylook_patch", None) == PATCH_GDN_PACKED_STATES:
        return True
    if (
        not callable(upstream)
        or tuple(inspect.signature(upstream).parameters) != GDN_WITH_STATES_PARAMS
        or not callable(getattr(gd, "_compute_g_beta", None))
    ):
        logging.warning(
            "%s stood down: qwen3_5.gated_delta.gated_delta_update_with_states signature changed",
            PATCH_GDN_PACKED_STATES,
        )
        return False

    compute_g_beta = gd._compute_g_beta
    kernel_packed = _make_gated_delta_packed_with_states_kernel()
    if kernel_packed is None:
        return False

    def gated_delta_update_with_states(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state=None,
        mask=None,
        use_kernel: bool = True,
        state_steps: Optional[int] = None,
    ):
        B, T, Hk, Dk = k.shape
        Hv, Dv = v.shape[-2:]
        eff_steps = T if state_steps is None else int(state_steps)
        if (
            mask is None
            and use_kernel
            and T > 1
            and 0 < eff_steps <= T
            and Dk == 128
            and Dv % 8 == 0
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            g, beta = compute_g_beta(A_log, a, b, dt_bias)
            eff_state = (
                mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
                if state is None
                else state
            )
            if (
                g.ndim == 3
                and g.dtype == mx.float32
                and eff_state.dtype == mx.float32
            ):
                input_type = q.dtype
                state_type = eff_state.dtype
                return kernel_packed(
                    inputs=[q, k, v, g, beta, eff_state, T],
                    template=[
                        ("InT", input_type),
                        ("StT", state_type),
                        ("Dk", Dk),
                        ("Dv", Dv),
                        ("Hk", Hk),
                        ("Hv", Hv),
                        ("StateT", eff_steps),
                    ],
                    grid=(32, Dv // 8, B * Hv),
                    threadgroup=(32, 2, 1),
                    output_shapes=[
                        (B, T, Hv, Dv),
                        eff_state.shape,
                        (B, eff_steps, Hv, Dv, Dk),
                    ],
                    output_dtypes=[input_type, state_type, state_type],
                )
        return upstream(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state=state,
            mask=mask,
            use_kernel=use_kernel,
            state_steps=state_steps,
        )

    setattr(gated_delta_update_with_states, "_heylook_patch", PATCH_GDN_PACKED_STATES)
    setattr(gated_delta_update_with_states, "_heylook_upstream", upstream)
    gd.gated_delta_update_with_states = gated_delta_update_with_states
    return True


def _is_8bit_affine_unbiased(linear, x_dtype) -> bool:
    import mlx.nn as nn

    if not isinstance(linear, nn.QuantizedLinear):
        return False
    if (
        getattr(linear, "bits", None) != 8
        or getattr(linear, "mode", None) != "affine"
        or _has_additive_bias(linear)
    ):
        return False
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    return (
        scales is not None
        and biases is not None
        and scales.dtype == x_dtype
        and biases.dtype == x_dtype
    )


def _verify_linear(verifier, linear, x, *, native_wins: bool):
    if _is_8bit_affine_unbiased(linear, x.dtype):
        if native_wins:
            return linear(x)
        out = gemma4_qmv_multi(linear, x, min_out_rows=1)
        if out is not None:
            return out
    return verifier._linear(linear, x)


def _verify_linears(verifier, linears, x, *, native_wins: bool):
    if len(linears) <= 4 and all(_is_8bit_affine_unbiased(l, x.dtype) for l in linears):
        if native_wins:
            return tuple(l(x) for l in linears)
        out = gemma4_qmv_fused(tuple(linears), x, min_out_rows=1)
        if out is not None:
            return out
    return verifier._linears(linears, x)


def _qwen3_5_lm_looks_compatible(lm) -> bool:
    if _is_mock(lm):
        return False
    mod = type(lm).__module__
    if not any(tag in mod for tag in _QWEN_LM_MODULES):
        return False
    for name, params in (
        ("speculative_verify_hidden", ("inputs", "cache")),
        ("speculative_logits_from_hidden", ("hidden",)),
        ("speculative_argmax_from_hidden", ("hidden",)),
    ):
        fn = getattr(lm, name, None)
        try:
            if not callable(fn) or tuple(inspect.signature(fn).parameters) != params:
                return False
        except (TypeError, ValueError):
            return False

    inner = getattr(lm, "model", None)
    layers = getattr(inner, "layers", None)
    if (
        inner is None
        or not isinstance(layers, list)
        or not layers
        or not callable(getattr(inner, "embed_tokens", None))
        or not callable(getattr(inner, "norm", None))
        or not isinstance(getattr(inner, "fa_idx", None), int)
        or not isinstance(getattr(inner, "ssm_idx", None), int)
        or getattr(lm, "args", None) is None
    ):
        return False

    for layer in layers:
        if not hasattr(layer, "is_linear"):
            return False
        if not all(
            hasattr(layer, attr)
            for attr in ("input_layernorm", "post_attention_layernorm", "mlp")
        ):
            return False
        if layer.is_linear:
            la = getattr(layer, "linear_attn", None)
            req = (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "conv_kernel_size",
                "conv_dim",
                "conv1d",
                "key_dim",
                "num_k_heads",
                "num_v_heads",
                "head_k_dim",
                "head_v_dim",
                "_normalize_qk",
                "A_log",
                "dt_bias",
                "norm",
                "out_proj",
            )
            if la is None or not all(hasattr(la, a) for a in req):
                return False
        else:
            sa = getattr(layer, "self_attn", None)
            req = (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "_prepare_projected_qkv",
                "scale",
                "rotary_emb",
            )
            if sa is None or not all(hasattr(sa, a) for a in req):
                return False
    return True


def _bind_qwen3_5_verify(lm) -> bool:
    import types

    if not _qwen3_5_lm_looks_compatible(lm):
        return False

    cur_verify = getattr(lm, "speculative_verify_hidden", None)
    if getattr(getattr(cur_verify, "__func__", None), "_heylook_patch", None) == _PATCH_VERIFY:
        return True

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.activations import swiglu
        from mlx_vlm.models.base import (
            cache_position_offset,
            scaled_dot_product_attention,
        )
        from mlx_vlm.models.cache import ArraysCache, KVCache
        from mlx_vlm.models.qwen3_5 import language as qwen_lang
        from mlx_vlm.speculative.cache_state import (
            iter_leaf_caches,
            start_speculative_cache,
        )
    except ImportError:
        return False

    verifier = getattr(qwen_lang, "_EXACT_SPECULATIVE_VERIFIER", None)
    advance_lengths_info = getattr(qwen_lang, "_qwen3_5_advance_lengths_info", None)
    gdn_update = getattr(qwen_lang, "gated_delta_update", None)
    if verifier is None or not callable(advance_lengths_info) or not callable(gdn_update):
        return False

    upstream_verify = cur_verify
    upstream_logits = lm.speculative_logits_from_hidden
    upstream_argmax = lm.speculative_argmax_from_hidden

    def _can_run_fast_b1(lm_self, inputs, cache) -> bool:
        if (
            getattr(inputs, "ndim", 0) != 2
            or int(inputs.shape[0]) != 1
            or not (MIN_VERIFY_T <= int(inputs.shape[1]) <= MAX_VERIFY_T)
        ):
            return False
        if not cache or len(cache) != len(lm_self.model.layers):
            return False
        if getattr(lm_self, "_rope_deltas", None) is None:
            return False
        fa_idx = int(lm_self.model.fa_idx)
        if not (0 <= fa_idx < len(cache)) or cache[fa_idx] is None:
            return False
        for c in iter_leaf_caches(cache):
            if type(c) not in (KVCache, ArraysCache):
                return False
            if getattr(c, "bits", None) is not None:
                return False
            if getattr(c, "left_padding", None) is not None:
                return False
            if getattr(c, "_lengths", None) is not None:
                return False
        return True

    def _compute_verify_position_ids(lm_self, seq_len: int, fa_cache):
        cache_offset = int(cache_position_offset(fa_cache))
        rd = lm_self._rope_deltas
        if isinstance(rd, mx.array) and rd.ndim > 0 and int(rd.shape[0]) > 1:
            rd = rd[:1]
        delta = cache_offset + rd
        if isinstance(delta, mx.array):
            if delta.ndim == 0:
                delta = delta.reshape(1, 1)
            elif delta.ndim == 1:
                delta = delta[:, None]
        else:
            delta = mx.array(delta).reshape(1, 1)
        pos_ids = mx.arange(seq_len).reshape(1, -1) + delta
        stored_pos = getattr(lm_self, "_position_ids", None)
        if stored_pos is not None and getattr(stored_pos, "ndim", 0) == 3:
            pos_ids = mx.broadcast_to(pos_ids[None, ...], (3, 1, seq_len))
        return pos_ids

    def _pick_probe_module(lm_self):
        head = getattr(lm_self, "lm_head", None)
        if isinstance(head, nn.QuantizedLinear) and getattr(head, "bits", None) == 8:
            return head
        first_mlp = getattr(lm_self.model.layers[0], "mlp", None)
        gate_proj = getattr(first_mlp, "gate_proj", None)
        if isinstance(gate_proj, nn.QuantizedLinear) and getattr(gate_proj, "bits", None) == 8:
            return gate_proj
        return None

    def speculative_verify_hidden(self, inputs, cache):
        if not _can_run_fast_b1(self, inputs, cache):
            return upstream_verify(inputs, cache)

        seq_len = int(inputs.shape[1])
        fa_cache = cache[self.model.fa_idx]
        position_ids = _compute_verify_position_ids(self, seq_len, fa_cache)

        probe_mod = _pick_probe_module(self)
        native_wins = (
            native_8bit_qmv_beats_hoisted_for_t(probe_mod, seq_len)
            if probe_mod is not None
            else False
        )

        tx = start_speculative_cache(cache, seq_len)
        try:
            h = self.model.embed_tokens(inputs)
            position_embeddings = None
            for layer in self.model.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
                    break

            for layer, layer_cache in zip(self.model.layers, cache):
                normed = layer.input_layernorm(h)
                if layer.is_linear:
                    la = layer.linear_attn
                    mixed_qkv, z, b, a = _verify_linears(
                        verifier,
                        (la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a),
                        normed,
                        native_wins=native_wins,
                    )
                    z = z.reshape(1, seq_len, -1, la.head_v_dim)
                    if layer_cache is not None and layer_cache[0] is not None:
                        conv_state = layer_cache[0]
                        if conv_state.shape[0] != 1:
                            conv_state = mx.zeros(
                                (1, la.conv_kernel_size - 1, la.conv_dim),
                                dtype=normed.dtype,
                            )
                    else:
                        conv_state = mx.zeros(
                            (1, la.conv_kernel_size - 1, la.conv_dim),
                            dtype=normed.dtype,
                        )
                    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
                    if layer_cache is not None:
                        layer_cache.update_window(
                            0, conv_input, la.conv_kernel_size - 1, lengths=layer_cache.lengths
                        )
                    conv_out = nn.silu(la.conv1d(conv_input))
                    q, k, v = [
                        val.reshape(1, seq_len, heads, width)
                        for val, heads, width in zip(
                            mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
                            [la.num_k_heads, la.num_k_heads, la.num_v_heads],
                            [la.head_k_dim, la.head_k_dim, la.head_v_dim],
                        )
                    ]
                    q, k = la._normalize_qk(q, k)
                    gdn_out, _ = gdn_update(
                        q,
                        k,
                        v,
                        a,
                        b,
                        la.A_log,
                        la.dt_bias,
                        mask=None,
                        use_kernel=not la.training,
                        cache=layer_cache,
                    )
                    if layer_cache is not None and hasattr(layer_cache, "advance"):
                        layer_cache.advance(seq_len)
                        advance_lengths_info(layer_cache, seq_len)
                    normed_gdn = la.norm(gdn_out, z).reshape(1, seq_len, -1)
                    residual = _verify_linear(
                        verifier, la.out_proj, normed_gdn, native_wins=native_wins
                    )
                else:
                    sa = layer.self_attn
                    q_proj_out, keys, values = _verify_linears(
                        verifier,
                        (sa.q_proj, sa.k_proj, sa.v_proj),
                        normed,
                        native_wins=native_wins,
                    )
                    queries, keys, values, gate, _ = sa._prepare_projected_qkv(
                        q_proj_out,
                        keys,
                        values,
                        layer_cache,
                        position_ids,
                        position_embeddings,
                        None,
                    )
                    attn_out = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=layer_cache,
                        scale=sa.scale,
                        mask="causal",
                    )
                    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
                    residual = _verify_linear(
                        verifier,
                        sa.o_proj,
                        attn_out * mx.sigmoid(gate),
                        native_wins=native_wins,
                    )

                h = h + residual
                mlp_in = layer.post_attention_layernorm(h)
                mlp = layer.mlp
                if (
                    not hasattr(mlp, "switch_mlp")
                    and hasattr(mlp, "gate_proj")
                    and hasattr(mlp, "up_proj")
                    and hasattr(mlp, "down_proj")
                ):
                    gate_out, up_out = _verify_linears(
                        verifier, (mlp.gate_proj, mlp.up_proj), mlp_in, native_wins=native_wins
                    )
                    mlp_out = _verify_linear(
                        verifier, mlp.down_proj, swiglu(gate_out, up_out), native_wins=native_wins
                    )
                else:
                    mlp_out = verifier._feed_forward(mlp, mlp_in)
                h = h + mlp_out

            return self.model.norm(h), {}, tx
        except BaseException:
            tx.abort()
            raise

    def _head_logits_b1_8bit(lm_self, hidden):
        if (
            getattr(lm_self.args, "tie_word_embeddings", True)
            or getattr(hidden, "ndim", 0) != 3
            or int(hidden.shape[0]) != 1
            or not (MIN_VERIFY_T <= int(hidden.shape[1]) <= MAX_VERIFY_T)
        ):
            return None
        head = getattr(lm_self, "lm_head", None)
        if not _is_8bit_affine_unbiased(head, hidden.dtype):
            return None
        seq_len = int(hidden.shape[1])
        if native_8bit_qmv_beats_hoisted_for_t(head, seq_len):
            return head(hidden)
        return gemma4_qmv_multi(head, hidden, min_out_rows=1)

    def speculative_logits_from_hidden(self, hidden):
        logits = _head_logits_b1_8bit(self, hidden)
        if logits is not None:
            return logits
        return upstream_logits(hidden)

    def speculative_argmax_from_hidden(self, hidden):
        logits = _head_logits_b1_8bit(self, hidden)
        if logits is not None:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        return upstream_argmax(hidden)

    setattr(speculative_verify_hidden, "_heylook_patch", _PATCH_VERIFY)
    setattr(speculative_verify_hidden, "_heylook_upstream", upstream_verify)
    setattr(speculative_logits_from_hidden, "_heylook_patch", _PATCH_LOGITS)
    setattr(speculative_logits_from_hidden, "_heylook_upstream", upstream_logits)
    setattr(speculative_argmax_from_hidden, "_heylook_patch", _PATCH_ARGMAX)
    setattr(speculative_argmax_from_hidden, "_heylook_upstream", upstream_argmax)

    lm.speculative_verify_hidden = types.MethodType(speculative_verify_hidden, lm)
    lm.speculative_logits_from_hidden = types.MethodType(speculative_logits_from_hidden, lm)
    lm.speculative_argmax_from_hidden = types.MethodType(speculative_argmax_from_hidden, lm)
    return True


def _vision_looks_compatible(vt) -> bool:
    if _is_mock(vt):
        return False
    if not any(tag in type(vt).__module__ for tag in _QWEN_VISION_MODULES):
        return False
    cls = type(vt)
    try:
        interp_params = tuple(inspect.signature(cls.fast_pos_embed_interpolate).parameters)
        call_params = tuple(inspect.signature(cls.__call__).parameters)
    except (TypeError, ValueError):
        return False
    if interp_params != ("self", "grid_thw") or call_params[:3] != (
        "self",
        "hidden_states",
        "grid_thw",
    ):
        return False
    req = (
        "patch_embed",
        "pos_embed",
        "rot_pos_emb",
        "blocks",
        "merger",
        "deepstack_visual_indexes",
        "deepstack_merger_list",
        "spatial_merge_size",
        "num_grid_per_side",
    )
    if not all(hasattr(vt, a) for a in req):
        return False
    for blk in getattr(vt, "blocks", ()):
        attn = getattr(blk, "attn", None)
        if (
            attn is None
            or not all(hasattr(blk, a) for a in ("norm1", "norm2", "mlp"))
            or not all(hasattr(attn, a) for a in ("qkv", "proj", "num_heads", "scale"))
        ):
            return False
    return True


def bind_qwen3_5_vision(model) -> bool:
    """Bind sync-free GPU grid interpolation and hoisted vision trig onto ``vision_tower``."""
    vt = getattr(model, "vision_tower", None)
    if vt is None or _is_mock(vt):
        return False
    if getattr(vt, "_heylook_qwen3_5_vision", False):
        return True
    if not _vision_looks_compatible(vt):
        return False

    try:
        from itertools import accumulate
        import mlx.core as mx
        from mlx_vlm.models.base import ensure_fused_sdpa
        from mlx_vlm.models.rope_utils import rotate_half
    except ImportError:
        return False

    base_cls = type(vt)
    sub_cls = _VISION_CLASSES.get(base_cls)
    if sub_cls is None:

        class _HeylookQwenVisionModel(base_cls):
            def fast_pos_embed_interpolate(self, grid_thw):
                grid_thw_list = grid_thw.tolist()
                if not grid_thw_list:
                    return super().fast_pos_embed_interpolate(grid_thw)

                idx_parts = [[] for _ in range(4)]
                weight_parts = [[] for _ in range(4)]
                num_grid = self.num_grid_per_side

                for _t, h, w in grid_thw_list:
                    h = int(h)
                    w = int(w)

                    h_idxs = mx.linspace(0, num_grid - 1, h)
                    w_idxs = mx.linspace(0, num_grid - 1, w)

                    h_idxs_floor = h_idxs.astype(mx.int32)
                    w_idxs_floor = w_idxs.astype(mx.int32)
                    h_idxs_ceil = mx.minimum(h_idxs_floor + 1, num_grid - 1)
                    w_idxs_ceil = mx.minimum(w_idxs_floor + 1, num_grid - 1)

                    dh = h_idxs - h_idxs_floor.astype(mx.float32)
                    dw = w_idxs - w_idxs_floor.astype(mx.float32)

                    base_h = h_idxs_floor * num_grid
                    base_h_ceil = h_idxs_ceil * num_grid

                    indices = (
                        (base_h[:, None] + w_idxs_floor[None, :]).reshape(-1),
                        (base_h[:, None] + w_idxs_ceil[None, :]).reshape(-1),
                        (base_h_ceil[:, None] + w_idxs_floor[None, :]).reshape(-1),
                        (base_h_ceil[:, None] + w_idxs_ceil[None, :]).reshape(-1),
                    )
                    weights = (
                        ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                        ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                        (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                        (dh[:, None] * dw[None, :]).reshape(-1),
                    )
                    for corner in range(4):
                        idx_parts[corner].append(indices[corner])
                        weight_parts[corner].append(weights[corner])

                idx_rows = [
                    parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
                    for parts in idx_parts
                ]
                weight_rows = [
                    parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
                    for parts in weight_parts
                ]
                idx_tensor = mx.stack(idx_rows, axis=0).astype(mx.int32)
                pos_embeds = self.pos_embed(idx_tensor)
                weight_tensor = mx.stack(weight_rows, axis=0).astype(pos_embeds.dtype)
                pos_embeds = pos_embeds * weight_tensor[:, :, None]
                patch_pos_embeds = (
                    pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
                )

                split_sizes = [int(h * w) for _t, h, w in grid_thw_list]
                if len(split_sizes) > 1:
                    split_indices = list(accumulate(split_sizes[:-1]))
                    patch_pos_embeds_split = mx.split(
                        patch_pos_embeds, split_indices, axis=0
                    )
                else:
                    patch_pos_embeds_split = [patch_pos_embeds]

                merge_size = self.spatial_merge_size
                patch_pos_embeds_permute = []
                for pos_embed, (t, h, w) in zip(patch_pos_embeds_split, grid_thw_list):
                    t, h, w = int(t), int(h), int(w)
                    feature_dim = pos_embed.shape[-1]
                    pos_embed = mx.tile(pos_embed, (t, 1))
                    pos_embed = pos_embed.reshape(t, h, w, feature_dim)
                    pos_embed = (
                        pos_embed.reshape(
                            t,
                            h // merge_size,
                            merge_size,
                            w // merge_size,
                            merge_size,
                            feature_dim,
                        )
                        .transpose(0, 1, 3, 2, 4, 5)
                        .reshape(-1, feature_dim)
                    )
                    patch_pos_embeds_permute.append(pos_embed)

                return mx.concatenate(patch_pos_embeds_permute)

            def __call__(self, hidden_states, grid_thw, **kwargs):
                hidden_states = self.patch_embed(hidden_states)
                pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
                hidden_states = hidden_states + pos_embeds
                rotary_pos_emb = self.rot_pos_emb(grid_thw)

                seq_len = hidden_states.shape[0]
                hidden_states = hidden_states.reshape(seq_len, -1)
                rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

                batch_size = grid_thw.shape[0]
                cu_seqlens = []
                for i in range(batch_size):
                    seq_len_i = grid_thw[i, 1] * grid_thw[i, 2]
                    cu_seqlens.append(mx.repeat(seq_len_i, int(grid_thw[i, 0])))
                cu_seqlens = mx.concatenate(cu_seqlens)
                cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
                cu_seqlens = mx.pad(
                    cu_seqlens, (1, 0), mode="constant", constant_values=0
                )

                # Hoist host sync + trig cos/sin out of every block's attention call.
                split_indices = cu_seqlens[1:-1].tolist()
                cos = mx.expand_dims(
                    mx.tile(mx.expand_dims(mx.cos(rotary_pos_emb), axis=1), (1, 1, 2)),
                    axis=0,
                )
                sin = mx.expand_dims(
                    mx.tile(mx.expand_dims(mx.sin(rotary_pos_emb), axis=1), (1, 1, 2)),
                    axis=0,
                )

                deepstack_set = set(self.deepstack_visual_indexes)
                deepstack_feature_lists = []
                for layer_num, blk in enumerate(self.blocks):
                    attn = blk.attn
                    qkv = (
                        attn.qkv(blk.norm1(hidden_states))
                        .reshape(seq_len, 3, attn.num_heads, -1)
                        .transpose(1, 0, 2, 3)
                    )
                    q, k, v = mx.split(qkv, 3)
                    q_dtype = q.dtype
                    q = ((q * cos) + (rotate_half(q) * sin)).astype(q_dtype)
                    k = ((k * cos) + (rotate_half(k) * sin)).astype(q_dtype)

                    q = q.transpose(0, 2, 1, 3)
                    k = k.transpose(0, 2, 1, 3)
                    v = v.transpose(0, 2, 1, 3)

                    if not split_indices:
                        attn_out = ensure_fused_sdpa(q, k, v, attn.scale)
                    else:
                        splits = [
                            mx.split(tensor, split_indices, axis=2)
                            for tensor in (q, k, v)
                        ]
                        attn_out = mx.concatenate(
                            [
                                ensure_fused_sdpa(qs, ks, vs, attn.scale)
                                for qs, ks, vs in zip(*splits)
                            ],
                            axis=2,
                        )
                    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, -1)
                    hidden_states = hidden_states + attn.proj(attn_out)
                    hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

                    if layer_num in deepstack_set:
                        idx = self.deepstack_visual_indexes.index(layer_num)
                        deepstack_feature_lists.append(
                            self.deepstack_merger_list[idx](hidden_states)
                        )

                hidden_states = self.merger(hidden_states)
                return hidden_states, deepstack_feature_lists

        _HeylookQwenVisionModel.__name__ = base_cls.__name__
        _HeylookQwenVisionModel.__qualname__ = base_cls.__qualname__
        _HeylookQwenVisionModel.__module__ = base_cls.__module__
        _VISION_CLASSES[base_cls] = _HeylookQwenVisionModel
        sub_cls = _HeylookQwenVisionModel

    vt.__class__ = sub_cls
    vt._heylook_qwen3_5_vision = True
    return True


def bind_qwen3_5_runtime(model) -> bool:
    """Bind all Heylook-owned Qwen 3.5 / 3.8 runtime optimizations onto ``model``."""
    if _is_mock(model):
        return False
    lm = getattr(model, "language_model", None)
    if _is_mock(lm):
        return False
    lm_mod = type(lm).__module__
    if not any(tag in lm_mod for tag in _QWEN_LM_MODULES):
        return False

    gdn_ok = install_qwen3_5_packed_gdn_with_states()
    verify_ok = _bind_qwen3_5_verify(lm)
    vision_ok = bind_qwen3_5_vision(model)

    inner = getattr(lm, "model", None)
    model_mod = type(model).__module__
    if (
        getattr(inner, "embed_tokens", None) is not None
        and callable(getattr(lm, "get_rope_index", None))
        and (".qwen3_5." in model_mod or ".qwen3_5_moe." in model_mod or any(tag in lm_mod for tag in _QWEN_LM_MODULES))
    ):
        lm._heylook_qwen3_5_suffix_embed = True

    bound = bool(gdn_ok or verify_ok or vision_ok)
    if bound:
        lm._heylook_qwen3_5_runtime = True
    return bound
