"""Heylook-owned runtime optimizations for Gemma 4 target models and MTP drafters.

Speeds up both Gemma 4 31B-Dense and 26B-A4B MoE (with or without MTP) without
maintaining a fork of mlx-vlm:
- ``gemma4_qmv_multi`` / ``gemma4_qmv_fused``: dispatches multi-token
  verification projections (2 <= T <= 8) with wide output matrices (N >= 32768)
  through ``quantized_verifier``'s split-K QMV Metal kernels for both
  ``K=2816`` (26B) and ``K=5376`` (31B), bypassing upstream's ``K % 512 == 0``
  and ``QuantizedEmbedding`` exclusions. Fires on both models' 262k LM head
  and on 31B's 60 fused ``gate_proj + up_proj`` MLP pairs (43008 rows), while
  leaving narrow matrices on MLX's faster ``qmm`` path.
- Scalar RoPE offset on ``Attention`` layers: avoids per-layer
  ``mx.array(cache.offset)`` construction and CPU-GPU syncs for single-row
  unpadded caches, including transaction-safe delta tracking on
  ``BatchRotatingKVCache``.
- Cached pre-scaled norm weights on ``Router`` layers (26B MoE, 30 layers).
- Unpadded B=1 view extraction in ``LanguageModel.shared_kv_from_cache`` without
  ``left_padding.tolist()`` syncs.
- Dynamic-subclass ``Gemma4AssistantDraftModel.draft_block`` hoisting
  ``make_drafter_masks`` out of the K-step draft loop and keeping scalar
  positions free of ``.item()`` syncs across ``copy()``.
"""

import inspect
import logging
from typing import Any, Optional

from .gemma4_vision import _is_unpadded_single_row_batch_cache, bind_text_masks

MIN_QMV_OUT_ROWS = 32768
MIN_VERIFY_T = 2
MAX_VERIFY_T = 8
SUPPORTED_AFFINE_BITS = (4, 8)
SUPPORTED_GROUP_SIZES = (32, 64, 128)
N_TILE_ALIGNMENT = 8

_ATTENTION_CLASSES: dict[type, type] = {}
_MLP_CLASSES: dict[type, type] = {}
_ROUTER_CLASSES: dict[type, type] = {}
_DRAFTER_CLASSES: dict[type, type] = {}

_PATCH_SHARED_KV = "heylook#gemma4-shared-kv"
_PATCH_LOGITS = "heylook#gemma4-logits"


def _is_mock(obj: Any) -> bool:
    return obj is None or "unittest.mock" in type(obj).__module__


def _can_run_qmv(x) -> bool:
    import mlx.core as mx

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    return (
        x.ndim == 3
        and MIN_VERIFY_T <= x.shape[1] <= MAX_VERIFY_T
        and x.dtype in (mx.bfloat16, mx.float16)
    )


def _quantized_module_spec(mod, expected_dtype, expected_k: int):
    import mlx.core as mx

    bits = getattr(mod, "bits", None)
    group_size = getattr(mod, "group_size", None)
    if (
        getattr(mod, "mode", None) != "affine"
        or bits not in SUPPORTED_AFFINE_BITS
        or group_size not in SUPPORTED_GROUP_SIZES
    ):
        return None

    weight = getattr(mod, "weight", None)
    scales = getattr(mod, "scales", None)
    biases = getattr(mod, "biases", None)
    if weight is None or scales is None or biases is None:
        return None
    if (
        weight.ndim != 2
        or weight.dtype != mx.uint32
        or scales.dtype != expected_dtype
        or biases.dtype != expected_dtype
    ):
        return None

    pack_factor = 32 // bits
    values_per_thread = pack_factor * 2
    block_size = values_per_thread * 32
    n_size = int(weight.shape[0])
    k_size = int(weight.shape[1]) * pack_factor
    if (
        k_size != expected_k
        or k_size % block_size != 0
        or n_size % N_TILE_ALIGNMENT != 0
        or group_size % values_per_thread != 0
    ):
        return None
    if scales.shape != (n_size, k_size // group_size) or biases.shape != scales.shape:
        return None
    return bits, group_size, n_size, k_size, weight, scales, biases


def _has_additive_bias(mod) -> bool:
    return isinstance(mod, dict) and "bias" in mod or (
        hasattr(mod, "__contains__")
        and callable(getattr(mod, "get", None))
        and "bias" in mod
    )


def gemma4_qmv_multi(
    mod, x, *, min_out_rows: int = MIN_QMV_OUT_ROWS
):
    """Run split-K multi-token QMV for eligible affine 4/8-bit layers and tied heads."""
    if not _can_run_qmv(x):
        return None
    try:
        from mlx_vlm.models.quantized_verifier import _target_verify_qmv_kernel
    except ImportError:
        return None
    import mlx.core as mx

    batch, verify_t, k_in = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    spec = _quantized_module_spec(mod, x.dtype, k_in)
    if spec is None:
        return None
    bits, group_size, n_size, k_size, weight, scales, biases = spec
    if n_size < min_out_rows:
        return None

    kernel = _target_verify_qmv_kernel(
        bits, group_size, x.dtype, verify_t, k_size, n_size
    )
    out = kernel(
        inputs=[mx.contiguous(x), weight, scales, biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", verify_t),
            ("K_SIZE", k_size),
            ("N_SIZE", n_size),
        ],
        grid=(32, 2 * (n_size // 8), batch),
        threadgroup=(32, 2, 1),
        output_shapes=[(batch, verify_t, n_size)],
        output_dtypes=[x.dtype],
    )[0]
    if _has_additive_bias(mod):
        out = out + mod["bias"]
    return out


def gemma4_qmv_fused(
    linears, x, *, min_out_rows: int = MIN_QMV_OUT_ROWS
):
    """Run fused split-K multi-token QMV over 2-4 sibling affine quantized projections."""
    if not (2 <= len(linears) <= 4) or not _can_run_qmv(x):
        return None
    try:
        from mlx_vlm.models.quantized_verifier import _target_verify_fused_qmv_kernel
    except ImportError:
        return None
    import mlx.core as mx
    import mlx.nn as nn

    batch, verify_t, k_in = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    specs = []
    for linear in linears:
        if not isinstance(linear, nn.QuantizedLinear) or _has_additive_bias(linear):
            return None
        spec = _quantized_module_spec(linear, x.dtype, k_in)
        if spec is None:
            return None
        specs.append(spec)

    bits0, gs0 = specs[0][0], specs[0][1]
    if any(s[0] != bits0 or s[1] != gs0 for s in specs[1:]):
        return None
    n_sizes = tuple(s[2] for s in specs)
    total_n = sum(n_sizes)
    if total_n < min_out_rows:
        return None

    kernel = _target_verify_fused_qmv_kernel(
        bits0, gs0, x.dtype, verify_t, k_in, n_sizes
    )
    inputs = [mx.contiguous(x)]
    for _, _, _, _, weight, scales, biases in specs:
        inputs.extend((weight, scales, biases))
    out = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("VERIFY_T", verify_t),
            ("K_SIZE", k_in),
            ("N_SIZE", total_n),
        ],
        grid=(32, 2 * (total_n // 8), batch),
        threadgroup=(32, 2, 1),
        output_shapes=[(batch, verify_t, total_n)],
        output_dtypes=[x.dtype],
    )[0]
    split_indices = []
    offset = 0
    for n_size in n_sizes[:-1]:
        offset += n_size
        split_indices.append(offset)
    return tuple(mx.split(out, split_indices, axis=-1))


def _scalar_rope_offset(cache):
    """Return an int RoPE offset for unpadded B=1 caches without array/sync churn.

    Keeps transaction rollback safety on ``BatchRotatingKVCache``: cached
    ``_heylook_rot_delta`` tracks ``offset[0] - _offset`` directly on the cache
    object rather than keying on ``id(left_padding)``, because
    ``_RotatingCacheTransaction.abort()`` clones array fields while keeping
    scalar fields intact.
    """
    if cache is None:
        return 0
    off = getattr(cache, "offset", None)
    if isinstance(off, int):
        return off

    import mlx.core as mx

    # If padded right-alignment lengths are active (B > 1 prompt processing),
    # invalidate any cached rotation delta before checking single-row eligibility.
    if getattr(cache, "_lengths", None) is not None:
        if hasattr(cache, "_heylook_rot_delta"):
            cache._heylook_rot_delta = None
        return mx.array(off)

    if not _is_unpadded_single_row_batch_cache(cache):
        if getattr(cache, "_heylook_rot_delta", None) is not None:
            cache._heylook_rot_delta = None
        return mx.array(off)

    # BatchKVCache (no max_size): unpadded B=1 holds offset == _idx algebraically.
    if not hasattr(cache, "max_size") and isinstance(getattr(cache, "_idx", None), int):
        return cache._idx

    rot_off = getattr(cache, "_offset", None)
    if hasattr(cache, "max_size") and isinstance(rot_off, int):
        delta = getattr(cache, "_heylook_rot_delta", None)
        if delta is None:
            if (
                cache.keys is None
                and not getattr(cache, "rotated", False)
                and rot_off == 0
            ):
                delta = 0
            else:
                delta = int(mx.array(off).reshape(-1)[0].item()) - rot_off
            cache._heylook_rot_delta = delta
        return rot_off + delta

    return mx.array(off)


def _attention_looks_compatible(attn) -> bool:
    cls = type(attn)
    try:
        params = tuple(inspect.signature(cls.__call__).parameters)
    except (TypeError, ValueError):
        return False
    if params[:6] != ("self", "x", "mask", "cache", "shared_kv", "offset"):
        return False
    required = ("q_proj", "o_proj", "q_norm", "rope", "use_k_eq_v", "n_heads", "n_kv_heads", "head_dim", "scale")
    if not all(hasattr(attn, attr) for attr in required):
        return False
    if hasattr(attn, "k_proj") and not getattr(attn, "kv_shared_only", False):
        if not (hasattr(attn, "k_norm") and hasattr(attn, "v_norm")):
            return False
        if not attn.use_k_eq_v and not hasattr(attn, "v_proj"):
            return False
    return True


def _bind_attention_subclass(attn) -> bool:
    if getattr(attn, "_heylook_gemma4_attn", False):
        return True
    if not _attention_looks_compatible(attn):
        return False

    from mlx_vlm.models.base import scaled_dot_product_attention

    base_cls = type(attn)
    sub_cls = _ATTENTION_CLASSES.get(base_cls)
    if sub_cls is None:

        class _HeylookGemma4Attention(base_cls):
            def __call__(
                self,
                x,
                mask=None,
                cache=None,
                shared_kv=None,
                offset=None,
            ):
                batch_size, seq_len, _ = x.shape
                queries = self.q_proj(x).reshape(
                    batch_size, seq_len, self.n_heads, self.head_dim
                )
                queries = self.q_norm(queries)

                if shared_kv is not None:
                    keys, values = shared_kv
                else:
                    keys = self.k_proj(x).reshape(
                        batch_size, seq_len, self.n_kv_heads, self.head_dim
                    )
                    if self.use_k_eq_v:
                        values = keys
                    else:
                        values = self.v_proj(x).reshape(
                            batch_size, seq_len, self.n_kv_heads, self.head_dim
                        )
                    offset = _scalar_rope_offset(cache)

                    keys = self.k_norm(keys).transpose(0, 2, 1, 3)
                    keys = self.rope(keys, offset=offset)

                    values = self.v_norm(values).transpose(0, 2, 1, 3)
                    if cache is not None:
                        keys, values = cache.update_and_fetch(keys, values)

                queries = queries.transpose(0, 2, 1, 3)
                queries = self.rope(queries, offset=offset)

                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
                return self.o_proj(output), (keys, values), offset

        _HeylookGemma4Attention.__name__ = base_cls.__name__
        _HeylookGemma4Attention.__qualname__ = base_cls.__qualname__
        _HeylookGemma4Attention.__module__ = base_cls.__module__
        _ATTENTION_CLASSES[base_cls] = _HeylookGemma4Attention
        sub_cls = _HeylookGemma4Attention

    attn.__class__ = sub_cls
    attn._heylook_gemma4_attn = True
    return True


def _bind_mlp_subclass(mlp, *, min_out_rows: int) -> bool:
    import mlx.nn as nn

    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if not (
        isinstance(gate_proj, nn.QuantizedLinear)
        and isinstance(up_proj, nn.QuantizedLinear)
        and down_proj is not None
    ):
        return False
    total_n = int(gate_proj.weight.shape[0]) + int(up_proj.weight.shape[0])
    if total_n < min_out_rows:
        return False

    mlp._heylook_mlp_min_out_rows = min_out_rows
    if getattr(mlp, "_heylook_gemma4_mlp", False):
        return True

    try:
        from mlx_vlm.models.gemma4.language import geglu
    except ImportError:
        return False

    base_cls = type(mlp)
    sub_cls = _MLP_CLASSES.get(base_cls)
    if sub_cls is None:

        class _HeylookGemma4MLP(base_cls):
            def __call__(self, x):
                threshold = getattr(self, "_heylook_mlp_min_out_rows", MIN_QMV_OUT_ROWS)
                fused = gemma4_qmv_fused(
                    (self.gate_proj, self.up_proj), x, min_out_rows=threshold
                )
                if fused is not None:
                    gate, up = fused
                    return self.down_proj(geglu(gate, up))
                return super().__call__(x)

        _HeylookGemma4MLP.__name__ = base_cls.__name__
        _HeylookGemma4MLP.__qualname__ = base_cls.__qualname__
        _HeylookGemma4MLP.__module__ = base_cls.__module__
        _MLP_CLASSES[base_cls] = _HeylookGemma4MLP
        sub_cls = _HeylookGemma4MLP

    mlp.__class__ = sub_cls
    mlp._heylook_gemma4_mlp = True
    return True


def _bind_router_subclass(router) -> bool:
    if getattr(router, "_heylook_gemma4_router", False):
        return True
    required = ("eps", "proj", "scale", "per_expert_scale", "_root_size", "config")
    if not all(hasattr(router, attr) for attr in required) or not hasattr(
        router.config, "top_k_experts"
    ):
        return False

    import mlx.core as mx

    base_cls = type(router)
    sub_cls = _ROUTER_CLASSES.get(base_cls)
    if sub_cls is None:

        class _HeylookGemma4Router(base_cls):
            def __call__(self, x):
                scale = self.scale
                root_size = self._root_size
                if (
                    getattr(self, "_heylook_scale_id", None) != id(scale)
                    or getattr(self, "_heylook_root_size", None) != root_size
                ):
                    norm_weight = scale * root_size
                    self._heylook_scaled_norm_weight = norm_weight
                    self._heylook_scale_id = id(scale)
                    self._heylook_root_size = root_size
                else:
                    norm_weight = self._heylook_scaled_norm_weight

                x = mx.fast.rms_norm(x, norm_weight, self.eps)
                expert_scores = self.proj(x)
                top_k = self.config.top_k_experts
                top_k_indices = mx.stop_gradient(
                    mx.argpartition(expert_scores, kth=-top_k, axis=-1)[..., -top_k:]
                )
                top_k_weights = mx.take_along_axis(
                    expert_scores, top_k_indices, axis=-1
                )
                top_k_weights = mx.softmax(top_k_weights, axis=-1)
                top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
                return top_k_indices, top_k_weights

        _HeylookGemma4Router.__name__ = base_cls.__name__
        _HeylookGemma4Router.__qualname__ = base_cls.__qualname__
        _HeylookGemma4Router.__module__ = base_cls.__module__
        _ROUTER_CLASSES[base_cls] = _HeylookGemma4Router
        sub_cls = _HeylookGemma4Router

    router.__class__ = sub_cls
    router._heylook_gemma4_router = True
    return True


def bind_gemma4_text_layers(
    text_model, *, min_out_rows: int = MIN_QMV_OUT_ROWS
) -> bool:
    """Bind per-layer Gemma 4 Attention, wide MLP, and Router fast paths."""
    if _is_mock(text_model):
        return False
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, list) or not layers:
        return False

    bound_any = False
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and ".gemma4." in type(attn).__module__:
            if _bind_attention_subclass(attn):
                bound_any = True
            else:
                logging.warning(
                    "gemma4 attention fast path stood down: unexpected Attention shape"
                )
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and ".gemma4." in type(mlp).__module__:
            if _bind_mlp_subclass(mlp, min_out_rows=min_out_rows):
                bound_any = True
        router = getattr(layer, "router", None)
        if router is not None and ".gemma4." in type(router).__module__:
            if _bind_router_subclass(router):
                bound_any = True
            else:
                logging.warning(
                    "gemma4 router fast path stood down: unexpected Router shape"
                )
    return bound_any


def _bind_logits_from_hidden(lm, *, min_out_rows: int) -> bool:
    import types

    lm._heylook_min_out_rows = min_out_rows
    current = getattr(lm, "logits_from_hidden", None)
    if getattr(getattr(current, "__func__", None), "_heylook_patch", None) == _PATCH_LOGITS:
        return True
    if not callable(current) or tuple(inspect.signature(current).parameters) != ("hidden",):
        return False
    text_model = getattr(lm, "model", None)
    if getattr(text_model, "embed_tokens", None) is None:
        return False
    try:
        from mlx_vlm.models.gemma4.language import logit_softcap
    except ImportError:
        return False

    def logits_from_hidden(self, hidden):
        threshold = getattr(self, "_heylook_min_out_rows", MIN_QMV_OUT_ROWS)
        logits = gemma4_qmv_multi(
            self.model.embed_tokens, hidden, min_out_rows=threshold
        )
        if logits is None:
            logits = self.model.embed_tokens.as_linear(hidden)
        if self.final_logit_softcapping is not None:
            logits = logit_softcap(self.final_logit_softcapping, logits)
        return logits

    setattr(logits_from_hidden, "_heylook_patch", _PATCH_LOGITS)
    lm.logits_from_hidden = types.MethodType(logits_from_hidden, lm)
    return True


def _extract_unpadded_single_row_kv(cache):
    import mlx.core as mx
    from mlx_vlm.models.cache import BatchRotatingKVCache

    if isinstance(cache, BatchRotatingKVCache):
        if cache.rotated:
            if cache.keys.shape[2] != cache.max_size:
                return None
            origin = cache._idx % cache.max_size
            if origin == 0:
                return cache.keys, cache.values
            return (
                mx.roll(cache.keys, -origin, axis=2),
                mx.roll(cache.values, -origin, axis=2),
            )
        end = cache._idx
        start = max(0, end - cache.max_size)
        return cache.keys[..., start:end, :], cache.values[..., start:end, :]
    return cache.keys[..., : cache._idx, :], cache.values[..., : cache._idx, :]


def _bind_shared_kv_from_cache(lm) -> bool:
    import types

    current = getattr(lm, "shared_kv_from_cache", None)
    if getattr(getattr(current, "__func__", None), "_heylook_patch", None) == _PATCH_SHARED_KV:
        return True
    if not callable(current) or tuple(inspect.signature(current).parameters) != ("caches",):
        return False

    import mlx.core as mx
    from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache

    upstream = current

    def shared_kv_from_cache(self, caches):
        selected = {
            layer.layer_type: cache
            for layer, cache in zip(self.model.layers, caches)
            if cache is not None
        }
        shared = {}
        for layer_type, cache in selected.items():
            if not isinstance(cache, (BatchKVCache, BatchRotatingKVCache)):
                return upstream(caches)
            if cache.keys is None or cache.values is None:
                continue
            if not _is_unpadded_single_row_batch_cache(cache):
                return upstream(caches)
            kv = _extract_unpadded_single_row_kv(cache)
            if kv is None:
                return upstream(caches)
            shared[layer_type] = kv
        return shared

    setattr(shared_kv_from_cache, "_heylook_patch", _PATCH_SHARED_KV)
    setattr(shared_kv_from_cache, "_heylook_upstream", upstream)
    lm.shared_kv_from_cache = types.MethodType(shared_kv_from_cache, lm)
    return True


def bind_gemma4_runtime(
    model, *, min_out_rows: int = MIN_QMV_OUT_ROWS
) -> bool:
    """Bind all heylook-owned Gemma 4 runtime optimizations onto a target model."""
    bind_text_masks(model)

    import mlx.core as mx

    if not isinstance(getattr(mx, "array", None), type):
        return False
    lm = getattr(model, "language_model", None)
    text_model = getattr(lm, "model", None)
    if _is_mock(lm) or _is_mock(text_model):
        return False
    if ".gemma4." not in type(text_model).__module__:
        return False

    layers_ok = bind_gemma4_text_layers(text_model, min_out_rows=min_out_rows)
    logits_ok = _bind_logits_from_hidden(lm, min_out_rows=min_out_rows)
    kv_ok = _bind_shared_kv_from_cache(lm)
    if not (logits_ok and kv_ok):
        logging.warning(
            "gemma4 runtime fast path stood down on language_model (logits=%s, shared_kv=%s)",
            logits_ok,
            kv_ok,
        )
    lm._heylook_gemma4_runtime = bool(logits_ok)
    return bool(layers_ok or logits_ok or kv_ok)


def _drafter_looks_compatible(drafter) -> bool:
    try:
        params = tuple(inspect.signature(type(drafter).draft_block).parameters)
    except (TypeError, ValueError):
        return False
    if params != (
        "self",
        "last_bonus",
        "hidden",
        "cache",
        "block_size",
        "sampler",
        "token_dtype",
        "greedy",
    ):
        return False
    text_cfg = getattr(getattr(drafter, "config", None), "text_config", None)
    inner = getattr(drafter, "model", None)
    return (
        hasattr(drafter, "pre_projection")
        and hasattr(drafter, "post_projection")
        and hasattr(text_cfg, "sliding_window")
        and isinstance(getattr(inner, "layers", None), list)
        and callable(getattr(inner, "norm", None))
        and hasattr(inner, "embed_tokens")
    )


def bind_gemma4_drafter(
    drafter, *, min_out_rows: int = MIN_QMV_OUT_ROWS
) -> bool:
    """Bind per-instance Gemma 4 MTP drafter optimizations via cached subclassing.

    Dynamic subclassing survives ``copy.copy(draft)`` in ``MTPBatch.__init__``
    without mutating upstream ``Gemma4AssistantDraftModel`` globally.
    """
    if _is_mock(drafter):
        return False
    if ".gemma4_assistant." not in type(drafter).__module__:
        return False
    if getattr(drafter, "_heylook_gemma4_drafter", False):
        return True
    if not _drafter_looks_compatible(drafter):
        logging.warning(
            "gemma4 drafter fast path stood down: drafter is not the shape it was written for"
        )
        return False

    try:
        import mlx.core as mx
        from mlx_vlm.speculative.drafters.gemma4_assistant import masks as drafter_masks
    except ImportError:
        return False

    make_drafter_masks = getattr(drafter_masks, "make_drafter_masks", None)
    if not callable(make_drafter_masks):
        return False

    bind_gemma4_text_layers(drafter.model, min_out_rows=min_out_rows)

    base_cls = type(drafter)
    sub_cls = _DRAFTER_CLASSES.get(base_cls)
    if sub_cls is None:

        class _HeylookGemma4DraftModel(base_cls):
            def draft_block(
                self,
                last_bonus,
                hidden,
                cache,
                block_size: int,
                sampler,
                token_dtype: mx.Dtype = mx.int32,
                greedy: bool = False,
            ) -> mx.array:
                pos = self._position
                is_single_pos = isinstance(pos, int) or getattr(pos, "shape", None) == (1,)
                is_single_bonus = (
                    isinstance(last_bonus, int)
                    or getattr(last_bonus, "shape", None) == (1,)
                )
                if (
                    self._shared_kv is None
                    or self._input_embed is None
                    or block_size <= 1
                    or not is_single_pos
                    or not is_single_bonus
                    or getattr(hidden, "shape", (0,))[0] != 1
                ):
                    return super().draft_block(
                        last_bonus,
                        hidden,
                        cache,
                        block_size,
                        sampler,
                        token_dtype=token_dtype,
                        greedy=greedy,
                    )

                del cache
                pos_offset = pos if isinstance(pos, int) else int(pos.item())
                shared_kv = self._shared_kv
                tok = (
                    mx.array([[last_bonus]], dtype=token_dtype)
                    if isinstance(last_bonus, int)
                    else last_bonus[:, None].astype(token_dtype)
                )
                h_prev = hidden
                tokens = []
                masks = None
                sliding_window = self.config.text_config.sliding_window
                kv_valid_len = self._kv_valid_len

                for _ in range(block_size - 1):
                    tok_embed = self._input_embed(tok) * self._input_embed_scale
                    inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)
                    h = self.pre_projection(inputs_embeds)
                    if masks is None:
                        masks = drafter_masks.make_drafter_masks(
                            shared_kv,
                            query_len=h.shape[1],
                            query_offset=pos_offset,
                            sliding_window=sliding_window,
                            dtype=h.dtype,
                            kv_valid_len=kv_valid_len,
                        )
                    for layer in self.model.layers:
                        h, _, _ = layer(
                            h,
                            mask=masks[layer.layer_type],
                            cache=None,
                            per_layer_input=None,
                            shared_kv=shared_kv[layer.layer_type],
                            offset=pos_offset,
                        )
                    h = self.model.norm(h)
                    h_prev = self.post_projection(h)
                    if greedy and self.masked_embedding is not None:
                        tok = self.masked_embedding.argmax(
                            h, self.model.embed_tokens.weight
                        )
                    else:
                        logits = (
                            self._lm_head_fn(h)
                            if self._lm_head_fn is not None
                            else self.model.embed_tokens.as_linear(h)
                        )
                        tok = sampler(logits)
                    tokens.append(tok)

                return mx.concatenate(tokens, axis=1)

        _HeylookGemma4DraftModel.__name__ = base_cls.__name__
        _HeylookGemma4DraftModel.__qualname__ = base_cls.__qualname__
        _HeylookGemma4DraftModel.__module__ = base_cls.__module__
        _DRAFTER_CLASSES[base_cls] = _HeylookGemma4DraftModel
        sub_cls = _HeylookGemma4DraftModel

    drafter.__class__ = sub_cls
    drafter._heylook_gemma4_drafter = True
    return True
