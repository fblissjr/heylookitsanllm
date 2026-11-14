"""Shared Gemma 4 model customizations for generative and embedding paths."""

import inspect
import logging
import types

VISUAL_PATCH_KEYS = (
    ("pixel_values", "image_position_ids"),
    ("pixel_values_videos", "video_position_ids"),
)


def _is_unpadded_single_row_batch_cache(c) -> bool:
    """True when ``c`` is a batch-1 cache with no left or pending right padding."""
    lp = getattr(c, "left_padding", None)
    if lp is None or getattr(lp, "shape", None) != (1,):
        return False
    if getattr(c, "_right_padding", None) is not None or getattr(c, "_lengths", None) is not None:
        c._heylook_unpadded_b1 = False
        c._heylook_lp_id = None
        return False
    lp_id = id(lp)
    cached = getattr(c, "_heylook_unpadded_b1", None)
    if cached is not None:
        if getattr(c, "_heylook_lp_id", None) == lp_id:
            return cached
        # BatchRotatingKVCache rebinds self.left_padding (-= S) when trimming or
        # rotating, which only makes left_padding more negative while keys stay resident.
        if cached and c.keys is not None and type(c).__name__ == "BatchRotatingKVCache":
            return True
    ok = int(lp.item()) <= 0
    c._heylook_unpadded_b1 = ok
    c._heylook_lp_id = lp_id
    return ok


def _cache_has_positive_offset(c) -> bool:
    """Check ``offset > 0`` without forcing a GPU-to-CPU sync on single-row caches."""
    if c is None:
        return False
    int_off = getattr(c, "_offset", None)
    if isinstance(int_off, int):
        if int_off <= 0:
            return False
        if _is_unpadded_single_row_batch_cache(c):
            return True
    off = getattr(c, "offset", 0)
    if isinstance(off, int):
        return off > 0
    if (
        isinstance(getattr(off, "shape", None), tuple)
        and off.shape == (1,)
        and _is_unpadded_single_row_batch_cache(c)
    ):
        idx = getattr(c, "_idx", None)
        if isinstance(idx, int):
            return idx > 0
    import mlx.core as mx

    return int(mx.max(mx.array(off)).item()) > 0


def gemma4_make_masks(self, h, cache, mm_token_type_ids=None):
    """Gemma 4 text attention masks without per-step syncs or redundant 4D masks.

    Upstream (mlx-vlm ``gemma4/language.py`` ``Gemma4TextModel._make_masks``)
    runs synchronous ``.item()`` checks on ``mm_token_type_ids`` and
    ``c.offset`` on prefill/verification steps, and lets ``BatchKVCache`` /
    ``BatchRotatingKVCache`` materialize 4D boolean masks even for an unpadded
    single row. This preserves the 4D sliding mask once ``BatchRotatingKVCache``
    rotates or crosses ``window_size`` with ``N > 1``, which
    ``Gemma4ExactSpeculativeVerifier`` requires past the window boundary.
    heylook-owned (VISION principle 6): pinned against upstream in
    ``test_mlxvlm_surface.TestOwnedGemma4TextMasks``.
    """
    import mlx.core as mx
    from mlx_vlm.models.base import create_attention_mask, create_causal_mask
    from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    n = h.shape[1]
    use_bidirectional_vision = (
        n > 1
        and mm_token_type_ids is not None
        and getattr(self.config, "use_bidirectional_attention", None) == "vision"
        and bool(mx.any((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item())
    )

    mask = {}
    masks = []
    for layer, c in zip(self.layers, cache):
        lt = layer.layer_type
        if lt not in mask:
            if lt == "full_attention":
                if not use_bidirectional_vision and (
                    c is None
                    or type(c) is KVCache
                    or (type(c) is BatchKVCache and _is_unpadded_single_row_batch_cache(c))
                ):
                    m = None if n == 1 else "causal"
                else:
                    return_array = (
                        use_bidirectional_vision
                        or getattr(c, "left_padding", None) is not None
                    )
                    m = create_attention_mask(h, c, return_array=return_array)
            elif lt == "sliding_attention":
                unpadded_b1 = (
                    not use_bidirectional_vision
                    and type(c) is BatchRotatingKVCache
                    and c.max_size <= self.window_size
                    and _is_unpadded_single_row_batch_cache(c)
                )
                if unpadded_b1 and (
                    n == 1
                    or (
                        not c.rotated
                        and min(c.max_size - 1, c._offset) + n <= self.window_size
                    )
                ):
                    m = None if n == 1 else "causal"
                else:
                    return_array = (
                        (n > 1 and _cache_has_positive_offset(c))
                        or use_bidirectional_vision
                    )
                    m = create_attention_mask(
                        h, c, window_size=self.window_size, return_array=return_array
                    )
            else:
                m = create_attention_mask(h, c, return_array=use_bidirectional_vision)

            if use_bidirectional_vision and isinstance(m, str) and m == "causal":
                window = self.window_size if lt == "sliding_attention" else None
                m = create_causal_mask(n, window_size=window)
            if use_bidirectional_vision and isinstance(m, mx.array):
                m = self._apply_blockwise_bidirectional_overlay(m, mm_token_type_ids)
            mask[lt] = m
        masks.append(mask[lt])
    return masks


def bind_text_masks(model) -> bool:
    """Bind ``gemma4_make_masks`` onto this Gemma 4 text model instance.

    Stands down silently on unrelated architectures, text-only ``gemma4_text``,
    embeddings, and mock MLX environments. Logs a warning and stands down if a
    multimodal Gemma 4 text model changes the signature or attributes this
    helper expects.
    """
    import mlx.core as mx

    if not isinstance(getattr(mx, "array", None), type):
        return False
    text_model = getattr(getattr(model, "language_model", None), "model", None)
    if text_model is None or "unittest.mock" in type(text_model).__module__:
        return False

    mod = type(text_model).__module__
    if ".gemma4." not in mod:
        return False

    method = getattr(text_model, "_make_masks", None)
    if getattr(method, "__func__", None) is gemma4_make_masks:
        return True

    valid_method = (
        callable(method)
        and tuple(inspect.signature(method).parameters)
        == ("h", "cache", "mm_token_type_ids")
    )
    valid_attrs = (
        callable(getattr(text_model, "_apply_blockwise_bidirectional_overlay", None))
        and isinstance(getattr(text_model, "window_size", None), int)
        and isinstance(getattr(text_model, "layers", None), list)
    )
    if not (valid_method and valid_attrs):
        logging.warning(
            "gemma4 text mask fast path stood down: text model is not the shape it was written for"
        )
        return False
    text_model._make_masks = types.MethodType(gemma4_make_masks, text_model)
    return True


def gathered_position_embeddings(self, patch_positions, padding_positions):
    """Gemma 4 vision position lookup as a row gather.

    Upstream (mlx-vlm gemma4/vision.py ``_position_embeddings``) builds a
    float32 one-hot over the 10240-row table and multiplies it in; a 32-frame
    window peaks several GB above resident for that alone. The gather gives
    bit-identical rows (internal/claude/embeddinggemma2_probes/posemb.py,
    posemb_memory.py). heylook-owned (VISION principle 6, owner 2026-11-04):
    no upstream PR; test_mlxvlm_surface pins it against the upstream method.
    """
    import mlx.core as mx

    pos = mx.maximum(patch_positions, 0)  # padding rows carry -1; masked below
    table = self.position_embedding_table
    rows = table[0][pos[..., 0]] + table[1][pos[..., 1]]
    return mx.where(mx.expand_dims(padding_positions, -1), 0.0, rows)


def bind_position_gather(model) -> bool:
    """Bind the gather onto this model's patch embedder instance, never the class.

    Stands down silently on text models, mock MLX environments, and unrelated VLM
    families that have a different ``patch_embedder`` layout. Logs a warning and
    stands down when a Gemma-family patch embedder does not have the shape it
    was written against: a 3-D two-axis ``mx.array`` table and a
    ``_position_embeddings(patch_positions, padding_positions)`` method.
    """
    import mlx.core as mx

    if not isinstance(getattr(mx, "array", None), type):
        return False
    embedder = getattr(getattr(model, "vision_tower", None), "patch_embedder", None)
    if embedder is None or "unittest.mock" in type(embedder).__module__:
        return False

    is_gemma = "gemma" in type(embedder).__module__ or "gemma" in str(
        getattr(getattr(model, "config", None), "model_type", "")
    )
    method = getattr(embedder, "_position_embeddings", None)
    if getattr(method, "__func__", None) is gathered_position_embeddings:
        return True
    if method is None and not is_gemma:
        return False

    table = getattr(embedder, "position_embedding_table", None)
    valid_method = (
        callable(method)
        and tuple(inspect.signature(method).parameters)
        == ("patch_positions", "padding_positions")
    )
    if (
        not isinstance(table, mx.array)
        or table.ndim != 3
        or table.shape[0] != 2
        or not valid_method
    ):
        logging.warning(
            "vision position gather stood down: patch embedder is not the shape it was written for"
        )
        return False
    embedder._position_embeddings = types.MethodType(gathered_position_embeddings, embedder)
    return True


def trim_visual_padding(inputs):
    """Drop trailing patch slots that no image or video frame uses.

    The processor pads every grid (3D image batches ``[B, L, D]`` or 4D video
    batches ``[N_videos, N_frames, L, D]``) to the largest budget (position id
    ``-1``); the vision tower masks those slots, so cutting them changes no
    valid patch and shortens attention. Rows keep their own padding up to the
    longest one. Measured: internal/claude/embeddinggemma2_probes/vision_candidates.txt (C4).
    """
    if not isinstance(inputs, dict):
        return inputs
    import mlx.core as mx

    if not isinstance(getattr(mx, "array", None), type):
        return inputs
    for values, positions in VISUAL_PATCH_KEYS:
        ids = inputs.get(positions)
        val = inputs.get(values)
        if (
            not isinstance(ids, mx.array)
            or not isinstance(val, mx.array)
            or ids.ndim not in (3, 4)
            or val.ndim != ids.ndim
            or val.shape[-2] != ids.shape[-2]
        ):
            continue
        n = int(mx.sum(mx.all(ids != -1, axis=-1), axis=-1).max().item())
        if 0 < n < ids.shape[-2]:
            inputs[values] = val[..., :n, :]
            inputs[positions] = ids[..., :n, :]
    return inputs
