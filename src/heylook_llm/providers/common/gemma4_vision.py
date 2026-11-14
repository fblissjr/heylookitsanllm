"""Shared Gemma 4 vision customizations for generative and embedding paths."""

import inspect
import logging
import types

VISUAL_PATCH_KEYS = (
    ("pixel_values", "image_position_ids"),
    ("pixel_values_videos", "video_position_ids"),
)


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
