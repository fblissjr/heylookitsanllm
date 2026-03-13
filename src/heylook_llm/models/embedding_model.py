"""Generic embedding model for MLX.

Loads any mlx-lm-supported transformer as a backbone and adds
mean/cls/none pooling + optional dense projection layers + L2 normalization.

The backbone is loaded dynamically via mlx_lm.utils._get_classes(), so any
architecture that mlx-lm supports (Gemma, Llama, Qwen, etc.) works here
without hard-coding import paths.

Architecture:
  input_ids -> backbone(embed_tokens -> transformer layers -> norm)
    -> pooling(hidden_states, attention_mask)
    -> Dense chain (optional)
    -> L2 normalize
"""

from typing import Dict, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn


def mean_pooling(
    hidden_states: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """Mean pool hidden states over the sequence dimension.

    Args:
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding.
            If None, pools over all tokens.

    Returns:
        (batch, hidden_size)
    """
    if attention_mask is None:
        return mx.mean(hidden_states, axis=1)

    mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)
    summed = mx.sum(hidden_states * mask_expanded, axis=1)
    counts = mx.maximum(mx.sum(mask_expanded, axis=1), 1e-9)
    return summed / counts


def normalize_embeddings(x: mx.array) -> mx.array:
    """L2 normalize along the last dimension.

    Args:
        x: (..., dim)

    Returns:
        (..., dim) with unit L2 norm along last axis.
    """
    norms = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    norms = mx.maximum(norms, 1e-12)
    return x / norms


def load_backbone(model_config: dict):
    """Load any mlx-lm transformer as an embedding backbone.

    Uses mlx_lm.utils._get_classes() for dynamic architecture resolution.
    The returned Model is a causal LM wrapper; we extract .model (the
    transformer body with embed_tokens, layers, norm).

    Args:
        model_config: dict with at minimum 'model_type' key.

    Returns:
        (backbone, args) where backbone is the transformer body (nn.Module)
        and args is the dataclass of model args.
    """
    try:
        from mlx_lm.utils import _get_classes
    except ImportError as exc:
        raise ImportError(
            "mlx-lm's _get_classes() API has moved or been removed. "
            "Update load_backbone() for the new mlx-lm version."
        ) from exc

    ModelClass, ArgsClass = _get_classes(model_config)

    # Filter config to only keys the ArgsClass accepts
    valid_keys = ArgsClass.__dataclass_fields__
    filtered = {k: v for k, v in model_config.items() if k in valid_keys}
    args = ArgsClass(**filtered)

    # Instantiate the full causal LM model, then extract the transformer body
    full_model = ModelClass(args)
    backbone = full_model.model
    del full_model  # release lm_head wrapper before weight materialization

    return backbone, args


class EmbeddingModel(nn.Module):
    """Generic MLX embedding encoder.

    Uses any mlx-lm transformer body as backbone, then adds pooling,
    optional dense projections, and L2 normalization.
    The key difference from the generative model: all attention is
    bidirectional (mask=None for non-padding tokens).
    """

    def __init__(
        self,
        backbone: nn.Module,
        args,
        pooling: Literal["mean", "cls", "none"] = "mean",
        dense_out_features: Optional[List[int]] = None,
    ):
        super().__init__()
        self.args = args
        self.model = backbone
        self.pooling = pooling

        # Pre-compute embedding scale factor (Gemma-family models need sqrt(hidden_size))
        model_type = getattr(args, "model_type", "")
        if model_type.startswith("gemma"):
            self._embed_scale = mx.array(args.hidden_size**0.5, dtype=mx.bfloat16)
        else:
            self._embed_scale = None

        # Dense projection layers (sentence-transformers 2_Dense, 3_Dense)
        self.dense_layers = []
        if dense_out_features:
            in_dim = args.hidden_size
            for out_dim in dense_out_features:
                self.dense_layers.append(nn.Linear(in_dim, out_dim, bias=False))
                in_dim = out_dim

    def _make_padding_mask(self, attention_mask: mx.array) -> Optional[mx.array]:
        """Create an additive attention mask that masks padding key positions.

        For bidirectional attention, we don't want a causal mask. We only need
        to prevent query tokens from attending to padding key positions.

        Args:
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding.

        Returns:
            (batch, 1, 1, seq_len) additive mask: 0.0 for real tokens, -inf for padding.
            Returns None if all tokens are real (no masking needed).
        """
        if mx.all(attention_mask).item():
            return None

        mask = attention_mask[:, None, None, :].astype(mx.float32)
        return mx.where(mask, 0.0, mx.finfo(mx.float32).min)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass: embed -> transformer (bidirectional) -> pool -> project -> normalize.

        Args:
            input_ids: (batch, seq_len) integer token IDs.
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding.

        Returns:
            If pooling is "mean" or "cls": (batch, output_dim) L2-normalized embeddings.
            If pooling is "none": (batch, seq_len, output_dim) per-token embeddings.
        """
        # Token embeddings (+ Gemma-family scaling if applicable)
        h = self.model.embed_tokens(input_ids)
        if self._embed_scale is not None:
            h = h * self._embed_scale.astype(h.dtype)

        # Build padding mask for attention (None if no padding present)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = self._make_padding_mask(attention_mask)

        # Forward through all layers -- bidirectional (no causal mask),
        # but padding positions are masked out via attn_mask
        for layer in self.model.layers:
            h = layer(h, mask=attn_mask, cache=None)

        # Final RMS norm
        h = self.model.norm(h)

        # Pooling
        if self.pooling == "mean":
            h = mean_pooling(h, attention_mask)
        elif self.pooling == "cls":
            h = h[:, 0, :]
        # "none" returns per-token embeddings (B, seq, dim)

        # Dense projections (Identity activation = just linear)
        for dense in self.dense_layers:
            h = dense(h)

        # L2 normalize
        return normalize_embeddings(h)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap HF checkpoint keys to our model structure.

        Handles two weight layouts:
        1. Flat keys from HF checkpoint: embed_tokens.weight, layers.0.*, norm.weight
           -> Need 'model.' prefix added (our self.model is the backbone)
        2. Already prefixed: model.embed_tokens.weight, etc.
           -> Pass through

        Also removes lm_head weights (we don't use them).
        """
        result = {}
        for key, value in weights.items():
            # Drop LM head -- we're an embedding model
            if key.startswith("lm_head"):
                continue

            # If key is flat (no model. prefix) and matches transformer structure,
            # add model. prefix
            if not key.startswith("model.") and not key.startswith("dense_layers."):
                transformer_prefixes = ("embed_tokens.", "layers.", "norm.")
                if any(key.startswith(p) for p in transformer_prefixes):
                    key = f"model.{key}"

            result[key] = value

        return result
