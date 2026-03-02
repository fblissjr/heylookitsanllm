"""EmbeddingGemma model for MLX.

Reuses mlx-lm's Gemma3 transformer internals (optimized for Apple Silicon)
but replaces causal attention with bidirectional attention and adds
mean pooling + dense projection layers + L2 normalization.

Architecture (from google/embeddinggemma-300m):
  input_ids -> embed_tokens -> scale(sqrt(hidden_size))
    -> 24 transformer layers (mask=None for bidirectional)
    -> final RMSNorm
    -> mean_pooling(hidden_states, attention_mask)
    -> Dense(768, 3072, no bias, no activation)
    -> Dense(3072, 768, no bias, no activation)
    -> L2 normalize
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma3_text import Gemma3Model, ModelArgs


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

    # Expand mask to (batch, seq_len, 1) for broadcasting
    mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)
    summed = mx.sum(hidden_states * mask_expanded, axis=1)
    # Clamp to avoid division by zero
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


@dataclass
class EmbeddingGemmaModelArgs(ModelArgs):
    """Extends Gemma3 ModelArgs with embedding-specific fields."""

    dense_out_features: List[int] = field(default_factory=lambda: [3072, 768])

    def __post_init__(self):
        # ModelArgs uses _sliding_window_pattern but the HF config key is
        # _sliding_window_pattern. Handle both names.
        if not hasattr(self, "sliding_window_pattern") or self.sliding_window_pattern is None:
            swp = getattr(self, "_sliding_window_pattern", None)
            if swp is not None:
                self.sliding_window_pattern = swp


class EmbeddingGemmaModel(nn.Module):
    """Pure MLX EmbeddingGemma encoder.

    Uses Gemma3Model from mlx-lm for the transformer body (embed_tokens +
    layers + norm), then adds mean pooling, dense projections, and L2 norm.
    The key difference from the generative Gemma3 model: all attention is
    bidirectional (mask=None).
    """

    def __init__(self, args: EmbeddingGemmaModelArgs):
        super().__init__()
        self.args = args
        self.model = Gemma3Model(args)

        # Dense projection layers (sentence-transformers 2_Dense, 3_Dense)
        in_dim = args.hidden_size
        self.dense_layers = []
        for out_dim in args.dense_out_features:
            self.dense_layers.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass: embed -> transformer (bidirectional) -> pool -> project -> normalize.

        Args:
            input_ids: (batch, seq_len) integer token IDs.
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding.
                Used for mean pooling only; attention layers see all tokens.

        Returns:
            (batch, output_dim) L2-normalized embeddings.
        """
        # Token embeddings + scaling (same as Gemma3Model)
        h = self.model.embed_tokens(input_ids)
        h = h * mx.array(self.args.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)

        # Forward through all layers with mask=None (bidirectional attention)
        for layer in self.model.layers:
            h = layer(h, mask=None, cache=None)

        # Final RMS norm
        h = self.model.norm(h)

        # Mean pooling
        h = mean_pooling(h, attention_mask)

        # Dense projections (Identity activation = just linear)
        for dense in self.dense_layers:
            h = dense(h)

        # L2 normalize
        return normalize_embeddings(h)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap HF checkpoint keys to our model structure.

        Handles two weight layouts:
        1. Flat keys from HF checkpoint: embed_tokens.weight, layers.0.*, norm.weight
           -> Need 'model.' prefix added (our self.model is a Gemma3Model)
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
