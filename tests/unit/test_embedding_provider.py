"""Integration tests for MLXEmbeddingProvider.load_model().

Exercises the full provider load path: config.json -> load_backbone() ->
weight sanitization -> model.load_weights() -> tokenizer, using a tiny
Gemma3-text backbone with synthetic safetensors weights on disk.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
from mlx.utils import tree_flatten
import orjson


def _build_tiny_model_dir(tmp_path: Path, *, with_dense: bool = False):
    """Create a minimal model directory with config.json and safetensors.

    Returns the path and the model config dict.
    """
    config = {
        "model_type": "gemma3_text",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
    }

    # Write config.json
    (tmp_path / "config.json").write_bytes(orjson.dumps(config))

    # Build weights matching the Gemma3-text architecture
    from heylook_llm.models.embedding_model import EmbeddingModel, load_backbone

    backbone, args = load_backbone(config)

    dense_out_features = [16] if with_dense else None
    model = EmbeddingModel(
        backbone=backbone,
        args=args,
        pooling="mean",
        dense_out_features=dense_out_features,
    )

    # Materialize random weights and save as safetensors
    # mx.eval on model parameters is MLX graph materialization, not Python eval
    mx.eval(model.parameters())  # noqa: S307

    # tree_flatten handles MLX's nested param dicts (including list indices)
    flat_weights = dict(tree_flatten(model.parameters()))

    # Split: transformer weights go to root, dense weights go to 2_Dense/
    transformer_weights = {}
    dense_weights = {}
    for k, v in flat_weights.items():
        if k.startswith("dense_layers."):
            # Remap dense_layers.0.weight -> linear.weight for the on-disk format
            dense_key = k.replace("dense_layers.0.", "linear.")
            dense_weights[dense_key] = v
        else:
            transformer_weights[k] = v

    mx.save_safetensors(str(tmp_path / "model.safetensors"), transformer_weights)

    if with_dense and dense_weights:
        dense_dir = tmp_path / "2_Dense"
        dense_dir.mkdir()
        (dense_dir / "config.json").write_bytes(
            orjson.dumps({"in_features": 32, "out_features": 16})
        )
        mx.save_safetensors(str(dense_dir / "model.safetensors"), dense_weights)

    return config


class TestProviderLoadModel:
    """Test MLXEmbeddingProvider.load_model() end-to-end."""

    def test_load_model_basic(self, tmp_path):
        """Provider loads model from disk with config.json + safetensors."""
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        _build_tiny_model_dir(tmp_path)

        provider = MLXEmbeddingProvider(
            model_id="test-embedding",
            config={"model_path": str(tmp_path), "pooling": "mean"},
            verbose=False,
        )

        # Mock the tokenizer since we don't have real tokenizer files
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            provider.load_model()

        assert provider.model is not None
        assert provider.tokenizer is mock_tokenizer

        # Model should produce embeddings
        input_ids = mx.array([[1, 2, 3]])
        result = provider.model(input_ids)
        # MLX graph materialization (not Python eval)
        mx.eval(result)  # noqa: S307
        assert result.shape == (1, 32)

    def test_load_model_with_dense(self, tmp_path):
        """Provider loads model with Dense projection layers."""
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        _build_tiny_model_dir(tmp_path, with_dense=True)

        provider = MLXEmbeddingProvider(
            model_id="test-embedding-dense",
            config={"model_path": str(tmp_path), "pooling": "mean"},
            verbose=False,
        )

        mock_tokenizer = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            provider.load_model()

        input_ids = mx.array([[1, 2, 3]])
        result = provider.model(input_ids)
        # MLX graph materialization (not Python eval)
        mx.eval(result)  # noqa: S307
        # Output dimension should be 16 (dense projection from 32 -> 16)
        assert result.shape == (1, 16)

    def test_load_model_cls_pooling(self, tmp_path):
        """Provider respects pooling config."""
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        _build_tiny_model_dir(tmp_path)

        provider = MLXEmbeddingProvider(
            model_id="test-embedding-cls",
            config={"model_path": str(tmp_path), "pooling": "cls"},
            verbose=False,
        )

        mock_tokenizer = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            provider.load_model()

        input_ids = mx.array([[1, 2, 3]])
        result = provider.model(input_ids)
        # MLX graph materialization (not Python eval)
        mx.eval(result)  # noqa: S307
        assert result.shape == (1, 32)

    def test_get_embeddings(self, tmp_path):
        """Full get_embeddings() path: text -> tokenize -> embed -> list."""
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        _build_tiny_model_dir(tmp_path)

        provider = MLXEmbeddingProvider(
            model_id="test-embedding",
            config={"model_path": str(tmp_path), "pooling": "mean"},
            verbose=False,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            provider.load_model()

        embeddings = provider.get_embeddings(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 32
        # Verify L2 normalized (norm ~= 1.0)
        norm = math.sqrt(sum(x * x for x in embeddings[0]))
        assert abs(norm - 1.0) < 1e-4

    def test_sanitize_handles_rotary_and_vision_keys(self, tmp_path):
        """Weights with rotary freqs and vision keys are cleaned during load."""
        _build_tiny_model_dir(tmp_path)

        # Add rotary and vision keys to the safetensors file
        existing = mx.load(str(tmp_path / "model.safetensors"))
        existing["model.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((16,))
        existing["vision_tower.encoder.weight"] = mx.zeros((32, 32))
        mx.save_safetensors(str(tmp_path / "model.safetensors"), existing)

        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

        provider = MLXEmbeddingProvider(
            model_id="test-embedding",
            config={"model_path": str(tmp_path), "pooling": "mean"},
            verbose=False,
        )

        mock_tokenizer = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            # Should not raise despite extra keys
            provider.load_model()

        assert provider.model is not None
