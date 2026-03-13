"""Tests for EmbeddingModel (embedding_model.py).

Covers:
- mean_pooling correctness with and without attention mask
- L2 normalization
- Forward pass output shapes with a tiny model (mean, cls, none pooling)
- Weight sanitize key remapping
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


def _make_tiny_model(pooling="mean", dense_out_features=None):
    """Create a tiny EmbeddingModel using load_backbone for testing."""
    from heylook_llm.models.embedding_model import EmbeddingModel, load_backbone

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
    backbone, args = load_backbone(config)
    model = EmbeddingModel(
        backbone=backbone,
        args=args,
        pooling=pooling,
        dense_out_features=dense_out_features or [64, 32],
    )
    for v in model.parameters().values():
        mx.eval(v)  # noqa: S307  -- MLX graph materialization, not Python eval
    return model


class TestMeanPooling:
    """Test mean pooling over sequence dimension with attention mask."""

    def test_mean_pool_no_mask(self):
        """Without mask, pools over entire sequence."""
        from heylook_llm.models.embedding_model import mean_pooling

        hidden = mx.array([[[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]]])
        result = mean_pooling(hidden, attention_mask=None)
        expected = mx.mean(hidden, axis=1)
        assert result.shape == (1, 4)
        assert mx.allclose(result, expected).item()

    def test_mean_pool_with_mask(self):
        """With mask, only pools over non-padding tokens."""
        from heylook_llm.models.embedding_model import mean_pooling

        hidden = mx.array([[[1.0, 2.0],
                            [3.0, 4.0],
                            [99.0, 99.0]]])
        mask = mx.array([[1, 1, 0]])
        result = mean_pooling(hidden, attention_mask=mask)
        expected = mx.array([[2.0, 3.0]])
        assert result.shape == (1, 2)
        assert mx.allclose(result, expected).item()

    def test_mean_pool_batch(self):
        """Batched pooling with different pad lengths."""
        from heylook_llm.models.embedding_model import mean_pooling

        hidden = mx.array([
            [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ])
        mask = mx.array([
            [1, 1, 0],
            [1, 0, 0],
        ])
        result = mean_pooling(hidden, attention_mask=mask)
        assert result.shape == (2, 2)
        assert mx.allclose(result[0], mx.array([4.0, 6.0])).item()
        assert mx.allclose(result[1], mx.array([1.0, 1.0])).item()

    def test_mean_pool_all_masked_returns_zeros(self):
        """Edge case: all tokens masked should return zeros (not NaN)."""
        from heylook_llm.models.embedding_model import mean_pooling

        hidden = mx.array([[[5.0, 5.0]]])
        mask = mx.array([[0]])
        result = mean_pooling(hidden, attention_mask=mask)
        assert result.shape == (1, 2)
        assert not mx.any(mx.isnan(result)).item()


class TestNormalize:
    """Test L2 normalization."""

    def test_unit_norm(self):
        from heylook_llm.models.embedding_model import normalize_embeddings

        x = mx.array([[3.0, 4.0]])
        result = normalize_embeddings(x)
        norms = mx.sqrt(mx.sum(result * result, axis=-1))
        assert mx.allclose(norms, mx.ones_like(norms), atol=1e-5).item()

    def test_batch_norm(self):
        from heylook_llm.models.embedding_model import normalize_embeddings

        x = mx.array([[3.0, 4.0], [0.0, 5.0]])
        result = normalize_embeddings(x)
        norms = mx.sqrt(mx.sum(result * result, axis=-1))
        assert mx.allclose(norms, mx.ones(2), atol=1e-5).item()

    def test_zero_vector_no_nan(self):
        from heylook_llm.models.embedding_model import normalize_embeddings

        x = mx.array([[0.0, 0.0]])
        result = normalize_embeddings(x)
        assert not mx.any(mx.isnan(result)).item()


class TestWeightSanitize:
    """Test weight key remapping for HF -> MLX structure."""

    def test_remap_model_prefix(self):
        model = _make_tiny_model()
        weights = {
            "model.embed_tokens.weight": mx.zeros((128, 32)),
            "lm_head.weight": mx.zeros((128, 32)),
        }
        sanitized = model.sanitize(weights)
        assert "lm_head.weight" not in sanitized
        assert "model.embed_tokens.weight" in sanitized

    def test_flat_keys_get_model_prefix(self):
        model = _make_tiny_model()
        weights = {
            "embed_tokens.weight": mx.zeros((128, 32)),
            "layers.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
            "norm.weight": mx.zeros((32,)),
        }
        sanitized = model.sanitize(weights)
        assert "model.embed_tokens.weight" in sanitized
        assert "model.layers.0.self_attn.q_proj.weight" in sanitized
        assert "model.norm.weight" in sanitized

    def test_strips_rotary_emb_inv_freq(self):
        """Precomputed rotary frequencies should be dropped (Llama, Qwen2, etc.)."""
        model = _make_tiny_model()
        weights = {
            "model.embed_tokens.weight": mx.zeros((128, 32)),
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((16,)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
        }
        sanitized = model.sanitize(weights)
        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
        assert "model.layers.0.self_attn.q_proj.weight" in sanitized

    def test_strips_vision_tower_keys(self):
        """Vision encoder weights should be dropped for text-only embedding."""
        model = _make_tiny_model()
        weights = {
            "model.embed_tokens.weight": mx.zeros((128, 32)),
            "vision_tower.encoder.layers.0.weight": mx.zeros((64, 64)),
            "vision_model.embeddings.weight": mx.zeros((32, 32)),
            "multi_modal_projector.linear.weight": mx.zeros((32, 64)),
        }
        sanitized = model.sanitize(weights)
        assert "model.embed_tokens.weight" in sanitized
        assert not any(k.startswith("vision_tower.") for k in sanitized)
        assert not any(k.startswith("vision_model.") for k in sanitized)
        assert not any(k.startswith("multi_modal_projector.") for k in sanitized)

    def test_preserves_dense_layer_keys(self):
        """Dense projection layer keys should pass through unchanged."""
        model = _make_tiny_model()
        weights = {
            "model.embed_tokens.weight": mx.zeros((128, 32)),
            "dense_layers.0.weight": mx.zeros((64, 32)),
            "dense_layers.1.weight": mx.zeros((32, 64)),
        }
        sanitized = model.sanitize(weights)
        assert "dense_layers.0.weight" in sanitized
        assert "dense_layers.1.weight" in sanitized

    def test_mixed_llama_style_weights(self):
        """Full Llama-style weight set: flat keys, rotary freqs, lm_head."""
        model = _make_tiny_model()
        weights = {
            "embed_tokens.weight": mx.zeros((128, 32)),
            "layers.0.self_attn.q_proj.weight": mx.zeros((32, 32)),
            "layers.0.self_attn.k_proj.weight": mx.zeros((32, 32)),
            "layers.0.self_attn.v_proj.weight": mx.zeros((32, 32)),
            "layers.0.self_attn.o_proj.weight": mx.zeros((32, 32)),
            "layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((16,)),
            "layers.0.mlp.gate_proj.weight": mx.zeros((64, 32)),
            "layers.0.mlp.up_proj.weight": mx.zeros((64, 32)),
            "layers.0.mlp.down_proj.weight": mx.zeros((32, 64)),
            "norm.weight": mx.zeros((32,)),
            "lm_head.weight": mx.zeros((128, 32)),
        }
        sanitized = model.sanitize(weights)
        # lm_head dropped
        assert "lm_head.weight" not in sanitized
        # rotary freqs dropped
        assert not any("rotary_emb" in k for k in sanitized)
        # all transformer keys get model. prefix
        assert "model.embed_tokens.weight" in sanitized
        assert "model.layers.0.self_attn.q_proj.weight" in sanitized
        assert "model.norm.weight" in sanitized
        # 9 keys total (11 - lm_head - rotary)
        assert len(sanitized) == 9


class TestForwardPass:
    """Test EmbeddingModel forward pass with tiny weights."""

    @pytest.fixture
    def tiny_model(self):
        return _make_tiny_model()

    def test_output_shape(self, tiny_model):
        input_ids = mx.array([[1, 2, 3]])
        result = tiny_model(input_ids)
        assert result.shape == (1, 32)

    def test_output_normalized(self, tiny_model):
        input_ids = mx.array([[1, 2, 3]])
        result = tiny_model(input_ids)
        norms = mx.sqrt(mx.sum(result * result, axis=-1))
        assert mx.allclose(norms, mx.ones_like(norms), atol=1e-5).item()

    def test_batch_output_shape(self, tiny_model):
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        result = tiny_model(input_ids)
        assert result.shape == (2, 32)

    def test_attention_mask_respected(self, tiny_model):
        ids_no_pad = mx.array([[1, 2, 3]])
        ids_with_pad = mx.array([[1, 2, 3, 0, 0]])
        mask_no_pad = mx.array([[1, 1, 1]])
        mask_with_pad = mx.array([[1, 1, 1, 0, 0]])

        result_no_pad = tiny_model(ids_no_pad, attention_mask=mask_no_pad)
        result_with_pad = tiny_model(ids_with_pad, attention_mask=mask_with_pad)

        assert result_no_pad.shape == (1, 32)
        assert result_with_pad.shape == (1, 32)
        norms_no_pad = mx.sqrt(mx.sum(result_no_pad * result_no_pad, axis=-1))
        norms_with_pad = mx.sqrt(mx.sum(result_with_pad * result_with_pad, axis=-1))
        assert mx.allclose(norms_no_pad, mx.ones_like(norms_no_pad), atol=1e-5).item()
        assert mx.allclose(norms_with_pad, mx.ones_like(norms_with_pad), atol=1e-5).item()
        assert mx.allclose(result_no_pad, result_with_pad, atol=1e-4).item()


class TestPoolingModes:
    """Test different pooling configurations."""

    def test_cls_pooling(self):
        model = _make_tiny_model(pooling="cls")
        input_ids = mx.array([[1, 2, 3]])
        result = model(input_ids)
        assert result.shape == (1, 32)

    def test_none_pooling(self):
        model = _make_tiny_model(pooling="none")
        input_ids = mx.array([[1, 2, 3]])
        result = model(input_ids)
        assert result.shape == (1, 3, 32)
