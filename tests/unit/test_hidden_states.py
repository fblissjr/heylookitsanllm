# tests/unit/test_hidden_states.py
"""Unit tests for hidden states extraction module."""
import base64

import numpy as np
import pytest

from heylook_llm.hidden_states import (
    HiddenStatesRequest,
    HiddenStatesResponse,
    encode_hidden_states_base64,
    LlamaCppHiddenStatesExtractor,
)


class TestHiddenStatesRequest:
    """Tests for HiddenStatesRequest model."""

    def test_minimal_request(self):
        """Test request with only required fields."""
        req = HiddenStatesRequest(input="test prompt", model="test-model")
        assert req.input == "test prompt"
        assert req.model == "test-model"
        assert req.layer == -2  # default
        assert req.max_length == 512  # default
        assert req.return_attention_mask is False  # default
        assert req.encoding_format == "float"  # default

    def test_full_request(self):
        """Test request with all fields specified."""
        req = HiddenStatesRequest(
            input=["prompt 1", "prompt 2"],
            model="Qwen3-4B-mxfp4-mlx",
            layer=-3,
            max_length=256,
            return_attention_mask=True,
            encoding_format="base64",
        )
        assert req.input == ["prompt 1", "prompt 2"]
        assert req.model == "Qwen3-4B-mxfp4-mlx"
        assert req.layer == -3
        assert req.max_length == 256
        assert req.return_attention_mask is True
        assert req.encoding_format == "base64"

    def test_string_input(self):
        """Test that string input is accepted."""
        req = HiddenStatesRequest(input="single prompt", model="test")
        assert req.input == "single prompt"

    def test_list_input(self):
        """Test that list input is accepted."""
        req = HiddenStatesRequest(input=["a", "b", "c"], model="test")
        assert req.input == ["a", "b", "c"]

    def test_positive_layer_index(self):
        """Test positive layer index."""
        req = HiddenStatesRequest(input="test", model="test", layer=5)
        assert req.layer == 5

    def test_negative_layer_index(self):
        """Test negative layer index."""
        req = HiddenStatesRequest(input="test", model="test", layer=-1)
        assert req.layer == -1


class TestHiddenStatesResponse:
    """Tests for HiddenStatesResponse model."""

    def test_float_response(self):
        """Test response with float format."""
        resp = HiddenStatesResponse(
            hidden_states=[[0.1, 0.2], [0.3, 0.4]],
            shape=[2, 2],
            model="test-model",
            layer=-2,
            dtype="float32",
        )
        assert resp.hidden_states == [[0.1, 0.2], [0.3, 0.4]]
        assert resp.shape == [2, 2]
        assert resp.model == "test-model"
        assert resp.layer == -2
        assert resp.dtype == "float32"
        assert resp.encoding_format is None
        assert resp.attention_mask is None

    def test_base64_response(self):
        """Test response with base64 format."""
        resp = HiddenStatesResponse(
            hidden_states="SGVsbG8gV29ybGQ=",
            shape=[10, 256],
            model="test-model",
            layer=-2,
            dtype="bfloat16",
            encoding_format="base64",
        )
        assert resp.hidden_states == "SGVsbG8gV29ybGQ="
        assert resp.encoding_format == "base64"

    def test_with_attention_mask(self):
        """Test response with attention mask."""
        resp = HiddenStatesResponse(
            hidden_states=[[0.1, 0.2]],
            shape=[1, 2],
            model="test",
            layer=-2,
            dtype="float32",
            attention_mask=[1, 1, 1, 0, 0],
        )
        assert resp.attention_mask == [1, 1, 1, 0, 0]


class TestBase64Encoding:
    """Tests for base64 encoding/decoding."""

    def test_encode_small_array(self):
        """Test encoding a small array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        encoded = encode_hidden_states_base64(arr)

        # Verify it's valid base64
        decoded_bytes = base64.b64decode(encoded)
        decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(2, 2)

        np.testing.assert_array_equal(decoded_arr, arr)

    def test_encode_preserves_values(self):
        """Test that encoding preserves exact values."""
        arr = np.array([[0.123456, -0.789012]], dtype=np.float32)
        encoded = encode_hidden_states_base64(arr)

        decoded_bytes = base64.b64decode(encoded)
        decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(1, 2)

        np.testing.assert_array_almost_equal(decoded_arr, arr, decimal=6)

    def test_encode_large_array(self):
        """Test encoding a larger array (simulating real hidden states)."""
        # 21 tokens x 2560 hidden dim (Qwen3-4B size)
        arr = np.random.randn(21, 2560).astype(np.float32)
        encoded = encode_hidden_states_base64(arr)

        # Verify round-trip
        decoded_bytes = base64.b64decode(encoded)
        decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(21, 2560)

        np.testing.assert_array_equal(decoded_arr, arr)

    def test_encode_converts_to_float32(self):
        """Test that non-float32 arrays are converted."""
        arr = np.array([[1.0, 2.0]], dtype=np.float64)
        encoded = encode_hidden_states_base64(arr)

        decoded_bytes = base64.b64decode(encoded)
        # Should decode as float32
        decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(1, 2)

        np.testing.assert_array_almost_equal(decoded_arr, arr.astype(np.float32))


class TestLlamaCppExtractor:
    """Tests for LlamaCppHiddenStatesExtractor."""

    def test_raises_not_implemented(self):
        """Test that llama.cpp extractor raises NotImplementedError."""
        extractor = LlamaCppHiddenStatesExtractor(model=None)

        with pytest.raises(NotImplementedError) as exc_info:
            extractor.extract(["test"])

        assert "llama.cpp" in str(exc_info.value).lower()
        assert "not supported" in str(exc_info.value).lower()

    def test_error_message_includes_issue_link(self):
        """Test that error message includes GitHub issue link."""
        extractor = LlamaCppHiddenStatesExtractor(model=None)

        with pytest.raises(NotImplementedError) as exc_info:
            extractor.extract(["test"])

        assert "github.com" in str(exc_info.value)
        assert "1695" in str(exc_info.value)


class TestLayerIndexNormalization:
    """Tests for layer index handling logic."""

    def test_negative_index_calculation(self):
        """Test that negative indices are calculated correctly."""
        # This tests the logic, not the actual extraction
        n_layers = 36  # Qwen3-4B has 36 layers

        # -1 should be layer 35 (last)
        layer_idx = -1
        target = n_layers + layer_idx
        assert target == 35

        # -2 should be layer 34 (second-to-last)
        layer_idx = -2
        target = n_layers + layer_idx
        assert target == 34

        # -36 should be layer 0 (first)
        layer_idx = -36
        target = n_layers + layer_idx
        assert target == 0

    def test_positive_index_passthrough(self):
        """Test that positive indices pass through unchanged."""
        n_layers = 36
        layer_idx = 5
        target = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        assert target == 5
