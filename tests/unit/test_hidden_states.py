# tests/unit/test_hidden_states.py
"""Unit tests for hidden states extraction module."""
import base64

import numpy as np
import pytest

from heylook_llm.hidden_states import (
    HiddenStatesRequest,
    HiddenStatesResponse,
    StructuredHiddenStatesRequest,
    StructuredHiddenStatesResponse,
    TokenBoundary,
    ChatTemplateTokenizer,
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


class TestStructuredHiddenStatesRequest:
    """Tests for StructuredHiddenStatesRequest model."""

    def test_minimal_request(self):
        """Test request with only required fields."""
        req = StructuredHiddenStatesRequest(
            model="Qwen3-4B",
            user_prompt="A cat sleeping"
        )
        assert req.model == "Qwen3-4B"
        assert req.user_prompt == "A cat sleeping"
        assert req.system_prompt is None
        assert req.thinking_content is None
        assert req.assistant_content is None
        assert req.enable_thinking is True  # default
        assert req.layer == -2  # default
        assert req.max_length == 512  # default
        assert req.encoding_format == "float"  # default
        assert req.return_token_boundaries is False  # default
        assert req.return_formatted_prompt is False  # default

    def test_full_request(self):
        """Test request with all fields specified."""
        req = StructuredHiddenStatesRequest(
            model="Qwen3-4B",
            user_prompt="A cat sleeping",
            system_prompt="You generate images",
            thinking_content="This is a simple scene",
            assistant_content="A tabby cat curled up",
            enable_thinking=True,
            layer=-3,
            max_length=256,
            encoding_format="base64",
            return_token_boundaries=True,
            return_formatted_prompt=True,
        )
        assert req.model == "Qwen3-4B"
        assert req.user_prompt == "A cat sleeping"
        assert req.system_prompt == "You generate images"
        assert req.thinking_content == "This is a simple scene"
        assert req.assistant_content == "A tabby cat curled up"
        assert req.enable_thinking is True
        assert req.layer == -3
        assert req.max_length == 256
        assert req.encoding_format == "base64"
        assert req.return_token_boundaries is True
        assert req.return_formatted_prompt is True


class TestStructuredHiddenStatesResponse:
    """Tests for StructuredHiddenStatesResponse model."""

    def test_minimal_response(self):
        """Test response with only required fields."""
        resp = StructuredHiddenStatesResponse(
            hidden_states=[[0.1, 0.2], [0.3, 0.4]],
            shape=[2, 2],
            model="Qwen3-4B",
            layer=-2,
            dtype="bfloat16",
        )
        assert resp.hidden_states == [[0.1, 0.2], [0.3, 0.4]]
        assert resp.shape == [2, 2]
        assert resp.model == "Qwen3-4B"
        assert resp.layer == -2
        assert resp.dtype == "bfloat16"
        assert resp.token_boundaries is None
        assert resp.token_counts is None
        assert resp.formatted_prompt is None

    def test_with_token_boundaries(self):
        """Test response with token boundaries."""
        resp = StructuredHiddenStatesResponse(
            hidden_states="SGVsbG8gV29ybGQ=",
            shape=[120, 2560],
            model="Qwen3-4B",
            layer=-2,
            dtype="bfloat16",
            encoding_format="base64",
            token_boundaries={
                "system": TokenBoundary(start=0, end=35),
                "user": TokenBoundary(start=35, end=80),
            },
            token_counts={"system": 35, "user": 45, "total": 120},
            formatted_prompt="<|im_start|>system\nYou generate images...",
        )
        assert resp.token_boundaries["system"].start == 0
        assert resp.token_boundaries["system"].end == 35
        assert resp.token_boundaries["user"].start == 35
        assert resp.token_counts["system"] == 35
        assert resp.token_counts["total"] == 120
        assert resp.formatted_prompt is not None


class TestTokenBoundary:
    """Tests for TokenBoundary model."""

    def test_boundary_creation(self):
        """Test creating a token boundary."""
        boundary = TokenBoundary(start=10, end=50)
        assert boundary.start == 10
        assert boundary.end == 50

    def test_boundary_dict_conversion(self):
        """Test converting boundary to/from dict."""
        boundary = TokenBoundary(start=0, end=100)
        d = boundary.model_dump()
        assert d == {"start": 0, "end": 100}

        # Recreate from dict
        boundary2 = TokenBoundary(**d)
        assert boundary2.start == 0
        assert boundary2.end == 100


class TestChatTemplateTokenizer:
    """Tests for ChatTemplateTokenizer class."""

    def test_initialization(self):
        """Test that tokenizer initializes correctly."""
        # Create a mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

            def apply_chat_template(self, messages, **kwargs):
                # Simple mock implementation
                parts = []
                for msg in messages:
                    parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
                if kwargs.get("add_generation_prompt"):
                    parts.append("<|im_start|>assistant\n")
                return "".join(parts)

        tokenizer = MockTokenizer()
        template_tokenizer = ChatTemplateTokenizer(tokenizer)

        assert template_tokenizer.tokenizer is tokenizer
        assert template_tokenizer.config == {}

    def test_special_token_constants(self):
        """Test that Qwen3 special token IDs are correct."""
        assert ChatTemplateTokenizer.THINK_START_TOKEN == 151667
        assert ChatTemplateTokenizer.THINK_END_TOKEN == 151668
        assert ChatTemplateTokenizer.IM_START_TOKEN == 151644
        assert ChatTemplateTokenizer.IM_END_TOKEN == 151645

    def test_build_prompt_user_only(self):
        """Test building prompt with only user message."""
        class MockTokenizer:
            def encode(self, text):
                # Simple: 1 token per word
                return list(range(len(text.split())))

            def apply_chat_template(self, messages, **kwargs):
                parts = []
                for msg in messages:
                    parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
                if kwargs.get("add_generation_prompt"):
                    parts.append("<|im_start|>assistant\n")
                return "".join(parts)

        tokenizer = MockTokenizer()
        template_tokenizer = ChatTemplateTokenizer(tokenizer)

        formatted_prompt, token_ids, boundaries = template_tokenizer.build_prompt_with_boundaries(
            user_prompt="Hello world",
        )

        assert "Hello world" in formatted_prompt
        assert "<|im_start|>user" in formatted_prompt
        assert "user" in boundaries
        assert "system" not in boundaries

    def test_build_prompt_with_system(self):
        """Test building prompt with system and user messages."""
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

            def apply_chat_template(self, messages, **kwargs):
                parts = []
                for msg in messages:
                    parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
                if kwargs.get("add_generation_prompt"):
                    parts.append("<|im_start|>assistant\n")
                return "".join(parts)

        tokenizer = MockTokenizer()
        template_tokenizer = ChatTemplateTokenizer(tokenizer)

        formatted_prompt, token_ids, boundaries = template_tokenizer.build_prompt_with_boundaries(
            user_prompt="Hello world",
            system_prompt="You are helpful",
        )

        assert "You are helpful" in formatted_prompt
        assert "Hello world" in formatted_prompt
        assert "system" in boundaries
        assert "user" in boundaries
        assert boundaries["system"]["start"] == 0
        assert boundaries["user"]["start"] > 0
