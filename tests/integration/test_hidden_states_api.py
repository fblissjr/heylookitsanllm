# tests/integration/test_hidden_states_api.py
"""Integration tests for /v1/hidden_states endpoint.

These tests require a running heylookitsanllm server with an MLX model loaded.
Run with: pytest tests/integration/test_hidden_states_api.py

Recommended model: Qwen3-4B-mxfp4-mlx (hidden_dim=2560, layers=36)
"""
import base64
import os

import numpy as np
import pytest
import requests

# Server configuration
BASE_URL = os.environ.get("HEYLOOK_TEST_URL", "http://localhost:8080")
# Default test model - override with HEYLOOK_TEST_MODEL env var
TEST_MODEL = os.environ.get("HEYLOOK_TEST_MODEL", "Qwen3-4B-mxfp4-mlx")

# Expected hidden dimension for Qwen3-4B
EXPECTED_HIDDEN_DIM = 2560


@pytest.fixture
def api_client():
    """Simple API client for testing."""
    class Client:
        def __init__(self, base_url):
            self.base_url = base_url

        def post(self, endpoint, json=None):
            return requests.post(f"{self.base_url}{endpoint}", json=json)

        def get(self, endpoint):
            return requests.get(f"{self.base_url}{endpoint}")

    return Client(BASE_URL)


@pytest.fixture
def check_server(api_client):
    """Skip tests if server is not running."""
    try:
        resp = api_client.get("/v1/models")
        if resp.status_code != 200:
            pytest.skip("Server not responding correctly")
    except requests.ConnectionError:
        pytest.skip("Server not running")


@pytest.fixture
def check_model_available(api_client, check_server):
    """Skip tests if test model is not available."""
    resp = api_client.get("/v1/models")
    models = resp.json().get("data", [])
    model_ids = [m.get("id") for m in models]
    if TEST_MODEL not in model_ids:
        pytest.skip(f"Test model {TEST_MODEL} not available. Available: {model_ids}")


@pytest.mark.integration
class TestHiddenStatesEndpoint:
    """Integration tests for /v1/hidden_states endpoint."""

    def test_basic_extraction(self, api_client, check_model_available):
        """Test basic hidden states extraction with defaults."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "Hello, world!",
            "model": TEST_MODEL,
        })

        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "hidden_states" in data
        assert "shape" in data
        assert "model" in data
        assert "layer" in data
        assert "dtype" in data

        # Check shape format
        assert len(data["shape"]) == 2
        seq_len, hidden_dim = data["shape"]
        assert seq_len > 0
        assert hidden_dim == EXPECTED_HIDDEN_DIM

        # Check hidden states match shape
        hs = data["hidden_states"]
        assert len(hs) == seq_len
        assert len(hs[0]) == hidden_dim

        # Check model and layer
        assert data["model"] == TEST_MODEL
        assert data["layer"] == -2  # default

    def test_custom_layer(self, api_client, check_model_available):
        """Test extraction from a specific layer."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "Test prompt",
            "model": TEST_MODEL,
            "layer": -1,  # last layer
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == -1

    def test_positive_layer_index(self, api_client, check_model_available):
        """Test extraction with positive layer index."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "Test prompt",
            "model": TEST_MODEL,
            "layer": 10,  # layer 10 of 36
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == 10

    def test_with_attention_mask(self, api_client, check_model_available):
        """Test extraction with attention mask returned."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "Short text",
            "model": TEST_MODEL,
            "return_attention_mask": True,
        })

        assert resp.status_code == 200
        data = resp.json()

        assert "attention_mask" in data
        assert len(data["attention_mask"]) == data["shape"][0]
        # All tokens should be 1 (no padding for single sequence)
        assert all(m == 1 for m in data["attention_mask"])

    def test_base64_encoding(self, api_client, check_model_available):
        """Test base64 encoding format."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "Test for base64",
            "model": TEST_MODEL,
            "encoding_format": "base64",
        })

        assert resp.status_code == 200
        data = resp.json()

        assert data["encoding_format"] == "base64"
        assert isinstance(data["hidden_states"], str)

        # Verify we can decode it
        decoded_bytes = base64.b64decode(data["hidden_states"])
        arr = np.frombuffer(decoded_bytes, dtype=np.float32)

        # Should match shape
        expected_size = data["shape"][0] * data["shape"][1]
        assert len(arr) == expected_size

        # Reshape and verify
        reshaped = arr.reshape(data["shape"])
        assert reshaped.shape == tuple(data["shape"])

    def test_max_length_truncation(self, api_client, check_model_available):
        """Test that long inputs are truncated to max_length."""
        # Create a long input (should be > 50 tokens)
        long_text = "Hello world! " * 50

        resp = api_client.post("/v1/hidden_states", json={
            "input": long_text,
            "model": TEST_MODEL,
            "max_length": 32,
        })

        assert resp.status_code == 200
        data = resp.json()

        # Sequence length should not exceed max_length
        assert data["shape"][0] <= 32

    def test_chat_template_prompt(self, api_client, check_model_available):
        """Test with Z-Image style chat template prompt."""
        prompt = """<|im_start|>user
A beautiful sunset over the ocean<|im_end|>
<|im_start|>assistant
<think>
</think>
"""
        resp = api_client.post("/v1/hidden_states", json={
            "input": prompt,
            "model": TEST_MODEL,
            "layer": -2,
        })

        assert resp.status_code == 200
        data = resp.json()

        # Should have reasonable token count (around 20-25 for this prompt)
        assert 10 < data["shape"][0] < 50
        assert data["shape"][1] == EXPECTED_HIDDEN_DIM


@pytest.mark.integration
class TestHiddenStatesErrors:
    """Error handling tests for /v1/hidden_states endpoint."""

    def test_missing_model(self, api_client, check_server):
        """Test error when model field is missing."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "test",
        })

        assert resp.status_code in [400, 422, 500]

    def test_missing_input(self, api_client, check_server):
        """Test error when input field is missing."""
        resp = api_client.post("/v1/hidden_states", json={
            "model": TEST_MODEL,
        })

        assert resp.status_code in [400, 422, 500]

    def test_unknown_model(self, api_client, check_server):
        """Test error when requesting unknown model."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "test",
            "model": "nonexistent-model-xyz",
        })

        # Should return 404 or 500 for unknown model
        assert resp.status_code in [404, 500]

    def test_invalid_layer_out_of_range(self, api_client, check_model_available):
        """Test error when layer index is out of range."""
        resp = api_client.post("/v1/hidden_states", json={
            "input": "test",
            "model": TEST_MODEL,
            "layer": 1000,  # Way out of range
        })

        # Should return error
        assert resp.status_code in [400, 500]


@pytest.mark.integration
class TestHiddenStatesLlamaCpp:
    """Tests for llama.cpp model behavior (should return error)."""

    def test_gguf_model_not_supported(self, api_client, check_server):
        """Test that GGUF models return NotImplementedError."""
        # Skip if no GGUF model is configured
        resp = api_client.get("/v1/models")
        models = resp.json().get("data", [])

        gguf_model = None
        for m in models:
            model_id = m.get("id", "")
            # Look for GGUF indicators
            if "gguf" in model_id.lower() or "llama" in model_id.lower():
                gguf_model = model_id
                break

        if gguf_model is None:
            pytest.skip("No GGUF model available for testing")

        resp = api_client.post("/v1/hidden_states", json={
            "input": "test",
            "model": gguf_model,
        })

        # Should return 422 (Unprocessable Entity) for unsupported model type
        assert resp.status_code == 422
        assert "not supported" in resp.json().get("detail", "").lower()


@pytest.mark.integration
class TestStructuredHiddenStatesEndpoint:
    """Integration tests for /v1/hidden_states/structured endpoint."""

    def test_basic_structured_extraction(self, api_client, check_model_available):
        """Test basic structured hidden states extraction."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "A cat sleeping in sunlight",
        })

        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "hidden_states" in data
        assert "shape" in data
        assert "model" in data
        assert "layer" in data
        assert "dtype" in data

        # Check shape
        assert len(data["shape"]) == 2
        assert data["shape"][1] == EXPECTED_HIDDEN_DIM

    def test_with_token_boundaries(self, api_client, check_model_available):
        """Test that token boundaries are returned when requested."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "A beautiful sunset",
            "system_prompt": "You are an image generator",
            "return_token_boundaries": True,
        })

        assert resp.status_code == 200
        data = resp.json()

        # Token boundaries should be present
        assert "token_boundaries" in data
        assert "user" in data["token_boundaries"]

        # System boundary should exist since we provided system_prompt
        assert "system" in data["token_boundaries"]

        # Boundaries should have start and end
        assert "start" in data["token_boundaries"]["system"]
        assert "end" in data["token_boundaries"]["system"]

        # Token counts should be present
        assert "token_counts" in data
        assert "total" in data["token_counts"]

    def test_with_formatted_prompt(self, api_client, check_model_available):
        """Test that formatted prompt is returned when requested."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "A cat",
            "return_formatted_prompt": True,
        })

        assert resp.status_code == 200
        data = resp.json()

        assert "formatted_prompt" in data
        assert "<|im_start|>user" in data["formatted_prompt"]
        assert "A cat" in data["formatted_prompt"]

    def test_with_system_and_user(self, api_client, check_model_available):
        """Test with both system and user prompts."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "A sunset over mountains",
            "system_prompt": "Generate detailed images",
            "return_token_boundaries": True,
            "return_formatted_prompt": True,
        })

        assert resp.status_code == 200
        data = resp.json()

        # Both prompts should be in formatted output
        assert "Generate detailed images" in data["formatted_prompt"]
        assert "A sunset over mountains" in data["formatted_prompt"]

        # Boundaries should reflect system comes before user
        boundaries = data["token_boundaries"]
        assert boundaries["system"]["start"] < boundaries["user"]["start"]

    def test_base64_encoding(self, api_client, check_model_available):
        """Test base64 encoding for structured extraction."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "Test prompt",
            "encoding_format": "base64",
        })

        assert resp.status_code == 200
        data = resp.json()

        assert data["encoding_format"] == "base64"
        assert isinstance(data["hidden_states"], str)

        # Verify we can decode it
        decoded_bytes = base64.b64decode(data["hidden_states"])
        arr = np.frombuffer(decoded_bytes, dtype=np.float32)
        expected_size = data["shape"][0] * data["shape"][1]
        assert len(arr) == expected_size

    def test_custom_layer(self, api_client, check_model_available):
        """Test structured extraction from a specific layer."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
            "user_prompt": "Test",
            "layer": -3,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["layer"] == -3


@pytest.mark.integration
class TestStructuredHiddenStatesErrors:
    """Error handling tests for /v1/hidden_states/structured endpoint."""

    def test_missing_user_prompt(self, api_client, check_server):
        """Test error when user_prompt is missing."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": TEST_MODEL,
        })

        assert resp.status_code in [400, 422]

    def test_missing_model(self, api_client, check_server):
        """Test error when model is missing."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "user_prompt": "test",
        })

        assert resp.status_code in [400, 422]

    def test_unknown_model(self, api_client, check_server):
        """Test error when requesting unknown model."""
        resp = api_client.post("/v1/hidden_states/structured", json={
            "model": "nonexistent-model-xyz",
            "user_prompt": "test",
        })

        assert resp.status_code in [404, 500]


@pytest.mark.integration
class TestModelCapabilities:
    """Tests for model capabilities in /v1/models endpoint."""

    def test_models_endpoint_returns_provider(self, api_client, check_server):
        """Test that /v1/models includes provider type."""
        resp = api_client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()

        assert "data" in data
        # At least some models should have provider
        models_with_provider = [m for m in data["data"] if "provider" in m]
        assert len(models_with_provider) > 0

        # Provider should be valid
        for model in models_with_provider:
            assert model["provider"] in ["mlx", "llama_cpp", "gguf", "coreml_stt", "mlx_stt"]

    def test_models_endpoint_structure(self, api_client, check_server):
        """Test /v1/models response structure."""
        resp = api_client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "list"
        assert "data" in data

        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "owned_by" in model

    def test_capabilities_field_present_when_configured(self, api_client, check_server):
        """Test that capabilities field is present when configured."""
        resp = api_client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()

        # This test just verifies the structure is correct
        # Capabilities may or may not be configured
        for model in data["data"]:
            if "capabilities" in model:
                assert isinstance(model["capabilities"], list)
                for cap in model["capabilities"]:
                    assert isinstance(cap, str)
