# tests/test_router.py
import pytest
from unittest.mock import MagicMock
import yaml # We need to mock this

# This test can now directly import from `src` because installing
# with `pip install -e .` makes the src package available.
from src.router import ModelRouter

# A fake models.yaml configuration for testing purposes
FAKE_CONFIG = {
    "models": [
        {
            "id": "qwen-vl-mlx",
            "provider": "mlx",
            "config": {"model_path": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"}
        },
        {
            "id": "llama-3-gguf",
            "provider": "llama_cpp",
            "config": {
                "model_path": "/path/to/llama3.gguf",
                "chat_format": "llama-3",
                "n_gpu_layers": -1
            }
        }
    ]
}

@pytest.fixture
def mock_providers(mocker):
    """A pytest fixture to mock the MLX and LlamaCpp providers."""
    # Why: We mock the providers to avoid actually loading heavy models during tests.
    # We replace the real classes with MagicMock objects that record how they are called.
    mocker.patch('src.router.MLXProvider', new_callable=MagicMock)
    mocker.patch('src.router.LlamaCppProvider', new_callable=MagicMock)

    # We also mock yaml.safe_load to return our fake config
    mocker.patch('yaml.safe_load', return_value=FAKE_CONFIG)

def test_router_initialization(mock_providers):
    """Test that the router initializes correctly and loads the first model from the config."""
    from src.router import MLXProvider

    # When the router is created...
    router = ModelRouter(config_path="dummy_path.yaml", log_level=20) # 20 = INFO

    # Then it should have loaded the first model.
    assert router.current_provider_id == "qwen-vl-mlx"
    assert "qwen-vl-mlx" in router.providers

    # And it should have called the MLXProvider constructor exactly once with the correct config.
    MLXProvider.assert_called_once()
    MLXProvider.assert_called_with(
        model_id="qwen-vl-mlx",
        config=FAKE_CONFIG["models"][0]["config"],
        verbose=False
    )

def test_router_get_same_model(mock_providers):
    """Test that requesting the same model twice doesn't trigger a reload."""
    from src.router import MLXProvider

    router = ModelRouter(config_path="dummy_path.yaml", log_level=20)
    MLXProvider.reset_mock() # Reset call count after initialization

    # When we request the same model again...
    provider = router.get_provider("qwen-vl-mlx")

    # Then the constructor should NOT have been called again.
    MLXProvider.assert_not_called()
    assert router.current_provider_id == "qwen-vl-mlx"
    assert provider is not None

def test_router_hot_swap_model(mock_providers):
    """Test the hot-swapping functionality between different providers."""
    from src.router import MLXProvider, LlamaCppProvider

    router = ModelRouter(config_path="dummy_path.yaml", log_level=10) # 10 = DEBUG

    # The first model is loaded on init
    MLXProvider.assert_called_once()
    LlamaCppProvider.assert_not_called()

    # When we request the second model (which uses a different provider)...
    provider = router.get_provider("llama-3-gguf")

    # Then the LlamaCppProvider constructor should have been called with its config.
    LlamaCppProvider.assert_called_once()
    LlamaCppProvider.assert_called_with(
        model_id="llama-3-gguf",
        config=FAKE_CONFIG["models"][1]["config"],
        verbose=True # Because log_level was DEBUG
    )

    # And the current provider should have been updated.
    assert router.current_provider_id == "llama-3-gguf"
    assert provider is not None

    # And the original MLX provider should have been removed from the active providers dict.
    assert "qwen-vl-mlx" not in router.providers
