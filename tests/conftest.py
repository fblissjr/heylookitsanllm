# tests/conftest.py
#
# Root conftest providing shared fixtures for all test modules.

import sys
from unittest.mock import patch

import pytest

from heylook_llm.config import (
    ChatMessage,
    ChatRequest,
    ImageContentPart,
    ImageUrl,
    TextContentPart,
)

from helpers.mlx_mock import create_mlx_module_mocks


# ---------------------------------------------------------------------------
# Marker registration (supplements pyproject.toml markers)
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "mlx_perf: MLX performance tests")


# ---------------------------------------------------------------------------
# MLX mocking fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mlx():
    """Patch sys.modules so MLX provider code can be imported on any platform."""
    modules = create_mlx_module_mocks()
    with patch.dict(sys.modules, modules):
        yield modules


@pytest.fixture
def mock_mlx_provider(mock_mlx):
    """Return an MLXProvider instance with all MLX deps mocked.

    The provider is constructed but load_model() is NOT called -- tests should
    call it explicitly or set up model/processor manually.
    """
    # Import inside fixture so the sys.modules patch is active
    from heylook_llm.providers.mlx_provider import MLXProvider

    provider = MLXProvider(
        model_id="test-model",
        config={"model_path": "/fake/model", "vision": False},
        verbose=False,
    )
    return provider


@pytest.fixture
def mock_vlm_provider(mock_mlx):
    """Return an MLXProvider configured as a VLM (vision=True)."""
    from heylook_llm.providers.mlx_provider import MLXProvider

    provider = MLXProvider(
        model_id="test-vlm",
        config={"model_path": "/fake/vlm", "vision": True},
        verbose=False,
    )
    return provider


# ---------------------------------------------------------------------------
# Request / message fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chat_request() -> ChatRequest:
    """Basic text-only ChatRequest."""
    return ChatRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hello, how are you?"),
        ],
        max_tokens=128,
    )


@pytest.fixture
def sample_chat_request_with_thinking() -> ChatRequest:
    """ChatRequest with a thinking field on an assistant message."""
    return ChatRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(
                role="assistant",
                content="The answer is 4.",
                thinking="I need to add 2 and 2 together.",
            ),
            ChatMessage(role="user", content="Are you sure?"),
        ],
        max_tokens=128,
    )


@pytest.fixture
def sample_messages():
    """List of ChatMessage objects covering common scenarios."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="What is 1+1?"),
    ]


@pytest.fixture
def sample_multimodal_request() -> ChatRequest:
    """ChatRequest with image content."""
    return ChatRequest(
        model="test-vlm",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TextContentPart(type="text", text="What is in this image?"),
                    ImageContentPart(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,iVBOR..."),
                    ),
                ],
            ),
        ],
        max_tokens=256,
    )


@pytest.fixture
def temp_models_toml(tmp_path):
    """Write a minimal models.toml and return its path."""
    toml_content = """
[[models]]
id = "test-mlx"
provider = "mlx"
enabled = true
[models.config]
model_path = "/fake/mlx-model"
"""
    p = tmp_path / "models.toml"
    p.write_text(toml_content)
    return p
