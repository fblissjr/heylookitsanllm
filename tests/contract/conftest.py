# tests/contract/conftest.py
#
# Shared fixtures for contract tests. Creates a FastAPI TestClient backed by
# mock router and service objects -- no real models, no MLX hardware needed.

import sys
from collections import OrderedDict
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from helpers.mlx_mock import FakeChunk, create_mlx_module_mocks
from heylook_llm.config import AppConfig
from heylook_llm.model_service import load_profiles


# ---------------------------------------------------------------------------
# Test model configs (one MLX, one GGUF)
# ---------------------------------------------------------------------------

TEST_MODELS_DATA = {
    "models": [
        {
            "id": "test-mlx-model",
            "provider": "mlx",
            "description": "Test MLX model for contract tests",
            "tags": ["test"],
            "enabled": True,
            "config": {"model_path": "/fake/mlx-model", "vision": False},
        },
        {
            "id": "test-gguf-model",
            "provider": "gguf",
            "description": "Test GGUF model for contract tests",
            "tags": ["test", "gguf"],
            "enabled": True,
            "config": {"model_path": "/fake/model.gguf"},
        },
    ],
    "default_model": "test-mlx-model",
    "max_loaded_models": 2,
}


# ---------------------------------------------------------------------------
# Fake provider that yields canned chunks
# ---------------------------------------------------------------------------

class FakeProvider:
    """Minimal provider mock that yields pre-set FakeChunks."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = None

    def create_chat_completion(self, request):
        """Return a generator of FakeChunks."""
        return iter([
            FakeChunk("Hello", token_id=1),
            FakeChunk(", ", token_id=2),
            FakeChunk("world!", token_id=3),
        ])


# ---------------------------------------------------------------------------
# Mock ModelRouter
# ---------------------------------------------------------------------------

class MockRouter:
    """Mimics the ModelRouter interface for contract tests."""

    def __init__(self):
        self.app_config = AppConfig(**TEST_MODELS_DATA)
        self.providers = OrderedDict()
        self.config_path = "models.toml"
        self.log_level = 40  # ERROR -- suppress logging noise in tests
        self.max_loaded_models = 2

    def list_available_models(self):
        return [m.id for m in self.app_config.get_enabled_models()]

    def get_provider(self, model_id):
        model_config = self.app_config.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model '{model_id}' not found or not enabled")
        if model_id not in self.providers:
            self.providers[model_id] = FakeProvider(model_id)
        return self.providers[model_id]

    def get_loaded_models(self):
        return dict(self.providers)

    def get_model_status(self, model_id):
        loaded = model_id in self.providers
        return {
            "loaded": loaded,
            "memory_mb": 100.0 if loaded else None,
            "context_used": 0 if loaded else None,
            "context_capacity": 4096 if loaded else None,
            "requests_active": 0 if loaded else None,
        }

    def reload_config(self):
        pass

    def unload_model(self, model_id):
        return self.providers.pop(model_id, None) is not None


# ---------------------------------------------------------------------------
# Mock ModelService
# ---------------------------------------------------------------------------

class MockModelService:
    """Mimics ModelService for admin endpoint tests."""

    def __init__(self):
        self.app_config = AppConfig(**TEST_MODELS_DATA)
        self._profiles = load_profiles()

    def list_configs(self):
        return list(self.app_config.models)

    def get_config(self, model_id):
        return self.app_config.get_model_config(model_id)

    def get_profiles(self):
        return {
            name: {"name": p.name, "description": p.description}
            for name, p in self._profiles.items()
        }

    def scan_paths(self, paths=None, scan_hf=True):
        return []  # No real scanning in tests

    def apply_profile(self, config, profile_name, model_info):
        profile = self._profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")
        return profile.apply(config, model_info)


# ---------------------------------------------------------------------------
# App + TestClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mlx_mocks():
    """Patch sys.modules so MLX imports don't fail on non-Apple hardware."""
    modules = create_mlx_module_mocks()
    with patch.dict(sys.modules, modules):
        yield modules


@pytest.fixture(scope="session")
def app(mlx_mocks):
    """Create and configure the FastAPI app with mock router and service."""
    from heylook_llm.api import app as fastapi_app

    mock_router = MockRouter()
    mock_service = MockModelService()

    fastapi_app.state.router_instance = mock_router
    fastapi_app.state.model_service = mock_service

    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """TestClient for making in-process HTTP requests."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mock_router(app):
    """Access the mock router from app state (for per-test assertions)."""
    return app.state.router_instance


@pytest.fixture
def mock_service(app):
    """Access the mock service from app state (for per-test assertions)."""
    return app.state.model_service
