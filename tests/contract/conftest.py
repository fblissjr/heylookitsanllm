# tests/contract/conftest.py
#
# Shared fixtures for contract tests. Creates a FastAPI TestClient backed by
# mock router and service objects -- no real models, no MLX hardware needed.

import sys
from collections import OrderedDict
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from helpers.mlx_mock import FakeChunk, create_mlx_module_mocks, real_mlx_available
from heylook_llm.config import AppConfig
from heylook_llm.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Test model configs
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
    ],
    "default_model": "test-mlx-model",
    "max_loaded_models": 2,
}


# ---------------------------------------------------------------------------
# Fake provider that yields canned chunks
# ---------------------------------------------------------------------------

class FakeProvider(BaseProvider):
    """Minimal provider that yields pre-set FakeChunks.

    SUBCLASSES BaseProvider rather than duck-typing it. It used to hand-roll
    the surface (provider_name / check_capacity / template_info), which meant
    every new obligation on the provider contract had to be remembered here
    too -- and the first one that wasn't (``effective_thinking``, 2026-08-07)
    turned every route test into a 500 with no hint of why. Inheriting means
    a contract addition arrives with its default already in place, and only a
    DELIBERATE difference needs to be written down.
    """

    provider_name = "mlx"  # telemetry's provider type

    def __init__(self, model_id: str, config: dict | None = None):
        # Real config shape, so cascade-derived answers (effective_thinking)
        # match what a real MLX provider would say for the same request.
        super().__init__(model_id, config or {"model_path": "/fake/mlx-model", "vision": False}, False)
        self.processor = None

    def load_model(self):
        """Nothing to load; the fixtures hand out ready providers."""

    def template_info(self):
        """None -> pass-through reasoning parser. Stated explicitly because
        it is a CHOICE (these tests assert unparsed passthrough), not just
        the inherited default."""
        return None

    def create_chat_completion(self, request, abort_event=None):
        """Yield FakeChunks. A real generator (not a list_iterator) so the
        route's generator.close() -- which releases the generation gate -- works.
        ``abort_event`` matches the provider contract (per-request cancel signal).
        """
        yield FakeChunk("Hello", token_id=1)
        yield FakeChunk(", ", token_id=2)
        yield FakeChunk("world!", token_id=3)


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
        # Imported lazily: router pulls in the MLX stack, which is only mocked
        # once the fixtures have run.
        from heylook_llm.router import ModelNotFound

        model_config = self.app_config.get_model_config(model_id)
        if not model_config:
            # ModelNotFound, matching the real router: id-resolution failure is
            # the client's fault (400), while a bare ValueError from here means
            # the LOAD failed and must stay a 500.
            raise ModelNotFound(f"Model '{model_id}' not found or not enabled")
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

    def pin_model(self, model_id):
        pass

    def unpin_model(self, model_id):
        pass

    def reload_config(self):
        pass

    def is_loading(self, model_id):
        return False

    def clear_cache(self):
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

    def list_configs(self):
        return list(self.app_config.models)

    def get_config(self, model_id):
        return self.app_config.get_model_config(model_id)

    def get_samplers(self):
        from heylook_llm.samplers import get_sampler_registry

        return {info["name"]: info for info in get_sampler_registry().list_info()}

    def scan_paths(self, paths=None, scan_hf=True):
        return []  # No real scanning in tests


# ---------------------------------------------------------------------------
# App + TestClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mlx_mocks():
    """Patch sys.modules so MLX imports don't fail on non-Apple hardware.

    CONDITIONAL, and that is the whole point (2026-08-18): contract tests never
    run a real generation path -- they drive FakeProvider -- so the mock exists
    ONLY so ``import heylook_llm.api`` (which pulls the whole provider stack,
    every module of which imports mlx at module level) succeeds where MLX is
    absent. Where MLX is really installed, patching is pure damage and made
    INVOCATION ORDER LOAD-BEARING two ways:

    - This fixture is session-scoped, so the patch stands until the whole run
      ends. ``pytest tests/contract/ tests/unit/`` therefore ran every unit
      test under MagicMock ``mlx`` -- ~57 failures + 8 collection errors that
      all passed in isolation.
    - Worse than scope: a heylook module FIRST imported while the patch is
      active binds MagicMocks into its own namespace PERMANENTLY (the module
      object outlives the patch), so narrowing the scope alone would not have
      fixed it.

    Skipping the patch outright on Apple hardware removes both. It also fixes
    isolated contract runs, which the mock tree was actively breaking: it
    shadowed the real ``mlx_lm`` with a MagicMock that has no
    ``tokenizer_utils`` submodule.
    """
    if real_mlx_available():
        yield None
        return
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
