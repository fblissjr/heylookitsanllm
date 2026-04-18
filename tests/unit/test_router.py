"""Router tests.

Fixtures use TOML because `router.py:_load_config` is TOML-only. The tests
exercise real LRU eviction, hot-swap, and max_loaded_models behavior with
a MockProvider, so they're structural tests of the router itself and worth
keeping.
"""

import logging
import os
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

from heylook_llm.providers.base import BaseProvider
from heylook_llm.router import ModelRouter


class MockProvider(BaseProvider):
    """Minimal provider that satisfies BaseProvider's abstract contract.

    Overrides the constructor to skip MLX initialization. Attaches a MagicMock
    `unload` per instance so tests can assert it was called during LRU eviction.
    """

    def __init__(self, model_id, model_config, is_debug):
        self.model_id = model_id
        self.model_config = model_config
        self.is_debug = is_debug
        self.unload = MagicMock()

    def load_model(self):
        pass

    def create_chat_completion(self, request):
        pass


_BASE_TOML = textwrap.dedent("""
    default_model = "{default_model}"
    max_loaded_models = {max_loaded_models}

    [[models]]
    id = "model1-mlx"
    provider = "mlx"
    enabled = {model1_enabled}
    config = {{ model_path = "/fake/path/model1" }}

    [[models]]
    id = "model2-llama"
    provider = "mlx"
    enabled = {model2_enabled}
    config = {{ model_path = "/fake/path/model2" }}

    [[models]]
    id = "model3-mlx"
    provider = "mlx"
    enabled = {model3_enabled}
    config = {{ model_path = "/fake/path/model3" }}
""").strip()


def _render_config(
    *,
    default_model: str = "model1-mlx",
    max_loaded_models: int = 2,
    model1_enabled: bool = True,
    model2_enabled: bool = True,
    model3_enabled: bool = True,
) -> str:
    return _BASE_TOML.format(
        default_model=default_model,
        max_loaded_models=max_loaded_models,
        model1_enabled=str(model1_enabled).lower(),
        model2_enabled=str(model2_enabled).lower(),
        model3_enabled=str(model3_enabled).lower(),
    )


@patch('heylook_llm.router.MLXProvider', new=MockProvider)
class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.temp_config_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.toml'
        )
        self.temp_config_file.write(_render_config())
        self.temp_config_file.close()
        self.config_path = self.temp_config_file.name

    def tearDown(self):
        os.unlink(self.config_path)

    def _rewrite_config(self, **overrides) -> None:
        with open(self.config_path, 'w') as f:
            f.write(_render_config(**overrides))

    def test_initialization(self):
        """Router inits cleanly with no enabled models and no default."""
        self._rewrite_config(
            default_model="",
            model1_enabled=False,
            model2_enabled=False,
            model3_enabled=False,
        )
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        self.assertEqual(router.list_available_models(), [])
        self.assertEqual(len(router.providers), 0)

    def test_get_provider_loads_and_caches(self):
        """Second get_provider for the same id returns the cached instance."""
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None
        )

        provider1 = router.get_provider('model1-mlx')
        self.assertEqual(provider1.model_id, 'model1-mlx')
        self.assertEqual(len(router.providers), 1)

        provider1_cached = router.get_provider('model1-mlx')
        self.assertIs(provider1, provider1_cached)
        self.assertEqual(len(router.providers), 1)

    def test_lru_eviction(self):
        """Third load evicts the oldest non-pinned provider and calls unload()."""
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None
        )

        provider1 = router.get_provider('model1-mlx')
        provider2 = router.get_provider('model2-llama')
        self.assertEqual(len(router.providers), 2)

        router.get_provider('model3-mlx')
        self.assertNotIn('model1-mlx', router.providers)
        self.assertIn('model2-llama', router.providers)
        self.assertIn('model3-mlx', router.providers)
        self.assertEqual(len(router.providers), 2)

        provider1.unload.assert_called_once()
        provider2.unload.assert_not_called()

    def test_hot_swapping(self):
        """Re-accessing already-loaded models is a cache hit (no reload)."""
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None
        )

        router.get_provider('model1-mlx')
        router.get_provider('model2-llama')
        router.get_provider('model1-mlx')
        router.get_provider('model2-llama')

        self.assertEqual(len(router.providers), 2)

    def test_max_loaded_models_one(self):
        """With max=1, a new load always evicts the previous."""
        self._rewrite_config(max_loaded_models=1)
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.DEBUG, initial_model_id=None
        )

        provider1 = router.get_provider('model1-mlx')
        self.assertIn('model1-mlx', router.providers)
        self.assertEqual(len(router.providers), 1)

        router.get_provider('model2-llama')
        self.assertNotIn('model1-mlx', router.providers)
        self.assertIn('model2-llama', router.providers)
        self.assertEqual(len(router.providers), 1)
        provider1.unload.assert_called_once()


if __name__ == '__main__':
    unittest.main()
