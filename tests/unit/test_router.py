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
from unittest.mock import patch

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider


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

    def test_no_startup_preload_without_explicit_request(self):
        """A configured `default_model` alone must NOT load anything at startup.

        Claim: startup preload is opt-in. Delete this and the server silently
        goes back to pinning a multi-GB model into RAM on every boot just
        because models.toml names a routing default.
        """
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        self.assertEqual(len(router.providers), 0)

    def test_explicit_initial_model_is_preloaded(self):
        """`--model-id` is the one thing that still pre-warms at startup.

        Claim: opting out of the implicit preload must not remove the explicit
        one.
        """
        router = ModelRouter(
            config_path=self.config_path,
            log_level=logging.INFO,
            initial_model_id='model2-llama',
        )
        self.assertIn('model2-llama', router.providers)

    def test_literal_none_default_model_is_treated_as_unset(self):
        """`default_model = "none"` means UNSET, not a model called "none".

        Claim: `model_importer` and `model_service` both write the literal
        string "none" when a scan finds no models, and it is truthy -- so it
        sailed past the "no default configured" branch and every model-less
        request died on `Model 'none' not found`. Delete this and that
        default-shipped config produces a nonsense error instead of the
        actionable "no default configured. Available: [...]".
        """
        self._rewrite_config(default_model="none")
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        self.assertIsNone(router.app_config.default_model)
        with self.assertRaises(ValueError) as ctx:
            router.get_provider('')
        self.assertIn('no default configured', str(ctx.exception))

    def test_stale_default_model_warns_at_startup_without_loading(self):
        """A `default_model` naming nothing real is reported at boot.

        Claim: startup used to validate the default because it preloaded it.
        Now that preload is opt-in, the only signal left would be a failed
        request at runtime -- so validate (warn) without loading. Delete this
        and a typo'd default is invisible until someone sends a model-less
        request.
        """
        self._rewrite_config(default_model="typo-model")
        with self.assertLogs(level=logging.WARNING) as logs:
            router = ModelRouter(
                config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
            )
        self.assertEqual(len(router.providers), 0)  # warned, did NOT load
        self.assertTrue(
            any('typo-model' in m for m in logs.output),
            f"expected a warning naming the stale default, got: {logs.output}",
        )

    def test_valid_default_model_does_not_warn(self):
        """The happy path stays quiet -- a real default is not a problem."""
        with self.assertNoLogs(level=logging.WARNING):
            ModelRouter(
                config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
            )

    def test_default_model_still_routes_unspecified_requests(self):
        """`default_model` keeps its routing role: a request with no model id
        resolves to it (and only then loads it).

        Claim: the preload change must not turn `default_model` into dead
        config.
        """
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        provider = router.get_provider('')
        self.assertEqual(provider.model_id, 'model1-mlx')


if __name__ == '__main__':
    unittest.main()
