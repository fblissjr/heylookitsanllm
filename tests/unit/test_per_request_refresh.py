"""reload_config() must push per_request defaults into LOADED providers.

A provider is constructed with a snapshot of its config dict and reads
per_request defaults from it at request time. Without the refresh, a PATCH to
default_sampler/temperature/enable_thinking on a loaded model reported "no
reload required" while the loaded process kept serving the old default -- the
stale-snapshot lie the effect classification exists to prevent, relocated
into the per_request bucket (found by the 2026-08-11 review; the first real
consumer of the classes, the v3 config editor, rendered the false promise).

requires_reload keys must NOT refresh: the reported reload is their real cost.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from heylook_llm.router import ModelRouter
from heylook_llm.providers.base import BaseProvider


class _ConfigDictProvider(BaseProvider):
    """Follows the BaseProvider convention (self.config = the dict), unlike
    the shared MockProvider, which renames it -- the refresh keys off the
    real attribute."""

    def __init__(self, model_id, model_config, is_debug):
        super().__init__(model_id, model_config, is_debug)

    def load_model(self):
        pass

    def create_chat_completion(self, request, abort_event=None):  # pragma: no cover
        yield from ()

    def unload(self):
        pass


def _toml(temperature=None, max_kv_size=None, model_id="m1"):
    lines = [
        f'default_model = "{model_id}"',
        "max_loaded_models = 1",
        "",
        "[[models]]",
        f'id = "{model_id}"',
        'provider = "mlx"',
        "enabled = true",
        "",
        "[models.config]",
        'model_path = "/fake/path/m1"',
    ]
    if temperature is not None:
        lines.append(f"temperature = {temperature}")
    if max_kv_size is not None:
        lines.append(f"max_kv_size = {max_kv_size}")
    return "\n".join(lines) + "\n"


@patch("heylook_llm.router.MLXProvider", new=_ConfigDictProvider)
class TestPerRequestRefresh(unittest.TestCase):
    def setUp(self):
        f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml")
        f.write(_toml())
        f.close()
        self.config_path = f.name

    def tearDown(self):
        os.unlink(self.config_path)

    def _rewrite(self, **kwargs):
        with open(self.config_path, "w") as f:
            f.write(_toml(**kwargs))

    def _router_with_loaded_provider(self):
        router = ModelRouter(
            config_path=self.config_path, log_level=logging.INFO, initial_model_id=None
        )
        provider = router.get_provider("m1")
        return router, provider

    def test_per_request_default_reaches_loaded_provider(self):
        router, provider = self._router_with_loaded_provider()
        self.assertIsNone(provider.config.get("temperature"))

        self._rewrite(temperature=0.9)
        router.reload_config()

        self.assertEqual(provider.config["temperature"], 0.9)

    def test_clearing_a_per_request_default_reaches_loaded_provider(self):
        self._rewrite(temperature=0.9)
        router, provider = self._router_with_loaded_provider()
        self.assertEqual(provider.config["temperature"], 0.9)

        self._rewrite()  # key removed = back to the default (None)
        router.reload_config()

        self.assertIsNone(provider.config["temperature"])

    def test_requires_reload_key_stays_a_snapshot(self):
        router, provider = self._router_with_loaded_provider()

        self._rewrite(max_kv_size=4096)
        router.reload_config()

        # The loaded process really does keep the old value; refreshing the
        # dict would make the dict lie in the OPPOSITE direction (claiming a
        # live change the process never saw).
        self.assertIsNone(provider.config.get("max_kv_size"))

    def test_removed_entry_is_skipped(self):
        router, provider = self._router_with_loaded_provider()
        # m1 vanishes from the config while its provider is still loaded.
        self._rewrite(model_id="m2")
        router.reload_config()  # must not raise
        self.assertIsNone(provider.config.get("temperature"))


if __name__ == "__main__":
    unittest.main()
