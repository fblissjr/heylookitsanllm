"""reload_config() must push per_request defaults into LOADED providers.

A provider is constructed with a snapshot of its config dict and reads
per_request defaults from it at request time. Without the refresh, a PATCH to
temperature/enable_thinking on a loaded model reported "no
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
    real attribute AND the declared provider_name (the class-match guard)."""

    provider_name = "mlx"

    def __init__(self, model_id, model_config, is_debug):
        super().__init__(model_id, model_config, is_debug)

    def load_model(self):
        pass

    def create_chat_completion(self, request, abort_event=None):  # pragma: no cover
        yield from ()

    def unload(self, *, drain: bool = True):
        pass


def _toml(temperature=None, context_length=None, model_id="m1", enabled=True,
          model_path="/fake/path/m1"):
    lines = [
        f'default_model = "{model_id}"',
        "max_loaded_models = 1",
        "",
        "[[models]]",
        f'id = "{model_id}"',
        'provider = "mlx"',
        f"enabled = {str(enabled).lower()}",
        "",
        "[models.config]",
        f'model_path = "{model_path}"',
    ]
    if temperature is not None:
        lines.append(f"temperature = {temperature}")
    if context_length is not None:
        lines.append(f"context_length = {context_length}")
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

        self._rewrite(context_length=4096)
        router.reload_config()

        # The loaded process really does keep the old value; refreshing the
        # dict would make the dict lie in the OPPOSITE direction (claiming a
        # live change the process never saw).
        self.assertIsNone(provider.config.get("context_length"))

    def test_removed_entry_is_skipped(self):
        router, provider = self._router_with_loaded_provider()
        # m1 vanishes from the config while its provider is still loaded.
        self._rewrite(model_id="m2")
        router.reload_config()  # must not raise
        self.assertIsNone(provider.config.get("temperature"))

    def test_disabled_but_loaded_model_still_refreshes(self):
        # POST /{id}/toggle disables WITHOUT unloading, and re-enabling must
        # not resurrect stale defaults -- so the refresh must look the entry
        # up regardless of `enabled` (get_model_config filters on it).
        router, provider = self._router_with_loaded_provider()

        self._rewrite(temperature=0.9, enabled=False)
        router.reload_config()

        self.assertEqual(provider.config["temperature"], 0.9)

    def test_stale_reload_fields_tracks_the_loaded_snapshot(self):
        router, provider = self._router_with_loaded_provider()
        self.assertEqual(router.stale_reload_fields("m1"), [])

        self._rewrite(context_length=4096)
        router.reload_config()

        # requires_reload change on a loaded model -> reported stale...
        self.assertEqual(router.stale_reload_fields("m1"), ["context_length"])
        # ...while a per_request change never is (it refreshes live).
        self._rewrite(context_length=4096, temperature=0.9)
        router.reload_config()
        self.assertEqual(router.stale_reload_fields("m1"), ["context_length"])
        # Unloaded models report nothing regardless of saved diffs.
        self.assertEqual(router.stale_reload_fields("m2"), [])

    def test_status_says_busy_or_idle_on_any_engine(self):
        """/status's requests_active was null for every model; a client told
        busy from idle by timing a generation, which queued behind the one
        it was measuring."""
        router, provider = self._router_with_loaded_provider()
        self.assertEqual(router.get_model_status("m1")["requests_active"], 0)
        with provider.generation_active():
            self.assertEqual(router.get_model_status("m1")["requests_active"], 1)
        self.assertNotIn("requests_active", router.get_model_status("m2"))

    def test_a_template_file_swap_is_stale_though_no_field_moved(self):
        """The template binds at load but lives in a file beside the
        weights; editing it changes what a respawn uses with the config
        untouched, and the row must say reload."""
        with tempfile.TemporaryDirectory() as model_dir:
            template = os.path.join(model_dir, "chat_template.jinja")
            with open(template, "w") as f:
                f.write("{{ messages }} A")
            self._rewrite(model_path=model_dir)
            router, provider = self._router_with_loaded_provider()
            provider.loaded_chat_template = "{{ messages }} A"
            self.assertEqual(router.stale_reload_fields("m1"), [])
            with open(template, "w") as f:
                f.write("{{ messages }} B")
            self.assertEqual(router.stale_reload_fields("m1"), ["chat_template"])


if __name__ == "__main__":
    unittest.main()
