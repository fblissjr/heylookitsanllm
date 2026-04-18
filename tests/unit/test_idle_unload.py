"""Tests for idle-based model unloading (C2 of Slice 1.5).

``ModelRouter.unload_idle_models(now_ts)`` scans loaded providers, skips
pinned ones, computes each model's effective idle-unload threshold (per-model
override beats global default), and unloads any model whose last-used
timestamp is further in the past than its threshold. ``MemoryManager.tick``
drives this from the existing 60s resource-snapshot loop.

Tests use a fake clock injected via a ``now_ts`` argument to
``unload_idle_models`` and direct mutation of ``router._last_used_ts`` --
no real sleep, no real MLX.
"""

from __future__ import annotations

import logging
import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider as _MockProvider


# Template has explicit unload_after_idle_seconds placeholders per model +
# a global idle_unload_seconds at the top.
_IDLE_TOML = textwrap.dedent("""
    default_model = "model-fast"
    max_loaded_models = 3
    idle_unload_seconds = {global_idle}

    [[models]]
    id = "model-fast"
    provider = "mlx"
    enabled = true
    config = {{ model_path = "/fake/fast", unload_after_idle_seconds = {fast_override} }}

    [[models]]
    id = "model-slow"
    provider = "mlx"
    enabled = true
    config = {{ model_path = "/fake/slow", unload_after_idle_seconds = {slow_override} }}

    [[models]]
    id = "model-global"
    provider = "mlx"
    enabled = true
    config = {{ model_path = "/fake/global" }}
""").strip()


def _render(
    *,
    global_idle: int = 1800,
    fast_override: str = "60",
    slow_override: str = "3600",
) -> str:
    return _IDLE_TOML.format(
        global_idle=global_idle,
        fast_override=fast_override,
        slow_override=slow_override,
    )


@patch("heylook_llm.router.MLXProvider", new=_MockProvider)
class TestIdleUnload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml")
        self.tmp.write(_render())
        self.tmp.close()
        self.config_path = self.tmp.name

    def tearDown(self):
        os.unlink(self.config_path)

    def _rewrite(self, **kwargs) -> None:
        with open(self.config_path, "w") as f:
            f.write(_render(**kwargs))

    def _router(self) -> ModelRouter:
        return ModelRouter(
            config_path=self.config_path,
            log_level=logging.INFO,
            initial_model_id=None,
        )

    def test_loads_populate_last_used(self):
        router = self._router()
        router.get_provider("model-fast")
        self.assertIn("model-fast", router._last_used_ts)

    def test_idle_model_unloads_after_window(self):
        router = self._router()
        fast = router.get_provider("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0

        # 120s past its 60s window.
        router.unload_idle_models(now_ts=1_120.0)

        self.assertNotIn("model-fast", router.providers)
        fast.unload.assert_called_once()

    def test_recently_used_model_stays(self):
        router = self._router()
        router.get_provider("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0

        # 30s into its 60s window.
        router.unload_idle_models(now_ts=1_030.0)

        self.assertIn("model-fast", router.providers)

    def test_pinned_model_exempt(self):
        router = self._router()
        fast = router.get_provider("model-fast")
        router.pin_model("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0

        router.unload_idle_models(now_ts=10_000.0)

        self.assertIn("model-fast", router.providers)
        fast.unload.assert_not_called()

    def test_per_model_override_wins_over_global(self):
        """model-fast has override=60s, global is 1800s. At 120s idle it should
        unload; model-global (no override) shouldn't."""
        router = self._router()
        fast = router.get_provider("model-fast")
        glob = router.get_provider("model-global")
        router._last_used_ts["model-fast"] = 1_000.0
        router._last_used_ts["model-global"] = 1_000.0

        router.unload_idle_models(now_ts=1_120.0)

        self.assertNotIn("model-fast", router.providers)
        self.assertIn("model-global", router.providers)
        fast.unload.assert_called_once()
        glob.unload.assert_not_called()

    def test_per_model_zero_disables_unload(self):
        """unload_after_idle_seconds=0 means never idle-unload this model."""
        self._rewrite(fast_override="0")
        router = self._router()
        fast = router.get_provider("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0

        router.unload_idle_models(now_ts=1_000_000.0)

        self.assertIn("model-fast", router.providers)
        fast.unload.assert_not_called()

    def test_global_zero_disables_everything(self):
        """idle_unload_seconds=0 globally disables idle unload. Per-model
        overrides still apply for models that set their own non-zero value."""
        self._rewrite(global_idle=0, fast_override="60", slow_override="0")
        router = self._router()
        fast = router.get_provider("model-fast")
        slow = router.get_provider("model-slow")
        glob = router.get_provider("model-global")
        router._last_used_ts["model-fast"] = 1_000.0
        router._last_used_ts["model-slow"] = 1_000.0
        router._last_used_ts["model-global"] = 1_000.0

        router.unload_idle_models(now_ts=1_000_000.0)

        # model-fast still unloaded (explicit per-model override).
        self.assertNotIn("model-fast", router.providers)
        fast.unload.assert_called_once()
        # model-slow and model-global stay (slow's own 0 + global 0).
        self.assertIn("model-slow", router.providers)
        self.assertIn("model-global", router.providers)
        slow.unload.assert_not_called()
        glob.unload.assert_not_called()

    def test_cache_hit_refreshes_last_used(self):
        router = self._router()
        router.get_provider("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0

        # Cache hit should refresh the timestamp past the original 1000.0.
        import time as _time

        before = _time.time()
        router.get_provider("model-fast")
        self.assertGreaterEqual(router._last_used_ts["model-fast"], before)


class TestMaxLoadedModelsDefault(unittest.TestCase):
    """AppConfig schema default flipped from 2 to 1 in C2. A models.toml that
    doesn't set ``max_loaded_models`` should default to 1, matching the user's
    already-explicit value in production."""

    def test_schema_default_is_one(self):
        from heylook_llm.config import AppConfig

        cfg = AppConfig(models=[])
        self.assertEqual(cfg.max_loaded_models, 1)


if __name__ == "__main__":
    unittest.main()
