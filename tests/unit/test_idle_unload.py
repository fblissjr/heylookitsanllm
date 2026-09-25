"""Tests for idle-based model unloading (C2 of Slice 1.5).

``ModelRouter.unload_idle_models(now_ts)`` scans loaded providers, computes
each model's effective idle-unload threshold (per-model
override beats global default), and unloads any model whose last-used
timestamp is further in the past than its threshold. ``MemoryManager.tick``
drives this from the existing 60s resource-snapshot loop.

Tests use a fake clock injected via a ``now_ts`` argument to
``unload_idle_models`` and direct mutation of ``router._last_used_ts`` --
no real sleep, no real MLX.
"""

from __future__ import annotations

import logging
import textwrap
import time
import unittest
from unittest.mock import patch

import pytest

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider as _MockProvider


# Template has explicit unload_after_idle_seconds placeholders per model +
# a global idle_unload_seconds at the top.
_IDLE_TOML = textwrap.dedent("""
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


@pytest.fixture
def make_router(tmp_path):
    """A router over _IDLE_TOML rendered with the given overrides, with
    MockProvider standing in for MLXProvider."""
    config_path = tmp_path / "models.toml"

    def _make(**render_kwargs) -> ModelRouter:
        config_path.write_text(_render(**render_kwargs))
        return ModelRouter(
            config_path=str(config_path),
            log_level=logging.INFO,
            initial_model_id=None,
        )

    with patch("heylook_llm.router.MLXProvider", new=_MockProvider):
        yield _make


# Every get_provider stamps last-used: a first load populates it, and a cache
# hit refreshes it past an old stamp.
@pytest.mark.parametrize("cache_hit", [False, True], ids=["load_populates", "cache_hit_refreshes"])
def test_get_provider_stamps_last_used(make_router, cache_hit):
    router = make_router()
    if cache_hit:
        router.get_provider("model-fast")
        router._last_used_ts["model-fast"] = 1_000.0
    before = time.time()
    router.get_provider("model-fast")
    assert router._last_used_ts["model-fast"] >= before


# Effective threshold = per-model override, else global; 0 disables at either
# level; a model unloads iff idle longer than its threshold. Every loaded
# model was last used at t=1000. Defaults: global 1800s, fast 60s, slow 3600s.
# - override_wins_over_global: at 120s idle fast (60s) goes, global (1800s)
#   stays.
# - global_zero: idle_unload_seconds=0 disables idle unload globally, but a
#   model that sets its own non-zero value still unloads; slow's own 0 and
#   global's inherited 0 stay.
@pytest.mark.parametrize("render, loaded, now_ts, unloaded", [
    ({}, ["model-fast"], 1_120.0, {"model-fast"}),
    ({}, ["model-fast"], 1_030.0, set()),
    ({}, ["model-fast", "model-global"], 1_120.0, {"model-fast"}),
    ({"fast_override": "0"}, ["model-fast"], 1_000_000.0, set()),
    ({"global_idle": 0, "fast_override": "60", "slow_override": "0"},
     ["model-fast", "model-slow", "model-global"], 1_000_000.0, {"model-fast"}),
], ids=["idle_past_window_unloads", "recently_used_stays", "override_wins_over_global",
        "per_model_zero_disables", "global_zero_disables_all_but_explicit_override"])
def test_idle_threshold(make_router, render, loaded, now_ts, unloaded):
    router = make_router(**render)
    providers = {mid: router.get_provider(mid) for mid in loaded}
    for mid in loaded:
        router._last_used_ts[mid] = 1_000.0

    router.unload_idle_models(now_ts=now_ts)

    for mid, provider in providers.items():
        if mid in unloaded:
            assert mid not in router.providers
            provider.unload.assert_called_once()
        else:
            assert mid in router.providers
            provider.unload.assert_not_called()


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
