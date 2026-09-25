"""Tests for idle-based model unloading (C2 of Slice 1.5).

``ModelRouter.unload_idle_models(now_ts)`` scans loaded providers, computes
each model's effective idle-unload threshold (per-model
override beats global default), and unloads any model whose last-used
timestamp is further in the past than its threshold. ``MemoryManager.tick``
drives this from the existing 60s resource-snapshot loop.

Tests use a fake clock injected via a ``now_ts`` argument to
``unload_idle_models`` (plus the router's clock for the stamps
``get_provider`` writes, and direct setting of ``router._last_used_ts`` in the
threshold table) -- no real sleep, no real MLX.
"""

from __future__ import annotations

import logging
import textwrap
import time
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


class _Clock:
    """The router's `time` module with a settable `time()`; everything else
    passes through to the real module."""

    def __init__(self, now: float):
        self.now = now

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


# Every get_provider counts as use, observed as the idle unload it defers
# (model-fast's threshold is 60s): a first load starts the window, and a
# cache hit after the window has passed starts it again.
@pytest.mark.parametrize("cache_hit", [False, True], ids=["load_populates", "cache_hit_refreshes"])
def test_get_provider_stamps_last_used(make_router, cache_hit):
    router = make_router()
    clock = _Clock(1_000.0)
    with patch("heylook_llm.router.time", new=clock):
        router.get_provider("model-fast")
        if cache_hit:
            clock.now = 1_100.0  # past the 60s window since the load
            router.get_provider("model-fast")
    used_at = clock.now
    assert router.unload_idle_models(now_ts=used_at + 59) == []
    assert "model-fast" in router.providers
    assert router.unload_idle_models(now_ts=used_at + 61) == ["model-fast"]


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
