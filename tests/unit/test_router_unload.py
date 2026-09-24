# tests/unit/test_router_unload.py
"""
Unit tests for ModelRouter.unload_model on a minimal router.
(Model pinning, which this file also covered, was removed in v2.0.126.)

Note: Eviction/unload calls gc.collect + mx.clear_cache which segfault when
mixing real MLX Metal state with MagicMock providers. We test selection logic
directly and use careful fixture scoping to avoid the crash.
"""

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

# ModelRouter pulls in MLX at module load. Import it REAL, skipping this
# module entirely where MLX is absent. The previous spelling imported it
# under an import-scoped `patch.dict(sys.modules, mocks)` -- which looked
# safe but was an ACTIVE defect (ordering review 2026-08-18): patch.dict
# restores by clear+update, so any module FIRST-imported inside the window
# (the whole provider chain, and real numpy through it) was
# EVICTED from sys.modules on exit. When this file was the first importer
# of the router chain (any per-file invocation: IDE run-this-file,
# changed-files CI, file-list sharding), later re-imports crashed --
# `ImportError: cannot load module more than once per process` across the
# router files, and a hard SEGFAULT paired with test_generation_core. Full
# runs were healed only by collection luck (test_admin_offloop importing
# the chain unmocked first). These tests never touch MLX at runtime (their
# providers are `.provider == "test"`), so faking the stack bought nothing.
pytest.importorskip("mlx.core", reason="router import pulls the MLX provider stack")
from heylook_llm.router import ModelRouter  # noqa: E402


def _make_fresh_router():
    """Create a minimal ModelRouter with fake providers."""
    r = object.__new__(ModelRouter)
    r.providers = OrderedDict()
    r.cache_lock = threading.RLock()
    r.max_loaded_models = 2
    r.log_level = 40

    for name in ["model-a", "model-b"]:
        p = MagicMock()
        p.provider = "test"
        r.providers[name] = p

    return r


@pytest.fixture
def router():
    return _make_fresh_router()


def test_unload_a_loaded_model_succeeds(router):
    assert router.unload_model("model-a") is True
    assert "model-a" not in router.providers


def test_unload_nonexistent_returns_false(router):
    assert router.unload_model("nonexistent") is False
