# tests/unit/test_router_pinning.py
"""
Unit tests for model pinning in ModelRouter.
Tests pin_model, unpin_model, eviction selection, and unload guard logic.

Note: Eviction/unload calls gc.collect + mx.clear_cache which segfault when
mixing real MLX Metal state with MagicMock providers. We test selection logic
directly and use careful fixture scoping to avoid the crash.
"""

import atexit
import sys
import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from helpers.mlx_mock import create_mlx_module_mocks

# Keep patch alive for entire module to avoid segfault from MLX C cleanup.
# atexit safety net: if the process tears down before pytest cleanup,
# stopping the patch prevents MLX C cleanup from colliding with MagicMock.
_mlx_mocks = create_mlx_module_mocks()
_patch = patch.dict(sys.modules, _mlx_mocks)
_patch.start()
atexit.register(_patch.stop)

from heylook_llm.router import ModelRouter  # noqa: E402 -- must import after patch


def _make_fresh_router():
    """Create a minimal ModelRouter with fake providers for pinning tests."""
    r = object.__new__(ModelRouter)
    r.providers = OrderedDict()
    r.cache_lock = threading.RLock()
    r._pinned = set()
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


class TestPinModel:
    def test_pin_loaded_model(self, router):
        router.pin_model("model-a")
        assert "model-a" in router._pinned

    def test_pin_unloaded_raises(self, router):
        with pytest.raises(ValueError, match="not currently loaded"):
            router.pin_model("nonexistent")

    def test_pin_idempotent(self, router):
        router.pin_model("model-a")
        router.pin_model("model-a")
        assert "model-a" in router._pinned


class TestUnpinModel:
    def test_unpin_pinned_model(self, router):
        router.pin_model("model-a")
        router.unpin_model("model-a")
        assert "model-a" not in router._pinned

    def test_unpin_nonexistent_is_safe(self, router):
        router.unpin_model("nonexistent")

    def test_unpin_holds_lock(self, router):
        """Verify unpin_model acquires cache_lock (thread safety fix)."""
        router.pin_model("model-a")
        lock_acquired = threading.Event()
        unpin_done = threading.Event()

        def hold_lock():
            with router.cache_lock:
                lock_acquired.set()
                unpin_done.wait(timeout=2.0)

        t = threading.Thread(target=hold_lock)
        t.start()
        lock_acquired.wait()

        def do_unpin():
            router.unpin_model("model-a")
            unpin_done.set()

        t2 = threading.Thread(target=do_unpin)
        t2.start()

        import time
        time.sleep(0.05)
        assert "model-a" in router._pinned

        unpin_done.set()
        t.join(timeout=2.0)
        t2.join(timeout=2.0)


class TestEvictionSelection:
    """Test which model _evict_lru_model would select, without calling it.

    _evict_lru_model iterates providers in LRU order and picks the first
    non-pinned model. We verify the same selection logic directly.
    """

    def _find_eviction_candidate(self, router):
        for model_id in router.providers:
            if model_id not in router._pinned:
                return model_id
        return None

    def test_eviction_skips_pinned(self, router):
        router.pin_model("model-a")
        candidate = self._find_eviction_candidate(router)
        assert candidate == "model-b"

    def test_eviction_picks_lru(self, router):
        candidate = self._find_eviction_candidate(router)
        assert candidate == "model-a"  # inserted first = LRU

    def test_all_pinned_no_candidate(self, router):
        router.pin_model("model-a")
        router.pin_model("model-b")
        candidate = self._find_eviction_candidate(router)
        assert candidate is None

    def test_all_pinned_raises_on_evict(self, router):
        router.pin_model("model-a")
        router.pin_model("model-b")
        with router.cache_lock:
            with pytest.raises(RuntimeError, match="All.*pinned"):
                router._evict_lru_model()


class TestUnloadPinnedGuard:
    def test_unload_pinned_without_force_raises(self, router):
        router.pin_model("model-a")
        with pytest.raises(RuntimeError, match="pinned"):
            router.unload_model("model-a")

    def test_unload_unpinned_succeeds(self, router):
        result = router.unload_model("model-a")
        assert result is True
        assert "model-a" not in router.providers

    def test_unload_nonexistent_returns_false(self, router):
        result = router.unload_model("nonexistent")
        assert result is False
