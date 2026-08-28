# tests/unit/test_router_races.py
"""Phase 1 item 5: router TOCTOU + idle-unload race.

Two defects, both rare solo but with OOM/crash-class consequences:

1. TOCTOU over-capacity load: the capacity check + evict ran under
   cache_lock, but the load and the publish ran OUTSIDE it. Two concurrent
   requests for two DIFFERENT models both passed the check and loaded
   simultaneously -- two full model weights in memory on a box budgeted for
   max_loaded_models. Fix: reserve capacity in the router's _loading
   side-set under cache_lock before loading (self.providers keeps meaning
   "real, loaded providers"); concurrent loaders wait until a slot frees.

2. Idle unload vs queued requests: a request that hit the cache and is
   WAITING at the FIFO generation gate counts as neither "active" (that
   starts after gate acquire) nor recently-used (last_used was stamped at
   cache hit, and gate waits can exceed the idle threshold). The idle tick
   could tear the provider down under it. Fix: the unload decision checks
   the provider's generation queue stats (active + waiting) under the SAME
   cache_lock as the pop.
"""

from __future__ import annotations

import logging
import os
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

from heylook_llm.providers.base import BaseProvider
from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider


_RACE_TOML = textwrap.dedent("""
    max_loaded_models = 1
    idle_unload_seconds = 60

    [[models]]
    id = "model-a"
    provider = "mlx"
    enabled = true
    config = { model_path = "/fake/a" }

    [[models]]
    id = "model-b"
    provider = "mlx"
    enabled = true
    config = { model_path = "/fake/b" }
""").strip()


class SlowTrackingProvider(BaseProvider):
    """Tracks how many models hold (or are loading) weights concurrently.

    Weights start consuming memory when load_model BEGINS, so the live set
    is entered at load start and left at unload.
    """

    _lock = threading.Lock()
    live: set[str] = set()
    peak = 0
    load_seconds = 0.15

    def __init__(self, model_id, model_config, is_debug):
        self.model_id = model_id
        self.model_config = model_config
        self.is_debug = is_debug

    def load_model(self):
        cls = SlowTrackingProvider
        with cls._lock:
            cls.live.add(self.model_id)
            cls.peak = max(cls.peak, len(cls.live))
        time.sleep(cls.load_seconds)

    def unload(self):
        cls = SlowTrackingProvider
        with cls._lock:
            cls.live.discard(self.model_id)

    def create_chat_completion(self, request, abort_event=None):  # pragma: no cover
        pass

    @classmethod
    def reset(cls):
        with cls._lock:
            cls.live = set()
            cls.peak = 0


class QueueStatsProvider(MockProvider):
    """MockProvider variant with a controllable generation queue snapshot."""

    def __init__(self, model_id, model_config, is_debug):
        super().__init__(model_id, model_config, is_debug)
        self.queue_stats = {"active": 0, "waiting": 0, "max_waiting": 10, "capacity": 1}

    def generation_queue_stats(self):
        return self.queue_stats


class _RouterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml")
        self.tmp.write(_RACE_TOML)
        self.tmp.close()
        self.config_path = self.tmp.name

    def tearDown(self):
        os.unlink(self.config_path)

    def _router(self) -> ModelRouter:
        return ModelRouter(
            config_path=self.config_path,
            log_level=logging.INFO,
            initial_model_id=None,
        )


@patch("heylook_llm.router.MLXProvider", new=SlowTrackingProvider)
class TestLoadCapacityTOCTOU(_RouterTestBase):
    def setUp(self):
        super().setUp()
        SlowTrackingProvider.reset()

    def test_concurrent_different_model_loads_respect_capacity(self):
        router = self._router()
        errors = []

        def load(model_id):
            try:
                router.get_provider(model_id)
            except Exception as e:  # pragma: no cover - failure detail
                errors.append(e)

        t_a = threading.Thread(target=load, args=("model-a",))
        t_b = threading.Thread(target=load, args=("model-b",))
        t_a.start()
        time.sleep(0.03)  # A is mid-load when B arrives
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)
        self.assertFalse(t_a.is_alive() or t_b.is_alive(), "loader deadlocked")

        self.assertEqual(errors, [])
        # Never two models' weights in memory at once (max_loaded_models=1).
        self.assertLessEqual(
            SlowTrackingProvider.peak, 1,
            f"{SlowTrackingProvider.peak} models held weights concurrently "
            f"with max_loaded_models=1 -- capacity check is check-then-act",
        )
        # And the cache itself never ends up over capacity.
        self.assertLessEqual(len(router.providers), 1)

    def test_slot_reservation_invisible_to_readers(self):
        """While a model loads, reader APIs must not surface the reservation."""
        router = self._router()
        seen = {}

        def load():
            router.get_provider("model-a")

        t = threading.Thread(target=load)
        t.start()
        time.sleep(0.05)  # mid-load
        seen["loaded_models"] = dict(router.get_loaded_models())
        seen["current"] = router.get_current_model_id()
        seen["status_loaded"] = router.get_model_status("model-a")["loaded"]
        t.join(timeout=10)
        self.assertFalse(t.is_alive())

        for key, value in seen.items():
            with self.subTest(reader=key):
                self.assertFalse(
                    value,
                    f"{key} exposed a mid-load reservation as a real provider: {value!r}",
                )

    def test_unload_model_ignores_in_progress_load(self):
        """unload_model on a mid-load id reports not-loaded and must not
        destroy the reservation (the load still completes and publishes)."""
        router = self._router()

        t = threading.Thread(target=lambda: router.get_provider("model-a"))
        t.start()
        time.sleep(0.05)
        self.assertFalse(router.unload_model("model-a"))
        t.join(timeout=10)
        self.assertFalse(t.is_alive())
        self.assertIn("model-a", router.providers)


@patch("heylook_llm.router.MLXProvider", new=QueueStatsProvider)
class TestIdleUnloadRespectsQueue(_RouterTestBase):
    def _idle_router_with(self, active=0, waiting=0):
        router = self._router()
        provider = router.get_provider("model-a")
        provider.queue_stats = {
            "active": active, "waiting": waiting, "max_waiting": 10, "capacity": 1,
        }
        router._last_used_ts["model-a"] = 1_000.0
        return router, provider

    def test_waiting_request_blocks_idle_unload(self):
        # A request has been queued at the generation gate longer than the
        # idle window (big model, deep queue). Tearing the provider down now
        # would delete weights the queued request is about to use.
        router, provider = self._idle_router_with(waiting=1)
        unloaded = router.unload_idle_models(now_ts=2_000.0)
        self.assertEqual(unloaded, [])
        self.assertIn("model-a", router.providers)
        provider.unload.assert_not_called()

    def test_active_generation_blocks_idle_unload(self):
        router, provider = self._idle_router_with(active=1)
        unloaded = router.unload_idle_models(now_ts=2_000.0)
        self.assertEqual(unloaded, [])
        provider.unload.assert_not_called()

    def test_quiescent_provider_still_unloads(self):
        router, provider = self._idle_router_with()
        unloaded = router.unload_idle_models(now_ts=2_000.0)
        self.assertEqual(unloaded, ["model-a"])
        provider.unload.assert_called_once()

    def test_provider_without_queue_stats_still_unloads(self):
        # Providers with no generation gate (stats None) keep old behavior.
        router, provider = self._idle_router_with()
        provider.generation_queue_stats = lambda: None
        unloaded = router.unload_idle_models(now_ts=2_000.0)
        self.assertEqual(unloaded, ["model-a"])

    def test_a_gateless_provider_generating_is_not_unloaded(self):
        """The gguf shape: no generation gate, so queue stats are None.

        `unload_idle_models` skips a generating model in its candidate scan,
        then RELEASES the lock before `_unload_idle` re-acquires it and pops.
        Everything that becomes true in that window has to be caught by the
        re-check -- and the re-check tested only queue stats, which are None
        for a provider that queues its own requests (llama-server does). So a
        gguf generation starting inside the window took a SIGTERM under an
        open stream: exactly the "gguf had strictly less protection" hole
        `_is_generating` exists to close, still open on the path that does the
        popping.

        Driven through `_unload_idle` directly, because that IS the window:
        going through `unload_idle_models` would let the candidate scan catch
        it and prove nothing about the re-check.
        """
        router, provider = self._idle_router_with()
        provider.generation_queue_stats = lambda: None   # no gate, like gguf
        # The REAL counter, via the context manager the providers use, rather
        # than a patched attribute: active_generations is a read-only property
        # and the point is that the router reads what a live run actually sets.
        with provider.generation_active():               # a run began in the window
            self.assertFalse(router._unload_idle("model-a"))
        self.assertIn("model-a", router.providers)
        provider.unload.assert_not_called()

    def test_a_model_pinned_inside_the_window_is_not_unloaded(self):
        # Same window, the other condition the scan skips on. Pinning is how a
        # batch job holds a model; losing it mid-job is the thing pin prevents.
        router, provider = self._idle_router_with()
        provider.generation_queue_stats = lambda: None
        router._pinned.add("model-a")
        self.assertFalse(router._unload_idle("model-a"))
        self.assertIn("model-a", router.providers)
        provider.unload.assert_not_called()

    def test_generating_counts_as_use_so_the_next_tick_does_not_unload(self):
        """Skipping without re-stamping only DEFERRED the unload by a tick.

        `_last_used_ts` is written at REQUEST time, so a generation longer than
        the idle threshold is already stale for its whole second half. The tick
        after it finishes would then tear the model down the instant the reply
        lands -- immediately before the user's next message.
        """
        router, provider = self._idle_router_with(active=1)
        self.assertEqual(router.unload_idle_models(now_ts=2_000.0), [])
        # The run ends. `_unload_idle` stamps with the wall clock rather than
        # the injected one, so assert the STAMP MOVED rather than pinning a
        # fake-clock value the production path never sees.
        self.assertGreater(router._last_used_ts["model-a"], 1_000.0)
        provider.queue_stats = {"active": 0, "waiting": 0, "max_waiting": 10, "capacity": 1}
        self.assertIn("model-a", router.providers)


if __name__ == "__main__":
    unittest.main()
