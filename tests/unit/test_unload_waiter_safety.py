# tests/unit/test_unload_waiter_safety.py
"""Post-review follow-ups 1-2 from the 2026-07-06 /code-review pass.

1. Teardown waiter-safety at the right depth: MLXProvider.unload() waited
   only for ACTIVE generations, but the active counter decrements BEFORE
   gate.release() admits the next waiter -- so eviction/clear_cache (which
   never got _unload_idle's queue check) could free weights exactly as a
   woken waiter starts generating. Fix lives in unload() itself so every
   teardown caller (evict, clear_cache, explicit unload, idle) inherits it.

2. Bounded reservation wait: get_provider's capacity-reservation loop
   waited forever on other threads' in-flight loads -- a wedged load
   blocked admission of every OTHER model indefinitely and starved the
   all-pinned RuntimeError. The loop now raises after a deadline.
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

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider


# ---------------------------------------------------------------------------
# 1. unload() waits for gate waiters, not just actives
# ---------------------------------------------------------------------------

def _bare_provider(gate_stats_fn):
    """MLXProvider skeleton with just the attrs unload() touches.

    Avoids load_model()/Metal entirely; unload() must run on exactly this
    surface: _active_lock, _active_generations, generation_queue_stats(),
    _strategies, model_id.
    """
    from heylook_llm.providers.mlx_provider import MLXProvider

    p = object.__new__(MLXProvider)
    p.model_id = "waiter-safety-test"
    p._active_lock = threading.Lock()
    p._active_generations = 0
    p._strategies = {}
    p.generation_queue_stats = gate_stats_fn
    return p


class TestUnloadWaitsForWaiters(unittest.TestCase):
    def test_unload_waits_until_gate_has_no_waiters(self):
        # Gate reports one waiter for the first 0.3s, then quiescent --
        # models the decrement-before-release window where a queued request
        # is about to start generating on these weights.
        flip_at = time.monotonic() + 0.3

        def stats():
            waiting = 1 if time.monotonic() < flip_at else 0
            return {"active": 0, "waiting": waiting, "max_waiting": 10, "capacity": 1}

        p = _bare_provider(stats)
        start = time.monotonic()
        p.unload()
        elapsed = time.monotonic() - start

        self.assertGreaterEqual(
            elapsed, 0.25,
            "unload() returned while the generation gate still had a waiter "
            "-- weights freed under a request about to run",
        )

    def test_unload_immediate_when_quiescent(self):
        p = _bare_provider(
            lambda: {"active": 0, "waiting": 0, "max_waiting": 10, "capacity": 1}
        )
        start = time.monotonic()
        p.unload()
        self.assertLess(time.monotonic() - start, 1.0)


# ---------------------------------------------------------------------------
# 2. Bounded reservation wait in get_provider
# ---------------------------------------------------------------------------

_TOML = textwrap.dedent("""
    max_loaded_models = 1

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


@patch("heylook_llm.router.MLXProvider", new=MockProvider)
class TestReservationWaitBounded(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml")
        self.tmp.write(_TOML)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _router(self) -> ModelRouter:
        return ModelRouter(
            config_path=self.tmp.name, log_level=logging.INFO, initial_model_id=None
        )

    def test_wedged_inflight_load_times_out_instead_of_hanging(self):
        router = self._router()
        router._reservation_wait_timeout = 0.2
        # Simulate another thread's load that never publishes (wedged).
        router._loading.add("ghost-model")

        start = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            router.get_provider("model-a")
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 5.0, "timeout did not bound the wait")
        self.assertIn("ghost-model", str(ctx.exception))

    def test_all_pinned_still_raises_immediately(self):
        router = self._router()
        router.get_provider("model-a")
        router.pin_model("model-a")

        start = time.monotonic()
        with self.assertRaises(RuntimeError) as ctx:
            router.get_provider("model-b")
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertIn("pinned", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
