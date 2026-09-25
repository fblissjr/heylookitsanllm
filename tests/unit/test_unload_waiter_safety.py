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
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import patch

import pytest

from heylook_llm.router import ModelRouter

from _mock_provider import MockProvider


# ---------------------------------------------------------------------------
# 1. unload() waits for gate waiters, not just actives
# ---------------------------------------------------------------------------

def _text_engine_blocking_until(provider, monkeypatch, release: threading.Event):
    """Stand the text engine in with one that yields a chunk only once
    ``release`` is set, and give the provider a processor, so a real
    ``create_chat_completion`` holds the gate and the active count."""
    from helpers.mlx_mock import create_mock_model, create_mock_processor

    mod = sys.modules[type(provider).__module__]

    class _Chunk:
        text = "x"

    def generate(self, *a, **k):
        release.wait(5)
        yield _Chunk()

    monkeypatch.setattr(mod.UnifiedTextStrategy, "generate", generate)
    provider.model = create_mock_model()
    provider.processor = create_mock_processor()
    provider._compile_strategies()


def _hold_a_waiter(provider, monkeypatch, release, events):
    """Another caller holds the process-global gate's slot and one more
    queues behind it: waiting=1, and nothing active on THIS provider -- the
    decrement-before-release window, where an actives-only wait returns."""
    from heylook_llm.providers.common.generation_gate import get_process_gate

    gate = get_process_gate(8)
    gate.acquire()

    def waiter():
        gate.acquire()
        events["drained"] = time.monotonic()
        gate.release()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    _until(lambda: provider.generation_queue_stats()["waiting"] == 1)

    def finish():
        gate.release()
        t.join(5)
    return finish


def _hold_a_generation(provider, monkeypatch, release, events):
    """A real generation on this provider, parked mid-decode."""
    from heylook_llm.config import ChatMessage, ChatRequest

    _text_engine_blocking_until(provider, monkeypatch, release)
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])

    def run():
        list(provider.create_chat_completion(req))
        events["drained"] = time.monotonic()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    _until(lambda: provider.active_generations == 1)

    def finish():
        release.set()
        t.join(5)
    return finish


def _until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "setup never reached its state"
        time.sleep(0.01)


@pytest.mark.parametrize("hold", [_hold_a_waiter, _hold_a_generation],
                         ids=["unload_waits_until_gate_has_no_waiters",
                              "unload_waits_for_active_generations"])
def test_unload_returns_only_after_the_traffic_drains(mock_mlx_provider, monkeypatch, hold):
    """The real provider and the real process gate: unload() on its own
    thread must still be waiting while the work is in flight, and return
    only after it drained -- freeing weights under a request about to run
    (or running) is the Metal fault."""
    release = threading.Event()
    events = {}
    finish = hold(mock_mlx_provider, monkeypatch, release, events)

    def unload():
        mock_mlx_provider.unload()
        events["unloaded"] = time.monotonic()

    u = threading.Thread(target=unload, daemon=True)
    u.start()
    try:
        u.join(0.4)
        assert u.is_alive(), "unload() returned while the work was still in flight"
    finally:
        finish()
    u.join(5)
    assert not u.is_alive(), "unload() never returned after the work drained"
    assert events["unloaded"] >= events["drained"]


def _idle_loaded_mock_provider(request):
    from helpers.mlx_mock import create_mock_model, create_mock_processor

    p = request.getfixturevalue("mock_mlx_provider")
    p.model = create_mock_model()
    p.processor = create_mock_processor()
    return p


# The control for the wait above: with nothing active or queued, unload()
# returns at once (a mocked provider holding a model and processor).
@pytest.mark.parametrize("build, bound_s", [
    (_idle_loaded_mock_provider, 0.5),
], ids=["unload_immediate_when_idle"])
def test_unload_returns_at_once_when_nothing_waits(request, build, bound_s):
    p = build(request)
    start = time.monotonic()
    p.unload()
    assert time.monotonic() - start < bound_s


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


class _WedgedLoadProvider(MockProvider):
    """A load that does not finish until the test lets it."""

    release = threading.Event()

    def load_model(self):
        self.release.wait(10)


@patch("heylook_llm.router.MLXProvider", new=_WedgedLoadProvider)
class TestReservationWaitBounded(unittest.TestCase):
    def setUp(self):
        _WedgedLoadProvider.release = threading.Event()
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
        """A real load of model-a wedges on its own thread; model-b, which
        needs that slot (max_loaded_models=1), must get a RuntimeError naming
        the wedged load within the bound instead of waiting forever."""
        router = self._router()
        router._reservation_wait_timeout = 0.2
        loader = threading.Thread(
            target=lambda: router.get_provider("model-a"), daemon=True)
        loader.start()
        try:
            deadline = time.monotonic() + 5
            while not router.is_loading("model-a"):
                self.assertLess(time.monotonic(), deadline, "model-a never started loading")
                time.sleep(0.01)

            start = time.monotonic()
            with self.assertRaises(RuntimeError) as ctx:
                router.get_provider("model-b")
            elapsed = time.monotonic() - start
        finally:
            _WedgedLoadProvider.release.set()
            loader.join(5)

        self.assertLess(elapsed, 5.0, "timeout did not bound the wait")
        self.assertIn("model-a", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
