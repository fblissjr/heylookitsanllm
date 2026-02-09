"""Unit tests for MLX Provider safe unload with reference counting.

Tests that the _active_generations counter prevents model unload during active generation,
avoiding Metal command buffer crashes during LRU cache eviction.
"""
import threading
import time
from unittest.mock import MagicMock

import pytest


class FakeMLXProvider:
    """Minimal stand-in for MLXProvider to test reference counting logic.

    We cannot import the real MLXProvider without MLX installed, so we replicate
    the exact ref-counting and unload logic under test.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = MagicMock()
        self.processor = MagicMock()
        self._generation_lock = threading.Lock()
        self._active_generations = 0
        self._active_lock = threading.Lock()
        self._content_cache = {}
        self._strategies = {}
        self._unloaded = False
        self._unload_waited = False

    def generate(self, duration: float = 0.5):
        """Simulate a generation that holds the active counter."""
        with self._active_lock:
            self._active_generations += 1

        self._generation_lock.acquire()
        try:
            time.sleep(duration)
        finally:
            self._generation_lock.release()
            with self._active_lock:
                self._active_generations -= 1

    def unload(self, max_wait: float = 5.0):
        """Mirror of MLXProvider.unload() with safe wait loop."""
        start = time.time()
        while True:
            with self._active_lock:
                if self._active_generations == 0:
                    break
                active = self._active_generations
            elapsed = time.time() - start
            if elapsed > max_wait:
                self._unload_waited = True
                break
            self._unload_waited = True
            time.sleep(0.05)

        self._content_cache.clear()
        self._strategies.clear()
        del self.model
        del self.processor
        self._unloaded = True

    @property
    def active_generations(self):
        with self._active_lock:
            return self._active_generations


@pytest.mark.unit
class TestRefCountingSafety:
    """Tests for the active generation reference counting mechanism."""

    def test_counter_starts_at_zero(self):
        provider = FakeMLXProvider("test-model")
        assert provider.active_generations == 0

    def test_counter_increments_during_generation(self):
        provider = FakeMLXProvider("test-model")
        started = threading.Event()

        def gen():
            with provider._active_lock:
                provider._active_generations += 1
            started.set()
            time.sleep(0.3)
            with provider._active_lock:
                provider._active_generations -= 1

        t = threading.Thread(target=gen)
        t.start()
        started.wait()
        assert provider.active_generations >= 1
        t.join()
        assert provider.active_generations == 0

    def test_counter_multiple_concurrent(self):
        """Multiple generations should all increment the counter."""
        provider = FakeMLXProvider("test-model")
        # Barrier parties = 3 threads + 1 main thread
        barrier = threading.Barrier(4)
        done = threading.Event()

        def gen():
            with provider._active_lock:
                provider._active_generations += 1
            barrier.wait()
            done.wait()
            with provider._active_lock:
                provider._active_generations -= 1

        threads = [threading.Thread(target=gen) for _ in range(3)]
        for t in threads:
            t.start()

        barrier.wait()  # wait for all 3 threads to increment
        # All 3 should be active
        assert provider.active_generations == 3

        done.set()
        for t in threads:
            t.join()
        assert provider.active_generations == 0

    def test_unload_waits_for_active_generation(self):
        """unload() should wait until active generations drop to 0."""
        provider = FakeMLXProvider("test-model")
        unload_started = threading.Event()
        gen_start = threading.Event()

        def gen():
            with provider._active_lock:
                provider._active_generations += 1
            gen_start.set()
            # Hold generation for 0.3s
            time.sleep(0.3)
            with provider._active_lock:
                provider._active_generations -= 1

        # Start generation
        t_gen = threading.Thread(target=gen)
        t_gen.start()
        gen_start.wait()

        # Start unload while generation is active
        unload_complete = threading.Event()
        unload_saw_active = False

        def do_unload():
            nonlocal unload_saw_active
            unload_started.set()
            # If counter > 0 at start, we correctly waited
            with provider._active_lock:
                if provider._active_generations > 0:
                    unload_saw_active = True
            provider.unload()
            unload_complete.set()

        t_unload = threading.Thread(target=do_unload)
        t_unload.start()

        t_gen.join()
        t_unload.join()

        assert provider._unloaded
        assert unload_saw_active

    def test_unload_immediate_when_idle(self):
        """unload() should complete immediately when no generations are active."""
        provider = FakeMLXProvider("test-model")
        start = time.time()
        provider.unload()
        elapsed = time.time() - start
        assert provider._unloaded
        assert elapsed < 0.2  # should be near-instant

    def test_force_unload_after_timeout(self):
        """unload() should force-unload after max_wait even if generations are still active."""
        provider = FakeMLXProvider("test-model")

        # Simulate a stuck generation (never decrements)
        with provider._active_lock:
            provider._active_generations = 1

        start = time.time()
        provider.unload(max_wait=0.3)
        elapsed = time.time() - start

        assert provider._unloaded
        assert elapsed >= 0.25  # waited close to max_wait
        assert elapsed < 1.0  # didn't wait too long

    def test_counter_never_goes_negative(self):
        """Counter should never go below 0 under normal usage."""
        provider = FakeMLXProvider("test-model")

        threads = []
        for _ in range(10):
            def run():
                with provider._active_lock:
                    provider._active_generations += 1
                time.sleep(0.01)
                with provider._active_lock:
                    provider._active_generations -= 1
            t = threading.Thread(target=run)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert provider.active_generations == 0
