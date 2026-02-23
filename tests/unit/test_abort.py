# tests/unit/test_abort.py
"""Tests for the cooperative abort mechanism."""

import threading
import time

import pytest

from heylook_llm.providers.abort import AbortEvent


class TestAbortEvent:
    """AbortEvent basic behavior."""

    def test_initial_state_is_clear(self):
        abort = AbortEvent()
        assert not abort.is_set()

    def test_set_makes_is_set_true(self):
        abort = AbortEvent()
        abort.set()
        assert abort.is_set()

    def test_clear_resets_after_set(self):
        abort = AbortEvent()
        abort.set()
        abort.clear()
        assert not abort.is_set()

    def test_multiple_sets_are_idempotent(self):
        abort = AbortEvent()
        abort.set()
        abort.set()
        assert abort.is_set()

    def test_multiple_clears_are_idempotent(self):
        abort = AbortEvent()
        abort.clear()
        abort.clear()
        assert not abort.is_set()

    def test_repr_clear(self):
        abort = AbortEvent()
        assert "clear" in repr(abort)

    def test_repr_set(self):
        abort = AbortEvent()
        abort.set()
        assert "set" in repr(abort)


class TestAbortEventThreadSafety:
    """AbortEvent cross-thread signaling."""

    def test_set_from_another_thread(self):
        abort = AbortEvent()
        result = {"seen": False}

        def worker():
            # Wait briefly, then set abort
            time.sleep(0.01)
            abort.set()

        t = threading.Thread(target=worker)
        t.start()

        # Busy-wait with timeout
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if abort.is_set():
                result["seen"] = True
                break
            time.sleep(0.001)

        t.join(timeout=2.0)
        assert result["seen"], "Abort signal not seen from main thread"

    def test_clear_set_sequence_under_lock(self):
        """Simulate the preemption pattern: clear() after lock acquire, set() to abort."""
        abort = AbortEvent()
        lock = threading.Lock()
        sequence = []

        def generation_loop():
            """Simulates a generation that checks abort per token."""
            lock.acquire()
            abort.clear()
            try:
                for i in range(100):
                    if abort.is_set():
                        sequence.append(f"aborted_at_{i}")
                        return
                    time.sleep(0.001)
                sequence.append("completed")
            finally:
                lock.release()

        def preempt():
            """Simulates a new request preempting the current one."""
            time.sleep(0.02)  # Let generation start
            # Cannot acquire lock (generation holds it), so signal abort
            acquired = lock.acquire(blocking=False)
            if not acquired:
                abort.set()
                # Wait for generation to finish and release lock
                lock.acquire()
            abort.clear()
            sequence.append("preempted")
            lock.release()

        t1 = threading.Thread(target=generation_loop)
        t2 = threading.Thread(target=preempt)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Generation should have been aborted, not completed
        assert any("aborted_at" in s for s in sequence), f"Expected abort, got: {sequence}"
        assert "preempted" in sequence

    def test_rapid_set_clear_cycles(self):
        """Stress test: rapid set/clear from multiple threads."""
        abort = AbortEvent()
        errors = []

        def setter():
            for _ in range(1000):
                abort.set()
                abort.clear()

        threads = [threading.Thread(target=setter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Should not raise any exceptions
        assert not errors
        # Final state should be clear (all threads did set then clear)
        assert not abort.is_set()
