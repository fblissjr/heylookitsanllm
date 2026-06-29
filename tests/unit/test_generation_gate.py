# tests/unit/test_generation_gate.py
"""Unit tests for GenerationGate -- the FIFO admission gate for generation.

Pure threading; no MLX required.
"""
import threading
import time

import pytest

from heylook_llm.providers.common.generation_gate import GenerationGate, ModelBusyError


@pytest.mark.unit
class TestGenerationGateBasics:
    def test_acquire_release_single(self):
        gate = GenerationGate(max_waiting=4)
        gate.acquire()
        assert gate.busy is True
        gate.release()
        assert gate.busy is False

    def test_waiting_starts_zero(self):
        gate = GenerationGate(max_waiting=4)
        assert gate.waiting == 0

    def test_model_busy_error_message_contains_marker(self):
        # The API layer maps errors to HTTP 503 via `"MODEL_BUSY" in str(e)`.
        gate = GenerationGate(max_waiting=0)
        gate.acquire()
        try:
            with pytest.raises(ModelBusyError) as exc:
                gate.check_capacity()
            assert "MODEL_BUSY" in str(exc.value)
        finally:
            gate.release()

    def test_model_busy_is_runtimeerror(self):
        # Routes catch `except RuntimeError as e`.
        assert issubclass(ModelBusyError, RuntimeError)

    def test_negative_max_waiting_rejected(self):
        with pytest.raises(ValueError):
            GenerationGate(max_waiting=-1)


@pytest.mark.unit
class TestGenerationGateCapacity:
    def test_check_capacity_ok_when_idle(self):
        gate = GenerationGate(max_waiting=2)
        gate.check_capacity()  # no raise

    def test_check_capacity_ok_while_active_but_no_queue(self):
        # The active holder does not count as "waiting".
        gate = GenerationGate(max_waiting=2)
        gate.acquire()
        try:
            gate.check_capacity()  # 0 waiting < 2 -> ok
        finally:
            gate.release()

    def test_check_capacity_raises_when_queue_full(self):
        gate = GenerationGate(max_waiting=1)
        gate.acquire()  # active holder
        blocked = threading.Thread(target=gate.acquire)  # becomes the 1 waiter
        blocked.start()
        try:
            # Wait until the second thread is actually queued.
            _wait_for(lambda: gate.waiting == 1)
            with pytest.raises(ModelBusyError):
                gate.check_capacity()
        finally:
            gate.release()       # let the waiter through
            blocked.join(timeout=2)
            gate.release()       # release the waiter's slot

    def test_max_waiting_zero_is_single_flight(self):
        gate = GenerationGate(max_waiting=0)
        gate.acquire()
        try:
            with pytest.raises(ModelBusyError):
                gate.check_capacity()
        finally:
            gate.release()


@pytest.mark.unit
class TestGenerationGateFifo:
    def test_fifo_order(self):
        """Waiters are served strictly in arrival order, never preempted."""
        gate = GenerationGate(max_waiting=16)
        order = []
        order_lock = threading.Lock()
        started = []

        gate.acquire()  # main thread holds the slot first

        def worker(n):
            # Record arrival, then block for the slot.
            with order_lock:
                started.append(n)
            gate.acquire()
            with order_lock:
                order.append(n)
            time.sleep(0.01)
            gate.release()

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            # Stagger starts so arrival order is deterministic.
            _wait_for(lambda i=i: len(started) == i + 1)
            _wait_for(lambda i=i: gate.waiting == i + 1)

        # Release the main slot; waiters should drain in 0,1,2,3,4 order.
        gate.release()
        for t in threads:
            t.join(timeout=2)

        assert order == [0, 1, 2, 3, 4]

    def test_release_from_other_thread(self):
        """release() may run on a different thread than acquire() (streaming path)."""
        gate = GenerationGate(max_waiting=2)
        gate.acquire()
        done = threading.Event()

        def releaser():
            gate.release()
            done.set()

        t = threading.Thread(target=releaser)
        t.start()
        assert done.wait(timeout=2)
        assert gate.busy is False
        t.join(timeout=2)


def _wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")
