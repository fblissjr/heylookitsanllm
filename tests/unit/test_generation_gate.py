# tests/unit/test_generation_gate.py
"""Unit tests for GenerationGate -- the FIFO admission gate for generation.

Pure threading; no MLX required.
"""
import threading
import time
from contextlib import contextmanager

import pytest

from heylook_llm.providers.common.generation_gate import (
    GenerationGate,
    ModelBusyError,
    GenerationCancelled,
)


@contextmanager
def _gate_with(max_waiting, holders, waiters):
    """A gate with ``holders`` (0 or 1) active and ``waiters`` threads queued
    behind it; on exit every slot is released and every waiter drained."""
    gate = GenerationGate(max_waiting=max_waiting)
    threads = []
    if holders:
        gate.acquire()
    try:
        for n in range(1, waiters + 1):
            t = threading.Thread(target=gate.acquire)
            t.start()
            threads.append(t)
            # Wait until the thread is actually queued.
            _wait_for(lambda n=n: gate.waiting == n)
        yield gate
    finally:
        if holders:
            gate.release()           # let the first waiter through
            for t in threads:
                t.join(timeout=2)
                gate.release()       # release that waiter's slot


@pytest.mark.unit
class TestGenerationGateBasics:
    def test_negative_max_waiting_rejected(self):
        with pytest.raises(ValueError):
            GenerationGate(max_waiting=-1)


@pytest.mark.unit
class TestGenerationGateCapacity:
    # check_capacity raises iff a holder exists and waiting >= max_waiting.
    # The active holder does not count as "waiting".
    @pytest.mark.parametrize("max_waiting, holders, waiters, raises", [
        pytest.param(2, 0, 0, False, id="check_capacity_ok_when_idle"),
        pytest.param(2, 1, 0, False, id="check_capacity_ok_while_active_but_no_queue"),
        pytest.param(1, 1, 1, True, id="check_capacity_raises_when_queue_full"),
        # Regression: an idle gate with max_waiting=0 must still admit the
        # first request (it becomes active, it doesn't wait).
        pytest.param(0, 0, 0, False, id="max_waiting_zero_admits_first_request_when_idle"),
        # One active, no room behind it.
        pytest.param(0, 1, 0, True, id="max_waiting_zero_is_single_flight"),
    ])
    def test_check_capacity(self, max_waiting, holders, waiters, raises):
        with _gate_with(max_waiting, holders, waiters) as gate:
            if raises:
                with pytest.raises(ModelBusyError):
                    gate.check_capacity()
            else:
                gate.check_capacity()  # no raise


@pytest.mark.unit
class TestGenerationGateCancel:
    def test_cancel_check_raises_before_turn(self):
        gate = GenerationGate(max_waiting=8)
        gate.acquire()  # hold the slot so the next acquire must wait
        try:
            with pytest.raises(GenerationCancelled):
                # cancel_check already true -> never gets the turn
                gate.acquire(cancel_check=lambda: True)
            # The cancelled waiter must not poison the queue.
            assert gate.waiting == 0
        finally:
            gate.release()

    def test_cancel_while_waiting_frees_the_queue(self):
        gate = GenerationGate(max_waiting=8)
        gate.acquire()  # active holder
        cancel = {"v": False}
        result = {}

        def waiter():
            try:
                gate.acquire(cancel_check=lambda: cancel["v"])
                result["ok"] = True
                gate.release()
            except GenerationCancelled:
                result["cancelled"] = True

        t = threading.Thread(target=waiter)
        t.start()
        _wait_for(lambda: gate.waiting == 1)
        cancel["v"] = True  # request's client "disconnected" while queued
        t.join(timeout=2)

        assert result.get("cancelled") is True
        assert gate.waiting == 0
        # A subsequent waiter still gets through cleanly (queue not poisoned).
        gate.release()
        gate.acquire()
        assert gate.busy is True
        gate.release()


@pytest.mark.unit
class TestGenerationGateSnapshot:
    # snapshot/busy/waiting track acquire, queue and release; capacity is
    # max_waiting + 1. Every row also checks the gate is idle after release.
    @pytest.mark.parametrize("max_waiting, holders, waiters", [
        pytest.param(4, 1, 0, id="acquire_release_single"),
        pytest.param(4, 0, 0, id="waiting_starts_zero"),
        pytest.param(8, 0, 0, id="idle_snapshot"),
        pytest.param(8, 1, 0, id="active_snapshot"),
        pytest.param(8, 1, 1, id="snapshot_counts_waiters"),
    ])
    def test_snapshot(self, max_waiting, holders, waiters):
        with _gate_with(max_waiting, holders, waiters) as gate:
            assert gate.snapshot() == {"active": holders, "waiting": waiters,
                                       "max_waiting": max_waiting,
                                       "capacity": max_waiting + 1}
            assert gate.busy is bool(holders)
            assert gate.waiting == waiters
        assert gate.busy is False
        assert gate.waiting == 0


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
