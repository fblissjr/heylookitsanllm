# tests/unit/test_streaming_executor_pool.py
#
# Regression test for the process abort on generation-thread exit
# (macOS crash report 2026-07-06: pthread_exit -> TLS cleanup ->
# mlx CompilerCache destructor -> tupledealloc without the GIL ->
# Py_FatalError -> SIGTRAP).
#
# MLX keeps thread-local state (streams, compiler cache) whose destructors
# can drop Python objects. Any thread that ran MLX work must therefore NEVER
# be torn down while the process lives. streaming_utils used to create a
# fresh single-worker executor per request and shut it down at stream end --
# one dying MLX thread per request. The pool below leases persistent
# single-thread executors instead: pinning is preserved (one generation runs
# start-to-finish on its leased worker), but threads are reused, not killed.

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from heylook_llm.streaming_utils import _executor_pool, async_generator_with_abort


def _thread_ident_gen(n):
    """Generator yielding the ident of the thread each next() runs on."""
    for _ in range(n):
        yield threading.get_ident()


def _drain(gen):
    async def run():
        return [chunk async for chunk in async_generator_with_abort(gen, None, None)]
    return asyncio.run(run())


def test_generation_pinned_to_one_thread():
    idents = _drain(_thread_ident_gen(5))
    assert len(idents) == 5
    assert len(set(idents)) == 1, "all next() calls must run on one thread"
    assert idents[0] != threading.get_ident()


def test_sequential_generations_reuse_thread():
    first = _drain(_thread_ident_gen(3))
    second = _drain(_thread_ident_gen(3))
    assert set(first) == set(second), (
        "sequential generations must reuse the pooled worker, not spawn+kill "
        "a thread per request (dying MLX threads abort the process)"
    )


def test_pool_release_returns_executor():
    ex = _executor_pool.acquire()
    _executor_pool.acquire()  # concurrent lease gets a different executor
    _executor_pool.release(ex)
    assert _executor_pool.acquire() is ex


def test_concurrent_leases_are_distinct():
    a = _executor_pool.acquire()
    b = _executor_pool.acquire()
    assert a is not b
    _executor_pool.release(a)
    _executor_pool.release(b)


def test_executor_survives_stream_end():
    _drain(_thread_ident_gen(2))
    ex = _executor_pool.acquire()
    try:
        # A shut-down executor raises RuntimeError on submit; a pooled one
        # must still accept work.
        assert ex.submit(lambda: 42).result(timeout=5) == 42
    finally:
        _executor_pool.release(ex)


class TestQuarantine:
    """A close-timed-out executor must be held forever, never reused.

    If it were simply dropped, GC would fire ThreadPoolExecutor's weakref
    callback, enqueue the shutdown sentinel, and the wedged worker would
    eventually EXIT its thread -- the exact TLS-teardown abort this pool
    exists to prevent. Quarantine = strong ref for the process lifetime.
    """

    def test_quarantine_holds_strong_ref_and_never_reissues(self):
        ex = _executor_pool.acquire()
        _executor_pool.quarantine(ex)
        assert ex in _executor_pool._quarantined
        assert ex not in _executor_pool._free
        # Subsequent leases never hand the quarantined executor back out.
        others = [_executor_pool.acquire() for _ in range(3)]
        try:
            assert ex not in others
        finally:
            for o in others:
                _executor_pool.release(o)

    def test_close_timeout_routes_executor_to_quarantine(self):
        leased = []
        real_acquire = _executor_pool.acquire

        def tracking_acquire():
            got = real_acquire()
            leased.append(got)
            return got

        # Make the generator-close wait appear to time out. Only the close
        # path awaits wait_for here (no http_request/abort_event given).
        with patch.object(_executor_pool, "acquire", tracking_acquire), \
             patch("heylook_llm.streaming_utils.asyncio.wait_for",
                   MagicMock(side_effect=asyncio.TimeoutError)):
            _drain(_thread_ident_gen(2))

        assert len(leased) == 1
        assert leased[0] in _executor_pool._quarantined, (
            "close-timed-out executor was dropped instead of quarantined -- "
            "GC will fire the shutdown sentinel and kill an MLX thread"
        )
        assert leased[0] not in _executor_pool._free
