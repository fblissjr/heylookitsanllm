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


# That one generation runs start-to-finish on one worker thread is
# test_streaming_utils.py TestThreadPinning::test_all_chunks_produced_on_single_thread.


def _a_second_generation_reuses_the_worker(first):
    second = _drain(_thread_ident_gen(3))
    assert set(first) == set(second), (
        "sequential generations must reuse the pooled worker, not spawn+kill "
        "a thread per request (dying MLX threads abort the process)"
    )


def _a_pooled_executor_still_accepts_work(first):  # noqa: ARG001
    ex = _executor_pool.acquire()
    try:
        # A shut-down executor raises RuntimeError on submit; a pooled one
        # must still accept work.
        assert ex.submit(lambda: 42).result(timeout=5) == 42
    finally:
        _executor_pool.release(ex)


@pytest.mark.parametrize("then", [
    pytest.param(_a_second_generation_reuses_the_worker, id="sequential_generations_reuse_thread"),
    pytest.param(_a_pooled_executor_still_accepts_work, id="executor_survives_stream_end"),
])
def test_a_finished_generation_leaves_its_worker_in_the_pool(then):
    then(_drain(_thread_ident_gen(3)))


@pytest.mark.parametrize("released", [
    pytest.param(0, id="pool_release_returns_executor"),
    pytest.param(1, id="concurrent_leases_are_distinct"),
])
def test_concurrent_leases_differ_and_a_released_one_is_reissued(released):
    leases = [_executor_pool.acquire(), _executor_pool.acquire()]
    assert leases[0] is not leases[1]
    _executor_pool.release(leases[released])
    again = _executor_pool.acquire()
    try:
        assert again is leases[released]
    finally:
        for ex in leases:
            _executor_pool.release(ex)


def test_a_worker_whose_close_timed_out_is_kept_alive_and_never_reused(monkeypatch):
    """A generation whose close() really blocks past the close timeout: its
    worker must never serve another generation (it is wedged mid-close) and
    must outlive every reference the stream held, through a gc.

    If it were simply dropped, GC would fire ThreadPoolExecutor's weakref
    callback, enqueue the shutdown sentinel, and the wedged worker would EXIT
    its thread once the close finished -- the TLS-teardown abort this pool
    exists to prevent. The 30s close timeout is shortened, not faked: the
    real wait_for runs, and really times out.
    """
    import gc
    import heylook_llm.streaming_utils as su

    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(su.asyncio, "wait_for",
                        lambda fut, timeout: real_wait_for(fut, timeout=0.2))

    unblock = threading.Event()
    seen = []

    def wedges_on_close():
        try:
            seen.append(threading.current_thread())
            yield "a"
            yield "b"
        finally:
            unblock.wait(10)

    async def first_chunk_then_leave():
        agen = async_generator_with_abort(wedges_on_close(), None, None)
        assert await agen.__anext__() == "a"
        await agen.aclose()   # the close runs on the worker and blocks there

    try:
        asyncio.run(first_chunk_then_leave())
        (wedged,) = seen

        # Later generations run -- promptly, and on other workers.
        after = [_drain(_thread_ident_gen(1))[0] for _ in range(3)]
        assert wedged.ident not in after, "a wedged worker was handed a new generation"

        gc.collect()
    finally:
        unblock.set()   # the close finishes; the worker goes idle
    wedged.join(0.5)
    assert wedged.is_alive(), (
        "the close-timed-out worker's thread exited -- the executor was dropped "
        "instead of quarantined, and GC shut it down")
