# src/heylook_llm/streaming_utils.py
"""Shared streaming utilities used by api.py and messages_api.py."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import Request


class _PinnedExecutorPool:
    """Lease persistent single-thread executors for generation pinning.

    Threads that ran MLX work must NEVER be torn down while the process
    lives: MLX keeps thread-local state (streams, its compiler cache) whose
    destructors can hold Python objects, and a pthread's TLS cleanup runs
    after its Python thread state is gone -- deallocating those objects
    without the GIL is a Py_FatalError -> SIGTRAP process abort (hit in
    production 2026-07-06 with compiled sampler fns on the quantized-KV
    path; see tests/unit/test_streaming_executor_pool.py).

    So instead of one executor per request (shut down at stream end), this
    pool leases single-worker executors and reuses them. The pinning
    invariant is unchanged: a leased executor serves exactly one generation
    at a time, so each generation still runs start-to-finish on one thread.
    The pool grows to the max number of concurrently admitted requests
    (bounded by the generation gate's capacity) and never shrinks.
    """

    def __init__(self):
        self._free: list[ThreadPoolExecutor] = []
        self._lock = threading.Lock()

    def acquire(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._free:
                return self._free.pop()
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stream")

    def release(self, executor: ThreadPoolExecutor) -> None:
        with self._lock:
            self._free.append(executor)


_executor_pool = _PinnedExecutorPool()


class KeepaliveMarker:
    """Sentinel yielded during long prefill to keep SSE connections alive.

    The API layer checks for this type and emits an SSE comment (`: keepalive`)
    instead of a data chunk. SSE comments are ignored by conformant parsers
    but prevent HTTP connection timeouts during long prompt processing.
    """
    pass


KEEPALIVE_MARKER = KeepaliveMarker()


async def async_generator_with_abort(
    sync_gen,
    http_request: Request | None,
    abort_event,
    log_prefix: str = "",
) -> AsyncGenerator:
    """Wrap a synchronous generator for async iteration with client disconnect detection.

    Yields chunks from *sync_gen* via a thread-pool executor. While waiting for
    the next chunk it polls ``http_request.is_disconnected()`` and, if the
    client has gone away, sets *abort_event* so the provider can stop early.

    Args:
        sync_gen: Synchronous generator (e.g. from provider.create_chat_completion).
        http_request: The Starlette request, used for disconnect detection (may be None).
        abort_event: A ``threading.Event`` from the provider (may be None).
        log_prefix: Label used in log messages (e.g. "[API]" or "[MESSAGES]").
    """
    loop = asyncio.get_event_loop()

    def get_next():
        try:
            return next(sync_gen)
        except StopIteration:
            return None

    # Drive the whole generation on ONE dedicated thread. The default executor
    # is a multi-thread pool, so successive next() calls could otherwise run on
    # different threads -- fragile for MLX, whose per-generation stream and
    # wired_limit context are entered on the first next() and synchronized on
    # the last. Leased from the pool (not created fresh) so the thread is
    # REUSED across requests, never destroyed -- see _PinnedExecutorPool.
    gen_executor = _executor_pool.acquire()

    keepalive_interval = 5.0  # seconds between keepalive comments
    last_keepalive = loop.time()
    first_chunk_received = False

    try:
        while True:
            chunk_future = loop.run_in_executor(gen_executor, get_next)

            if http_request and abort_event:
                while not chunk_future.done():
                    if await http_request.is_disconnected():
                        logging.info(f"{log_prefix}Client disconnected during streaming")
                        abort_event.set()
                        # Wait (bounded) for the in-flight next() to observe the
                        # abort and unwind, so cleanup runs on a settled
                        # generator. Bounded so a non-cooperative generation can't
                        # pin this coroutine; the finally still closes it.
                        try:
                            await asyncio.wait_for(chunk_future, timeout=30)
                        except Exception:
                            pass
                        return
                    # Emit SSE keepalive comments during long prefill to
                    # prevent connection timeouts. Only before first chunk.
                    now = loop.time()
                    if not first_chunk_received and (now - last_keepalive) >= keepalive_interval:
                        yield KEEPALIVE_MARKER
                        last_keepalive = now
                    await asyncio.sleep(0.1)

            chunk = await chunk_future
            if chunk is None:
                break
            first_chunk_received = True
            yield chunk
    finally:
        # Close the provider generator so its finally blocks run immediately
        # (releases the generation gate, decrements _active_generations, clears
        # the MLX cache). Without this, close() only runs when GC collects the
        # abandoned generator -- which would hold the gate and stall the queue.
        # close() runs on the same pinned worker that drove generation.
        closed = False
        try:
            logging.debug(f"{log_prefix}Closing provider generator")
            close_future = loop.run_in_executor(gen_executor, sync_gen.close)
            await asyncio.wait_for(close_future, timeout=30)
            closed = True
        except Exception:
            pass
        finally:
            # Return the executor for reuse -- NEVER shut it down (a dying
            # MLX thread aborts the process; see _PinnedExecutorPool). If
            # close timed out the worker may be wedged mid-generation: leak
            # that executor rather than queueing a future request behind it.
            if closed:
                _executor_pool.release(gen_executor)
            else:
                logging.warning(f"{log_prefix}Generator close timed out; retiring its worker from the pool")
