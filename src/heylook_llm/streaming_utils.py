# src/heylook_llm/streaming_utils.py
"""Shared streaming utilities used by api.py and messages_api.py."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import AsyncGenerator

import anyio
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
        self._quarantined: list[ThreadPoolExecutor] = []
        self._lock = threading.Lock()

    def acquire(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._free:
                return self._free.pop()
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stream")

    def release(self, executor: ThreadPoolExecutor) -> None:
        with self._lock:
            self._free.append(executor)

    def quarantine(self, executor: ThreadPoolExecutor) -> None:
        """Hold a strong reference to a wedged executor forever; never reuse.

        Simply dropping it is not safe: GC fires ThreadPoolExecutor's weakref
        callback, which enqueues the shutdown sentinel, and the wedged worker
        eventually EXITS its thread -- the TLS-teardown abort this pool exists
        to prevent. The leak (one idle thread) is the intended cost.
        """
        with self._lock:
            self._quarantined.append(executor)


_executor_pool = _PinnedExecutorPool()


async def _await_worker_shielded(fut: asyncio.Future, *, timeout: float = 30.0) -> None:
    """Wait up to ``timeout`` seconds for ``fut`` even when the current task has
    a pending ``CancelledError`` from Starlette/AnyIO stream teardown."""
    deadline = asyncio.get_running_loop().time() + timeout
    absorbed_cancels = 0
    try:
        with anyio.CancelScope(shield=True):
            while not fut.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=remaining)
                    break
                except (TimeoutError, asyncio.TimeoutError):
                    break
                except asyncio.CancelledError:
                    absorbed_cancels += 1
                    if absorbed_cancels > 4:
                        break
                except Exception:
                    break
    finally:
        if absorbed_cancels:
            task = asyncio.current_task()
            if task is not None:
                for _ in range(absorbed_cancels):
                    task.cancel()


async def run_on_pinned_worker(fn, *, abort_event=None):
    """Run synchronous generation work on a persistent ``_executor_pool`` worker."""
    loop = asyncio.get_running_loop()
    gen_executor = _executor_pool.acquire()
    worker_idle = threading.Event()

    def _run():
        try:
            return fn()
        finally:
            worker_idle.set()

    fut = loop.run_in_executor(gen_executor, _run)
    try:
        return await fut
    except BaseException:
        if abort_event is not None and not abort_event.is_set():
            abort_event.set()
        if not worker_idle.is_set():
            await _await_worker_shielded(fut, timeout=30)
        raise
    finally:
        if worker_idle.is_set():
            _executor_pool.release(gen_executor)
        else:
            logging.warning(
                "Pinned worker still busy on exit; quarantining its worker (kept alive, never reused)"
            )
            _executor_pool.quarantine(gen_executor)


class KeepaliveMarker:
    """Sentinel yielded whenever the generation has been silent for
    ``KEEPALIVE_INTERVAL_S`` -- during prefill, and equally during a decode
    stall (v1.79.65; it used to stop at the first token, so a mid-generation
    pause got nothing). control_frame() spells it on the wire.
    """
    pass


KEEPALIVE_MARKER = KeepaliveMarker()


@dataclass(frozen=True, slots=True)
class PrefillProgress:
    """Sentinel yielded when the engine reports prefill progress: ``processed``
    of ``total`` prompt tokens done, cached prefix excluded (see
    AbortEvent.set_prefill_progress). One per CHANGE, read off the request's
    signal channel while the wrapper waits on the next chunk -- both engines
    report into that channel, so this is the only place progress is turned
    into a frame."""
    processed: int
    total: int


# Seconds of silence before a keepalive frame. A module constant rather than
# a local so a test can shorten it instead of sleeping through it.
KEEPALIVE_INTERVAL_S = 5.0

# ONE wire (v1.79.66: the OpenAI chat route, whose chunk grammar had no place
# for a control event and so spoke SSE comments, is gone). Keepalive is
# Anthropic's own `ping` event, which every Messages client already handles;
# prefill progress is a namespaced extension event, the same shape as
# heylook_saved. Every SSE consume loop guards with
# control_frame() BEFORE touching chunk fields -- the markers have none of
# GenerationChunk's, and forgetting the guard is a proven failure mode
# (/v1/messages carried the crash from its creation until 2026-08-13).
KEEPALIVE_FRAME = 'event: ping\ndata: {"type":"ping"}\n\n'


def control_frame(chunk) -> str | None:
    """A control marker's SSE frame, or None for a real chunk."""
    if isinstance(chunk, KeepaliveMarker):
        return KEEPALIVE_FRAME
    if isinstance(chunk, PrefillProgress):
        return ('event: heylook_progress\ndata: {"type":"heylook_progress",'
                f'"prefill":{{"processed":{chunk.processed},"total":{chunk.total}}}}}\n\n')
    return None


async def async_generator_with_abort(
    sync_gen,
    http_request: Request | None,
    abort_event,
    log_prefix: str = "",
    abort_on_disconnect: bool = True,
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
        abort_on_disconnect: Stop generating when the client goes away.
            TRUE for the stateless wire (/v1/messages):
            nothing is persisted there, so a disconnected client means the
            work has nowhere to go. FALSE for the conversation-generate saga,
            which persists server-side -- there a disconnect is a tab switch,
            not an abandonment, and killing the run threw away the answer the
            user came back for. Keepalives are emitted either way.
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

    last_keepalive = loop.time()
    # The engine reports prefill progress into the request's signal channel
    # (AbortEvent.set_prefill_progress); this wrapper is the ONE reader.
    # getattr, not a type check: tests pass a bare threading.Event here.
    read_progress = getattr(abort_event, "prefill_progress", None)
    last_progress = None

    exhausted = False
    try:
        while True:
            chunk_future = loop.run_in_executor(gen_executor, get_next)

            if http_request and abort_event:
                while not chunk_future.done():
                    if abort_on_disconnect and await http_request.is_disconnected():
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
                    now = loop.time()
                    progress = read_progress() if read_progress is not None else None
                    if progress is not None and progress != last_progress:
                        # One frame per change. It also breaks the silence, so
                        # the keepalive clock restarts from it.
                        last_progress = progress
                        last_keepalive = now
                        yield PrefillProgress(*progress)
                    elif (now - last_keepalive) >= KEEPALIVE_INTERVAL_S:
                        # Silence wherever it falls: a long prefill, or a decode
                        # stall (gguf swap, spec-decode hiccup) -- the clock is
                        # reset by every real chunk below, so this never fires
                        # into a flowing stream.
                        yield KEEPALIVE_MARKER
                        last_keepalive = now
                    # Block on the chunk itself, with a timeout that keeps the
                    # disconnect/keepalive cadence. A plain sleep here would
                    # quantize every chunk to the poll boundary (~10 chunks/s
                    # delivered and recorded, however fast the model decodes).
                    await asyncio.wait({chunk_future}, timeout=0.1)

            chunk = await chunk_future
            if chunk is None:
                exhausted = True
                break
            last_keepalive = loop.time()
            yield chunk
    finally:
        # Signal abort when exiting before natural exhaustion (for example when
        # Starlette/AnyIO cancels the streaming task while next() is in flight),
        # and shield close_future from an already-cancelled AnyIO CancelScope so
        # wait_for actually waits up to 30s instead of failing in <1ms and
        # falsely quarantining an idle worker.
        if not exhausted and abort_event is not None and not abort_event.is_set():
            abort_event.set()
        worker_idle = threading.Event()

        def _close_gen():
            try:
                sync_gen.close()
            finally:
                worker_idle.set()

        try:
            logging.debug(f"{log_prefix}Closing provider generator")
            close_future = loop.run_in_executor(gen_executor, _close_gen)
            await _await_worker_shielded(close_future, timeout=30)
        finally:
            if worker_idle.is_set():
                _executor_pool.release(gen_executor)
            else:
                logging.warning(f"{log_prefix}Generator close timed out; quarantining its worker (kept alive, never reused)")
                _executor_pool.quarantine(gen_executor)
