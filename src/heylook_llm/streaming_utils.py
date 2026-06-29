# src/heylook_llm/streaming_utils.py
"""Shared streaming utilities used by api.py and messages_api.py."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import Request


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
    # the last. Pinning to a single worker keeps a generation on one thread for
    # its entire life.
    gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-stream")

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
        try:
            logging.debug(f"{log_prefix}Closing provider generator")
            close_future = loop.run_in_executor(gen_executor, sync_gen.close)
            await asyncio.wait_for(close_future, timeout=30)
        except Exception:
            pass
        finally:
            gen_executor.shutdown(wait=False)
