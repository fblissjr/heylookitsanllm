# src/heylook_llm/streaming_utils.py
"""Shared streaming utilities used by api.py and messages_api.py."""

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import Request


class _KeepaliveMarker:
    """Sentinel yielded during long prefill to keep SSE connections alive.

    The API layer checks for this type and emits an SSE comment (`: keepalive`)
    instead of a data chunk. SSE comments are ignored by conformant parsers
    but prevent HTTP connection timeouts during long prompt processing.
    """
    pass


KEEPALIVE_MARKER = _KeepaliveMarker()
_KEEPALIVE_MARKER = KEEPALIVE_MARKER  # Module-level singleton


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

    keepalive_interval = 5.0  # seconds between keepalive comments
    last_keepalive = asyncio.get_event_loop().time()
    first_chunk_received = False

    try:
        while True:
            chunk_future = loop.run_in_executor(None, get_next)

            if http_request and abort_event:
                while not chunk_future.done():
                    if await http_request.is_disconnected():
                        logging.info(f"{log_prefix}Client disconnected during streaming")
                        abort_event.set()
                        try:
                            await chunk_future
                        except Exception:
                            pass
                        return
                    # Emit SSE keepalive comments during long prefill to
                    # prevent connection timeouts. Only before first chunk.
                    now = asyncio.get_event_loop().time()
                    if not first_chunk_received and (now - last_keepalive) >= keepalive_interval:
                        yield _KEEPALIVE_MARKER
                        last_keepalive = now
                    await asyncio.sleep(0.1)

            chunk = await chunk_future
            if chunk is None:
                break
            first_chunk_received = True
            yield chunk
    finally:
        # Close the provider generator so its finally blocks run immediately
        # (releases _generation_lock, decrements _active_generations, clears MLX cache).
        # Without this, close() only runs when GC collects the abandoned generator.
        try:
            logging.debug(f"{log_prefix}Closing provider generator")
            sync_gen.close()
        except Exception:
            pass
