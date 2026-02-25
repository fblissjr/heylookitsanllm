# src/heylook_llm/streaming_utils.py
"""Shared streaming utilities used by api.py and messages_api.py."""

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import Request


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
                await asyncio.sleep(0.1)

        chunk = await chunk_future
        if chunk is None:
            break
        yield chunk
