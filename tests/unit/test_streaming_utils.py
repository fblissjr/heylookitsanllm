# tests/unit/test_streaming_utils.py
"""Tests for async_generator_with_abort in streaming_utils."""

import asyncio
import threading
from unittest.mock import MagicMock

from heylook_llm.streaming_utils import async_generator_with_abort


def _tracked_gen(chunks, tracker: dict):
    """Generator whose finally block sets tracker['closed'] = True.

    This simulates a provider generator that releases a lock in its finally block.
    """
    try:
        for c in chunks:
            yield c
    finally:
        tracker["closed"] = True


async def _collect(agen):
    """Collect all chunks from an async generator."""
    result = []
    async for chunk in agen:
        result.append(chunk)
    return result


class TestSyncGenClose:
    """sync_gen.close() must run in all exit paths to release provider locks."""

    def test_finally_runs_after_normal_completion(self):
        """Generator exhaustion: finally block runs."""
        tracker = {"closed": False}
        gen = _tracked_gen(["a", "b"], tracker)

        chunks = asyncio.run(_collect(async_generator_with_abort(gen, None, None)))

        assert chunks == ["a", "b"]
        assert tracker["closed"], "Generator finally block did not run after normal completion"

    def test_finally_runs_after_client_disconnect(self):
        """Client disconnect: finally block must run so the provider releases its lock."""
        lock = threading.Lock()
        lock.acquire()

        def locked_gen():
            try:
                yield "first"
                yield "second"
            finally:
                lock.release()

        gen = locked_gen()

        # Simulate disconnect after first chunk
        call_count = 0

        async def disconnect_after_first():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        request = MagicMock()
        request.is_disconnected = disconnect_after_first
        abort_event = threading.Event()

        chunks = asyncio.run(_collect(async_generator_with_abort(gen, request, abort_event)))

        assert chunks == ["first"]
        assert not lock.locked(), "Provider lock still held -- generator finally block didn't run"

    def test_finally_runs_on_aclose(self):
        """Starlette calls aclose() on the async generator -- finally block must still run."""
        tracker = {"closed": False}
        gen = _tracked_gen(["a", "b", "c"], tracker)

        async def consume_one_then_aclose():
            agen = async_generator_with_abort(gen, None, None)
            chunk = await agen.__anext__()
            assert chunk == "a"
            await agen.aclose()

        asyncio.run(consume_one_then_aclose())

        assert tracker["closed"], "Generator finally block did not run after aclose()"


class TestThreadPinning:
    """The whole generation must run on ONE thread.

    MLX's per-generation stream and wired_limit context are entered on the first
    next() and synchronized on the last; if next() calls hop pool threads the
    sync happens on the wrong thread. The streaming bridge pins each generation
    to a dedicated single-thread executor.
    """

    def test_all_chunks_produced_on_single_thread(self):
        thread_ids = []

        def thread_recording_gen():
            for i in range(6):
                thread_ids.append(threading.get_ident())
                yield i

        gen = thread_recording_gen()
        chunks = asyncio.run(_collect(async_generator_with_abort(gen, None, None)))

        assert chunks == [0, 1, 2, 3, 4, 5]
        assert len(set(thread_ids)) == 1, (
            f"generation hopped threads: {set(thread_ids)}"
        )

    def test_close_runs_on_same_thread_as_generation(self):
        """close() must run on the pinned worker, not the event-loop thread --
        a generator can't be closed from a different thread while suspended."""
        gen_thread = {}
        close_thread = {}

        def recording_gen():
            try:
                for i in range(3):
                    gen_thread["id"] = threading.get_ident()
                    yield i
            finally:
                close_thread["id"] = threading.get_ident()

        async def consume_one_then_aclose():
            agen = async_generator_with_abort(recording_gen(), None, None)
            await agen.__anext__()
            await agen.aclose()

        asyncio.run(consume_one_then_aclose())

        assert close_thread["id"] == gen_thread["id"]
        assert close_thread["id"] != threading.get_ident()  # not the main/loop thread
