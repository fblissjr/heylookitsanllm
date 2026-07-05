# tests/unit/test_snapshot_thread_affinity.py
#
# Regression test for "RuntimeError: There is no Stream(gpu, N) in current
# thread" on radix cache reuse.
#
# The generation path runs each request on a fresh single-worker executor
# thread (streaming_utils.async_generator_with_abort), and generation_stream
# is thread-local (mx.new_thread_local_stream). KVCache.state returns LAZY
# slice nodes scheduled on the creating thread's stream. If snapshot_kv
# publishes those lazy nodes into the shared radix tree, a later request on a
# different thread hits mlx_lm's `mx.eval([c.state for c in prompt_cache])`
# and MLX cannot resolve the dead thread's stream.
#
# Invariant under test: snapshot_kv returns MATERIALIZED arrays -- safe to
# evaluate from any thread, including after the creating thread (and its
# thread-local stream) is gone.
#
# Requires the GPU device: only GPU thread-local streams are torn down with
# their thread (CPU streams survive, so a CPU version of this test cannot go
# red). Verified by direct probe: eval of a lazy slice scheduled on a dead
# thread's ThreadLocalStream(gpu) raises; the same on cpu passes.

import threading

import mlx.core as mx
import pytest

from heylook_llm.providers.common.cache_helpers import snapshot_kv

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="needs Metal GPU thread-local streams"
)


class _FakeKVLayer:
    """Minimal stand-in for mlx_lm KVCache: .state yields lazy slices."""

    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset

    def empty(self):
        return self.keys is None

    @property
    def state(self):
        # Mirrors mlx_lm KVCache.state: a lazy slice up to the current offset,
        # scheduled on whatever stream is active on the calling thread.
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )


def _snapshot_on_worker_thread(result):
    """Build lazy cache state under a thread-local stream, snapshot it."""
    local_stream = mx.new_thread_local_stream(mx.gpu)
    with mx.stream(local_stream):
        keys = mx.random.normal((1, 2, 8, 4)) * 2.0
        values = mx.random.normal((1, 2, 8, 4)) + 1.0
        layer = _FakeKVLayer(keys, values, offset=6)
        result["snapshot"] = snapshot_kv([layer])


def test_snapshot_kv_is_safe_to_eval_from_another_thread():
    result = {}
    worker = threading.Thread(target=_snapshot_on_worker_thread, args=(result,))
    worker.start()
    worker.join()
    assert worker.is_alive() is False

    snapshot = result["snapshot"]
    assert len(snapshot) == 1 and snapshot[0] is not None

    # This is exactly what mlx_lm's generate_step does with restored cache
    # state on the NEXT request's thread. Must not raise.
    mx.eval([s for s in snapshot if s is not None])

    keys, values = snapshot[0]
    assert keys.shape == (1, 2, 6, 4)
    assert values.shape == (1, 2, 6, 4)


def test_snapshot_kv_preserves_empty_layers():
    class _EmptyLayer:
        def empty(self):
            return True

        state = None

    snap = snapshot_kv([_EmptyLayer()])
    assert snap == [None]
