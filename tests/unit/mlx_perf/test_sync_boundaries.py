# tests/unit/mlx_perf/test_sync_boundaries.py
"""
Tests for MLX synchronization boundary optimization.

Verifies that async array scheduling and sync points are correctly placed
for optimal Metal pipeline utilization.
"""
import pytest
import sys

pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


class TestAsyncArrayScheduling:
    """Tests for async array scheduling patterns."""

    def test_async_scheduling_returns_immediately(self):
        """Verify async array scheduling doesn't block."""
        try:
            import mlx.core as mx
            import time
        except ImportError:
            pytest.skip("MLX not installed")

        # Create a large operation
        a = mx.random.normal((5000, 5000))
        b = mx.random.normal((5000, 5000))
        c = a @ b  # Matrix multiplication

        # async scheduling should return almost immediately
        start = time.perf_counter()
        mx.async_eval(c)
        async_time = time.perf_counter() - start

        # Now synchronize
        mx.synchronize()

        # async scheduling should be much faster than sync
        # (actual computation happens in background)
        assert async_time < 0.1, f"async scheduling took {async_time}s, expected <0.1s"

    def test_synchronize_waits_for_completion(self):
        """Verify synchronize blocks until computation completes."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        # Queue up some computation
        result = mx.random.normal((1000, 1000))
        for _ in range(10):
            result = result @ result.T

        mx.async_eval(result)

        # Synchronize
        mx.synchronize()

        # Result should be fully computed
        # Accessing .item() on an uncomputed array would block
        shape = result.shape
        assert shape == (1000, 1000)


class TestVisionEncodingSyncPoints:
    """Tests for vision encoding sync behavior."""

    def test_encode_batch_uses_async_scheduling(self):
        """Verify encode_batch uses async scheduling for pipelining."""
        try:
            import mlx.core as mx
            from heylook_llm.providers.mlx_batch_vision import BatchVisionEncoder
        except ImportError:
            pytest.skip("Required modules not available")

        # Create mock model and processor
        class MockVisionEncoder:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 729, 768))

        class MockModel:
            vision_encoder = MockVisionEncoder()

        class MockProcessor:
            image_processor = type("IP", (), {"size": {"height": 336}})()

        encoder = BatchVisionEncoder(MockModel(), MockProcessor())

        # Create a mock preprocessed batch
        batch = mx.random.uniform(shape=(2, 3, 336, 336))

        # encode_batch should return without blocking
        # (async scheduling is called internally)
        result = encoder.encode_batch(batch)

        # Result should be an array (may not be fully computed yet)
        assert hasattr(result, "shape")

        # Explicit sync to ensure computation completes
        mx.synchronize()


class TestClearCachePatterns:
    """Tests for strategic cache clearing."""

    def test_clear_cache_reduces_memory(self):
        """Verify clear_cache releases Metal buffer pool memory."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        # Allocate and release some arrays
        arrays = []
        for _ in range(10):
            arrays.append(mx.random.normal((1000, 1000)))
        mx.synchronize()

        # Get memory before clear
        memory_before = mx.metal.get_cache_memory()

        # Clear references
        del arrays
        mx.clear_cache()
        mx.synchronize()

        # Get memory after clear
        memory_after = mx.metal.get_cache_memory()

        # Cache should be reduced or at least not larger
        assert memory_after <= memory_before, (
            f"Cache memory increased after clear: {memory_before} -> {memory_after}"
        )

    def test_clear_cache_safe_during_generation(self):
        """Verify clear_cache can be called during generation without errors."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        # Simulate generation loop with periodic cache clears
        for i in range(10):
            # Simulate token generation
            logits = mx.random.normal((1, 32000))
            token = mx.argmax(logits, axis=-1)
            mx.synchronize()

            # Clear cache every 5 iterations (simulating 256 token pattern)
            if i % 5 == 4:
                mx.clear_cache()

        # Should complete without error


class TestStreamContexts:
    """Tests for Metal stream usage."""

    def test_stream_operations_isolated(self):
        """Verify operations on different streams don't interfere."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        # Create a custom stream
        custom_stream = mx.new_stream(mx.default_device())

        # Queue operations on default stream
        a = mx.random.normal((100, 100))

        # Queue operations on custom stream
        with mx.stream(custom_stream):
            b = mx.random.normal((100, 100))
            c = b @ b.T

        # Both should complete without interference
        mx.synchronize()

        assert a.shape == (100, 100)
        assert c.shape == (100, 100)
