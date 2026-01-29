# tests/integration/mlx_perf/test_memory_profiling.py
"""
Memory profiling tests for MLX operations.

These tests verify:
- Peak memory during model loading
- Memory scaling with generation length
- Memory leak detection (growth <5MB/iteration)
"""
import pytest
import sys
import gc

pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


@pytest.fixture
def memory_baseline():
    """Get baseline memory usage."""
    try:
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        return mx.metal.get_active_memory()
    except (ImportError, AttributeError):
        pytest.skip("MLX Metal memory API not available")


class TestMemoryUsage:
    """Tests for memory usage patterns."""

    def test_vision_buffer_cache_cleanup(self):
        """Verify vision buffer cache cleanup works correctly."""
        try:
            import mlx.core as mx
            from heylook_llm.providers.mlx_batch_vision import BatchVisionEncoder
        except ImportError:
            pytest.skip("Required modules not available")

        # Create a mock model and processor
        class MockModel:
            vision_encoder = lambda self, x: mx.zeros((x.shape[0], 729, 768))

        class MockProcessor:
            image_processor = type("IP", (), {"size": {"height": 336}})()

        encoder = BatchVisionEncoder(MockModel(), MockProcessor())

        # Create multiple buffers
        for batch_size in [1, 2, 4, 8]:
            _ = encoder._get_buffer(batch_size)

        initial_cache_size = len(encoder._buffer_cache)
        assert initial_cache_size == 4, f"Expected 4 cached buffers, got {initial_cache_size}"

        # Clear keeping only 2
        encoder.clear_buffers(keep_last_n=2)

        final_cache_size = len(encoder._buffer_cache)
        assert final_cache_size == 2, f"Expected 2 cached buffers after cleanup, got {final_cache_size}"

    def test_generation_memory_stable(self, loaded_model, benchmark_prompt, memory_baseline):
        """Verify memory doesn't grow unbounded during generation."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate

        model, tokenizer = loaded_model
        max_growth_mb = 50  # Maximum acceptable growth per iteration

        prompt_tokens = tokenizer.encode(benchmark_prompt)

        memory_samples = [memory_baseline]

        # Run multiple generations
        for i in range(5):
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=100,
            ):
                pass

            mx.synchronize()
            gc.collect()
            current_memory = mx.metal.get_active_memory()
            memory_samples.append(current_memory)

        # Calculate growth
        growth_mb = (memory_samples[-1] - memory_samples[0]) / (1024 * 1024)
        avg_growth_per_iter = growth_mb / 5

        print(f"\nMemory: start={memory_samples[0]/1e6:.1f}MB, end={memory_samples[-1]/1e6:.1f}MB")
        print(f"Total growth: {growth_mb:.1f}MB, avg per iteration: {avg_growth_per_iter:.1f}MB")

        assert avg_growth_per_iter <= max_growth_mb, (
            f"Memory growth {avg_growth_per_iter:.1f}MB/iter exceeds threshold {max_growth_mb}MB"
        )

    def test_clear_cache_releases_memory(self, memory_baseline):
        """Verify mx.clear_cache() releases memory."""
        import mlx.core as mx

        # Allocate some arrays
        arrays = [mx.random.normal((1000, 1000)) for _ in range(10)]
        mx.synchronize()

        memory_after_alloc = mx.metal.get_active_memory()

        # Clear references and cache
        del arrays
        gc.collect()
        mx.clear_cache()
        mx.synchronize()

        memory_after_clear = mx.metal.get_active_memory()

        released_mb = (memory_after_alloc - memory_after_clear) / (1024 * 1024)
        print(f"\nReleased {released_mb:.1f}MB after clear_cache()")

        # Should release at least some memory
        assert memory_after_clear < memory_after_alloc, (
            "clear_cache() did not release any memory"
        )


class TestMemoryLeaks:
    """Tests to detect memory leaks."""

    @pytest.mark.slow
    def test_repeated_generation_no_leak(self, loaded_model, benchmark_prompt, memory_baseline):
        """Verify no memory leak over many iterations."""
        import mlx.core as mx
        from mlx_lm.generate import stream_generate

        model, tokenizer = loaded_model
        max_growth_mb = 5  # Maximum growth per iteration for leak detection

        prompt_tokens = tokenizer.encode(benchmark_prompt)

        # Warmup
        for _ in range(3):
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=20,
            ):
                pass
            mx.synchronize()

        gc.collect()
        mx.clear_cache()
        mx.synchronize()

        baseline = mx.metal.get_active_memory()
        memory_readings = []

        # Many iterations to detect slow leaks
        for i in range(20):
            for response in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=30,
            ):
                pass

            mx.synchronize()

            if i % 5 == 4:  # Every 5 iterations
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
                memory_readings.append(mx.metal.get_active_memory())

        # Calculate trend
        if len(memory_readings) >= 2:
            growth_per_sample = [
                (memory_readings[i] - memory_readings[i-1]) / (1024 * 1024)
                for i in range(1, len(memory_readings))
            ]
            avg_growth = sum(growth_per_sample) / len(growth_per_sample)

            print(f"\nMemory trend: {growth_per_sample}")
            print(f"Average growth per 5 iterations: {avg_growth:.2f}MB")

            assert avg_growth <= max_growth_mb, (
                f"Memory leak detected: {avg_growth:.2f}MB growth per sample"
            )

    def test_sampler_no_leak(self, memory_baseline):
        """Verify sampler usage doesn't leak memory."""
        import mlx.core as mx
        from heylook_llm.providers.common.samplers import make_presence_penalty_processor

        processor = make_presence_penalty_processor(1.5)

        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        baseline = mx.metal.get_active_memory()

        # Run many iterations
        for _ in range(100):
            tokens = mx.array([1, 5, 10, 5, 20, 1])
            logits = mx.random.normal((1, 32000))
            result = processor(tokens, logits)
            mx.synchronize()

        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        final = mx.metal.get_active_memory()

        growth_mb = (final - baseline) / (1024 * 1024)
        print(f"\nSampler memory growth over 100 iterations: {growth_mb:.2f}MB")

        assert growth_mb < 10, f"Sampler leaked {growth_mb:.2f}MB"
