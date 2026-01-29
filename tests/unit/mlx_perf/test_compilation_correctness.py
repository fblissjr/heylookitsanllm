# tests/unit/mlx_perf/test_compilation_correctness.py
"""
Tests to verify compiled MLX functions produce correct outputs.

These tests ensure that @mx.compile decorated functions produce
identical results to their uncompiled equivalents within tolerance.
"""
import pytest
import sys

# Skip all tests if not on macOS (MLX is macOS-only)
pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


@pytest.fixture
def mlx_arrays():
    """Create test arrays for MLX operations."""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed")

    return {
        # Note: mlx-lm logits processors receive 1D arrays (vocab_size,), not 2D
        "logits": mx.random.normal((32000,)),  # Typical vocab size, 1D as in mlx-lm
        "tokens": mx.array([1, 5, 10, 5, 20, 1]),  # Sample token sequence with duplicates
        "image_hwc": mx.random.uniform(shape=(336, 336, 3)),  # HWC image
    }


class TestPresencePenaltyCompilation:
    """Tests for presence penalty processor compilation correctness."""

    def test_compiled_matches_uncompiled(self, mlx_arrays):
        """Verify compiled presence penalty matches uncompiled version."""
        import mlx.core as mx

        try:
            from heylook_llm.providers.common.samplers import _apply_presence_penalty_compiled, _mlx_unique
        except ImportError as e:
            pytest.skip(f"heylook_llm samplers not importable: {e}")

        tokens = mlx_arrays["tokens"]
        penalty = 1.5

        # Get unique tokens using pure MLX implementation
        unique_tokens = _mlx_unique(tokens)

        # Create two independent logits arrays for fair comparison
        # Note: mlx-lm passes 1D logits (vocab_size,) to processors
        logits1 = mx.random.normal((32000,))
        mx.synchronize()  # Force evaluation before copying
        logits2 = logits1 * 1.0  # Copy via multiplication

        # Uncompiled reference implementation
        def reference_presence_penalty(logits, unique_tokens, penalty):
            return logits.at[unique_tokens].add(-penalty)

        # Run both versions on separate arrays
        reference_result = reference_presence_penalty(logits1, unique_tokens, penalty)
        compiled_result = _apply_presence_penalty_compiled(logits2, unique_tokens, penalty)

        # Force evaluation
        mx.synchronize()

        # Verify results match
        diff = mx.abs(reference_result - compiled_result)
        max_diff = mx.max(diff).item()

        assert max_diff < 1e-5, f"Compiled result differs from reference by {max_diff}"

    def test_zero_penalty_noop(self, mlx_arrays):
        """Verify zero penalty returns unchanged logits."""
        import mlx.core as mx

        try:
            from heylook_llm.providers.common.samplers import make_presence_penalty_processor
        except ImportError as e:
            pytest.skip(f"heylook_llm samplers not importable: {e}")

        logits = mlx_arrays["logits"]
        tokens = mlx_arrays["tokens"]

        # Zero penalty processor
        processor = make_presence_penalty_processor(0.0)
        result = processor(tokens, mx.array(logits))  # Use mx.array() to copy

        mx.synchronize()

        # Should be identical
        diff = mx.max(mx.abs(result - logits)).item()
        assert diff == 0.0, "Zero penalty should not modify logits"

    def test_empty_tokens_noop(self, mlx_arrays):
        """Verify empty token sequence returns unchanged logits."""
        import mlx.core as mx

        try:
            from heylook_llm.providers.common.samplers import make_presence_penalty_processor
        except ImportError as e:
            pytest.skip(f"heylook_llm samplers not importable: {e}")

        logits = mlx_arrays["logits"]
        empty_tokens = mx.array([], dtype=mx.int32)

        processor = make_presence_penalty_processor(1.5)
        result = processor(empty_tokens, mx.array(logits))  # Use mx.array() to copy

        mx.synchronize()

        diff = mx.max(mx.abs(result - logits)).item()
        assert diff == 0.0, "Empty tokens should not modify logits"


class TestVisionPreprocessingCompilation:
    """Tests for vision preprocessing compilation correctness."""

    def test_normalize_and_transpose_correctness(self, mlx_arrays):
        """Verify compiled normalize+transpose matches manual implementation."""
        import mlx.core as mx
        import numpy as np

        img_hwc = mlx_arrays["image_hwc"]

        # Reference implementation (NumPy-style)
        mean = mx.array([0.485, 0.456, 0.406])
        std = mx.array([0.229, 0.224, 0.225])
        reference = (img_hwc - mean) / std
        reference = reference.transpose(2, 0, 1)

        # Import compiled version
        from heylook_llm.providers.mlx_batch_vision import _normalize_and_transpose

        compiled_result = _normalize_and_transpose(img_hwc)

        mx.synchronize()

        # Verify shape is CHW
        assert compiled_result.shape == (3, 336, 336), f"Expected CHW shape, got {compiled_result.shape}"

        # Verify values match
        diff = mx.max(mx.abs(reference - compiled_result)).item()
        assert diff < 1e-5, f"Compiled normalize differs from reference by {diff}"

    def test_output_dtype_preserved(self, mlx_arrays):
        """Verify compiled function preserves input dtype."""
        import mlx.core as mx
        from heylook_llm.providers.mlx_batch_vision import _normalize_and_transpose

        # Test with float32
        img_f32 = mlx_arrays["image_hwc"].astype(mx.float32)
        result_f32 = _normalize_and_transpose(img_f32)
        mx.synchronize()

        # float32 input should give float32 output (ImageNet constants are float32)
        assert result_f32.dtype == mx.float32, f"Expected float32, got {result_f32.dtype}"


class TestShapelessCompilation:
    """Tests for shapeless compilation with variable inputs."""

    def test_varying_batch_sizes(self, mlx_arrays):
        """Verify compiled functions work with different batch sizes."""
        import mlx.core as mx

        try:
            from heylook_llm.providers.common.samplers import _apply_presence_penalty_compiled, _mlx_unique
        except ImportError as e:
            pytest.skip(f"heylook_llm samplers not importable: {e}")

        penalty = 1.0

        for batch_tokens in [5, 10, 50, 100]:
            tokens = mx.arange(batch_tokens, dtype=mx.int32)
            logits = mx.random.normal((32000,))  # 1D as in mlx-lm
            unique = _mlx_unique(tokens)  # Pure MLX unique

            # Should not raise
            result = _apply_presence_penalty_compiled(logits, unique, penalty)
            mx.synchronize()

            # Basic sanity check
            assert result.shape == logits.shape

    def test_varying_image_sizes(self):
        """Verify compiled normalize works with different image sizes."""
        import mlx.core as mx
        from heylook_llm.providers.mlx_batch_vision import _normalize_and_transpose

        for size in [224, 336, 448]:
            img = mx.random.uniform(shape=(size, size, 3))
            result = _normalize_and_transpose(img)
            mx.synchronize()

            assert result.shape == (3, size, size), f"Expected (3, {size}, {size}), got {result.shape}"
