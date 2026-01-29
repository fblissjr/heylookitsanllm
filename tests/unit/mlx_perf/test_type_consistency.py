# tests/unit/mlx_perf/test_type_consistency.py
"""
Tests to verify dtype preservation through MLX operations.

MLX uses weak typing for Python scalars, but explicit mx.array() wrappers
can cause unexpected type promotion. These tests ensure dtypes are preserved.
"""
import pytest
import sys

pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


class TestScalarTypePreservation:
    """Tests for Python scalar weak typing behavior."""

    def test_python_float_preserves_fp16(self):
        """Verify Python float multiplication preserves float16 dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        arr_fp16 = mx.ones((10,), dtype=mx.float16)
        result = arr_fp16 * 2.0  # Python float - should preserve dtype

        assert result.dtype == mx.float16, f"Expected float16, got {result.dtype}"

    def test_python_float_preserves_bfloat16(self):
        """Verify Python float multiplication preserves bfloat16 dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        arr_bf16 = mx.ones((10,), dtype=mx.bfloat16)
        result = arr_bf16 * 2.0

        assert result.dtype == mx.bfloat16, f"Expected bfloat16, got {result.dtype}"

    def test_scale_calculation_preserves_dtype(self):
        """Verify attention scale calculation preserves input dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        # Simulate attention scale calculation
        head_dim = 64
        scale = 1.0 / (head_dim ** 0.5)  # Python float

        query_fp16 = mx.random.normal((1, 8, 128, head_dim)).astype(mx.float16)
        scaled = query_fp16 * scale

        assert scaled.dtype == mx.float16, f"Scale operation changed dtype to {scaled.dtype}"


class TestArrayOperationDtypes:
    """Tests for dtype preservation in array operations."""

    def test_softmax_preserves_dtype(self):
        """Verify softmax preserves input dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        logits_fp16 = mx.random.normal((1, 1000)).astype(mx.float16)
        probs = mx.softmax(logits_fp16, axis=-1)

        mx.synchronize()
        assert probs.dtype == mx.float16, f"Softmax changed dtype to {probs.dtype}"

    def test_presence_penalty_preserves_dtype(self):
        """Verify presence penalty preserves logits dtype."""
        try:
            import mlx.core as mx
            from heylook_llm.providers.common.samplers import make_presence_penalty_processor
        except ImportError:
            pytest.skip("MLX or heylook_llm not installed")

        # Test with float16 logits
        logits_fp16 = mx.random.normal((1, 32000)).astype(mx.float16)
        tokens = mx.array([1, 5, 10], dtype=mx.int32)

        processor = make_presence_penalty_processor(1.5)
        result = processor(tokens, logits_fp16)

        mx.synchronize()
        assert result.dtype == mx.float16, f"Presence penalty changed dtype to {result.dtype}"

    def test_vision_normalization_output_dtype(self):
        """Verify vision normalization produces expected dtype."""
        try:
            import mlx.core as mx
            from heylook_llm.providers.mlx_batch_vision import _normalize_and_transpose
        except ImportError:
            pytest.skip("MLX or heylook_llm not installed")

        # Input as float32 (typical for images)
        img = mx.random.uniform(shape=(336, 336, 3)).astype(mx.float32)
        result = _normalize_and_transpose(img)

        mx.synchronize()
        # ImageNet constants are float32, so output should be float32
        assert result.dtype == mx.float32, f"Expected float32, got {result.dtype}"


class TestBroadcastDtypeRules:
    """Tests for MLX broadcast dtype rules."""

    def test_broadcast_to_preserves_dtype(self):
        """Verify mx.broadcast_to preserves dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        template = mx.array([[1, 24, 24]], dtype=mx.int32)
        broadcasted = mx.broadcast_to(template, (4, 3))

        assert broadcasted.dtype == mx.int32, f"broadcast_to changed dtype to {broadcasted.dtype}"
        assert broadcasted.shape == (4, 3)

    def test_stack_preserves_dtype(self):
        """Verify mx.stack preserves dtype."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not installed")

        arrays_fp16 = [mx.ones((3, 336, 336), dtype=mx.float16) for _ in range(4)]
        stacked = mx.stack(arrays_fp16)

        assert stacked.dtype == mx.float16, f"stack changed dtype to {stacked.dtype}"
        assert stacked.shape == (4, 3, 336, 336)
