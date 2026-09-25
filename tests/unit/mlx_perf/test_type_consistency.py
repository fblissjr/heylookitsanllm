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




class TestArrayOperationDtypes:
    """Tests for dtype preservation in array operations."""


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

    # test_vision_normalization_output_dtype removed: it imported
    # `heylook_llm.providers.mlx_batch_vision._normalize_and_transpose`, a module
    # deleted when batch vision moved to apps/batch-labeler/ (v1.23.0). The test
    # skipped with a misleading "MLX not installed" message while testing nothing;
    # vision normalization is now covered by apps/batch-labeler's own suite.


