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

    def test_penalty_lowers_seen_token_logits(self, mlx_arrays):
        """Presence penalty subtracts from the logits of tokens already seen."""
        import mlx.core as mx

        from heylook_llm.providers.common.samplers import make_presence_penalty_processor

        logits = mlx_arrays["logits"]
        tokens = mlx_arrays["tokens"]  # contains tokens 1, 5, 10, 20
        penalty = 1.5

        processor = make_presence_penalty_processor(penalty)
        result = processor(tokens, mx.array(logits))
        mx.synchronize()

        # Seen tokens drop by exactly `penalty`; unseen tokens are unchanged.
        seen = mx.array([1, 5, 10, 20], dtype=mx.int32)
        seen_delta = (result[seen] - logits[seen])
        assert mx.max(mx.abs(seen_delta + penalty)).item() < 1e-5
        # A token never in the sequence (e.g. 2) is untouched.
        assert abs((result[2] - logits[2]).item()) < 1e-6

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


# TestVisionPreprocessingCompilation was removed in Phase 2 when batch
# vision labeling moved to the standalone apps/batch-labeler/ client
# (see CHANGELOG v1.23.0). The tests targeted the deleted module
# `heylook_llm.providers.mlx_batch_vision`.


class TestShapelessCompilation:
    """Tests for shapeless compilation with variable inputs."""

    def test_varying_batch_sizes(self, mlx_arrays):  # noqa: ARG002
        """The presence-penalty processor handles varying token-sequence lengths."""
        import mlx.core as mx

        from heylook_llm.providers.common.samplers import make_presence_penalty_processor

        penalty = 1.0
        processor = make_presence_penalty_processor(penalty)

        for batch_tokens in [5, 10, 50, 100]:
            tokens = mx.arange(batch_tokens, dtype=mx.int32)
            logits = mx.random.normal((32000,))  # 1D as in mlx-lm

            # Should not raise and must preserve shape across input sizes.
            result = processor(tokens, logits)
            mx.synchronize()
            assert result.shape == logits.shape
