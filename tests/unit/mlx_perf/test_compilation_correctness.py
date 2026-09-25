# tests/unit/mlx_perf/test_compilation_correctness.py
"""
The presence-penalty processor on 1-D logits (mlx-lm hands processors a
(vocab,) row), on the cases test_samplers.py does not state: a nonzero
baseline (subtract, not assign) and varying token-sequence lengths.

Zero penalty and an empty sequence (identity) live in test_samplers.py
TestPresencePenaltyProcessor; the (1, vocab) shape and float16 survival in
its test_mlx_lm_logits_shape_is_penalized_on_the_vocab_axis.
"""
import pytest
import sys

# Skip all tests if not on macOS (MLX is macOS-only)
pytestmark = [
    pytest.mark.mlx_perf,
    pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS"),
]


@pytest.mark.parametrize("token_lists, penalty", [
    # Random-normal logits: seen ids drop by exactly `penalty` (subtract, not
    # assign -- a zeros baseline cannot tell them apart); duplicates count once.
    pytest.param([[1, 5, 10, 5, 20, 1]], 1.5, id="penalty_lowers_seen_token_logits"),
    # The processor must handle any token-sequence length and keep the
    # (vocab,) shape (it was once shapeless-compiled).
    pytest.param([list(range(n)) for n in (5, 10, 50, 100)], 1.0, id="varying_batch_sizes"),
])
def test_seen_ids_drop_by_exactly_the_penalty(token_lists, penalty):
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed")
    from heylook_llm.providers.common.samplers import make_presence_penalty_processor

    processor = make_presence_penalty_processor(penalty)
    for toks in token_lists:
        logits = mx.random.normal((32000,))
        result = processor(mx.array(toks, dtype=mx.int32), mx.array(logits))
        mx.synchronize()
        assert result.shape == logits.shape
        seen = mx.array(sorted(set(toks)), dtype=mx.int32)
        assert mx.max(mx.abs(result[seen] - logits[seen] + penalty)).item() < 1e-5
        unseen = max(toks) + 1  # a token never in the sequence is untouched
        assert abs((result[unseen] - logits[unseen]).item()) < 1e-6
