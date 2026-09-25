# tests/unit/test_samplers.py
"""Tests for pure-MLX sampler utilities.

Covers the presence penalty processor.
"""

import mlx.core as mx
import pytest

from heylook_llm.providers.common.samplers import make_presence_penalty_processor


class TestPresencePenaltyProcessor:
    """make_presence_penalty_processor correctness."""

    @pytest.mark.parametrize(
        "penalty, tokens, vocab, base",
        [
            (0.0, [1, 2, 3], 100, 1.0),
            (1.5, [], 100, 1.0),
            (1.0, [5, 10, 15], 20, 0.0),
            # presence penalty = fixed per token, regardless of count
            # (token 5 lowered by 2.0, not 10.0)
            (2.0, [5, 5, 5, 5, 5], 20, 0.0),
            (0.5, [3], 10, 0.0),
        ],
        ids=["zero-penalty-is-noop", "empty-tokens-is-noop", "applied-to-seen-tokens",
             "duplicates-penalized-once", "value-scales"],
    )
    def test_each_seen_id_is_lowered_by_exactly_the_penalty(self, penalty, tokens, vocab, base):
        """1-D logits: every distinct seen id drops by the penalty, every other
        id is untouched; zero penalty or no tokens is the identity."""
        proc = make_presence_penalty_processor(penalty)
        result = proc(mx.array(tokens, dtype=mx.int32), mx.full(vocab, base)).tolist()
        seen = set(tokens)
        for i, value in enumerate(result):
            assert value == pytest.approx(base - penalty if i in seen else base), i

    def test_mlx_lm_logits_shape_is_penalized_on_the_vocab_axis(self):
        """mlx-lm hands processors ``logits[:, -1, :]`` -- shape (1, vocab).
        The scatter used to run along axis 0 (size 1), so every token id past
        0 was an out-of-bounds GPU write: memory corruption, then a Metal
        fault mid-generation, then a poisoned process (gemma-4-26B, 2026-09-04).
        The 1-D tests above never saw it. Pin the real shape."""
        proc = make_presence_penalty_processor(1.0)
        tokens = mx.array([5, 10])
        logits = mx.zeros((1, 20))
        result = proc(tokens, logits)
        assert result.shape == (1, 20)
        row = result.tolist()[0]
        assert row[5] == pytest.approx(-1.0) and row[10] == pytest.approx(-1.0)
        assert row[0] == pytest.approx(0.0) and row[19] == pytest.approx(0.0)
        # and the dtype the model produces survives the penalty
        half = proc(tokens, mx.zeros((1, 20), dtype=mx.float16))
        assert half.dtype == mx.float16


def test_knobs_off_build_no_processors():
    """Every OFF knob builds nothing: any logits processor makes the engine
    read the previous tokens back before each forward (a GPU sync that costs
    decode rate), so an identity processor is pure cost. The floor's values
    are the ones every request without an opinion carries."""
    from heylook_llm.providers.common.samplers import build
    from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR

    _, processors = build(dict(GLOBAL_SAMPLER_FLOOR))
    assert processors == []
    _, processors = build({**GLOBAL_SAMPLER_FLOOR, "repetition_penalty": 1.1})
    assert len(processors) == 1
