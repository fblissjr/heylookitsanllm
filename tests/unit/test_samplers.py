# tests/unit/test_samplers.py
"""Tests for pure-MLX sampler utilities.

Covers the presence penalty processor.
"""

import mlx.core as mx
import pytest

from heylook_llm.providers.common.samplers import make_presence_penalty_processor


class TestPresencePenaltyProcessor:
    """make_presence_penalty_processor correctness."""

    def test_zero_penalty_is_noop(self):
        proc = make_presence_penalty_processor(0.0)
        tokens = mx.array([1, 2, 3])
        logits = mx.ones(100)
        result = proc(tokens, logits)
        result_list = result.tolist()
        assert all(v == pytest.approx(1.0) for v in result_list)

    def test_empty_tokens_is_noop(self):
        proc = make_presence_penalty_processor(1.5)
        tokens = mx.array([], dtype=mx.int32)
        logits = mx.ones(100)
        result = proc(tokens, logits)
        result_list = result.tolist()
        assert all(v == pytest.approx(1.0) for v in result_list)

    def test_penalty_applied_to_seen_tokens(self):
        proc = make_presence_penalty_processor(1.0)
        tokens = mx.array([5, 10, 15])
        logits = mx.zeros(20)
        result = proc(tokens, logits)
        result_list = result.tolist()
        # Tokens 5, 10, 15 should have -1.0 penalty
        assert result_list[5] == pytest.approx(-1.0)
        assert result_list[10] == pytest.approx(-1.0)
        assert result_list[15] == pytest.approx(-1.0)
        # Other tokens should be 0.0
        assert result_list[0] == pytest.approx(0.0)
        assert result_list[1] == pytest.approx(0.0)

    def test_duplicate_tokens_penalized_once(self):
        """Presence penalty = fixed per token, regardless of count."""
        proc = make_presence_penalty_processor(2.0)
        tokens = mx.array([5, 5, 5, 5, 5])
        logits = mx.zeros(20)
        result = proc(tokens, logits)
        # Token 5 should be penalized exactly once (-2.0, not -10.0)
        assert result.tolist()[5] == pytest.approx(-2.0)

    def test_penalty_value_scales(self):
        proc = make_presence_penalty_processor(0.5)
        tokens = mx.array([3])
        logits = mx.zeros(10)
        result = proc(tokens, logits)
        assert result.tolist()[3] == pytest.approx(-0.5)
