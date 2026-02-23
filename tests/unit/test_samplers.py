# tests/unit/test_samplers.py
"""Tests for pure-MLX sampler utilities.

Covers _mlx_unique correctness and the presence penalty processor.
"""

import mlx.core as mx
import numpy as np
import pytest

from heylook_llm.providers.common.samplers import (
    _mlx_unique,
    make_presence_penalty_processor,
)


class TestMlxUnique:
    """_mlx_unique must match numpy's np.unique for all inputs."""

    def _assert_matches_numpy(self, values: list[int]):
        """Helper: compare _mlx_unique output against np.unique."""
        arr = mx.array(values)
        result = _mlx_unique(arr)
        expected = np.unique(np.array(values))
        np.testing.assert_array_equal(
            np.array(result.tolist()),
            expected,
            err_msg=f"Mismatch for input {values}",
        )

    def test_empty_array(self):
        result = _mlx_unique(mx.array([], dtype=mx.int32))
        assert result.size == 0

    def test_single_element(self):
        self._assert_matches_numpy([42])

    def test_two_same(self):
        self._assert_matches_numpy([7, 7])

    def test_two_different(self):
        self._assert_matches_numpy([3, 1])

    def test_already_unique_sorted(self):
        self._assert_matches_numpy([1, 2, 3, 4, 5])

    def test_already_unique_unsorted(self):
        self._assert_matches_numpy([5, 3, 1, 4, 2])

    def test_all_same(self):
        self._assert_matches_numpy([99] * 20)

    def test_duplicates_mixed(self):
        self._assert_matches_numpy([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

    def test_large_random(self):
        """100 random inputs drawn from token-sized range."""
        rng = np.random.default_rng(seed=12345)
        for _ in range(100):
            size = rng.integers(1, 200)
            values = rng.integers(0, 32000, size=size).tolist()
            self._assert_matches_numpy(values)

    def test_output_is_sorted(self):
        """np.unique returns sorted output; ours should too."""
        arr = mx.array([10, 3, 7, 3, 10, 1])
        result = _mlx_unique(arr)
        result_list = result.tolist()
        assert result_list == sorted(result_list)

    def test_dtype_preserved(self):
        arr = mx.array([1, 2, 3], dtype=mx.int32)
        result = _mlx_unique(arr)
        assert result.dtype == mx.int32


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
