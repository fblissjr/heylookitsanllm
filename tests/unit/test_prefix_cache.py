# tests/unit/test_prefix_cache.py
"""
Unit tests for prefix cache construction and forking.
Mock-based -- no real MLX model needed.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from helpers.mlx_mock import create_mlx_module_mocks, create_mock_model, create_mock_tokenizer

# Keep patch alive for entire module
_mlx_mocks = create_mlx_module_mocks()
_patch = patch.dict(sys.modules, _mlx_mocks)
_patch.start()

from heylook_llm.providers.common.generation_core import (  # noqa: E402
    PrefixCache, build_prefix_cache, fork_prefix_cache,
)
from heylook_llm.providers.common.cache_helpers import (  # noqa: E402
    snapshot_kv, restore_kv_from_snapshot,
)


class TestPrefixCacheDataclass:
    def test_construction(self):
        fake_snapshot = [(MagicMock(), MagicMock())]
        pc = PrefixCache(
            tokens=[1, 2, 3],
            kv_snapshot=fake_snapshot,
        )
        assert pc.num_tokens == 3
        assert pc.kv_snapshot is fake_snapshot
        assert pc.tokens == [1, 2, 3]

    def test_num_tokens_from_tokens(self):
        pc = PrefixCache(tokens=[10, 20], kv_snapshot=[])
        assert pc.num_tokens == 2


class TestBuildPrefixCache:
    def test_calls_model_forward_and_snapshot(self):
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock make_cache to return a list of mock cache layers
        mock_cache = [MagicMock() for _ in range(2)]
        for c in mock_cache:
            c.empty.return_value = True

        from heylook_llm.providers.common import cache_helpers
        with patch.object(cache_helpers, 'make_cache', return_value=mock_cache), \
             patch.object(cache_helpers, 'snapshot_kv', return_value=[(MagicMock(), MagicMock())]):

            mx_mock = sys.modules['mlx'].core
            mx_mock.array.return_value = MagicMock()

            result = build_prefix_cache(model, tokenizer, "You are a labeler")

            assert result.num_tokens == 5
            assert result.tokens == [1, 2, 3, 4, 5]
            assert model.called


class TestForkPrefixCache:
    def test_calls_restore(self):
        fake_snapshot = [(MagicMock(), MagicMock())]
        prefix = PrefixCache(
            tokens=[1, 2, 3],
            kv_snapshot=fake_snapshot,
        )

        model = create_mock_model()

        from heylook_llm.providers.common import generation_core
        mock_new_cache = [MagicMock()]
        # Patch at the module where fork_prefix_cache uses the imported name
        with patch.object(generation_core, 'restore_kv_from_snapshot', return_value=mock_new_cache):
            result = fork_prefix_cache(prefix, model)
            assert result is mock_new_cache
            generation_core.restore_kv_from_snapshot.assert_called_once_with(
                fake_snapshot, model, {}
            )
