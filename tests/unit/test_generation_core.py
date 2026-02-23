# tests/unit/test_generation_core.py
"""Tests for the unified generation core.

Covers:
- _build_cache_config construction from effective_request
- _setup_prompt_cache with and without model_id
- run_generation end-to-end with mocked lm_stream_generate
- Abort event handling
- Speculative decoding acceptance tracking
"""

import pytest
from unittest.mock import MagicMock, patch


class FakeResponse:
    """Mimics mlx-lm generation response."""
    def __init__(self, text, token, from_draft=False):
        self.text = text
        self.token = token
        self.from_draft = from_draft


class TestBuildCacheConfig:
    """Verify _build_cache_config extracts correct fields."""

    def test_defaults(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import _build_cache_config

        config = _build_cache_config({})
        assert config['cache_type'] == 'standard'
        assert config['kv_bits'] is None
        assert config['kv_group_size'] == 64
        assert config['max_kv_size'] is None

    def test_explicit_values(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import _build_cache_config

        config = _build_cache_config({
            'cache_type': 'quantized',
            'kv_bits': 4,
            'kv_group_size': 32,
            'max_kv_size': 2048,
        })
        assert config['cache_type'] == 'quantized'
        assert config['kv_bits'] == 4
        assert config['kv_group_size'] == 32
        assert config['max_kv_size'] == 2048


class TestSetupPromptCache:
    """Verify _setup_prompt_cache behavior."""

    def test_no_model_id_returns_full_tokens(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import _setup_prompt_cache

        prompt_cache, tokens, gen_cache = _setup_prompt_cache(
            None, MagicMock(), [1, 2, 3], {}, MagicMock()
        )
        assert prompt_cache is None
        assert tokens == [1, 2, 3]
        assert gen_cache is None

    def test_with_model_id_uses_cache_manager(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import _setup_prompt_cache

        mock_manager = MagicMock()
        mock_working_cache = MagicMock()
        mock_working_cache.cache = [MagicMock()]
        mock_manager.get_or_create_cache.return_value = mock_working_cache

        with patch('heylook_llm.providers.common.generation_core.process_prompt_with_cache') as mock_process:
            mock_process.return_value = ([3], mock_working_cache)
            prompt_cache, tokens, gen_cache = _setup_prompt_cache(
                "test-model", MagicMock(), [1, 2, 3], {}, mock_manager
            )

        assert prompt_cache is mock_working_cache
        assert tokens == [3]
        assert gen_cache == mock_working_cache.cache


class TestRunGeneration:
    """Verify run_generation end-to-end behavior."""

    def test_yields_responses(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import run_generation

        responses = [
            FakeResponse("Hello", 1),
            FakeResponse(" world", 2),
        ]

        with patch('heylook_llm.providers.common.generation_core.lm_stream_generate', return_value=iter(responses)), \
             patch('heylook_llm.providers.common.generation_core.wired_limit') as mock_wired, \
             patch('heylook_llm.providers.common.generation_core._get_generation_stream', return_value=MagicMock()):
            mock_wired.return_value.__enter__ = MagicMock()
            mock_wired.return_value.__exit__ = MagicMock(return_value=False)

            effective = {'max_tokens': 100, 'num_draft_tokens': 3}
            results = list(run_generation(
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt_tokens=[1, 2, 3],
                effective_request=effective,
                sampler=MagicMock(),
                processors=[],
            ))

        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].text == " world"

    def test_leading_space_cleanup(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import run_generation

        responses = [FakeResponse(" Hello", 1)]

        with patch('heylook_llm.providers.common.generation_core.lm_stream_generate', return_value=iter(responses)), \
             patch('heylook_llm.providers.common.generation_core.wired_limit') as mock_wired, \
             patch('heylook_llm.providers.common.generation_core._get_generation_stream', return_value=MagicMock()):
            mock_wired.return_value.__enter__ = MagicMock()
            mock_wired.return_value.__exit__ = MagicMock(return_value=False)

            effective = {'max_tokens': 100}
            results = list(run_generation(
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt_tokens=[1],
                effective_request=effective,
                sampler=MagicMock(),
                processors=[],
            ))

        assert results[0].text == "Hello"

    def test_abort_stops_generation(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import run_generation

        abort = MagicMock()
        abort.is_set.return_value = True

        responses = [FakeResponse("Hello", 1), FakeResponse(" world", 2)]

        with patch('heylook_llm.providers.common.generation_core.lm_stream_generate', return_value=iter(responses)), \
             patch('heylook_llm.providers.common.generation_core.wired_limit') as mock_wired, \
             patch('heylook_llm.providers.common.generation_core._get_generation_stream', return_value=MagicMock()):
            mock_wired.return_value.__enter__ = MagicMock()
            mock_wired.return_value.__exit__ = MagicMock(return_value=False)

            effective = {'max_tokens': 100}
            results = list(run_generation(
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt_tokens=[1],
                effective_request=effective,
                sampler=MagicMock(),
                processors=[],
                abort_event=abort,
            ))

        assert len(results) == 0

    def test_acceptance_tracking(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import run_generation

        responses = [
            FakeResponse("a", 1, from_draft=True),
            FakeResponse("b", 2, from_draft=False),
            FakeResponse("c", 3, from_draft=True),
        ]

        with patch('heylook_llm.providers.common.generation_core.lm_stream_generate', return_value=iter(responses)), \
             patch('heylook_llm.providers.common.generation_core.wired_limit') as mock_wired, \
             patch('heylook_llm.providers.common.generation_core._get_generation_stream', return_value=MagicMock()):
            mock_wired.return_value.__enter__ = MagicMock()
            mock_wired.return_value.__exit__ = MagicMock(return_value=False)

            effective = {'max_tokens': 100}
            draft = MagicMock()
            results = list(run_generation(
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt_tokens=[1],
                effective_request=effective,
                sampler=MagicMock(),
                processors=[],
                draft_model=draft,
            ))

        assert len(results) == 3

    def test_cache_storage_on_completion(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import run_generation

        responses = [FakeResponse("hi", 10)]

        with patch('heylook_llm.providers.common.generation_core.lm_stream_generate', return_value=iter(responses)), \
             patch('heylook_llm.providers.common.generation_core.wired_limit') as mock_wired, \
             patch('heylook_llm.providers.common.generation_core._get_generation_stream', return_value=MagicMock()), \
             patch('heylook_llm.providers.common.generation_core.store_generation_cache') as mock_store:
            mock_wired.return_value.__enter__ = MagicMock()
            mock_wired.return_value.__exit__ = MagicMock(return_value=False)

            mock_manager = MagicMock()
            mock_working_cache = MagicMock()
            mock_working_cache.cache = [MagicMock()]
            mock_manager.get_or_create_cache.return_value = mock_working_cache

            with patch('heylook_llm.providers.common.generation_core.process_prompt_with_cache') as mock_process:
                mock_process.return_value = ([1, 2], mock_working_cache)

                effective = {'max_tokens': 100}
                list(run_generation(
                    model=MagicMock(),
                    tokenizer=MagicMock(),
                    prompt_tokens=[1, 2],
                    effective_request=effective,
                    sampler=MagicMock(),
                    processors=[],
                    model_id="test-model",
                    cache_manager=mock_manager,
                ))

            mock_store.assert_called_once()
            call_args = mock_store.call_args
            assert call_args[0][0] is mock_working_cache
            assert call_args[0][1] == [1, 2, 10]  # prompt + generated
