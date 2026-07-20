"""Tests for resolve_stop_tokens utility."""
import json

import pytest
from unittest.mock import MagicMock

from heylook_llm.providers.common.stop_tokens import (
    extend_eos_from_generation_config,
    resolve_stop_tokens,
)


@pytest.mark.unit
class TestExtendEosFromGenerationConfig:
    """Raw HF tokenizers on the mlx-vlm load path don't absorb
    generation_config.json's eos list (mlx-lm's TokenizerWrapper does).
    gemma-4: tokenizer says eos=1 while generation_config declares
    [1, 106, 50] incl. the <turn|> turn terminator -- without the union,
    generation runs past end-of-turn until <eos> or the token cap."""

    def _tok(self, **attrs):
        from types import SimpleNamespace
        return SimpleNamespace(**attrs)

    def test_unions_generation_config_ids(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106, 50]})
        )
        tok = self._tok(eos_token_id=1)
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1, 106, 50}

    def test_scalar_id_and_existing_set_preserved(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 106})
        )
        tok = self._tok(eos_token_ids={1, 2})
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1, 2, 106}

    def test_missing_file_is_noop(self, tmp_path):
        tok = self._tok(eos_token_id=1)
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1}

    def test_malformed_file_never_raises(self, tmp_path):
        (tmp_path / "generation_config.json").write_text("{not json")
        tok = self._tok(eos_token_id=1)
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1}


class _FakeHfTokenizer:
    """Minimal raw-HF-tokenizer stand-in for TokenizerWrapper construction."""
    eos_token_id = 1
    eos_token_ids = {1, 106, 50}  # as extended from generation_config at load
    chat_template = None

    def get_vocab(self):
        return {}


@pytest.mark.unit
class TestEnsureGenTokenizer:
    """run_generation must wrap raw tokenizers itself: mlx-lm's
    stream_generate auto-wrap uses only the single eos_token_id, silently
    dropping the extra terminators (gemma-4's <turn|>)."""

    def test_raw_tokenizer_wrapped_with_full_stop_set(self):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from heylook_llm.providers.common.generation_core import ensure_gen_tokenizer

        wrapped = ensure_gen_tokenizer(_FakeHfTokenizer())
        assert isinstance(wrapped, TokenizerWrapper)
        assert resolve_stop_tokens(wrapped) == {1, 106, 50}

    def test_already_wrapped_passes_through_untouched(self):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from heylook_llm.providers.common.generation_core import ensure_gen_tokenizer

        w = TokenizerWrapper(_FakeHfTokenizer(), eos_token_ids={7})
        assert ensure_gen_tokenizer(w) is w


@pytest.mark.unit
class TestResolveStopTokens:
    def test_plural_eos_token_ids(self):
        """eos_token_ids (plural) takes priority."""
        tok = MagicMock()
        tok.eos_token_ids = [1, 2, 3]
        tok.eos_token_id = 1
        assert resolve_stop_tokens(tok) == {1, 2, 3}

    def test_singular_eos_token_id(self):
        """Falls back to eos_token_id (singular) when plural is absent."""
        tok = MagicMock(spec=[])
        tok.eos_token_id = 42
        result = resolve_stop_tokens(tok)
        assert result == {42}

    def test_none_eos_token_ids_falls_back(self):
        """When eos_token_ids is None, falls back to eos_token_id."""
        tok = MagicMock()
        tok.eos_token_ids = None
        tok.eos_token_id = 7
        assert resolve_stop_tokens(tok) == {7}

    def test_empty_eos_token_ids_falls_back(self):
        """When eos_token_ids is empty, falls back to eos_token_id."""
        tok = MagicMock()
        tok.eos_token_ids = []
        tok.eos_token_id = 5
        assert resolve_stop_tokens(tok) == {5}

    def test_no_eos_attrs_returns_empty(self):
        """Returns empty set if tokenizer has neither attribute."""
        tok = MagicMock(spec=[])
        assert resolve_stop_tokens(tok) == set()

    def test_returns_set_type(self):
        """Always returns a set."""
        tok = MagicMock()
        tok.eos_token_ids = [10]
        result = resolve_stop_tokens(tok)
        assert isinstance(result, set)

    def test_eos_token_id_zero_is_valid(self):
        """eos_token_id=0 should be treated as valid (not falsy)."""
        tok = MagicMock(spec=[])
        tok.eos_token_id = 0
        assert resolve_stop_tokens(tok) == {0}
