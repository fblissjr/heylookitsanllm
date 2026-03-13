"""Tests for resolve_stop_tokens utility."""
import pytest
from unittest.mock import MagicMock

from heylook_llm.providers.common.stop_tokens import resolve_stop_tokens


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
