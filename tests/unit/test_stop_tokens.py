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

    @pytest.mark.parametrize(
        "body", [None, "{not json"], ids=["missing-file-is-noop", "malformed-never-raises"],
    )
    def test_no_usable_generation_config_leaves_the_set(self, tmp_path, body):
        if body is not None:
            (tmp_path / "generation_config.json").write_text(body)
        tok = self._tok(eos_token_id=1)
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1}

    def test_transformers5_setattr_interception(self, tmp_path):
        """transformers 5.x SpecialTokensMixin.__setattr__ intercepts
        special-token attr assignment and rejects non-string values
        ('Cannot set a non-string value as the eos_token'). The extension
        must not depend on being able to assign eos_token_ids on the
        tokenizer -- the live gemma-4 regression: <turn|> silently dropped
        from the stop set, every response ran to the token cap."""
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106, 50]})
        )
        tok = _SetattrGuardedTokenizer()
        extend_eos_from_generation_config(tok, tmp_path)
        assert resolve_stop_tokens(tok) == {1, 106, 50}


class _SetattrGuardedTokenizer:
    """Mimics transformers 5.x SpecialTokensMixin: special-token attribute
    assignment routes through validation that rejects non-string values."""

    def __init__(self):
        object.__setattr__(self, "eos_token_id", 1)

    def __setattr__(self, name, value):
        if name.startswith("eos_token") and not isinstance(value, str):
            raise ValueError("Cannot set a non-string value as the eos_token")
        object.__setattr__(self, name, value)


@pytest.mark.unit
class TestResolveStopTokens:
    """eos_token_ids (plural) takes priority; a None or empty plural falls
    back to eos_token_id (singular); id 0 is valid, not falsy; neither
    attribute gives an empty set. Always a set. ``spec=[]`` builds a
    tokenizer that has only the attributes the row sets."""

    @pytest.mark.parametrize(
        "bare, attrs, expected",
        [
            (False, {"eos_token_ids": [1, 2, 3], "eos_token_id": 1}, {1, 2, 3}),
            (True, {"eos_token_id": 42}, {42}),
            (False, {"eos_token_ids": None, "eos_token_id": 7}, {7}),
            (False, {"eos_token_ids": [], "eos_token_id": 5}, {5}),
            (True, {}, set()),
            (False, {"eos_token_ids": [10]}, {10}),
            (True, {"eos_token_id": 0}, {0}),
        ],
        ids=[
            "plural-wins", "singular", "none-plural-falls-back", "empty-plural-falls-back",
            "neither-is-empty", "returns-set-type", "zero-is-valid",
        ],
    )
    def test_resolve_stop_tokens(self, bare, attrs, expected):
        tok = MagicMock(spec=[]) if bare else MagicMock()
        for name, value in attrs.items():
            setattr(tok, name, value)
        result = resolve_stop_tokens(tok)
        assert isinstance(result, set)
        assert result == expected
