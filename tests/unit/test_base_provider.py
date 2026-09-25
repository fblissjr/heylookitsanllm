"""Tests for BaseProvider.get_tokenizer() method.

Covers all branches:
1. No processor attribute -> None
2. processor is None -> None
3. processor.tokenizer exists -> returns it
4. processor has decode() -> returns processor itself, even with a private
   ``_tokenizer`` backend (an HF tokenizer as a text model's processor)
5. processor has no decode() -> None
"""

import pytest

from heylook_llm.providers.base import BaseProvider
from heylook_llm.config import ChatRequest


class ConcreteProvider(BaseProvider):
    """Minimal concrete subclass for testing (BaseProvider is ABC)."""

    def load_model(self):
        pass

    def create_chat_completion(self, request: ChatRequest):
        pass


_ABSENT = object()  # leave provider.processor unset
_SENTINEL = object()


class _HfTokenizer:
    """A text model's processor IS the HF tokenizer; ``_tokenizer`` is its Rust
    backend, and returning THAT resolved an empty stop set (v2.0.86-88)."""
    _tokenizer = object()

    def decode(self, ids):
        return "decoded"


class _PublicTokenizer:
    tokenizer = _SENTINEL


class _Decoder:
    def decode(self, ids):
        return "decoded"


class _Bare:
    pass


class TestGetTokenizer:
    # expected: None, "processor" (the processor itself), or the object returned
    @pytest.mark.parametrize("make_processor, expected", [
        (lambda: _ABSENT, None),          # no processor attribute
        (lambda: None, None),             # processor is None
        (_HfTokenizer, "processor"),      # returned itself, not its backend
        (_PublicTokenizer, _SENTINEL),    # .tokenizer (no _tokenizer) wins
        (_Decoder, "processor"),          # decode() and no tokenizer attr
        (_Bare, None),                    # no tokenizer attr, no decode()
    ], ids=["no_processor_attribute", "processor_is_none",
            "a_tokenizer_processor_is_returned_not_its_backend",
            "processor_with_public_tokenizer", "processor_with_decode",
            "processor_without_decode"])
    def test_get_tokenizer(self, make_processor, expected):
        provider = ConcreteProvider("test-model", {}, verbose=False)
        proc = make_processor()
        if proc is not _ABSENT:
            provider.processor = proc
        got = provider.get_tokenizer()
        if expected == "processor":
            assert got is proc
        else:
            assert got is expected


class TestMockProcessorHelperContract:
    """Pin create_mock_processor against the real get_tokenizer() contract.

    The helper uses MagicMock attribute deletion to mirror real mlx-vlm
    processors; these tests guard that the mock state actually drives
    get_tokenizer() to the intended branch (and not, e.g., the .decode fallback).
    """

    def test_with_tokenizer_resolves_to_tokenizer(self):
        from helpers.mlx_mock import create_mock_processor
        provider = ConcreteProvider("test-model", {}, verbose=False)
        provider.processor = create_mock_processor(with_tokenizer=True)
        tok = provider.get_tokenizer()
        assert tok is provider.processor.tokenizer
        assert tok.encode("hi") == [1, 2, 3, 4]  # real list, not a phantom mock

    def test_without_tokenizer_resolves_to_none(self):
        from helpers.mlx_mock import create_mock_processor
        provider = ConcreteProvider("test-model", {}, verbose=False)
        provider.processor = create_mock_processor(with_tokenizer=False)
        # No _tokenizer, no tokenizer, no decode() -> get_tokenizer must be None,
        # not the processor mock itself.
        assert provider.get_tokenizer() is None


