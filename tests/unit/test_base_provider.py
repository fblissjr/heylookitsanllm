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
