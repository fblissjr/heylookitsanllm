"""Tests for BaseProvider.get_tokenizer() method.

Covers all branches:
1. No processor attribute -> None
2. processor is None -> None
3. processor._tokenizer exists -> returns it
4. processor.tokenizer exists -> returns it
5. processor has decode() -> returns processor itself
6. processor has no decode() -> None
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


class TestGetTokenizer:
    def test_no_processor_attribute(self):
        """Provider with no processor attribute returns None."""
        provider = ConcreteProvider("test-model", {}, verbose=False)
        assert provider.get_tokenizer() is None

    def test_processor_is_none(self):
        """Provider with processor=None returns None."""
        provider = ConcreteProvider("test-model", {}, verbose=False)
        provider.processor = None
        assert provider.get_tokenizer() is None

    def test_processor_with_private_tokenizer(self):
        """Processor with _tokenizer attr returns _tokenizer."""
        provider = ConcreteProvider("test-model", {}, verbose=False)
        sentinel = object()

        class FakeProcessor:
            _tokenizer = sentinel

        provider.processor = FakeProcessor()
        assert provider.get_tokenizer() is sentinel

    def test_processor_with_public_tokenizer(self):
        """Processor with tokenizer attr (no _tokenizer) returns tokenizer."""
        provider = ConcreteProvider("test-model", {}, verbose=False)
        sentinel = object()

        class FakeProcessor:
            tokenizer = sentinel

        provider.processor = FakeProcessor()
        assert provider.get_tokenizer() is sentinel

    def test_processor_with_decode(self):
        """Processor with decode() method but no tokenizer attr returns processor."""
        provider = ConcreteProvider("test-model", {}, verbose=False)

        class FakeProcessor:
            def decode(self, ids):
                return "decoded"

        proc = FakeProcessor()
        provider.processor = proc
        assert provider.get_tokenizer() is proc

    def test_processor_without_decode(self):
        """Processor with no tokenizer attr and no decode() returns None."""
        provider = ConcreteProvider("test-model", {}, verbose=False)

        class FakeProcessor:
            pass

        provider.processor = FakeProcessor()
        assert provider.get_tokenizer() is None
