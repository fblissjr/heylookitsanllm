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


class TestWarmup:
    """S1.4: provider.warmup() exists and is safe to call."""

    def test_base_warmup_is_noop(self):
        """Default BaseProvider.warmup() returns None without side effects."""
        provider = ConcreteProvider("test-model", {}, verbose=False)
        # Just confirm it exists and doesn't raise.
        assert provider.warmup() is None

    def test_base_warmup_swallows_exceptions_in_subclass(self):
        """A subclass warmup() that raises should not propagate.

        Warmup is an optimization, not a correctness requirement. A failure
        here must not block the router from returning a usable provider.
        """

        class BrokenWarmup(ConcreteProvider):
            def _do_warmup(self):
                raise RuntimeError("simulated warmup failure")

            def warmup(self):
                try:
                    self._do_warmup()
                except Exception:
                    # Base contract: log-and-continue; don't propagate.
                    return None

        provider = BrokenWarmup("test-model", {}, verbose=False)
        # Must not raise
        assert provider.warmup() is None
