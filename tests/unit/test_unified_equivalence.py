# tests/unit/test_unified_equivalence.py
"""Equivalence tests for UnifiedTextStrategy with is_vlm=True vs is_vlm=False.

Proves the whole point of v1.16.0's unification: both VLM-text and text-only
paths produce equivalent generate_text() calls for the same input. Detects
future drift between the two paths.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from heylook_llm.config import ChatMessage, ChatRequest


class FakeResponse:
    def __init__(self, text, token):
        self.text = text
        self.token = token


def _make_strategy(mock_mlx, is_vlm):
    """Create a UnifiedTextStrategy with the given is_vlm flag."""
    from heylook_llm.providers.mlx_provider import UnifiedTextStrategy
    return UnifiedTextStrategy(
        draft_model=MagicMock(name="draft"),
        model_id="test-model",
        model_config={"enable_thinking": True},
        is_vlm=is_vlm,
    )


def _make_request():
    return ChatRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        max_tokens=64,
    )


def _setup_model_and_processor():
    """Create mock model/processor suitable for both VLM and text-only paths."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.language_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.tokenizer.encode.return_value = [1, 2, 3]
    mock_processor.tokenizer.apply_chat_template.return_value = "formatted"
    return mock_model, mock_processor


@pytest.mark.parametrize("is_vlm", [True, False])
class TestUnifiedPathEquivalence:
    """Both VLM-text and text-only should call generate_text with same core params."""

    def test_generate_text_called(self, mock_mlx, is_vlm):
        """Both paths should call generate_text (not run_generation directly)."""
        strategy = _make_strategy(mock_mlx, is_vlm)
        request = _make_request()
        effective = {'max_tokens': 64, 'temperature': 0.5}
        mock_model, mock_processor = _setup_model_and_processor()
        responses = [FakeResponse("hi", 1)]

        with patch('heylook_llm.providers.mlx_provider.generate_text', return_value=iter(responses)) as mock_gen, \
             patch('heylook_llm.providers.mlx_provider.vlm_apply_chat_template', return_value="formatted"):
            list(strategy.generate(request, effective, mock_model, mock_processor))

        mock_gen.assert_called_once()

    def test_effective_request_passthrough(self, mock_mlx, is_vlm):
        """Both paths should pass effective_request unchanged to generate_text."""
        strategy = _make_strategy(mock_mlx, is_vlm)
        request = _make_request()
        effective = {'max_tokens': 64, 'temperature': 0.5, 'top_p': 0.9}
        mock_model, mock_processor = _setup_model_and_processor()
        responses = [FakeResponse("ok", 1)]

        with patch('heylook_llm.providers.mlx_provider.generate_text', return_value=iter(responses)) as mock_gen, \
             patch('heylook_llm.providers.mlx_provider.vlm_apply_chat_template', return_value="formatted"):
            list(strategy.generate(request, effective, mock_model, mock_processor))

        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs['effective_request'] is effective

    def test_model_id_passthrough(self, mock_mlx, is_vlm):
        """Both paths should pass model_id to generate_text."""
        strategy = _make_strategy(mock_mlx, is_vlm)
        request = _make_request()
        effective = {'max_tokens': 64}
        mock_model, mock_processor = _setup_model_and_processor()
        responses = [FakeResponse("ok", 1)]

        with patch('heylook_llm.providers.mlx_provider.generate_text', return_value=iter(responses)) as mock_gen, \
             patch('heylook_llm.providers.mlx_provider.vlm_apply_chat_template', return_value="formatted"):
            list(strategy.generate(request, effective, mock_model, mock_processor))

        assert mock_gen.call_args.kwargs['model_id'] == "test-model"

    def test_draft_model_passthrough(self, mock_mlx, is_vlm):
        """Both paths should pass draft_model to generate_text."""
        strategy = _make_strategy(mock_mlx, is_vlm)
        request = _make_request()
        effective = {'max_tokens': 64}
        mock_model, mock_processor = _setup_model_and_processor()
        responses = [FakeResponse("ok", 1)]

        with patch('heylook_llm.providers.mlx_provider.generate_text', return_value=iter(responses)) as mock_gen, \
             patch('heylook_llm.providers.mlx_provider.vlm_apply_chat_template', return_value="formatted"):
            list(strategy.generate(request, effective, mock_model, mock_processor))

        assert mock_gen.call_args.kwargs['draft_model'] is strategy.draft_model


class TestModelWrapping:
    """Verify is_vlm controls model wrapping correctly."""

    def test_vlm_wraps_with_logits_wrapper(self, mock_mlx):
        """is_vlm=True should use LanguageModelLogitsWrapper."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="test", is_vlm=True,
        )
        mock_model = MagicMock()
        mock_model.language_model = MagicMock()

        gen_model = strategy._get_generation_model(mock_model)
        # Should be a wrapper, not the raw model
        assert gen_model is not mock_model
        # Should wrap the language_model component
        assert hasattr(gen_model, 'model')

    def test_text_only_uses_raw_model(self, mock_mlx):
        """is_vlm=False should return the raw model unchanged."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="test", is_vlm=False,
        )
        mock_model = MagicMock()

        gen_model = strategy._get_generation_model(mock_model)
        assert gen_model is mock_model

    def test_wrapper_cached_across_calls(self, mock_mlx):
        """The LanguageModelLogitsWrapper should be cached per strategy instance."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="test", is_vlm=True,
        )
        mock_model = MagicMock()
        mock_model.language_model = MagicMock()

        wrapper1 = strategy._get_generation_model(mock_model)
        wrapper2 = strategy._get_generation_model(mock_model)
        assert wrapper1 is wrapper2
