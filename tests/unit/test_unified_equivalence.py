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


class TestApplyTemplateMissingTemplate:
    """A model folder with NO chat template anywhere (no chat_template.jinja,
    no embedded tokenizer_config template, no chat_template.json) makes
    transformers raise a raw ValueError deep inside apply_chat_template. The
    strategy must convert that into an actionable error naming the model and
    the fix, deciding from TOKENIZER STATE (chat_template/has_chat_template)
    rather than matching transformers' error prose -- the prose changes
    between versions and a string match silently reverts the fix."""

    def _tokenizer(self, *, chat_template, has_chat_template=False, error="boom"):
        tokenizer = MagicMock()
        tokenizer.chat_template = chat_template
        tokenizer.has_chat_template = has_chat_template
        tokenizer.apply_chat_template.side_effect = ValueError(error)
        return tokenizer

    def test_missing_template_error_is_actionable(self, mock_mlx):
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="no-template-model", is_vlm=False,
            model_config={},
        )
        # Deliberately NOT the current transformers message: the translation
        # must not depend on upstream prose.
        tokenizer = self._tokenizer(
            chat_template=None, error="some future upstream wording",
        )

        with pytest.raises(ValueError) as exc_info:
            strategy._apply_template(
                [{"role": "user", "content": "hi"}],
                tokenizer, MagicMock(), MagicMock(), {},
            )

        msg = str(exc_info.value)
        assert "no-template-model" in msg
        assert "chat_template" in msg

    def test_other_value_errors_still_propagate(self, mock_mlx):
        """Template-rendering ValueErrors from a PRESENT template (e.g. a
        template raising on bad message shape) keep their message."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="m", is_vlm=False, model_config={},
        )
        tokenizer = self._tokenizer(
            chat_template="{{ messages }}", error="roles must alternate",
        )

        with pytest.raises(ValueError, match="roles must alternate"):
            strategy._apply_template(
                [{"role": "user", "content": "hi"}],
                tokenizer, MagicMock(), MagicMock(), {},
            )

    def test_wrapper_level_python_template_errors_propagate(self, mock_mlx):
        """mlx-lm chat_template_type models render via the wrapper's python
        template while the HF chat_template attr stays None -- their errors
        must NOT be mislabeled as 'no chat template'."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            draft_model=None, model_id="m", is_vlm=False, model_config={},
        )
        tokenizer = self._tokenizer(
            chat_template=None, has_chat_template=True,
            error="python template render error",
        )

        with pytest.raises(ValueError, match="python template render error"):
            strategy._apply_template(
                [{"role": "user", "content": "hi"}],
                tokenizer, MagicMock(), MagicMock(), {},
            )
