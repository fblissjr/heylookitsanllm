# tests/unit/test_unified_equivalence.py
"""The MLX text-path template call: a missing template is refused with a
message naming the fix, and other template errors propagate."""

import pytest
from unittest.mock import MagicMock, patch, call

from heylook_llm.config import ChatMessage, ChatRequest


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
            model_id="no-template-model", is_vlm=False,
            model_config={},
        )
        # Deliberately NOT the current transformers message: the translation
        # must not depend on upstream prose.
        tokenizer = self._tokenizer(
            chat_template=None, error="some future upstream wording",
        )

        with pytest.raises(ValueError) as exc_info:
            strategy._render_template(
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
            model_id="m", is_vlm=False, model_config={},
        )
        tokenizer = self._tokenizer(
            chat_template="{{ messages }}", error="roles must alternate",
        )

        with pytest.raises(ValueError, match="roles must alternate"):
            strategy._render_template(
                [{"role": "user", "content": "hi"}],
                tokenizer, MagicMock(), MagicMock(), {},
            )

    def test_wrapper_level_python_template_errors_propagate(self, mock_mlx):
        """mlx-lm chat_template_type models render via the wrapper's python
        template while the HF chat_template attr stays None -- their errors
        must NOT be mislabeled as 'no chat template'."""
        from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

        strategy = UnifiedTextStrategy(
            model_id="m", is_vlm=False, model_config={},
        )
        tokenizer = self._tokenizer(
            chat_template=None, has_chat_template=True,
            error="python template render error",
        )

        with pytest.raises(ValueError, match="python template render error"):
            strategy._render_template(
                [{"role": "user", "content": "hi"}],
                tokenizer, MagicMock(), MagicMock(), {},
            )
