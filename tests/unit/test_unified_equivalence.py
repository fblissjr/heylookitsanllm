# tests/unit/test_unified_equivalence.py
"""The MLX text-path template call: a missing template is refused with a
message naming the fix, and other template errors propagate."""

import pytest

from heylook_llm.config import ChatMessage, ChatRequest


def _real_tokenizer(chat_template=None):
    """A real transformers tokenizer (no weights, no files), so the error
    comes from transformers itself rather than from a stub spelling it."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")))
    tok.chat_template = chat_template
    return tok


def _render(model_id, tokenizer):
    from heylook_llm.providers.mlx_provider import UnifiedTextStrategy

    request = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    return UnifiedTextStrategy(model_id=model_id, is_vlm=False).render_prompt(
        request, {}, None, tokenizer)


class TestApplyTemplateMissingTemplate:
    """A model folder with NO chat template anywhere (no chat_template.jinja,
    no embedded tokenizer_config template, no chat_template.json) makes
    transformers raise a raw ValueError deep inside apply_chat_template. The
    strategy must convert that into an actionable error naming the model and
    the fix, deciding from TOKENIZER STATE (chat_template)
    rather than matching transformers' error prose -- the prose changes
    between versions and a string match silently reverts the fix."""

    def test_missing_template_error_is_actionable(self):
        with pytest.raises(ValueError) as exc_info:
            _render("no-template-model", _real_tokenizer(chat_template=None))

        msg = str(exc_info.value)
        assert "no-template-model" in msg
        assert "chat_template" in msg

    def test_other_value_errors_still_propagate(self):
        """A ValueError from a tokenizer that HAS templates (here: several
        named ones and no default, which transformers refuses) keeps its own
        message instead of being reported as a missing template."""
        tok = _real_tokenizer(chat_template={"tool_use": "{{ messages[0]['content'] }}"})

        with pytest.raises(ValueError) as exc_info:
            _render("m", tok)
        assert "multiple chat templates" in str(exc_info.value)
        assert "has no chat template" not in str(exc_info.value)
