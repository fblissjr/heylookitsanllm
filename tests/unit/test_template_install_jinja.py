# tests/unit/test_template_install_jinja.py
"""The vision path renders the ladder's template, not the legacy .json one.

transformers fills a processor's template from chat_template.json; the MLX
ladder (owner rule) puts chat_template.jinja first. Under auto, a jinja
winner must reach the processor too, or thinking controls, the preview and
`stale` read one body while images render with another. The exceptions keep
a model's images: a non-jinja winner, and a jinja with no media handling
beside a processor template that has it.
"""
import pytest

from heylook_llm.providers.common.template_info import (
    JINJA, TOKENIZER_CONFIG, ModelTemplateInfo, install_chat_template)

JINJA_BODY = "{% for m in messages %}{% if m.type == 'image' %}<|vision_start|>{% endif %}{{ m.content }}{% endfor %}"
LEGACY_BODY = "{% for m in messages %}{% if m.type == 'image' %}<image>{% endif %}{{ m.content }}{% endfor %}"
TEXT_ONLY = "{% for m in messages %}{{ m.content }}{% endfor %}"


class _Holder:
    def __init__(self, template):
        self.chat_template = template


def _install(winner_body, source, processor_body, tokenizer_body="TOKENIZER"):
    tok, proc = _Holder(tokenizer_body), _Holder(processor_body)
    info = ModelTemplateInfo(chat_template=winner_body, template_source=source)
    install_chat_template(tok, info, force=False, processor=proc)
    return tok.chat_template, proc.chat_template


@pytest.mark.unit
@pytest.mark.parametrize("winner, source, processor_before, processor_after", [
    # the case the rule exists for: the jinja wins, the processor held the legacy copy
    (JINJA_BODY, JINJA, LEGACY_BODY, JINJA_BODY),
    # a tokenizer_config winner may be text-only: the processor keeps its own
    (TEXT_ONLY, TOKENIZER_CONFIG, LEGACY_BODY, LEGACY_BODY),
    # a jinja without media handling never replaces one that has it
    (TEXT_ONLY, JINJA, LEGACY_BODY, LEGACY_BODY),
])
def test_auto_puts_a_jinja_winner_on_the_processor(winner, source, processor_before, processor_after):
    tokenizer, processor = _install(winner, source, processor_before)
    assert processor == processor_after
    assert tokenizer == "TOKENIZER"  # auto still never stomps the tokenizer's own
