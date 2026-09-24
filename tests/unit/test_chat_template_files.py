# tests/unit/test_chat_template_files.py
"""chat_template_files checks that need no model directory."""


_CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
           "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_prefix_stability_flags_what_costs_every_turn_its_cache():
    """Plan W3's lint. A generation prompt sharing nothing with the next
    turn's render of the reply (the audit's Qwen3.8 sidecar leaked a newline
    there), and a history that re-renders differently, both fail; a template
    whose generation prompt pre-fills a block the history then drops (Qwen3.5,
    gemma-4) keeps its role header and passes."""
    from heylook_llm.chat_template_files import prefix_stability

    assert prefix_stability(_CHATML)[0] is True
    prefilled = _CHATML.replace("<|im_start|>assistant\n{% endif %}",
                                "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}")
    assert prefix_stability(prefilled)[0] is True
    # (an expression, not a literal newline: trim_blocks eats a newline right
    # after a block tag, the very whitespace rule the real defect tripped on)
    leaked = _CHATML.replace("{% if add_generation_prompt %}<|im_start|>",
                             "{% if add_generation_prompt %}{{ '\\n' }}<|im_start|>")
    stable, why = prefix_stability(leaked)
    assert stable is False and "shares nothing" in why
    rewrites = _CHATML.replace("{{ m.content }}", "{{ m.content }}{% if loop.last %} (brief){% endif %}")
    stable, why = prefix_stability(rewrites)
    assert stable is False and "differently" in why
    assert prefix_stability(None)[0] is None
