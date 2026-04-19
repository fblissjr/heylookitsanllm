"""Tests for template-info loader.

``read_template_info(model_dir, chat_template_source)`` reads the model's
on-disk chat template + ``tokenizer_config.json`` + ``tokenizer.json`` and
returns everything a downstream component needs to interpret the model's
output WITHOUT hardcoded special-token literals or format-name lookup.

The loader is the single place we look at these files. Every other piece
of the output pipeline (reasoning parser factory, harmony parser's strip
set, observability) reads from this info object.
"""

from __future__ import annotations

import json


_HARMONY_TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        "199998": {"content": "<|startoftext|>", "special": True},
        "199999": {"content": "<|endoftext|>", "special": True},
        "200002": {"content": "<|return|>", "special": True},
        "200005": {"content": "<|channel|>", "special": True},
        "200006": {"content": "<|start|>", "special": True},
        "200007": {"content": "<|end|>", "special": True},
        "200008": {"content": "<|message|>", "special": True},
        "200012": {"content": "<|call|>", "special": True},
        "12345": {"content": "<|normal_token|>", "special": False},
    },
    "chat_template": "{# embedded stub #}",
}

_HARMONY_JINJA = (
    '{{- "<|start|>system<|channel|>final<|message|>hello<|end|>" }}\n'
    '{%- if add_generation_prompt -%}\n'
    '<|start|>assistant\n'
    '{%- endif -%}\n'
)


def _write_model_dir(tmp_path, *, jinja=None, tokenizer_config=None):
    if jinja is not None:
        (tmp_path / "chat_template.jinja").write_text(jinja)
    if tokenizer_config is not None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return tmp_path


class TestReadTemplateInfoTokenizerJson:
    """``tokenizer.json``'s ``added_tokens`` array is the authoritative source
    for fast tokenizers. Some models don't populate ``tokenizer_config.json``
    ``added_tokens_decoder`` at all, so template_info must read both files
    and union the results."""

    def test_reads_specials_from_tokenizer_json(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        (tmp_path / "tokenizer.json").write_text(json.dumps({
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": True},
                {"id": 1, "content": "<eos>", "special": True},
                {"id": 100, "content": "<|channel>", "special": True},
                {"id": 101, "content": "<channel|>", "special": True},
                {"id": 200, "content": "regular_token", "special": False},
            ],
        }))

        info = read_template_info(tmp_path, source=None)

        assert "<pad>" in info.special_tokens
        assert "<|channel>" in info.special_tokens
        assert "<channel|>" in info.special_tokens
        assert "regular_token" not in info.special_tokens

    def test_unions_specials_from_both_files(self, tmp_path):
        """tokenizer_config.json and tokenizer.json may list DIFFERENT
        specials (some models split the set). Union the two."""
        from heylook_llm.providers.common.template_info import read_template_info

        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "added_tokens_decoder": {
                "1": {"content": "<|from_config|>", "special": True},
            },
            "chat_template": "{{ '' }}",
        }))
        (tmp_path / "tokenizer.json").write_text(json.dumps({
            "added_tokens": [
                {"id": 2, "content": "<|from_json|>", "special": True},
            ],
        }))

        info = read_template_info(tmp_path, source=None)

        assert "<|from_config|>" in info.special_tokens
        assert "<|from_json|>" in info.special_tokens


class TestReadTemplateInfoHarmonyFormat:
    def test_reads_jinja_when_present(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=_HARMONY_JINJA,
            tokenizer_config=_HARMONY_TOKENIZER_CONFIG,
        )
        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == _HARMONY_JINJA
        assert info.template_source == "jinja"

    def test_special_tokens_set_from_added_tokens_decoder(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=_HARMONY_JINJA,
            tokenizer_config=_HARMONY_TOKENIZER_CONFIG,
        )
        info = read_template_info(tmp_path, source=None)

        # Only tokens with special:true, not the 12345 one.
        assert "<|channel|>" in info.special_tokens
        assert "<|message|>" in info.special_tokens
        assert "<|start|>" in info.special_tokens
        assert "<|end|>" in info.special_tokens
        assert "<|return|>" in info.special_tokens
        assert "<|normal_token|>" not in info.special_tokens

    def test_detects_harmony_format_from_template(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=_HARMONY_JINJA,
            tokenizer_config=_HARMONY_TOKENIZER_CONFIG,
        )
        info = read_template_info(tmp_path, source=None)

        assert info.has_harmony_structure is True


class TestReadTemplateInfoThinkingMarkers:
    def test_detects_thinking_markers_from_template(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        jinja = (
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking %}<think>\n\n</think>\n\n"
            "{% endif %}{% endif %}"
        )
        tokenizer_config = {
            "added_tokens_decoder": {
                "151667": {"content": "<think>", "special": True},
                "151668": {"content": "</think>", "special": True},
                "151643": {"content": "<|im_start|>", "special": True},
            },
            "chat_template": jinja,
        }
        _write_model_dir(tmp_path, jinja=jinja, tokenizer_config=tokenizer_config)

        info = read_template_info(tmp_path, source=None)

        assert info.has_thinking_markers is True
        assert info.has_harmony_structure is False
        assert "<think>" in info.special_tokens
        assert "</think>" in info.special_tokens


class TestReadTemplateInfoFallbacks:
    def test_falls_back_to_embedded_template_when_jinja_missing(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=None,
            tokenizer_config={
                "added_tokens_decoder": {"1": {"content": "<|eos|>", "special": True}},
                "chat_template": "{{ 'embedded template body' }}",
            },
        )

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == "{{ 'embedded template body' }}"
        assert info.template_source == "tokenizer_config"

    def test_empty_when_nothing_available(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == ""
        assert info.special_tokens == frozenset()
        assert info.has_harmony_structure is False
        assert info.has_thinking_markers is False
        assert info.template_source == "auto"

    def test_source_jinja_forces_jinja_even_with_embedded_template(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path,
            jinja="{{ 'forced jinja' }}",
            tokenizer_config={
                "added_tokens_decoder": {},
                "chat_template": "{{ 'ignored embedded' }}",
            },
        )

        info = read_template_info(tmp_path, source="jinja")

        assert info.chat_template == "{{ 'forced jinja' }}"

    def test_source_tokenizer_config_forces_embedded_even_with_jinja(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path,
            jinja="{{ 'ignored jinja' }}",
            tokenizer_config={
                "added_tokens_decoder": {},
                "chat_template": "{{ 'forced embedded' }}",
            },
        )

        info = read_template_info(tmp_path, source="tokenizer_config")

        assert info.chat_template == "{{ 'forced embedded' }}"

    def test_source_absolute_path(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        custom = tmp_path / "my_template.jinja"
        custom.write_text("{{ 'custom from path' }}")
        _write_model_dir(
            tmp_path, jinja="{{ 'dir jinja' }}",
            tokenizer_config={"added_tokens_decoder": {}, "chat_template": "x"},
        )

        info = read_template_info(tmp_path, source=str(custom))

        assert info.chat_template == "{{ 'custom from path' }}"

    def test_malformed_tokenizer_config_does_not_raise(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        (tmp_path / "tokenizer_config.json").write_text("{{{ not json")
        info = read_template_info(tmp_path, source=None)

        assert info.special_tokens == frozenset()
