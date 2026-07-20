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


class TestReadTemplateInfoGemmaChannelFormat:
    _GEMMA_JINJA = (
        "{{ bos_token }}{% if enable_thinking %}<|think|>\n{% endif %}"
        "{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n"
        "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}{% endif %}"
    )

    def test_detects_gemma_channel_structure(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(tmp_path, jinja=self._GEMMA_JINJA)
        info = read_template_info(tmp_path, source=None)

        assert info.has_gemma_channel_structure is True
        assert info.has_harmony_structure is False
        assert info.has_thinking_markers is False

    def test_harmony_template_is_not_gemma_structure(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=_HARMONY_JINJA,
            tokenizer_config=_HARMONY_TOKENIZER_CONFIG,
        )
        info = read_template_info(tmp_path, source=None)

        assert info.has_gemma_channel_structure is False

    def test_detects_enable_thinking_toggle(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(tmp_path, jinja=self._GEMMA_JINJA)
        assert read_template_info(tmp_path, source=None).supports_enable_thinking is True

    def test_no_enable_thinking_reference_means_no_toggle(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path,
            jinja="{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}",
        )
        assert read_template_info(tmp_path, source=None).supports_enable_thinking is False


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


class TestChatTemplateJsonFallback:
    """Some VLM conversions ship the template only as ``chat_template.json``
    (the processor-side convention: ``{"chat_template": "..."}``). The
    tokenizer never sees that file, so template_info must read it as the
    last auto fallback -- otherwise a chat_template.json-only model looks
    template-less to us while the processor knows better."""

    def test_auto_falls_back_to_chat_template_json(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        (tmp_path / "chat_template.json").write_text(
            json.dumps({"chat_template": "{{ 'from chat_template.json' }}"})
        )

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == "{{ 'from chat_template.json' }}"
        assert info.template_source == "chat_template_json"

    def test_embedded_template_wins_over_chat_template_json(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(
            tmp_path, jinja=None,
            tokenizer_config={
                "added_tokens_decoder": {},
                "chat_template": "{{ 'embedded' }}",
            },
        )
        (tmp_path / "chat_template.json").write_text(
            json.dumps({"chat_template": "{{ 'json' }}"})
        )

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == "{{ 'embedded' }}"
        assert info.template_source == "tokenizer_config"

    def test_jinja_wins_over_chat_template_json(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(tmp_path, jinja="{{ 'jinja' }}", tokenizer_config=None)
        (tmp_path / "chat_template.json").write_text(
            json.dumps({"chat_template": "{{ 'json' }}"})
        )

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == "{{ 'jinja' }}"
        assert info.template_source == "jinja"

    def test_malformed_chat_template_json_does_not_raise(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        (tmp_path / "chat_template.json").write_text("{{{ not json")

        info = read_template_info(tmp_path, source=None)

        assert info.chat_template == ""


class _FakeTokenizer:
    def __init__(self, chat_template=None, inner=None):
        self.chat_template = chat_template
        if inner is not None:
            self._tokenizer = inner


class TestInstallChatTemplate:
    """``install_chat_template(tokenizer, info, force=...)`` is the single
    place the resolved template gets attached to a live tokenizer.

    - force=True (explicit ``chat_template_source``): always overwrite.
    - force=False (auto): only fill in a MISSING tokenizer template --
      covers chat_template.json-only models where AutoTokenizer loads
      nothing, without stomping on what transformers loaded natively.
    """

    def _info(self, template="{{ 'resolved' }}"):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo
        return ModelTemplateInfo(chat_template=template)

    def test_force_overwrites_existing_template(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        tok = _FakeTokenizer(chat_template="{{ 'native' }}")
        installed = install_chat_template(tok, self._info(), force=True)

        assert installed is True
        assert tok.chat_template == "{{ 'resolved' }}"

    def test_auto_fills_missing_template(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        tok = _FakeTokenizer(chat_template=None)
        installed = install_chat_template(tok, self._info(), force=False)

        assert installed is True
        assert tok.chat_template == "{{ 'resolved' }}"

    def test_auto_preserves_native_template(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        tok = _FakeTokenizer(chat_template="{{ 'native' }}")
        installed = install_chat_template(tok, self._info(), force=False)

        assert installed is False
        assert tok.chat_template == "{{ 'native' }}"

    def test_noop_when_no_resolved_template(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        tok = _FakeTokenizer(chat_template=None)
        installed = install_chat_template(tok, self._info(template=""), force=True)

        assert installed is False
        assert tok.chat_template is None

    def test_installs_on_inner_tokenizer_too(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        inner = _FakeTokenizer(chat_template=None)
        tok = _FakeTokenizer(chat_template=None, inner=inner)
        install_chat_template(tok, self._info(), force=True)

        assert tok.chat_template == "{{ 'resolved' }}"
        assert inner.chat_template == "{{ 'resolved' }}"

    def test_none_tokenizer_is_safe(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        assert install_chat_template(None, self._info(), force=True) is False


class TestDetectChatTemplateSource:
    """``detect_chat_template_source(model_dir)`` is the ONE import-time
    detection both the CLI wizard and the /v1/admin import route call --
    the two inline copies drifted (.exists() vs .is_file()) within a day
    of each other."""

    def test_returns_jinja_when_file_present(self, tmp_path):
        from heylook_llm.providers.common.template_info import detect_chat_template_source

        (tmp_path / "chat_template.jinja").write_text("{{ messages }}")

        assert detect_chat_template_source(tmp_path) == "jinja"

    def test_returns_none_when_absent(self, tmp_path):
        from heylook_llm.providers.common.template_info import detect_chat_template_source

        assert detect_chat_template_source(tmp_path) is None

    def test_directory_named_like_template_is_not_detected(self, tmp_path):
        from heylook_llm.providers.common.template_info import detect_chat_template_source

        (tmp_path / "chat_template.jinja").mkdir()

        assert detect_chat_template_source(tmp_path) is None

    def test_expands_tilde_paths(self, tmp_path, monkeypatch):
        from heylook_llm.providers.common.template_info import detect_chat_template_source

        monkeypatch.setenv("HOME", str(tmp_path))
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("{{ messages }}")

        assert detect_chat_template_source("~/m") == "jinja"  # path-privacy: ignore


class TestExplicitChatTemplateJsonSource:
    """'chat_template_json' appears as a resolved-source label in load logs,
    so it must also be an accepted explicit ``chat_template_source`` value --
    otherwise configuring what the log reports warns 'not recognized' and,
    worse, still force-installs the auto pick."""

    def test_source_chat_template_json_forces_json_file(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(tmp_path, jinja="{{ 'jinja' }}", tokenizer_config=None)
        (tmp_path / "chat_template.json").write_text(
            json.dumps({"chat_template": "{{ 'forced json' }}"})
        )

        info = read_template_info(tmp_path, source="chat_template_json")

        assert info.chat_template == "{{ 'forced json' }}"
        assert info.template_source == "chat_template_json"

    def test_source_chat_template_json_missing_falls_back_to_auto(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        _write_model_dir(tmp_path, jinja="{{ 'jinja' }}", tokenizer_config=None)

        info = read_template_info(tmp_path, source="chat_template_json")

        assert info.chat_template == "{{ 'jinja' }}"
        assert info.template_source == "jinja"


class TestIsExplicitSource:
    """force-install must engage only for a genuinely explicit source --
    the documented value \"auto\" is truthy but means the fill-only path."""

    def test_none_and_empty_are_not_explicit(self):
        from heylook_llm.providers.common.template_info import is_explicit_source

        assert is_explicit_source(None) is False
        assert is_explicit_source("") is False

    def test_auto_is_not_explicit(self):
        from heylook_llm.providers.common.template_info import is_explicit_source

        assert is_explicit_source("auto") is False
        assert is_explicit_source(" AUTO ") is False

    def test_named_sources_and_paths_are_explicit(self):
        from heylook_llm.providers.common.template_info import is_explicit_source

        assert is_explicit_source("jinja") is True
        assert is_explicit_source("tokenizer_config") is True
        assert is_explicit_source("chat_template_json") is True
        assert is_explicit_source("/abs/path/custom.jinja") is True


class TestMissingTemplateError:
    """``missing_template_error(tokenizer, model_id)`` decides 'the model
    truly has no chat template' from TOKENIZER STATE, not from matching
    transformers' error prose (which is version-fragile). It must respect
    mlx-lm's wrapper-level python templates (``has_chat_template``), which
    render fine while the HF ``chat_template`` attr stays None."""

    def test_returns_actionable_error_when_no_template(self):
        from heylook_llm.providers.common.template_info import missing_template_error

        tok = _FakeTokenizer(chat_template=None)
        err = missing_template_error(tok, "my-model")

        assert isinstance(err, ValueError)
        assert "my-model" in str(err)
        assert "chat_template" in str(err)

    def test_returns_none_when_template_present(self):
        from heylook_llm.providers.common.template_info import missing_template_error

        tok = _FakeTokenizer(chat_template="{{ x }}")

        assert missing_template_error(tok, "m") is None

    def test_returns_none_for_wrapper_level_python_template(self):
        from heylook_llm.providers.common.template_info import missing_template_error

        tok = _FakeTokenizer(chat_template=None)
        tok.has_chat_template = True

        assert missing_template_error(tok, "m") is None

    def test_message_is_generic_without_model_id(self):
        from heylook_llm.providers.common.template_info import missing_template_error

        err = missing_template_error(_FakeTokenizer(chat_template=None), None)

        assert isinstance(err, ValueError)
        assert "chat_template" in str(err)


class TestStopTokenValidation:
    """A stop-less chat template (renders none of the model's OWN stop tokens)
    is rejected + self-heals, so a broken/corrupted jinja can't cause runaway
    generation. The stop set is read from the model's config, never hardcoded."""

    # gemma-like: eos_token_id resolves via added_tokens_decoder to <eos> + <end_of_turn>
    _CFG = {
        "eos_token": "<eos>",
        "eos_token_id": [1, 106],
        "added_tokens_decoder": {
            "1": {"content": "<eos>", "special": True},
            "106": {"content": "<end_of_turn>", "special": True},
        },
    }

    def test_broken_jinja_rejected_no_valid_fallback(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info
        # renders <|turn>model -- none of the model's stop tokens; embedded has none either
        _write_model_dir(tmp_path, jinja="{{ '<|turn>model\\n' }}", tokenizer_config=self._CFG)
        info = read_template_info(tmp_path, source="jinja")
        assert info.chat_template == ""                       # broken jinja NOT installed
        assert info.template_source == "none(stopless)"

    def test_valid_jinja_kept(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info
        _write_model_dir(tmp_path, jinja="{{ '<start_of_turn>model\\n<end_of_turn>' }}",
                         tokenizer_config=self._CFG)
        info = read_template_info(tmp_path, source="jinja")
        assert "<end_of_turn>" in info.chat_template

    def test_self_heals_to_embedded(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info
        cfg = dict(self._CFG, chat_template="{{ '<end_of_turn>' }}")  # embedded IS valid
        _write_model_dir(tmp_path, jinja="{{ '<|turn>model' }}", tokenizer_config=cfg)
        info = read_template_info(tmp_path, source="jinja")
        assert "<end_of_turn>" in info.chat_template
        assert info.template_source == "tokenizer_config"

    def test_detect_skips_broken_jinja(self, tmp_path):
        from heylook_llm.providers.common.template_info import detect_chat_template_source
        _write_model_dir(tmp_path, jinja="{{ '<|turn>model' }}", tokenizer_config=self._CFG)
        assert detect_chat_template_source(tmp_path) is None

    def test_detect_keeps_valid_jinja(self, tmp_path):
        from heylook_llm.providers.common.template_info import detect_chat_template_source
        _write_model_dir(tmp_path, jinja="{{ '<end_of_turn>' }}", tokenizer_config=self._CFG)
        assert detect_chat_template_source(tmp_path) == "jinja"

    def test_unknown_stop_set_not_rejected(self, tmp_path):
        # can't determine the model's stop tokens -> DON'T reject (never break on uncertainty)
        from heylook_llm.providers.common.template_info import read_template_info
        _write_model_dir(tmp_path, jinja="{{ '<|turn>model' }}", tokenizer_config={})
        info = read_template_info(tmp_path, source="jinja")
        assert "<|turn>model" in info.chat_template

    def test_eos_ids_resolve_via_tokenizer_json(self, tmp_path):
        # gemma-4 shape: tokenizer_config has NO added_tokens_decoder; the
        # generation_config eos ids (incl. the <turn|> turn terminator the
        # canonical template renders) resolve only via tokenizer.json's
        # added_tokens. Before this resolution the canonical template was
        # wrongly rejected as stopless -> template_info emptied -> thinking
        # parser + capability sniffing silently disabled.
        from heylook_llm.providers.common.template_info import read_template_info
        _write_model_dir(
            tmp_path,
            jinja="{{ '<|turn>model\\n' }}{{ '<turn|>' }}",
            tokenizer_config={"eos_token": "<eos>"},
        )
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106]})
        )
        (tmp_path / "tokenizer.json").write_text(json.dumps({
            "added_tokens": [
                {"id": 1, "content": "<eos>", "special": True},
                {"id": 106, "content": "<turn|>", "special": True},
            ],
        }))
        info = read_template_info(tmp_path, source="jinja")
        assert "<turn|>" in info.chat_template
        assert info.template_source == "jinja"
