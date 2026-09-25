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

import pytest


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

_HARMONY_FILES = {
    "chat_template.jinja": _HARMONY_JINJA,
    "tokenizer_config.json": _HARMONY_TOKENIZER_CONFIG,
}

_THINK_JINJA = (
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)

_GEMMA_JINJA = (
    "{{ bos_token }}{% if enable_thinking %}<|think|>\n{% endif %}"
    "{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n"
    "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}{% endif %}"
)

_QWEN35_STYLE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\\n\\n</think>\\n\\n' }}{% else %}{{ '<think>\\n' }}{% endif %}{% endif %}"
)
_QWEN3_CLASSIC = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
)

# gemma-like: eos_token_id resolves via added_tokens_decoder to <eos> + <end_of_turn>
_STOP_CFG = {
    "eos_token": "<eos>",
    "eos_token_id": [1, 106],
    "added_tokens_decoder": {
        "1": {"content": "<eos>", "special": True},
        "106": {"content": "<end_of_turn>", "special": True},
    },
}


def _write_model_dir(tmp_path, *, jinja=None, tokenizer_config=None):
    if jinja is not None:
        (tmp_path / "chat_template.jinja").write_text(jinja)
    if tokenizer_config is not None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return tmp_path


def _read(tmp_path, files, source=None):
    """Write ``files`` (name -> str written raw, or dict written as JSON) into
    the model dir and read it back. ``{dir}`` in ``source`` is the model dir."""
    from heylook_llm.providers.common.template_info import read_template_info

    for name, body in files.items():
        (tmp_path / name).write_text(body if isinstance(body, str) else json.dumps(body))
    if source is not None:
        source = source.format(dir=tmp_path)
    return read_template_info(tmp_path, source=source)


class TestReadTemplateInfoSpecials:
    """``tokenizer.json``'s ``added_tokens`` array is the authoritative source
    for fast tokenizers. Some models don't populate ``tokenizer_config.json``
    ``added_tokens_decoder`` at all, and some split the set between the two
    files, so template_info reads both and unions them. Only special:true
    entries count."""

    @pytest.mark.parametrize(
        "files, present, absent",
        [
            ({"tokenizer.json": {"added_tokens": [
                {"id": 0, "content": "<pad>", "special": True},
                {"id": 1, "content": "<eos>", "special": True},
                {"id": 100, "content": "<|channel>", "special": True},
                {"id": 101, "content": "<channel|>", "special": True},
                {"id": 200, "content": "regular_token", "special": False},
            ]}},
             {"<pad>", "<|channel>", "<channel|>"}, {"regular_token"}),
            ({"tokenizer_config.json": {
                "added_tokens_decoder": {"1": {"content": "<|from_config|>", "special": True}},
                "chat_template": "{{ '' }}",
              },
              "tokenizer.json": {"added_tokens": [
                  {"id": 2, "content": "<|from_json|>", "special": True}]}},
             {"<|from_config|>", "<|from_json|>"}, set()),
            (_HARMONY_FILES,
             {"<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>"},
             {"<|normal_token|>"}),
        ],
        ids=["tokenizer-json", "union-of-both-files", "added-tokens-decoder"],
    )
    def test_specials_come_from_the_tokenizer_files(self, tmp_path, files, present, absent):
        specials = _read(tmp_path, files).special_tokens
        assert present <= specials
        assert not (absent & specials)


class TestReadTemplateInfoFormat:
    @pytest.mark.parametrize(
        "files, flags, specials",
        [
            (_HARMONY_FILES, {"has_harmony_structure": True}, set()),
            ({"chat_template.jinja": _THINK_JINJA, "tokenizer_config.json": {
                "added_tokens_decoder": {
                    "151667": {"content": "<think>", "special": True},
                    "151668": {"content": "</think>", "special": True},
                    "151643": {"content": "<|im_start|>", "special": True},
                },
                "chat_template": _THINK_JINJA,
              }},
             {"has_thinking_markers": True, "has_harmony_structure": False},
             {"<think>", "</think>"}),
            ({"chat_template.jinja": _GEMMA_JINJA},
             {"has_gemma_channel_structure": True, "has_harmony_structure": False,
              "has_thinking_markers": False}, set()),
            (_HARMONY_FILES, {"has_gemma_channel_structure": False}, set()),
        ],
        ids=["harmony", "thinking-markers", "gemma-channels", "harmony-is-not-gemma"],
    )
    def test_format_flags_are_read_from_the_template(self, tmp_path, files, flags, specials):
        info = _read(tmp_path, files)
        for attr, value in flags.items():
            assert getattr(info, attr) is value, attr
        assert specials <= info.special_tokens

    @pytest.mark.parametrize(
        "jinja, expected",
        [
            (_GEMMA_JINJA, True),
            ("{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}", False),
        ],
        ids=["references-enable-thinking", "no-reference-no-toggle"],
    )
    def test_enable_thinking_toggle(self, tmp_path, jinja, expected):
        info = _read(tmp_path, {"chat_template.jinja": jinja})
        assert info.supports_enable_thinking is expected

    @pytest.mark.parametrize(
        "jinja, expected",
        [(_QWEN35_STYLE, True), (_QWEN3_CLASSIC, False)],
        ids=["qwen35-prefill", "classic-empty-block-is-not-prefill"],
    )
    def test_prefills_thinking(self, tmp_path, jinja, expected):
        info = _read(tmp_path, {"chat_template.jinja": jinja})
        assert info.has_thinking_markers is True
        assert info.prefills_thinking is expected


class TestTemplatePrecedence:
    """Auto order: chat_template.jinja > the tokenizer_config.json embedded
    template > chat_template.json > nothing ("auto", empty).

    chat_template.json: some VLM conversions ship the template only there
    (the processor-side convention ``{"chat_template": "..."}``). The
    tokenizer never sees that file, so template_info reads it as the last
    auto fallback, or a chat_template.json-only model looks template-less to
    us while the processor knows better.

    An explicit source forces its file. 'chat_template_json' appears as a
    resolved-source label in load logs, so it must also be an accepted
    explicit value (otherwise configuring what the log reports warns 'not
    recognized' and still force-installs the auto pick); when that file is
    missing, the auto order applies."""

    @pytest.mark.parametrize(
        "files, source, expected",
        [
            (_HARMONY_FILES, None,
             {"chat_template": _HARMONY_JINJA, "template_source": "jinja"}),
            ({"tokenizer_config.json": {
                "added_tokens_decoder": {"1": {"content": "<|eos|>", "special": True}},
                "chat_template": "{{ 'embedded template body' }}"}},
             None,
             {"chat_template": "{{ 'embedded template body' }}",
              "template_source": "tokenizer_config"}),
            ({}, None,
             {"chat_template": "", "special_tokens": frozenset(),
              "has_harmony_structure": False, "has_thinking_markers": False,
              "template_source": "auto"}),
            ({"chat_template.jinja": "{{ 'forced jinja' }}",
              "tokenizer_config.json": {"added_tokens_decoder": {},
                                        "chat_template": "{{ 'ignored embedded' }}"}},
             "jinja", {"chat_template": "{{ 'forced jinja' }}"}),
            ({"chat_template.jinja": "{{ 'ignored jinja' }}",
              "tokenizer_config.json": {"added_tokens_decoder": {},
                                        "chat_template": "{{ 'forced embedded' }}"}},
             "tokenizer_config", {"chat_template": "{{ 'forced embedded' }}"}),
            ({"my_template.jinja": "{{ 'custom from path' }}",
              "chat_template.jinja": "{{ 'dir jinja' }}",
              "tokenizer_config.json": {"added_tokens_decoder": {}, "chat_template": "x"}},
             "{dir}/my_template.jinja", {"chat_template": "{{ 'custom from path' }}"}),
            ({"chat_template.json": {"chat_template": "{{ 'from chat_template.json' }}"}},
             None,
             {"chat_template": "{{ 'from chat_template.json' }}",
              "template_source": "chat_template_json"}),
            ({"tokenizer_config.json": {"added_tokens_decoder": {},
                                        "chat_template": "{{ 'embedded' }}"},
              "chat_template.json": {"chat_template": "{{ 'json' }}"}},
             None, {"chat_template": "{{ 'embedded' }}", "template_source": "tokenizer_config"}),
            ({"chat_template.jinja": "{{ 'jinja' }}",
              "chat_template.json": {"chat_template": "{{ 'json' }}"}},
             None, {"chat_template": "{{ 'jinja' }}", "template_source": "jinja"}),
            ({"chat_template.jinja": "{{ 'jinja' }}",
              "chat_template.json": {"chat_template": "{{ 'forced json' }}"}},
             "chat_template_json",
             {"chat_template": "{{ 'forced json' }}", "template_source": "chat_template_json"}),
            ({"chat_template.jinja": "{{ 'jinja' }}"}, "chat_template_json",
             {"chat_template": "{{ 'jinja' }}", "template_source": "jinja"}),
        ],
        ids=[
            "jinja-when-present", "embedded-when-jinja-missing", "empty-when-nothing",
            "source-jinja-forces-jinja", "source-tokenizer-config-forces-embedded",
            "source-absolute-path", "auto-falls-back-to-chat-template-json",
            "embedded-wins-over-chat-template-json", "jinja-wins-over-chat-template-json",
            "source-chat-template-json-forces-it", "source-chat-template-json-missing-is-auto",
        ],
    )
    def test_which_template_is_read(self, tmp_path, files, source, expected):
        info = _read(tmp_path, files, source)
        for attr, value in expected.items():
            assert getattr(info, attr) == value, attr

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("tokenizer_config.json", {"special_tokens": frozenset()}),
            ("chat_template.json", {"chat_template": ""}),
        ],
        ids=["tokenizer-config", "chat-template-json"],
    )
    def test_a_malformed_file_does_not_raise(self, tmp_path, name, expected):
        info = _read(tmp_path, {name: "{{{ not json"})
        for attr, value in expected.items():
            assert getattr(info, attr) == value


class _FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


_NO_TOKENIZER = object()


class TestInstallChatTemplate:
    """``install_chat_template(tokenizer, info, force=...)`` is the single
    place the resolved template gets attached to a live tokenizer.

    - force=True (explicit ``chat_template_source``): always overwrite.
    - force=False (auto): only fill in a MISSING tokenizer template --
      covers chat_template.json-only models where AutoTokenizer loads
      nothing, without stomping on what transformers loaded natively.
    - an empty resolved template installs nothing; a None tokenizer is safe.
    """

    @pytest.mark.parametrize(
        "native, resolved, force, installed, final",
        [
            ("{{ 'native' }}", "{{ 'resolved' }}", True, True, "{{ 'resolved' }}"),
            (None, "{{ 'resolved' }}", False, True, "{{ 'resolved' }}"),
            ("{{ 'native' }}", "{{ 'resolved' }}", False, False, "{{ 'native' }}"),
            (None, "", True, False, None),
            (_NO_TOKENIZER, "{{ 'resolved' }}", True, False, None),
        ],
        ids=[
            "force-overwrites-existing", "auto-fills-missing", "auto-preserves-native",
            "noop-when-no-resolved-template", "none-tokenizer-is-safe",
        ],
    )
    def test_install(self, native, resolved, force, installed, final):
        from heylook_llm.providers.common.template_info import (
            ModelTemplateInfo, install_chat_template)

        info = ModelTemplateInfo(chat_template=resolved)
        if native is _NO_TOKENIZER:
            assert install_chat_template(None, info, force=force) is installed
            return
        tok = _FakeTokenizer(chat_template=native)
        assert install_chat_template(tok, info, force=force) is installed
        assert tok.chat_template == final


class TestIsExplicitSource:
    """force-install must engage only for a genuinely explicit source --
    the documented value \"auto\" is truthy but means the fill-only path."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            ((None, ""), False),
            (("auto", " AUTO "), False),
            (("jinja", "tokenizer_config", "chat_template_json", "/abs/path/custom.jinja"),
             True),
        ],
        ids=["none-and-empty", "auto", "named-sources-and-paths"],
    )
    def test_is_explicit_source(self, values, expected):
        from heylook_llm.providers.common.template_info import is_explicit_source

        for value in values:
            assert is_explicit_source(value) is expected, value


class TestMissingTemplateError:
    """``missing_template_error(tokenizer, model_id)`` decides 'the model
    truly has no chat template' from TOKENIZER STATE, not from matching
    transformers' error prose (which is version-fragile). The error names
    the model when there is one and is generic otherwise."""

    @pytest.mark.parametrize(
        "template, model_id, message_has",
        [
            (None, "my-model", ("my-model", "chat_template")),
            ("{{ x }}", "m", None),
            (None, None, ("chat_template",)),
        ],
        ids=["actionable-error-when-no-template", "none-when-template-present",
             "generic-without-model-id"],
    )
    def test_missing_template_error(self, template, model_id, message_has):
        from heylook_llm.providers.common.template_info import missing_template_error

        err = missing_template_error(_FakeTokenizer(chat_template=template), model_id)
        if message_has is None:
            assert err is None
            return
        assert isinstance(err, ValueError)
        for text in message_has:
            assert text in str(err)


class TestStopTokenValidation:
    """A stop-less chat template (renders none of the model's OWN stop tokens)
    is rejected + self-heals, so a broken/corrupted jinja can't cause runaway
    generation. The stop set is read from the model's config, never hardcoded.

    Rows (all live-found): broken jinja with no valid fallback is NOT
    installed; a valid jinja is kept; a broken jinja self-heals to a valid
    embedded template; an undeterminable stop set is never rejected (never
    break on uncertainty); gemma-4's shape, where tokenizer_config has NO
    added_tokens_decoder and the generation_config eos ids (incl. the <turn|>
    terminator the canonical template renders) resolve only via
    tokenizer.json's added_tokens. Before that resolution the canonical
    template was wrongly rejected as stopless -> template_info emptied ->
    thinking parser + capability sniffing silently disabled."""

    @pytest.mark.parametrize(
        "files, in_template, expected",
        [
            ({"chat_template.jinja": "{{ '<|turn>model\\n' }}", "tokenizer_config.json": _STOP_CFG},
             None, {"chat_template": "", "template_source": "none(stopless)"}),
            ({"chat_template.jinja": "{{ '<start_of_turn>model\\n<end_of_turn>' }}",
              "tokenizer_config.json": _STOP_CFG},
             "<end_of_turn>", {}),
            ({"chat_template.jinja": "{{ '<|turn>model' }}",
              "tokenizer_config.json": dict(_STOP_CFG, chat_template="{{ '<end_of_turn>' }}")},
             "<end_of_turn>", {"template_source": "tokenizer_config"}),
            ({"chat_template.jinja": "{{ '<|turn>model' }}", "tokenizer_config.json": {}},
             "<|turn>model", {}),
            ({"chat_template.jinja": "{{ '<|turn>model\\n' }}{{ '<turn|>' }}",
              "tokenizer_config.json": {"eos_token": "<eos>"},
              "generation_config.json": {"eos_token_id": [1, 106]},
              "tokenizer.json": {"added_tokens": [
                  {"id": 1, "content": "<eos>", "special": True},
                  {"id": 106, "content": "<turn|>", "special": True},
              ]}},
             "<turn|>", {"template_source": "jinja"}),
        ],
        ids=[
            "broken-jinja-rejected-no-valid-fallback", "valid-jinja-kept",
            "self-heals-to-embedded", "unknown-stop-set-not-rejected",
            "eos-ids-resolve-via-tokenizer-json",
        ],
    )
    def test_stop_validation(self, tmp_path, files, in_template, expected):
        info = _read(tmp_path, files, "jinja")
        if in_template is not None:
            assert in_template in info.chat_template
        for attr, value in expected.items():
            assert getattr(info, attr) == value, attr


class TestReadsReasoningContent:
    """v1.79.63: whether the template reads `reasoning_content` decides how a
    history message's thinking is handed over (see vlm_inputs.thinking_for_template)."""

    def test_probe_on_and_off(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info
        reads = ("{% for m in messages %}{% if m.reasoning_content %}<think>{{ m.reasoning_content }}"
                 "</think>{% endif %}{{ m.content }}{% endfor %}")
        _write_model_dir(tmp_path, jinja=reads, tokenizer_config=_HARMONY_TOKENIZER_CONFIG)
        assert read_template_info(tmp_path, source=None).reads_reasoning_content is True
        other = tmp_path / "other"; other.mkdir()
        _write_model_dir(other, jinja="{% for m in messages %}{{ m.content }}{% endfor %}",
                         tokenizer_config=_HARMONY_TOKENIZER_CONFIG)
        assert read_template_info(other, source=None).reads_reasoning_content is False


def test_thinking_budget_markers_are_offered_only_where_one_token_closes_the_block():
    """Plan W7: the MLX budget forces a newline and ONE close token. Harmony
    leaves its analysis channel with a multi-token sequence, so it gets none;
    the capability report and the provider read this one answer."""
    from heylook_llm.providers.common.template_info import (
        ModelTemplateInfo, thinking_budget_markers)

    assert thinking_budget_markers(ModelTemplateInfo(has_thinking_markers=True)) == ("<think>", "</think>")
    assert thinking_budget_markers(
        ModelTemplateInfo(has_gemma_channel_structure=True)) == ("<|channel>", "<channel|>")
    assert thinking_budget_markers(
        ModelTemplateInfo(has_harmony_structure=True, has_thinking_markers=True)) is None
    assert thinking_budget_markers(ModelTemplateInfo()) is None
    assert thinking_budget_markers(None) is None
