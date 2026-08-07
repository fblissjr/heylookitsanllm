# tests/unit/test_thinking_capability.py
"""Thinking capability detection + enable_thinking template forwarding.

The cross-model mechanism: templates that reference ``enable_thinking``
support the toggle (Qwen3 renders <think> blocks, gemma-4 renders thought
channels); capabilities are sniffed from the model's own template so
/v1/models reports "thinking" without a manual models.toml flag, and the
VLM template path forwards the kwarg exactly like the text path.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

_GEMMA_JINJA = (
    "{{ bos_token }}{% if enable_thinking %}<|think|>\n{% endif %}"
    "{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n"
    "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}{% endif %}"
)


@pytest.mark.unit
class TestThinkingCapabilityFromTemplate:
    def _model_config(self, tmp_path):
        from heylook_llm.config import ModelConfig

        return ModelConfig(
            id="m", provider="mlx", config={"model_path": str(tmp_path)}
        )

    def test_template_toggle_reports_thinking(self, tmp_path):
        from heylook_llm.capabilities import infer_model_capabilities

        (tmp_path / "chat_template.jinja").write_text(_GEMMA_JINJA)
        caps = infer_model_capabilities(self._model_config(tmp_path))
        assert "thinking" in caps

    def test_no_toggle_no_thinking(self, tmp_path):
        from heylook_llm.capabilities import infer_model_capabilities

        (tmp_path / "chat_template.jinja").write_text(
            "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        caps = infer_model_capabilities(self._model_config(tmp_path))
        assert "thinking" not in caps

    def test_mlx_config_rejects_supports_thinking(self):
        """Claim: MLX thinking capability is DERIVED (template probe /
        enable_thinking / the explicit ModelConfig.capabilities override),
        never a hand-set flag -- a manual flag shadowing derivable truth is
        the same rot class as the dead enable_thinking cascade layer.
        supports_thinking stays GGUF-only (no pre-load template to probe).
        """
        from pydantic import ValidationError

        from heylook_llm.config import MLXModelConfig

        with pytest.raises(ValidationError):
            MLXModelConfig.model_validate(
                {"model_path": "/x", "supports_thinking": True}
            )


@pytest.mark.unit
class TestCapabilitiesReachBothSurfaces:
    """Derived capabilities have TWO readers -- /v1/models (what chat gates
    its UI on) and /v1/admin/models (what the Models page lists). Only the
    first ever ran the inference, so the admin list reported the stored
    ``capabilities`` override, which is empty on every entry that never set
    one -- i.e. all of them, and even more so after thin entries landed. The
    Models page therefore showed no capabilities at all.
    """

    def test_admin_response_reports_derived_capabilities(self, tmp_path):
        from heylook_llm.admin_api import _model_config_to_response
        from heylook_llm.config import ModelConfig

        (tmp_path / "chat_template.jinja").write_text(_GEMMA_JINJA)
        (tmp_path / "config.json").write_text('{"model_type": "gemma", "vision_config": {}}')
        mc = ModelConfig(id="m", provider="mlx", config={"model_path": str(tmp_path)})
        assert mc.capabilities == [], "fixture must not pre-set an override"

        caps = _model_config_to_response(mc, set()).capabilities
        assert "chat" in caps and "vision" in caps and "thinking" in caps, caps

    def test_explicit_override_still_wins(self, tmp_path):
        # Same short-circuit /v1/models honors: a hand-written capabilities
        # list is an override, not a hint.
        from heylook_llm.admin_api import _model_config_to_response
        from heylook_llm.config import ModelConfig

        (tmp_path / "chat_template.jinja").write_text(_GEMMA_JINJA)
        mc = ModelConfig(id="m", provider="mlx", config={"model_path": str(tmp_path)},
                         capabilities=["chat"])
        assert _model_config_to_response(mc, set()).capabilities == ["chat"]


@pytest.mark.unit
class TestThinkingFlagAgreesAcrossSurfaces:
    """PROPERTY, not an example: the flag the PROMPT was built with and the
    flag the PARSER is armed with must be equal for every request shape.

    They are two readings of one decision. A prompt built thinking-ON with a
    content-state parser misroutes the entire reasoning trace -- on a
    ``prefills_thinking`` template (Qwen3.5 pre-fills an unclosed ``<think>``)
    the model's output starts inside the block, so a content-state parser
    silently routes all of it to content. That is the v1.34.64 bug.

    One shared resolver was supposed to prevent this, but the two sides
    stopped feeding it the same INPUT: the prompt side reads the cascade
    OUTPUT (which includes the sampler layers) while the parser side read the
    RAW request (which does not). So any sampler that sets enable_thinking --
    the bundled `thinking` sampler by request, or a model's default_sampler --
    split them. Stated as a property over the cross-product because the
    divergence lives in specific COMBINATIONS, and an example-per-case test
    is exactly what missed it.
    """

    CONFIGS = [
        {"model_path": "/fake", "vision": False},
        {"model_path": "/fake", "vision": False, "enable_thinking": True},
        {"model_path": "/fake", "vision": False, "enable_thinking": False},
        {"model_path": "/fake", "vision": False, "default_sampler": "thinking"},
        {"model_path": "/fake", "vision": False, "default_sampler": "deterministic"},
    ]
    REQUESTS = [
        {},
        {"enable_thinking": True},
        {"enable_thinking": False},
        {"sampler": "thinking"},
        {"sampler": "deterministic"},
        {"sampler": "thinking", "enable_thinking": False},
        {"sampler": "deterministic", "enable_thinking": True},
    ]

    # Driven through LlamaServerProvider deliberately: `effective_thinking`
    # lives on BaseProvider, and this provider is pure stdlib (no MLX import),
    # so the property runs standalone. test_mlx_provider.py pins the MLX
    # provider's own prompt-side wrapper against the same rule -- it lives in
    # the file that already needs the batched MLX module mocks.
    def _provider(self, config):
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider

        return LlamaServerProvider("m", dict(config, model_path="/fake/model.gguf"), False)

    def test_prompt_and_parser_see_the_same_flag(self):
        from heylook_llm.config import ChatMessage, ChatRequest

        for cfg in self.CONFIGS:
            for kw in self.REQUESTS:
                provider = self._provider({k: v for k, v in cfg.items()
                                           if k not in ("model_path", "vision")})
                request = ChatRequest(
                    messages=[ChatMessage(role="user", content="hi")], **kw
                )
                # PROMPT side, read off the real thing this provider sends --
                # not a re-derivation, which is the whole failure being pinned.
                prompt_side = provider._build_payload(request)["chat_template_kwargs"]["enable_thinking"]
                # PARSER side (api.py / messages_api.py arm from this).
                parser_side = provider.effective_thinking(request)
                assert prompt_side == parser_side, (
                    f"config={cfg} request={kw}: prompt built with "
                    f"thinking={prompt_side} but parser armed with {parser_side}"
                )

    def test_a_sampler_that_turns_thinking_on_reaches_the_parser(self):
        """The specific combination that was broken, pinned on its own so a
        regression names itself instead of surfacing as one row of a matrix."""
        from heylook_llm.config import ChatMessage, ChatRequest

        msg = [ChatMessage(role="user", content="hi")]
        assert self._provider({}).effective_thinking(
            ChatRequest(messages=msg, sampler="thinking")) is True
        assert self._provider({"default_sampler": "thinking"}).effective_thinking(
            ChatRequest(messages=msg)) is True

    def test_a_gguf_entry_can_still_ask_for_thinking_by_default(self):
        """Claim: `enable_thinking` is settable on a gguf models.toml entry.

        Making unset mean OFF (v1.50.0) is right, but GGUFModelConfig is
        extra="forbid" and had no such field -- so a gguf model that used to
        inherit its template's thinking-ON default had NO way to ask for that
        back, short of `default_sampler = "thinking"`, which drags a
        presence_penalty change along with it.
        """
        from heylook_llm.config import ChatMessage, ChatRequest, GGUFModelConfig

        GGUFModelConfig(model_path="/fake/model.gguf", enable_thinking=True)  # must validate

        msg = [ChatMessage(role="user", content="hi")]
        on = self._provider({"enable_thinking": True})
        assert on.effective_thinking(ChatRequest(messages=msg)) is True
        # ...and an explicit request still wins over the model default.
        assert on.effective_thinking(
            ChatRequest(messages=msg, enable_thinking=False)) is False
        # An unset field stays off, and must not override the materialized value.
        assert self._provider({}).effective_thinking(ChatRequest(messages=msg)) is False

    def test_the_matrix_is_not_all_one_value(self):
        """Guard against a vacuous property: if every row of the cross-product
        resolved the same way, the agreement above would hold trivially."""
        from heylook_llm.config import ChatMessage, ChatRequest

        msg = [ChatMessage(role="user", content="hi")]
        seen = {
            self._provider({k: v for k, v in cfg.items()
                            if k not in ("model_path", "vision")})
                .effective_thinking(ChatRequest(messages=msg, **kw))
            for cfg in self.CONFIGS for kw in self.REQUESTS
        }
        assert seen == {True, False}, f"matrix only ever produced {seen}"


@pytest.mark.unit
class TestVlmTemplateThinkingForwarding:
    class _FakeTokenizer:
        def __init__(self):
            self.last_kwargs = None

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True, **kwargs):
            self.last_kwargs = kwargs
            return "PROMPT"

    def _run(self, **call_kwargs):
        from heylook_llm.providers import mlx_provider

        tok = self._FakeTokenizer()
        processor = SimpleNamespace(tokenizer=tok, image_token="<image>")
        messages = [{"role": "user", "content": "hi"}]
        with patch.object(
            mlx_provider, "mlx_vlm_apply_chat_template",
            side_effect=lambda p, c, m, num_images, return_messages: m,
        ):
            out = mlx_provider.vlm_apply_chat_template(
                processor, {}, messages, num_images=0, **call_kwargs
            )
        assert out == "PROMPT"
        return tok.last_kwargs

    def test_bool_is_forwarded(self):
        assert self._run(enable_thinking=False) == {"enable_thinking": False}
        assert self._run(enable_thinking=True) == {"enable_thinking": True}

    def test_none_omits_the_kwarg(self):
        assert self._run(enable_thinking=None) == {}
