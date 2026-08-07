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
