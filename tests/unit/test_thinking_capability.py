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
        from heylook_llm.api import _infer_model_capabilities

        (tmp_path / "chat_template.jinja").write_text(_GEMMA_JINJA)
        caps = _infer_model_capabilities(self._model_config(tmp_path))
        assert "thinking" in caps

    def test_no_toggle_no_thinking(self, tmp_path):
        from heylook_llm.api import _infer_model_capabilities

        (tmp_path / "chat_template.jinja").write_text(
            "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        caps = _infer_model_capabilities(self._model_config(tmp_path))
        assert "thinking" not in caps


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
