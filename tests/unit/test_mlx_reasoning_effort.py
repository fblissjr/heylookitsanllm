# tests/unit/test_mlx_reasoning_effort.py
"""The MLX half of thinking depth -- template kwargs on all three paths.

Claims (what breaks if a test is deleted):
- kwarg tests: thinking depth silently stops reaching MLX models, or reaches
  them under a name the template does not read (plan W2: the variable is the
  template's own, e.g. Muse's reasoning_strength).
- the retry rows (test_mlx_provider.py TestContinuationTemplate): someone
  moves the depth kwarg into `base_kwargs`, and every request to a model whose
  tokenizer wrapper has a narrow signature becomes a hard TypeError.
- the vision test: depth works on a text turn and reverts the moment an image
  is attached -- same model, same conversation, no error.
- the capability rows (test_thinking_capability.py
  TestThinkingCapabilityFromTemplate): depth is offered where the template
  has a depth variable, and not inferred from thinking.
"""
from pathlib import Path

import pytest

from heylook_llm.providers.mlx_provider import vlm_apply_chat_template


_TEMPLATES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_templates"


def _processor(template_name):
    """A processor whose tokenizer renders a REAL chat template fixture
    through the engines' jinja environment, so what is asserted is the
    prompt the model would see, not the kwargs a stub recorded."""
    from heylook_llm.chat_template_files import _engine_environment

    template = _engine_environment().from_string((_TEMPLATES / template_name).read_text())

    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            return template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                   bos_token="", eos_token="", **kw)

    class Proc:
        image_token = "<|image_pad|>"
        tokenizer = Tok()

    return Proc()


class FakeConfig(dict):
    """mlx_vlm's prompt_utils does `config["model_type"]`; heylook's own code
    reads it as an attribute. Support both."""

    def __init__(self):
        super().__init__(model_type="qwen3_5")
        self.model_type = "qwen3_5"


def _msgs():
    return [{"role": "user", "content": "hi"}]


_THINK_OPEN = "<|im_start|>assistant\n<think>\n"
_THINK_CLOSED = "<think>\n\n</think>\n\n"


@pytest.mark.unit
class TestVlmTemplateKwargs:
    """The VLM template path reaches the template like the text path: a bool
    enable_thinking is forwarded, None omits it (the template's own default
    applies -- Qwen3's default is ON, so omitting differs from False), and
    depth rides under the template's own variable (absent depth leaves the
    template's default). Asserted on the rendered prompt."""

    @pytest.mark.parametrize(
        "template, enable_thinking, depth, check",
        [
            ("qwen3_8_official.jinja", False, None, lambda out: out.endswith(_THINK_CLOSED)),
            ("qwen3_8_official.jinja", None, None, lambda out: out.endswith(_THINK_OPEN)),
            ("muse_glimmer.jinja", False, {"reasoning_strength": "low"},
             lambda out: "Reasoning strength: low." in out),
            ("muse_glimmer.jinja", False, None,
             lambda out: "Reasoning strength: high." in out),
        ],
        ids=["bool-forwarded", "none-omits-the-kwarg",
             "depth-under-the-templates-own-variable",
             "absent-depth-sends-no-kwarg"],
    )
    def test_the_rendered_prompt(self, template, enable_thinking, depth, check):
        out = vlm_apply_chat_template(_processor(template), FakeConfig(), _msgs(), num_images=0,
                                      enable_thinking=enable_thinking, depth=depth)
        assert check(out), out[-300:]

    def test_the_variable_comes_from_detection(self):
        from types import SimpleNamespace

        from heylook_llm.providers.mlx_provider import _depth_kwargs

        info = SimpleNamespace(chat_template="Reasoning strength: {{ reasoning_strength | default('high') }}")
        assert _depth_kwargs({"reasoning_effort": "low"}, info) == {"reasoning_strength": "low"}
        assert _depth_kwargs({}, info) is None


@pytest.mark.unit
def test_depth_reaches_the_template_through_the_vision_path():
    """prepare_vlm_inputs_parallel is the ONLY path an image-bearing request
    takes; the parameter existing on vlm_apply_chat_template is not enough.
    A real image, the real loader and the real template: the depth asked for
    is in the prompt the vision path builds."""
    import base64
    import io

    from PIL import Image

    from heylook_llm.providers.common.batch_vision import BatchVisionProcessor
    from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

    png = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(png, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()

    class Part:
        def __init__(self, type, text=None, image_url=None):
            self.type, self.text, self.image_url = type, text, image_url

    class Msg:
        role = "user"
        content = [Part("image_url", image_url=type("U", (), {"url": url})()),
                   Part("text", text="what is this?")]

    images, prompt, has_images, _ = prepare_vlm_inputs_parallel(
        [Msg()], _processor("muse_glimmer.jinja"), FakeConfig(),
        BatchVisionProcessor(max_workers=1), vlm_apply_chat_template,
        enable_thinking=True, depth={"reasoning_strength": "low"})
    assert has_images and len(images) == 1
    assert "Reasoning strength: low." in prompt


@pytest.mark.unit
def test_vision_strategy_derives_system_prefix_with_depth(monkeypatch, caplog):
    """VLMVisionStrategy._system_prefixes must call vlm_apply_chat_template with
    `depth=` (not `reasoning_effort=`), otherwise _system_prefixes silently caches
    None and every vision request with a system prompt misses its system boundary."""
    import base64
    import io
    import logging
    from types import SimpleNamespace

    import mlx.core as mx
    from PIL import Image

    from heylook_llm.config import ChatRequest
    from heylook_llm.providers import mlx_provider
    from heylook_llm.providers.common import vlm_engine
    from heylook_llm.providers.common.template_info import ModelTemplateInfo

    png = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(png, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()

    template_body = (_TEMPLATES / "muse_glimmer.jinja").read_text()
    proc = _processor("muse_glimmer.jinja")
    cfg = FakeConfig()
    model = SimpleNamespace(config=cfg)
    info = ModelTemplateInfo(chat_template=template_body)

    def fake_prepare_inputs(_processor_arg, images=None, prompts="", **_kwargs):
        ids = mx.array([[ord(ch) for ch in prompts]])
        out = {"input_ids": ids}
        if images:
            out["pixel_values"] = mx.zeros((len(images), 1, 1, 3))
        return out

    captured = {}

    def fake_vlm_generate(**kwargs):
        captured["shared_prefixes"] = kwargs.get("shared_prefixes")
        yield SimpleNamespace(text="ok", token=1, finish_reason="stop")

    monkeypatch.setattr(mlx_provider, "vlm_prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr(vlm_engine, "generate", fake_vlm_generate)

    strategy = mlx_provider.VLMVisionStrategy(template_info=info, model_id="test-vlm")
    req = ChatRequest.model_validate({
        "model": "test-vlm",
        "max_tokens": 8,
        "reasoning_effort": "low",
        "messages": [
            {"role": "system", "content": "Be brief and careful."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "describe this"},
            ]},
        ],
    })
    eff = {
        "max_tokens": 8,
        "temperature": 0.0,
        "enable_thinking": True,
        "reasoning_effort": "low",
    }

    with caplog.at_level(logging.INFO):
        chunks = list(strategy.generate(req, eff, model, proc))

    assert [c.text for c in chunks] == ["ok"]
    assert captured["shared_prefixes"] and len(captured["shared_prefixes"][0]) > 0
    prefix_str = "".join(chr(t) for t in captured["shared_prefixes"][0])
    assert "Be brief and careful." in prefix_str
    assert "Reasoning strength: low." in prefix_str
    assert not any("System-prompt checkpoint not derived" in r.message for r in caplog.records)
