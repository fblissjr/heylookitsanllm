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
import pytest

from heylook_llm.providers.mlx_provider import vlm_apply_chat_template


class FakeProcessor:
    """Records what the template was called with."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        return "PROMPT"


class FakeConfig(dict):
    """mlx_vlm's prompt_utils does `config["model_type"]`; heylook's own code
    reads it as an attribute. Support both."""

    def __init__(self):
        super().__init__(model_type="qwen2_vl")
        self.model_type = "qwen2_vl"


def _msgs():
    return [{"role": "user", "content": "hi"}]


@pytest.mark.unit
class TestVlmTemplateKwargs:
    """The VLM template path forwards the kwargs exactly like the text path:
    a bool enable_thinking is forwarded, None omits the kwarg (template
    default applies), and depth rides under the template's own variable
    (absent depth sends nothing extra). Exact kwargs, so an extra key fails."""

    @pytest.mark.parametrize(
        "enable_thinking, depth, sent",
        [
            (False, None, {"enable_thinking": False}),
            (None, None, {}),
            (False, {"reasoning_strength": "low"},
             {"enable_thinking": False, "reasoning_strength": "low"}),
            (True, None, {"enable_thinking": True}),
        ],
        ids=["bool-forwarded", "none-omits-the-kwarg",
             "depth-under-the-templates-own-variable", "absent-depth-sends-no-kwarg"],
    )
    def test_template_kwargs_sent(self, enable_thinking, depth, sent):
        p = FakeProcessor()
        vlm_apply_chat_template(p, FakeConfig(), _msgs(), num_images=0,
                                enable_thinking=enable_thinking, depth=depth)
        assert p.calls[-1] == {"tokenize": False, "add_generation_prompt": True, **sent}

    def test_the_variable_comes_from_detection(self):
        from types import SimpleNamespace

        from heylook_llm.providers.mlx_provider import _depth_kwargs

        info = SimpleNamespace(chat_template="Reasoning strength: {{ reasoning_strength | default('high') }}")
        assert _depth_kwargs({"reasoning_effort": "low"}, info) == {"reasoning_strength": "low"}
        assert _depth_kwargs({}, info) is None


@pytest.mark.unit
def test_depth_reaches_the_template_through_the_vision_path():
    """prepare_vlm_inputs_parallel is the ONLY path an image-bearing request
    takes; the parameter existing on vlm_apply_chat_template is not enough."""
    import inspect

    from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

    seen = {}

    def fake_template(processor, config, messages, num_images=None,
                      enable_thinking=None, depth=None, **kw):
        seen["depth"] = depth
        return "PROMPT"

    class Msg:
        role = "user"
        content = "hi"

    prepare_vlm_inputs_parallel([Msg()], FakeProcessor(), FakeConfig(), None, fake_template,
                                enable_thinking=True, depth={"reasoning_effort": "low"})
    assert seen == {"depth": {"reasoning_effort": "low"}}
    # guard the guard: dropped, the call above would pass it as **kw
    assert "depth" in inspect.signature(prepare_vlm_inputs_parallel).parameters
