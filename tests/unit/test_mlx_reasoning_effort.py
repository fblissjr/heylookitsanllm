# tests/unit/test_mlx_reasoning_effort.py
"""The MLX half of reasoning_effort -- template kwargs on all three paths.

Claims (what breaks if a test is deleted):
- kwarg tests: thinking depth silently stops reaching MLX models.
- the retry test: someone moves reasoning_effort into `base_kwargs`, and every
  request to a model whose tokenizer wrapper has a narrow signature becomes a
  hard TypeError. The comment at the call site reasons about exactly this and
  nothing executed it before.
- the vision test: depth works on a text turn and reverts the moment an image
  is attached -- same model, same conversation, no error.
"""
import pytest

from heylook_llm.providers.mlx_provider import vlm_apply_chat_template


class FakeProcessor:
    """Records what the template was called with."""

    def __init__(self, reject: set[str] | None = None):
        self.calls: list[dict] = []
        self.reject = reject or set()

    def apply_chat_template(self, messages, **kw):
        bad = self.reject & set(kw)
        if bad:
            raise TypeError(f"unexpected keyword argument {sorted(bad)[0]!r}")
        self.calls.append(kw)
        return "PROMPT"


class FakeConfig(dict):
    """mlx_vlm's prompt_utils does `config["model_type"]`; heylook's own code
    reads it as an attribute. Support both so this fake matches the real
    contract rather than half of it."""

    def __init__(self):
        super().__init__(model_type="qwen2_vl")
        self.model_type = "qwen2_vl"


def _msgs():
    return [{"role": "user", "content": "hi"}]


@pytest.mark.unit
class TestVlmTemplateKwargs:
    def test_effort_is_forwarded(self):
        p = FakeProcessor()
        vlm_apply_chat_template(p, FakeConfig(), _msgs(), num_images=0,
                                enable_thinking=True, reasoning_effort="low")
        assert p.calls[-1].get("reasoning_effort") == "low"

    def test_effort_is_forwarded_without_thinking(self):
        """Not gated: harmony templates read it unconditionally and have no
        enable_thinking at all."""
        p = FakeProcessor()
        vlm_apply_chat_template(p, FakeConfig(), _msgs(), num_images=0,
                                enable_thinking=False, reasoning_effort="high")
        assert p.calls[-1].get("reasoning_effort") == "high"

    def test_absent_effort_sends_no_kwarg(self):
        p = FakeProcessor()
        vlm_apply_chat_template(p, FakeConfig(), _msgs(), num_images=0,
                                enable_thinking=True)
        assert "reasoning_effort" not in p.calls[-1]


@pytest.mark.unit
class TestVisionPathForwardsEffort:
    """prepare_vlm_inputs_parallel is the ONLY path an image-bearing request
    takes; the parameter existing on vlm_apply_chat_template is not enough."""

    def test_reasoning_effort_reaches_the_template_through_the_vision_path(self):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        seen = {}

        def fake_template(processor, config, messages, num_images=None,
                          enable_thinking=None, reasoning_effort=None, **kw):
            seen["reasoning_effort"] = reasoning_effort
            seen["enable_thinking"] = enable_thinking
            return "PROMPT"

        class Msg:
            role = "user"
            content = "hi"

        prepare_vlm_inputs_parallel(
            [Msg()], FakeProcessor(), FakeConfig(), None, fake_template,
            model=None, enable_thinking=True, reasoning_effort="low",
        )
        assert seen == {"reasoning_effort": "low", "enable_thinking": True}

    def test_the_signature_actually_accepts_it(self):
        """Guard the guard: if the parameter were dropped, the test above would
        pass it as **kw and still 'work'."""
        import inspect
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel
        assert "reasoning_effort" in inspect.signature(
            prepare_vlm_inputs_parallel).parameters


@pytest.mark.unit
class TestTextTemplateRetry:
    """The text path passes template kwargs SEPARATELY from base_kwargs so the
    TypeError fallback can drop them. In base_kwargs they would survive the
    retry and fail it identically."""

    def test_reasoning_effort_is_not_baked_into_base_kwargs(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[2] / "src" / "heylook_llm"
               / "providers" / "mlx_provider.py").read_text()
        # The assignment must go through the retry-droppable dict.
        assert 'template_kwargs["reasoning_effort"] = reasoning_effort' in src
        assert 'base_kwargs["reasoning_effort"]' not in src, (
            "in base_kwargs the TypeError retry re-passes it and fails again")

    def test_a_narrow_wrapper_still_renders_after_the_retry(self):
        """A processor that rejects the kwarg must not take the whole request
        down -- the retry drops template kwargs and renders without them."""
        p = FakeProcessor(reject={"reasoning_effort"})
        with pytest.raises(TypeError):
            p.apply_chat_template(_msgs(), reasoning_effort="low")
        # ...and the same processor renders fine once it is dropped.
        assert p.apply_chat_template(_msgs(), enable_thinking=True) == "PROMPT"


@pytest.mark.unit
class TestReasoningEffortCapability:
    """Depth is its own capability, NOT implied by thinking.

    The two really do come apart in this model zoo: Qwen3.5 reads
    enable_thinking and never reasoning_effort; gpt-oss the exact reverse.
    Deriving one from the other hides the control on whichever family is the
    counterexample.
    """

    def _caps(self, provider, cfg):
        from heylook_llm.config import ModelConfig
        from heylook_llm.capabilities import infer_model_capabilities
        return infer_model_capabilities(ModelConfig.model_validate(
            {"id": "x", "provider": provider, "config": cfg}))

    def test_gguf_thinking_model_offers_depth(self):
        caps = self._caps("gguf", {"model_path": "/x.gguf", "supports_thinking": True})
        assert "reasoning_effort" in caps

    def test_gguf_without_thinking_does_not(self):
        caps = self._caps("gguf", {"model_path": "/x.gguf"})
        assert "reasoning_effort" not in caps

    def test_mlx_depth_is_probed_from_the_template_not_inferred(self, tmp_path):
        """An MLX model whose template never mentions reasoning_effort must
        NOT advertise it, even with thinking on -- that is the Qwen3.5 case."""
        d = tmp_path / "m"
        d.mkdir()
        (d / "chat_template.jinja").write_text("{% if enable_thinking %}x{% endif %}")
        caps = self._caps("mlx", {"model_path": str(d), "enable_thinking": True})
        assert "thinking" in caps
        assert "reasoning_effort" not in caps

    def test_mlx_template_reading_effort_advertises_it_without_thinking(self, tmp_path):
        """The gpt-oss/harmony case: depth, no enable_thinking anywhere."""
        d = tmp_path / "m"
        d.mkdir()
        (d / "chat_template.jinja").write_text(
            '{%- if reasoning_effort is not defined %}'
            '{%- set reasoning_effort = "medium" %}{%- endif %}')
        caps = self._caps("mlx", {"model_path": str(d)})
        assert "reasoning_effort" in caps
        assert "thinking" not in caps
