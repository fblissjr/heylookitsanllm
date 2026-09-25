# tests/unit/test_vlm_inputs.py
"""Tests for standalone VLM input preparation.

Covers:
- Image URL extraction from ContentPart objects
- Text-only messages (no images)
- Thinking reconstruction in assistant messages
- A chat-template failure raises (no fallback render)
- Parallel image loading delegation
"""

from unittest.mock import MagicMock

import pytest


class FakeContentPart:
    """Mimics ContentPart with .type, .text, .image_url attributes."""
    def __init__(self, type, text=None, image_url=None):
        self.type = type
        self.text = text
        self.image_url = image_url


class FakeImageUrl:
    def __init__(self, url):
        self.url = url


class FakeMessage:
    """Mimics ChatMessage with .role, .content, .thinking."""
    def __init__(self, role, content, thinking=None):
        self.role = role
        self.content = content
        self.thinking = thinking


class TestPrepareVlmInputsParallel:
    """Core tests for prepare_vlm_inputs_parallel."""

    def test_thinking_reconstruction(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [
            FakeMessage("user", "question"),
            FakeMessage("assistant", "answer", thinking="my reasoning"),
            FakeMessage("user", "follow up"),
        ]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()

        captured_messages = []

        def capture_template(proc, cfg, msgs, **kwargs):
            captured_messages.extend(msgs)
            return "formatted"

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, capture_template
        )

        # The assistant message should have thinking tags prepended
        assistant_msg = captured_messages[1]
        assert "<think>" in assistant_msg["content"]
        assert "my reasoning" in assistant_msg["content"]
        assert "answer" in assistant_msg["content"]


    def test_a_template_failure_raises_and_nothing_is_rendered_in_its_place(self, mock_mlx):
        """Until v2.0.58 any template exception was swallowed and a ladder
        walked instead: the tokenizer's template with a fresh generation
        prompt, then a bare "role: content" join. A broken template then
        produced a confident answer to a prompt with no template in it -- and,
        on a continuation, silently RESTARTED the message. Continuing or not,
        the failure now reaches the caller."""
        import pytest
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        def failing_template(proc, cfg, msgs, **kwargs):
            raise ValueError("template error")

        for continuing in (False, True):
            with pytest.raises(ValueError, match="template error"):
                prepare_vlm_inputs_parallel(
                    [FakeMessage("user", "hi"), FakeMessage("assistant", "The goat is")],
                    MagicMock(), MagicMock(), MagicMock(), failing_template,
                    continue_final_message=continuing)



class TestMediaAttribution:
    """Each image must be attributed to the message it was attached to.

    The bug this pins (fixed v2.0.15): every message was flattened to a plain
    string and the images went out as a bare ``num_images=`` total, which
    attributes nothing -- so mlx-vlm fell back to dumping every image on the
    LAST USER TURN. An image attached in turn 1 was announced to the model as
    if it had arrived in the latest turn, and two images from different turns
    arrived adjacent in load order. Nothing in the suite could see it: every
    other test here hands in a MagicMock template fn and asserts on the total.
    """

    @staticmethod
    def _capture(messages, model_type="qwen3_5"):
        """Run the real function, returning what it hands the template."""
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        seen = {}

        def spy(processor, config, msgs, num_images=0, **kwargs):
            seen["messages"] = msgs
            seen["num_images"] = num_images
            return "formatted"

        batch = MagicMock()
        batch.load_images_parallel.side_effect = lambda urls: [MagicMock() for _ in urls]
        prepare_vlm_inputs_parallel(
            messages, MagicMock(), {"model_type": model_type}, batch, spy)
        return seen

    @staticmethod
    def _markers(content):
        """Image markers in one message's content, however it travelled."""
        if not isinstance(content, list):
            return 0
        return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image")

    # ContentPart-shaped OBJECTS, as every production path hands them over
    # (ChatMessage validates dicts into them); the dict-form branch these used
    # to exercise was test-only surface and went in v2.0.58.
    def _img(self, url):
        return FakeContentPart("image_url", image_url=FakeImageUrl(url))

    def _txt(self, text):
        return FakeContentPart("text", text=text)

    def test_markers_land_on_the_message_that_carried_the_image(self, mock_mlx):
        # Image on turn 1, a later user turn with none -- the shape that used
        # to move the marker to the end.
        messages = [
            FakeMessage("user", [self._img("a.png"), self._txt("what is this?")]),
            FakeMessage("assistant", "a cat"),
            FakeMessage("user", [self._txt("and now?")]),
        ]
        seen = self._capture(messages)
        assert [self._markers(m["content"]) for m in seen["messages"]] == [1, 0, 0]

    def test_two_images_stay_on_their_own_turns(self, mock_mlx):
        messages = [
            FakeMessage("user", [self._img("a.png"), self._txt("first")]),
            FakeMessage("assistant", "ok"),
            FakeMessage("user", [self._img("b.png"), self._txt("second")]),
        ]
        seen = self._capture(messages)
        assert [self._markers(m["content"]) for m in seen["messages"]] == [1, 0, 1]
        # Load order must follow message order, or the markers point at the
        # wrong pixels even when they are on the right turns.
        assert seen["num_images"] == 2

    def test_text_only_conversation_is_untouched(self, mock_mlx):
        # The blast-radius claim in content_for_template's docstring: a
        # conversation with no images must render exactly as it did before.
        messages = [FakeMessage("user", "hello"), FakeMessage("assistant", "hi")]
        seen = self._capture(messages)
        assert [m["content"] for m in seen["messages"]] == ["hello", "hi"]


class TestUpstreamAttributionAssumption:
    """The fix above works by handing mlx-vlm block-form content and trusting
    it to attribute media PER MESSAGE. That trust is the load-bearing half and
    lives in a dependency pinned to a SHA, so it gets its own check.

    Deliberately NOT using ``mock_mlx``: that fixture replaces
    ``apply_chat_template`` with a stub returning a fixed string
    (tests/helpers/mlx_mock.py), so a cross-check written under it asserts on
    the stub and is green no matter what upstream does.
    """

    def test_block_form_content_is_attributed_to_its_own_message(self):
        import pytest
        prompt_utils = pytest.importorskip(
            "mlx_vlm.prompt_utils",
            reason="real mlx-vlm not importable here")

        messages = [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "what is this?"}]},
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
        ]
        built = prompt_utils.apply_chat_template(
            None, {"model_type": "qwen3_5"}, messages,
            num_images=1, return_messages=True)

        def markers(content):
            if not isinstance(content, list):
                return 0
            return sum(1 for b in content
                       if isinstance(b, dict) and b.get("type") == "image")

        # The marker stays on turn 1. If upstream ever reverts to pooling
        # media onto the last user turn this reads [0, 0, 1] and the
        # placement fix in vlm_inputs.py is silently doing nothing.
        assert [markers(m["content"]) for m in built] == [1, 0, 0]


_IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}


def _image_on(role):
    """A user text turn, then an image-bearing turn from ``role``."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "draw a cat"}]},
        {"role": role, "content": [_IMAGE_PART, {"type": "text", "text": "here it is"}]},
    ]


class TestNonUserImageGuard:
    """An image on a non-user turn must be REFUSED on MLX, not relocated.

    mlx-vlm gates the marker on ``role == "user"`` in three places, so such an
    image does not error there -- it silently moves to the latest user turn and
    the model is told it arrived in that message. Once the editor can attach
    media to an assistant message, that is a wrong answer produced from a
    plausible-looking request, so the provider refuses and names gguf.
    """

    @pytest.mark.parametrize(
        "cases",
        [
            [(_image_on("assistant"), ["assistant"]), (_image_on("system"), ["system"])],
            [([{"role": "user", "content": [_IMAGE_PART, {"type": "text", "text": "what is this?"}]},
               {"role": "assistant", "content": "a cat"}], [])],
        ],
        ids=["non-user-image-is-detected", "user-image-is-not-flagged"],
    )
    def test_flagged_roles(self, mock_mlx, cases):
        from heylook_llm.config import ChatRequest
        from heylook_llm.providers.mlx_provider import _non_user_image_roles

        for messages, flagged in cases:
            request = ChatRequest.model_validate({"messages": messages})
            assert _non_user_image_roles(request.messages) == flagged


def test_reasoning_content_survives_mlx_vlms_message_rebuild():
    """mlx-vlm's apply_chat_template(return_messages=True) rebuilds each
    message as role + content only. heylook's wrapper must put the rest back,
    or `reasoning_content` never reaches the template on the VLM path (found
    2026-09-24: gemma-4's continued turn lost its thought and degenerated).
    Driven through the REAL mlx-vlm rebuild with a recording tokenizer."""
    import pytest

    from heylook_llm.providers.mlx_provider import vlm_apply_chat_template

    config = {"model_type": "gemma4"}
    seen = {}

    class Recorder:
        def apply_chat_template(self, messages, **_kw):
            seen["messages"] = messages
            return "rendered"

    try:
        vlm_apply_chat_template(
            Recorder(), config,
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "partial", "reasoning_content": "THOUGHT"}],
            enable_thinking=True, continue_final_message=True)
    except Exception as e:  # a missing mlx-vlm model registry entry, say
        pytest.skip(f"mlx-vlm could not rebuild gemma4 messages here: {e}")
    assert seen["messages"][-1].get("reasoning_content") == "THOUGHT"


def test_the_models_own_template_draws_the_image_markup():
    """heylook used to flatten mlx-vlm's structured content to a string with a
    bare image token before rendering, so Qwen lost the
    <|vision_start|>/<|vision_end|> its template wraps an image in (and its
    mRoPE finds images by vision_start) and gemma-4 gained a stray space.
    The template now renders the structured content itself; only a template
    that cannot take list content is flattened. Both halves, through the
    real Qwen3.5 template fixture and a string-only one."""
    from pathlib import Path

    from heylook_llm.chat_template_files import _engine_environment
    from heylook_llm.providers.mlx_provider import vlm_apply_chat_template

    env = _engine_environment()

    class Tok:
        def __init__(self, body):
            self.template = env.from_string(body)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                        bos_token="", eos_token="", **kw)

    class Proc:
        image_token = "<|image_pad|>"

        def __init__(self, body):
            self.tokenizer = Tok(body)

    class Cfg(dict):  # the family the fixture template belongs to
        def __init__(self):
            super().__init__(model_type="qwen3_5")
            self.model_type = "qwen3_5"

    qwen = (Path(__file__).resolve().parents[1] / "fixtures" / "chat_templates" / "qwen3_5.jinja").read_text()
    out = vlm_apply_chat_template(Proc(qwen), Cfg(), [{"role": "user", "content": "x"}],
                                  num_images=1, enable_thinking=False)
    assert "<|vision_start|><|image_pad|><|vision_end|>x" in out
    # ...and a message with no image renders as its plain text: mlx-vlm's list
    # form of it picked up template artifacts (a trailing space on gemma-4's
    # system turn, which changed its replies).
    text_only = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}]
    out = vlm_apply_chat_template(Proc(qwen), Cfg(), text_only, num_images=0, enable_thinking=False)
    assert out == Tok(qwen).apply_chat_template(text_only, add_generation_prompt=True, enable_thinking=False)

    string_only = ("{% for m in messages %}{{ '<' + m['role'] + '>' + m['content'] }}{% endfor %}"
                   "{% if add_generation_prompt %}<assistant>{% endif %}")
    out = vlm_apply_chat_template(Proc(string_only), Cfg(), [{"role": "user", "content": "x"}],
                                  num_images=1, enable_thinking=False)
    assert out == "<user><|image_pad|> x<assistant>"


def test_images_reach_the_model_in_the_order_they_were_sent(monkeypatch):
    # The first image finishes loading last; the list must still line up
    # with the markers (it once came back in completion order).
    import time
    from types import SimpleNamespace

    from heylook_llm.providers.common import batch_vision

    delay = {"first": 0.15, "second": 0.0, "third": 0.05}

    def slow_load(url):
        time.sleep(delay[url])
        return SimpleNamespace(tag=url, width=1, height=1, size=(1, 1))

    monkeypatch.setattr(batch_vision, "load_image", slow_load)
    got = batch_vision.BatchVisionProcessor(max_workers=3).load_images_parallel(
        ["first", "second", "third"])
    assert [g.tag for g in got] == ["first", "second", "third"]


def test_an_unreadable_image_fails_the_request_instead_of_becoming_a_red_square():
    import pytest

    from heylook_llm.providers.base import InvalidGenerationRequest
    from heylook_llm.providers.common.batch_vision import BatchVisionProcessor
    from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

    msg = FakeMessage("user", [FakeContentPart("image_url", image_url=FakeImageUrl(
        "data:image/png;base64,bm90IGFuIGltYWdl")), FakeContentPart("text", text="what is this?")])
    with pytest.raises(InvalidGenerationRequest, match="could not be read"):
        prepare_vlm_inputs_parallel([msg], MagicMock(), {"model_type": "qwen3_5"},
                                    BatchVisionProcessor(max_workers=2), MagicMock())
