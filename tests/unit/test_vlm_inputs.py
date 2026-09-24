# tests/unit/test_vlm_inputs.py
"""Tests for standalone VLM input preparation.

Covers:
- Image URL extraction from ContentPart objects
- Text-only messages (no images)
- Thinking reconstruction in assistant messages
- A chat-template failure raises (no fallback render)
- Parallel image loading delegation
"""

from unittest.mock import MagicMock, patch


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

    def test_text_only_messages(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", "Hello world")]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()
        mock_template_fn = MagicMock(return_value="formatted")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, mock_template_fn
        )

        assert images == []
        assert has_images is False
        assert prompt == "formatted"
        mock_batch.load_images_parallel.assert_not_called()

    def test_image_url_extraction_object_format(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", [
            FakeContentPart("text", text="describe this"),
            FakeContentPart("image_url", image_url=FakeImageUrl("http://example.com/img.png")),
        ])]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()
        mock_batch.load_images_parallel.return_value = [MagicMock()]
        mock_template_fn = MagicMock(return_value="formatted with image")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, mock_template_fn
        )

        assert has_images is True
        assert len(images) == 1
        mock_batch.load_images_parallel.assert_called_once_with(["http://example.com/img.png"])

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

    def test_continuation_reaches_the_template(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        template_fn = MagicMock(return_value="open turn")
        prepare_vlm_inputs_parallel(
            [FakeMessage("user", "hi"), FakeMessage("assistant", "The goat is")],
            MagicMock(), MagicMock(), MagicMock(), template_fn,
            continue_final_message=True,
        )
        assert template_fn.call_args.kwargs["continue_final_message"] is True

    def test_a_template_failure_raises_and_nothing_is_rendered_in_its_place(self, mock_mlx):
        """Until v2.0.58 any template exception was swallowed and a ladder
        walked instead: the tokenizer's template with a fresh generation
        prompt, then a bare "role: content" join. A broken template then
        produced a confident answer to a prompt with no template in it -- and,
        on a continuation, silently RESTARTED the message. Continuing or not,
        the failure now reaches the caller and the tokenizer is never asked."""
        import pytest
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        def failing_template(proc, cfg, msgs, **kwargs):
            raise ValueError("template error")

        for continuing in (False, True):
            processor = MagicMock()
            with pytest.raises(ValueError, match="template error"):
                prepare_vlm_inputs_parallel(
                    [FakeMessage("user", "hi"), FakeMessage("assistant", "The goat is")],
                    processor, MagicMock(), MagicMock(), failing_template,
                    continue_final_message=continuing)
            processor.tokenizer.apply_chat_template.assert_not_called()

    def test_multiple_images(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", [
            FakeContentPart("text", text="compare these"),
            FakeContentPart("image_url", image_url=FakeImageUrl("http://a.com/1.png")),
            FakeContentPart("image_url", image_url=FakeImageUrl("http://b.com/2.png")),
        ])]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()
        mock_batch.load_images_parallel.return_value = [MagicMock(), MagicMock()]
        mock_template_fn = MagicMock(return_value="formatted")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, mock_template_fn
        )

        assert len(images) == 2
        mock_batch.load_images_parallel.assert_called_once_with([
            "http://a.com/1.png", "http://b.com/2.png"
        ])
        # num_images should be passed to template
        mock_template_fn.assert_called_once()
        assert mock_template_fn.call_args.kwargs.get('num_images') == 2


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


class TestNonUserImageGuard:
    """An image on a non-user turn must be REFUSED on MLX, not relocated.

    mlx-vlm gates the marker on ``role == "user"`` in three places, so such an
    image does not error there -- it silently moves to the latest user turn and
    the model is told it arrived in that message. Once the editor can attach
    media to an assistant message, that is a wrong answer produced from a
    plausible-looking request, so the provider refuses and names gguf.
    """

    @staticmethod
    def _req(role):
        from heylook_llm.config import ChatRequest
        return ChatRequest.model_validate({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "draw a cat"}]},
            {"role": role, "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                {"type": "text", "text": "here it is"}]},
        ]})

    def test_assistant_image_is_detected(self, mock_mlx):
        from heylook_llm.providers.mlx_provider import _non_user_image_roles

        assert _non_user_image_roles(self._req("assistant").messages) == ["assistant"]
        assert _non_user_image_roles(self._req("system").messages) == ["system"]

    def test_user_image_is_not_flagged(self, mock_mlx):
        from heylook_llm.config import ChatRequest
        from heylook_llm.providers.mlx_provider import _non_user_image_roles

        ok = ChatRequest.model_validate({"messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                {"type": "text", "text": "what is this?"}]},
            {"role": "assistant", "content": "a cat"},
        ]})
        assert _non_user_image_roles(ok.messages) == []


def test_reasoning_content_survives_mlx_vlms_message_rebuild():
    """mlx-vlm's apply_chat_template(return_messages=True) rebuilds each
    message as role + content only. heylook's wrapper must put the rest back,
    or `reasoning_content` never reaches the template on the VLM path (found
    2026-09-24: gemma-4's continued turn lost its thought and degenerated).
    Driven through the REAL mlx-vlm rebuild with a recording tokenizer."""
    import json
    from pathlib import Path

    import pytest

    from heylook_llm.providers.mlx_provider import vlm_apply_chat_template

    cfg_path = Path("modelzoo/google/gemma-4-26b-a4b-it-8bit-mlx/config.json")
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"model_type": "gemma4"}
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
