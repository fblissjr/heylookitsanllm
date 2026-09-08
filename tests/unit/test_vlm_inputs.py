# tests/unit/test_vlm_inputs.py
"""Tests for standalone VLM input preparation.

Covers:
- Image URL extraction from ContentPart objects and dict formats
- Text-only messages (no images)
- Thinking reconstruction in assistant messages
- Error recovery in chat template application
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

    def test_image_url_extraction_dict_format(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()
        mock_batch.load_images_parallel.return_value = [MagicMock()]
        mock_template_fn = MagicMock(return_value="formatted")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, mock_template_fn
        )

        assert has_images is True
        assert len(images) == 1
        mock_batch.load_images_parallel.assert_called_once_with(["data:image/png;base64,abc"])

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

    def test_template_error_recovery(self, mock_mlx):
        """When vlm template fails, fall back to tokenizer.apply_chat_template."""
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", "hello")]
        mock_processor = MagicMock()
        mock_processor.tokenizer.apply_chat_template.return_value = "tokenizer fallback"
        mock_config = MagicMock()
        mock_batch = MagicMock()

        def failing_template(proc, cfg, msgs, **kwargs):
            raise ValueError("template error")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, failing_template
        )

        assert prompt == "tokenizer fallback"
        mock_processor.tokenizer.apply_chat_template.assert_called_once()

    def test_template_total_fallback(self, mock_mlx):
        """When all template calls fail, fall back to manual formatting."""
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", "hello")]
        mock_processor = MagicMock()
        mock_processor.tokenizer.apply_chat_template.side_effect = TypeError("also fails")
        mock_config = MagicMock()
        mock_batch = MagicMock()

        def always_failing(proc, cfg, msgs, **kwargs):
            raise ValueError("always fails")

        images, prompt, has_images, image_urls = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, always_failing
        )

        assert "user: hello" in prompt

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

    def _img(self, url):
        return {"type": "image_url", "image_url": {"url": url}}

    def test_markers_land_on_the_message_that_carried_the_image(self, mock_mlx):
        # Image on turn 1, a later user turn with none -- the shape that used
        # to move the marker to the end.
        messages = [
            FakeMessage("user", [self._img("a.png"), {"type": "text", "text": "what is this?"}]),
            FakeMessage("assistant", "a cat"),
            FakeMessage("user", [{"type": "text", "text": "and now?"}]),
        ]
        seen = self._capture(messages)
        assert [self._markers(m["content"]) for m in seen["messages"]] == [1, 0, 0]

    def test_two_images_stay_on_their_own_turns(self, mock_mlx):
        messages = [
            FakeMessage("user", [self._img("a.png"), {"type": "text", "text": "first"}]),
            FakeMessage("assistant", "ok"),
            FakeMessage("user", [self._img("b.png"), {"type": "text", "text": "second"}]),
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
