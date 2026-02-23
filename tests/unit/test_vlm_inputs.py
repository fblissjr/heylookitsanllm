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

        images, prompt, has_images = prepare_vlm_inputs_parallel(
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

        images, prompt, has_images = prepare_vlm_inputs_parallel(
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

        images, prompt, has_images = prepare_vlm_inputs_parallel(
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

        images, prompt, has_images = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, capture_template
        )

        # The assistant message should have thinking tags prepended
        assistant_msg = captured_messages[1]
        assert "<think>" in assistant_msg["content"]
        assert "my reasoning" in assistant_msg["content"]
        assert "answer" in assistant_msg["content"]

    def test_template_error_recovery(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", "hello")]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()

        call_count = 0

        def failing_template(proc, cfg, msgs, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("template error")
            return "fallback formatted"

        images, prompt, has_images = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, failing_template
        )

        assert prompt == "fallback formatted"
        assert call_count == 2

    def test_template_total_fallback(self, mock_mlx):
        """When all template calls fail, fall back to manual formatting."""
        from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel

        messages = [FakeMessage("user", "hello")]
        mock_processor = MagicMock()
        mock_config = MagicMock()
        mock_batch = MagicMock()

        def always_failing(proc, cfg, msgs, **kwargs):
            raise ValueError("always fails")

        images, prompt, has_images = prepare_vlm_inputs_parallel(
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

        images, prompt, has_images = prepare_vlm_inputs_parallel(
            messages, mock_processor, mock_config, mock_batch, mock_template_fn
        )

        assert len(images) == 2
        mock_batch.load_images_parallel.assert_called_once_with([
            "http://a.com/1.png", "http://b.com/2.png"
        ])
        # num_images should be passed to template
        mock_template_fn.assert_called_once()
        assert mock_template_fn.call_args.kwargs.get('num_images') == 2


class TestReconstructThinking:
    """Test the module-level _reconstruct_thinking helper."""

    def test_assistant_with_thinking(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import _reconstruct_thinking

        msg = {"role": "assistant", "content": "answer", "thinking": "reasoning"}
        result = _reconstruct_thinking(msg)
        assert "<think>" in result["content"]
        assert "reasoning" in result["content"]
        assert "thinking" not in result

    def test_user_message_unchanged(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import _reconstruct_thinking

        msg = {"role": "user", "content": "question", "thinking": "irrelevant"}
        result = _reconstruct_thinking(msg)
        assert result["content"] == "question"

    def test_no_thinking_field(self, mock_mlx):
        from heylook_llm.providers.common.vlm_inputs import _reconstruct_thinking

        msg = {"role": "assistant", "content": "answer"}
        result = _reconstruct_thinking(msg)
        assert result["content"] == "answer"
