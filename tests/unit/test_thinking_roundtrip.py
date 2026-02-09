# tests/unit/test_thinking_roundtrip.py
"""
Unit tests for thinking round-trip: _reconstruct_thinking() and prefill conventions.

Tests the Glass Box backend logic that ensures:
1. Thinking content on assistant messages round-trips through <think> tags
2. Non-assistant messages ignore thinking field
3. Prefill detection (last message is assistant -> add_generation_prompt=False)
"""
import pytest


@pytest.fixture
def reconstruct_thinking(mock_mlx):  # noqa: ARG001 -- fixture dependency
    """Import _reconstruct_thinking with MLX mocked."""
    from heylook_llm.providers.mlx_provider import _reconstruct_thinking
    return _reconstruct_thinking


@pytest.mark.unit
class TestReconstructThinking:
    """Tests for _reconstruct_thinking() helper."""

    def test_assistant_with_thinking(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "hello", "thinking": "I should say hi"}
        result = reconstruct_thinking(msg)
        assert "<think>" in result["content"]
        assert "I should say hi" in result["content"]
        assert "thinking" not in result  # key removed via pop()

    def test_thinking_prepended_before_content(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "answer", "thinking": "let me think"}
        result = reconstruct_thinking(msg)
        assert result["content"].startswith("<think>")
        assert result["content"].endswith("answer")
        assert "let me think" in result["content"]

    def test_thinking_format(self, reconstruct_thinking):
        """Verify exact format: <think>\n{thinking}\n</think>\n{content}"""
        msg = {"role": "assistant", "content": "reply", "thinking": "reason"}
        result = reconstruct_thinking(msg)
        assert result["content"] == "<think>\nreason\n</think>\nreply"

    def test_user_message_thinking_ignored(self, reconstruct_thinking):
        msg = {"role": "user", "content": "hi", "thinking": "something"}
        result = reconstruct_thinking(msg)
        assert "<think>" not in result["content"]
        assert result["content"] == "hi"

    def test_system_message_thinking_ignored(self, reconstruct_thinking):
        msg = {"role": "system", "content": "You are helpful.", "thinking": "something"}
        result = reconstruct_thinking(msg)
        assert "<think>" not in result["content"]

    def test_none_thinking(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "hello", "thinking": None}
        result = reconstruct_thinking(msg)
        assert result["content"] == "hello"
        assert "thinking" not in result  # None gets popped

    def test_empty_thinking(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "hello", "thinking": ""}
        result = reconstruct_thinking(msg)
        assert result["content"] == "hello"
        assert "thinking" not in result  # empty string is falsy

    def test_no_thinking_key(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "hello"}
        result = reconstruct_thinking(msg)
        assert result["content"] == "hello"

    def test_mutates_input_dict(self, reconstruct_thinking):
        """_reconstruct_thinking uses pop() so it mutates the dict."""
        msg = {"role": "assistant", "content": "hi", "thinking": "reason"}
        reconstruct_thinking(msg)
        assert "thinking" not in msg

    def test_preserves_other_keys(self, reconstruct_thinking):
        msg = {"role": "assistant", "content": "hi", "thinking": "r", "name": "bot"}
        result = reconstruct_thinking(msg)
        assert result["name"] == "bot"
        assert result["role"] == "assistant"

    def test_multiline_thinking(self, reconstruct_thinking):
        thinking_text = "Step 1: analyze\nStep 2: compute\nStep 3: conclude"
        msg = {"role": "assistant", "content": "42", "thinking": thinking_text}
        result = reconstruct_thinking(msg)
        assert "Step 1: analyze" in result["content"]
        assert "Step 3: conclude" in result["content"]


@pytest.mark.unit
class TestPrefillConvention:
    """Tests for the prefill detection pattern used in generation strategies.

    When the last message is role=assistant, add_generation_prompt should be False
    (prefill/continue mode). When the last message is role=user, it should be True
    (normal generation).
    """

    def test_last_assistant_means_prefill(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "partial response"},
        ]
        last_is_assistant = messages[-1].get("role") == "assistant"
        add_gen_prompt = not last_is_assistant
        assert last_is_assistant is True
        assert add_gen_prompt is False

    def test_last_user_means_generate(self):
        messages = [
            {"role": "user", "content": "hi"},
        ]
        last_is_assistant = messages[-1].get("role") == "assistant"
        add_gen_prompt = not last_is_assistant
        assert last_is_assistant is False
        assert add_gen_prompt is True

    def test_empty_messages(self):
        messages = []
        last_is_assistant = messages[-1].get("role") == "assistant" if messages else False
        add_gen_prompt = not last_is_assistant
        assert last_is_assistant is False
        assert add_gen_prompt is True

    def test_system_then_user(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        last_is_assistant = messages[-1].get("role") == "assistant"
        assert last_is_assistant is False

    def test_multi_turn_ending_with_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": ""},
        ]
        last_is_assistant = messages[-1].get("role") == "assistant"
        assert last_is_assistant is True
