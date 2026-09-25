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
def reconstruct_thinking():
    """Import _reconstruct_thinking from canonical location."""
    from heylook_llm.providers.common.vlm_inputs import _reconstruct_thinking
    return _reconstruct_thinking


@pytest.mark.unit
class TestReconstructThinking:
    """Tests for _reconstruct_thinking() helper."""

    # Exact format: <think>\n{thinking}\n</think>\n{content}. The 'thinking'
    # key is removed via pop() (so the INPUT dict is mutated), other keys
    # survive, and multi-line thinking is carried verbatim.
    @pytest.mark.parametrize(
        "msg, expected",
        [
            ({"role": "assistant", "content": "reply", "thinking": "reason"},
             {"role": "assistant", "content": "<think>\nreason\n</think>\nreply"}),
            ({"role": "assistant", "content": "hello", "thinking": "I should say hi"},
             {"role": "assistant", "content": "<think>\nI should say hi\n</think>\nhello"}),
            ({"role": "assistant", "content": "answer", "thinking": "let me think"},
             {"role": "assistant", "content": "<think>\nlet me think\n</think>\nanswer"}),
            ({"role": "assistant", "content": "42",
              "thinking": "Step 1: analyze\nStep 2: compute\nStep 3: conclude"},
             {"role": "assistant", "content":
              "<think>\nStep 1: analyze\nStep 2: compute\nStep 3: conclude\n</think>\n42"}),
            ({"role": "assistant", "content": "hi", "thinking": "r", "name": "bot"},
             {"role": "assistant", "content": "<think>\nr\n</think>\nhi", "name": "bot"}),
            ({"role": "assistant", "content": "hi", "thinking": "reason"},
             {"role": "assistant", "content": "<think>\nreason\n</think>\nhi"}),
        ],
        ids=[
            "format", "assistant-with-thinking", "prepended-before-content",
            "multiline", "preserves-other-keys", "mutates-input-dict",
        ],
    )
    def test_thinking_format(self, reconstruct_thinking, msg, expected):
        result = reconstruct_thinking(msg)
        assert result == expected
        assert "thinking" not in msg

    # Non-assistant roles, and None / empty / missing thinking: content is
    # unchanged and the key is still popped.
    @pytest.mark.parametrize(
        "msg, expected",
        [
            ({"role": "user", "content": "hi", "thinking": "something"},
             {"role": "user", "content": "hi"}),
            ({"role": "system", "content": "You are helpful.", "thinking": "something"},
             {"role": "system", "content": "You are helpful."}),
            ({"role": "assistant", "content": "hello", "thinking": None},
             {"role": "assistant", "content": "hello"}),
            ({"role": "assistant", "content": "hello", "thinking": ""},
             {"role": "assistant", "content": "hello"}),
            ({"role": "assistant", "content": "hello"},
             {"role": "assistant", "content": "hello"}),
        ],
        ids=["user-ignored", "system-ignored", "none-thinking", "empty-thinking",
             "no-thinking-key"],
    )
    def test_content_unchanged_without_assistant_thinking(
        self, reconstruct_thinking, msg, expected
    ):
        assert reconstruct_thinking(msg) == expected


@pytest.mark.unit
class TestThinkingForTemplate:
    """v1.79.63: prior thinking goes to the template the way THAT template
    takes it. Three branches, and the None case keeps the old behaviour.

    A family with neither (gemma-4 before its template read
    reasoning_content, or any plain template) drops it: <think> text the
    model never emits is noise, not history. Non-assistant thinking is
    dropped everywhere."""

    @pytest.mark.parametrize(
        "msg, info, expected",
        [
            ({"role": "assistant", "content": "a", "thinking": "t"},
             dict(reads_reasoning_content=True, has_thinking_markers=True),
             {"role": "assistant", "content": "a", "reasoning_content": "t"}),
            ({"role": "assistant", "content": "a", "thinking": "t"},
             dict(has_thinking_markers=True),
             {"role": "assistant", "content": "<think>\nt\n</think>\na"}),
            ({"role": "assistant", "content": "a", "thinking": "t"}, {},
             {"role": "assistant", "content": "a"}),
            ({"role": "assistant", "content": "a", "thinking": "t"}, None,
             {"role": "assistant", "content": "<think>\nt\n</think>\na"}),
            ({"role": "user", "content": "q", "thinking": "t"},
             dict(reads_reasoning_content=True),
             {"role": "user", "content": "q"}),
        ],
        ids=[
            "reads-reasoning-content-gets-the-key", "markers-get-tags-reconstructed",
            "neither-drops-it", "none-keeps-legacy-reconstruction",
            "non-assistant-dropped-everywhere",
        ],
    )
    def test_thinking_for_template(self, msg, info, expected):
        from types import SimpleNamespace

        from heylook_llm.providers.common.vlm_inputs import thinking_for_template

        if info is not None:
            info = SimpleNamespace(**{
                "has_thinking_markers": False, "reads_reasoning_content": False, **info})
        assert thinking_for_template(msg, info) == expected
