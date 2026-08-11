# tests/unit/test_continuation.py
"""continue_final_message: the request-level resolution + the parser edge.

The provider-level halves live with their providers
(test_mlx_provider.TestContinuationTemplate, test_llama_server_provider.
TestContinuationEchoStrip/Guards). Here: the ONE flag-vs-convention
resolution every consumer must ask (ChatRequest.is_continuation), and the
parser rule that a continuation never starts inside a thinking block --
there is no generation prompt, so a ``prefills_thinking`` template opened
nothing, and an armed parser would misfile the whole continuation as
thinking.
"""

import pytest

from heylook_llm.config import ChatRequest
from heylook_llm.reasoning_parser import select_reasoning_parser
from heylook_llm.thinking_parser import HybridThinkingParser


def _req(messages, flag=None):
    return ChatRequest(model="m", messages=messages, continue_final_message=flag)


USER_LAST = [{"role": "user", "content": "hi"}]
ASSISTANT_LAST = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "he"}]


@pytest.mark.unit
class TestIsContinuation:
    def test_auto_trailing_assistant_continues(self):
        assert _req(ASSISTANT_LAST).is_continuation() is True

    def test_auto_trailing_user_does_not(self):
        assert _req(USER_LAST).is_continuation() is False

    def test_explicit_true_continues_any_role(self):
        assert _req(USER_LAST, flag=True).is_continuation() is True

    def test_explicit_false_never_continues(self):
        assert _req(ASSISTANT_LAST, flag=False).is_continuation() is False


class _PrefillingTemplate:
    has_thinking_markers = True
    prefills_thinking = True
    has_harmony_structure = False
    has_gemma_channel_structure = False
    special_tokens = frozenset()


@pytest.mark.unit
class TestParserContinuationEdge:
    @staticmethod
    def _channels(parser, text):
        deltas = parser.process_chunk(text) + parser.flush()
        content = "".join(t for ch, t in deltas if ch == "content")
        thinking = "".join(t for ch, t in deltas if ch == "thinking")
        return content, thinking

    def test_prefilled_thinking_arms_parser_normally(self):
        parser = select_reasoning_parser(_PrefillingTemplate(), thinking_enabled=True)
        assert isinstance(parser, HybridThinkingParser)
        content, thinking = self._channels(parser, "still inside the block")
        assert thinking == "still inside the block"
        assert content == ""

    def test_continuation_disarms_the_prefill_assumption(self):
        # Same template, same thinking flag -- but a continuation stream
        # starts inside the final message's CONTENT.
        parser = select_reasoning_parser(
            _PrefillingTemplate(), thinking_enabled=True, continuing=True)
        content, thinking = self._channels(parser, "continued content")
        assert content == "continued content"
        assert not thinking
