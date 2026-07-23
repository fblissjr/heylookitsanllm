# src/heylook_llm/thinking_parser.py
"""
Parser for <think>...</think> thinking blocks in generated text.

Text-based only, deliberately: the tag STRINGS are the format grammar
(like the harmony/gemma channel parsers in reasoning_parser.py), while
token IDs are per-model vocabulary -- an earlier token-level mode
hardcoded Qwen3's ids (151667/151668) and silently failed to split for
any other <think>-family vocabulary.

``initial_thinking=True`` supports templates that PRE-FILL ``<think>\\n``
into the generation prompt when thinking is enabled (Qwen3.5 style): the
model's output then starts inside the block and never emits the opening
tag itself. Selection is driven by ``ModelTemplateInfo.prefills_thinking``
+ the request's effective thinking flag (see select_reasoning_parser).
"""
from typing import Tuple, Optional, List


class StreamingThinkingParser:
    """
    Stateful parser for streaming thinking content.

    Tracks whether we're currently inside a <think> block and yields
    appropriate deltas for thinking vs content.
    """

    def __init__(self, initial_thinking: bool = False):
        self._initial_thinking = initial_thinking
        self.buffer = ""
        self.in_thinking = initial_thinking
        self.thinking_complete = False

    def process_chunk(self, chunk_text: str) -> list[Tuple[str, str]]:
        """
        Process a chunk of streamed text and return deltas.

        Args:
            chunk_text: The next chunk of generated text

        Returns:
            List of tuples (delta_type, text) where delta_type is 'thinking' or 'content'
        """
        if not chunk_text:
            return []

        self.buffer += chunk_text
        results = []

        while True:
            if not self.in_thinking and not self.thinking_complete:
                # Looking for <think> start
                if '<think>' in self.buffer:
                    idx = self.buffer.index('<think>')
                    # Emit any content before <think>
                    if idx > 0:
                        results.append(('content', self.buffer[:idx]))
                    self.buffer = self.buffer[idx + 7:]  # Skip <think>
                    self.in_thinking = True
                else:
                    # Check if buffer might contain partial <think> tag
                    # Keep last 6 chars in case of partial match
                    if len(self.buffer) > 6:
                        # No <think> found and buffer is long enough - emit as content
                        emit_len = len(self.buffer) - 6
                        results.append(('content', self.buffer[:emit_len]))
                        self.buffer = self.buffer[emit_len:]
                    break

            elif self.in_thinking:
                # Inside thinking block, looking for </think>
                if '</think>' in self.buffer:
                    idx = self.buffer.index('</think>')
                    # Emit thinking content
                    if idx > 0:
                        results.append(('thinking', self.buffer[:idx]))
                    self.buffer = self.buffer[idx + 8:]  # Skip </think>
                    self.in_thinking = False
                    self.thinking_complete = True
                else:
                    # Check for partial </think> tag
                    if len(self.buffer) > 8:
                        emit_len = len(self.buffer) - 8
                        results.append(('thinking', self.buffer[:emit_len]))
                        self.buffer = self.buffer[emit_len:]
                    break

            else:
                # Thinking complete, everything else is content
                if self.buffer:
                    results.append(('content', self.buffer))
                    self.buffer = ""
                break

        return results

    def flush(self) -> list[Tuple[str, str]]:
        """
        Flush any remaining buffer content.

        Call this when the stream ends to get any remaining content.

        Returns:
            List of tuples (delta_type, text)
        """
        results = []
        if self.buffer:
            if self.in_thinking:
                results.append(('thinking', self.buffer))
            else:
                results.append(('content', self.buffer))
            self.buffer = ""
        return results

    def reset(self):
        """Reset parser state for reuse."""
        self.buffer = ""
        self.in_thinking = self._initial_thinking
        self.thinking_complete = False


class HybridThinkingParser:
    """ReasoningParser adapter over StreamingThinkingParser (text-based;
    see module docstring).

    The name and the ``token_id`` parameter survive from the retired
    token-level mode so existing call sites and the ReasoningParser
    protocol stay stable; ``token_id`` is ignored.

    Declared-specials stripping is NOT here: its rolling holdback -- born in
    this class, because the inner parser's tag-holdback splits specials
    across deltas -- now serves all four parsers as
    ``reasoning_parser.StripSpecials``, which the factory composes over this
    one (docs/parser_strip_unification.md).
    """

    def __init__(self, initial_thinking: bool = False):
        self._parser = StreamingThinkingParser(initial_thinking=initial_thinking)

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        return [d for d in self._parser.process_chunk(text) if d[1]]

    def flush(self) -> List[Tuple[str, str]]:
        return [d for d in self._parser.flush() if d[1]]

    def reset(self):
        self._parser.reset()
