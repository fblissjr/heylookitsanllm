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
import re
from typing import Tuple, Optional, List


def parse_thinking_content(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse <think>...</think> blocks from generated text.

    Handles multiple thinking blocks by concatenating them with separators.
    Removes all thinking blocks from the content.

    Args:
        text: Raw generated text that may contain <think>...</think> blocks

    Returns:
        Tuple of (content_without_thinking, thinking_content_or_none)
        - content: The response text with all <think> blocks removed
        - thinking: Concatenated thinking content, or None if no blocks found
    """
    if not text:
        return "", None

    pattern = r'<think>\s*(.*?)\s*</think>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return text.strip(), None

    # Concatenate all thinking blocks (rare to have multiple, but handle it)
    non_empty_matches = [m.strip() for m in matches if m.strip()]
    if not non_empty_matches:
        # Had think tags but they were empty
        content = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        return content, None

    thinking = '\n---\n'.join(non_empty_matches)

    # Remove all think blocks from content
    content = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    return content, thinking


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
    """Streaming <think>-block parser (text-based; see module docstring).

    The name and the ``token_id`` parameter survive from the retired
    token-level mode so existing call sites and the ReasoningParser
    protocol stay stable; ``token_id`` is ignored.

    ``strip_tokens``: declared specials removed from ROUTED text, same
    defense as the harmony/gemma/pass-through parsers -- without it,
    non-think control tokens leak literally into user-visible output on
    detokenizer paths that emit specials as text.
    """

    def __init__(self, initial_thinking: bool = False,
                 strip_tokens: "frozenset[str]" = frozenset()):
        from heylook_llm.reasoning_parser import _compile_strip_pattern, _strip_specials
        self._parser = StreamingThinkingParser(initial_thinking=initial_thinking)
        self._strip_pattern = _compile_strip_pattern(strip_tokens)
        self._strip = _strip_specials
        # The inner parser's own buffer holdback can SPLIT a special token
        # across deltas, so per-delta sub() would miss it. Keep a per-kind
        # rolling pending buffer and hold back any tail that could be a
        # partial special (from its last '<' within max-token-length).
        self._max_special_len = max((len(t) for t in strip_tokens), default=0)
        self._pend_kind: Optional[str] = None
        self._pend = ""

    def _flush_pend(self, out: List[Tuple[str, str]]) -> None:
        if self._pend and self._pend_kind is not None:
            cleaned = self._strip(self._pend, self._strip_pattern)
            if cleaned:
                out.append((self._pend_kind, cleaned))
        self._pend = ""
        self._pend_kind = None

    def _cleaned(self, deltas: List[Tuple[str, str]], final: bool = False) -> List[Tuple[str, str]]:
        if self._strip_pattern is None:
            return [d for d in deltas if d[1]]
        out: List[Tuple[str, str]] = []
        for kind, text in deltas:
            if self._pend_kind is not None and kind != self._pend_kind:
                self._flush_pend(out)
            self._pend_kind = kind
            # strip removes only COMPLETE tokens, so carrying stripped text
            # forward is safe; partials survive into the held tail below
            buf = self._strip(self._pend + text, self._strip_pattern)
            keep = 0
            scan_from = max(0, len(buf) - (self._max_special_len - 1))
            lt = buf.rfind("<")
            if lt >= scan_from:
                keep = len(buf) - lt
            self._pend = buf[len(buf) - keep:] if keep else ""
            emit = buf[: len(buf) - keep] if keep else buf
            if emit:
                out.append((kind, emit))
        if final:
            self._flush_pend(out)
        return out

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        return self._cleaned(self._parser.process_chunk(text))

    def flush(self) -> List[Tuple[str, str]]:
        return self._cleaned(self._parser.flush(), final=True)

    def reset(self):
        self._parser.reset()
        self._pend = ""
        self._pend_kind = None
