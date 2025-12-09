# src/heylook_llm/thinking_parser.py
"""
Parser for Qwen3 thinking blocks in generated text.

Qwen3 models can output <think>...</think> blocks when reasoning.
This module extracts thinking content and separates it from the final response.
"""
import re
from typing import Tuple, Optional


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

    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
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
        self.in_thinking = False
        self.thinking_complete = False
