# src/heylook_llm/thinking_parser.py
"""
Parser for Qwen3 thinking blocks in generated text.

Qwen3 models can output <think>...</think> blocks when reasoning.
This module extracts thinking content and separates it from the final response.

Supports two parsing modes:
1. Token-level: Uses special token IDs for precise detection (recommended)
2. Text-based: Falls back to regex for models without token IDs

Qwen3 thinking token IDs:
- <think>: 151667
- </think>: 151668
"""
import re
from typing import Tuple, Optional, List

# Qwen3 special token IDs for thinking blocks
THINK_START_TOKEN_ID = 151667  # <think>
THINK_END_TOKEN_ID = 151668    # </think>


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


class TokenLevelThinkingParser:
    """
    Token-based parser for streaming thinking content.

    Uses special token IDs for precise detection of <think>/</ think> boundaries.
    More efficient than text-based parsing as it avoids buffering and regex.

    This parser is recommended for Qwen3 models where token IDs are available.
    Falls back gracefully if token IDs are not provided.
    """

    def __init__(self):
        self.in_thinking = False
        self.thinking_complete = False
        # Track if we've seen the <think> token (skip its text in output)
        self._skip_next_text = False

    def process_token(
        self, token_id: Optional[int], text: str
    ) -> List[Tuple[str, str]]:
        """
        Process a token with its ID and text.

        Args:
            token_id: The token ID (can be None if not available)
            text: The decoded text for this token

        Returns:
            List of tuples (delta_type, text) where delta_type is 'thinking' or 'content'
            Empty list if token should be suppressed (e.g., the special tokens themselves)
        """
        if not text:
            return []

        # Check for special token IDs
        if token_id == THINK_START_TOKEN_ID:
            # Entering thinking mode - suppress the <think> text
            self.in_thinking = True
            self._skip_next_text = True
            return []

        if token_id == THINK_END_TOKEN_ID:
            # Exiting thinking mode - suppress the </think> text
            self.in_thinking = False
            self.thinking_complete = True
            self._skip_next_text = True
            return []

        # Skip text that corresponds to special tokens (already handled above)
        if self._skip_next_text:
            self._skip_next_text = False
            # Only skip if text matches the special token text
            if text.strip() in ('<think>', '</think>'):
                return []

        # Determine delta type based on current state
        if self.in_thinking:
            return [('thinking', text)]
        else:
            return [('content', text)]

    def flush(self) -> List[Tuple[str, str]]:
        """
        Flush any remaining state.

        For token-level parsing, there's no buffering so this just resets state.

        Returns:
            Empty list (no buffered content)
        """
        return []

    def reset(self):
        """Reset parser state for reuse."""
        self.in_thinking = False
        self.thinking_complete = False
        self._skip_next_text = False


class HybridThinkingParser:
    """
    Hybrid parser that uses token IDs when available, falls back to text parsing.

    This is the recommended parser for production use as it:
    1. Uses precise token-level detection when token IDs are available
    2. Falls back to text-based parsing for compatibility with other models
    """

    def __init__(self):
        self._token_parser = TokenLevelThinkingParser()
        self._text_parser = StreamingThinkingParser()
        self._use_token_mode = False  # Will be set once we see a token ID

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Process a chunk of streamed text with optional token ID.

        Args:
            text: The next chunk of generated text
            token_id: Optional token ID for token-level parsing

        Returns:
            List of tuples (delta_type, text) where delta_type is 'thinking' or 'content'
        """
        # If we have a token ID, use token-level parsing
        if token_id is not None:
            self._use_token_mode = True
            return self._token_parser.process_token(token_id, text)

        # Once we've seen token IDs, continue using token parser
        # (this handles edge cases where token_id might be missing for some chunks)
        if self._use_token_mode:
            return self._token_parser.process_token(None, text)

        # Fall back to text-based parsing
        return self._text_parser.process_chunk(text)

    def flush(self) -> List[Tuple[str, str]]:
        """Flush any remaining buffer content."""
        if self._use_token_mode:
            return self._token_parser.flush()
        return self._text_parser.flush()

    def reset(self):
        """Reset parser state for reuse."""
        self._token_parser.reset()
        self._text_parser.reset()
        self._use_token_mode = False
