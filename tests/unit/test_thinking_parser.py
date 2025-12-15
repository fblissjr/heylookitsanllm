# tests/unit/test_thinking_parser.py
"""
Unit tests for the Qwen3 thinking token parser.

Tests the non-streaming parse_thinking_content function,
the StreamingThinkingParser class, and the token-level parsers.
"""
import pytest
from heylook_llm.thinking_parser import (
    parse_thinking_content,
    StreamingThinkingParser,
    TokenLevelThinkingParser,
    HybridThinkingParser,
    THINK_START_TOKEN_ID,
    THINK_END_TOKEN_ID,
)


class TestParseThinkingContent:
    """Tests for the non-streaming parse_thinking_content function."""

    def test_no_thinking_block(self):
        """Text without thinking block should return as-is."""
        text = "This is a response without thinking."
        content, thinking = parse_thinking_content(text)
        assert content == text
        assert thinking is None

    def test_single_thinking_block(self):
        """Single thinking block should be extracted."""
        text = "<think>Let me reason about this.</think>The answer is 42."
        content, thinking = parse_thinking_content(text)
        assert content == "The answer is 42."
        assert thinking == "Let me reason about this."

    def test_thinking_block_with_whitespace(self):
        """Whitespace around content should be stripped."""
        text = "<think>  \n\nThinking with spaces  \n</think>  \n\nContent here"
        content, thinking = parse_thinking_content(text)
        assert content == "Content here"
        assert thinking == "Thinking with spaces"

    def test_multiple_thinking_blocks(self):
        """Multiple thinking blocks should be concatenated."""
        text = "<think>First thought.</think>Response<think>Second thought.</think>More response"
        content, thinking = parse_thinking_content(text)
        assert content == "ResponseMore response"
        assert thinking == "First thought.\n---\nSecond thought."

    def test_empty_thinking_block(self):
        """Empty thinking block should result in None thinking."""
        text = "<think></think>Just content"
        content, thinking = parse_thinking_content(text)
        assert content == "Just content"
        assert thinking is None

    def test_whitespace_only_thinking_block(self):
        """Whitespace-only thinking block should result in None thinking."""
        text = "<think>   \n\n   </think>Just content"
        content, thinking = parse_thinking_content(text)
        assert content == "Just content"
        assert thinking is None

    def test_empty_input(self):
        """Empty input should return empty content and None thinking."""
        content, thinking = parse_thinking_content("")
        assert content == ""
        assert thinking is None

    def test_none_input_like(self):
        """Empty string input should be handled."""
        content, thinking = parse_thinking_content("")
        assert content == ""
        assert thinking is None

    def test_thinking_at_end(self):
        """Thinking block at end of text."""
        text = "Some content<think>Thinking at end</think>"
        content, thinking = parse_thinking_content(text)
        assert content == "Some content"
        assert thinking == "Thinking at end"

    def test_only_thinking(self):
        """Only thinking block, no content."""
        text = "<think>Just thinking, no response</think>"
        content, thinking = parse_thinking_content(text)
        assert content == ""
        assert thinking == "Just thinking, no response"

    def test_multiline_thinking(self):
        """Multiline thinking content should be preserved."""
        text = "<think>Line 1\nLine 2\nLine 3</think>Final answer"
        content, thinking = parse_thinking_content(text)
        assert content == "Final answer"
        assert "Line 1" in thinking
        assert "Line 2" in thinking
        assert "Line 3" in thinking

    def test_special_characters_in_thinking(self):
        """Special characters in thinking should be preserved."""
        text = "<think>2+2=4, x<y, a&b</think>Result"
        content, thinking = parse_thinking_content(text)
        assert content == "Result"
        assert "2+2=4" in thinking
        assert "x<y" in thinking
        assert "a&b" in thinking


class TestStreamingThinkingParser:
    """Tests for the StreamingThinkingParser class."""

    def test_no_thinking_stream(self):
        """Stream without thinking should all be content."""
        parser = StreamingThinkingParser()
        results = []
        for chunk in ["Hello", " world", "!"]:
            results.extend(parser.process_chunk(chunk))
        results.extend(parser.flush())

        # Filter empty results
        results = [(t, txt) for t, txt in results if txt]
        assert all(t == 'content' for t, _ in results)
        full_text = ''.join(txt for _, txt in results)
        assert "Hello world!" in full_text

    def test_complete_thinking_block(self):
        """Complete thinking block followed by content."""
        parser = StreamingThinkingParser()
        results = []

        # Send thinking block in chunks
        results.extend(parser.process_chunk("<think>"))
        results.extend(parser.process_chunk("Reasoning"))
        results.extend(parser.process_chunk("</think>"))
        results.extend(parser.process_chunk("Answer"))
        results.extend(parser.flush())

        # Filter non-empty results
        results = [(t, txt) for t, txt in results if txt]

        # Check we have thinking and content
        thinking_parts = [txt for t, txt in results if t == 'thinking']
        content_parts = [txt for t, txt in results if t == 'content']

        assert len(thinking_parts) > 0, "Should have thinking parts"
        assert len(content_parts) > 0, "Should have content parts"
        assert "Reasoning" in ''.join(thinking_parts)
        assert "Answer" in ''.join(content_parts)

    def test_split_tag_handling(self):
        """Tags split across chunks should be handled correctly."""
        parser = StreamingThinkingParser()
        results = []

        # Split <think> tag across chunks
        results.extend(parser.process_chunk("<thi"))
        results.extend(parser.process_chunk("nk>"))
        results.extend(parser.process_chunk("Thinking"))
        results.extend(parser.process_chunk("</th"))
        results.extend(parser.process_chunk("ink>"))
        results.extend(parser.process_chunk("Content"))
        results.extend(parser.flush())

        results = [(t, txt) for t, txt in results if txt]

        thinking_text = ''.join(txt for t, txt in results if t == 'thinking')
        content_text = ''.join(txt for t, txt in results if t == 'content')

        assert "Thinking" in thinking_text
        assert "Content" in content_text

    def test_reset(self):
        """Reset should allow parser reuse."""
        parser = StreamingThinkingParser()

        # First use
        parser.process_chunk("<think>First</think>Content1")
        parser.flush()

        # Reset and reuse
        parser.reset()
        results = []
        results.extend(parser.process_chunk("<think>Second</think>Content2"))
        results.extend(parser.flush())

        results = [(t, txt) for t, txt in results if txt]
        full_text = ''.join(txt for _, txt in results)
        assert "Second" in full_text or "Content2" in full_text
        assert "First" not in full_text

    def test_content_before_thinking(self):
        """Content before thinking block should be emitted as content."""
        parser = StreamingThinkingParser()
        results = []

        results.extend(parser.process_chunk("Prefix"))
        results.extend(parser.process_chunk("<think>"))
        results.extend(parser.process_chunk("Thinking"))
        results.extend(parser.process_chunk("</think>"))
        results.extend(parser.process_chunk("Suffix"))
        results.extend(parser.flush())

        results = [(t, txt) for t, txt in results if txt]

        content_text = ''.join(txt for t, txt in results if t == 'content')
        thinking_text = ''.join(txt for t, txt in results if t == 'thinking')

        assert "Prefix" in content_text
        assert "Suffix" in content_text
        assert "Thinking" in thinking_text


class TestEdgeCases:
    """Edge case tests."""

    def test_nested_angle_brackets(self):
        """Content with angle brackets but not think tags."""
        text = "Compare x<y and y>z in the expression"
        content, thinking = parse_thinking_content(text)
        assert content == text
        assert thinking is None

    def test_partial_think_tag(self):
        """Partial think tag should not be matched."""
        text = "<thin some random text with angle brackets"
        content, thinking = parse_thinking_content(text)
        assert content == text
        assert thinking is None

    def test_unclosed_think_tag(self):
        """Unclosed think tag should not match."""
        text = "<think>This is never closed and the response continues"
        content, thinking = parse_thinking_content(text)
        # Unclosed tag means no match
        assert content == text
        assert thinking is None

    def test_very_long_thinking(self):
        """Very long thinking content should be handled."""
        long_thinking = "A" * 10000
        text = f"<think>{long_thinking}</think>Short answer"
        content, thinking = parse_thinking_content(text)
        assert content == "Short answer"
        assert thinking == long_thinking
        assert len(thinking) == 10000


class TestTokenLevelThinkingParser:
    """Tests for the TokenLevelThinkingParser class."""

    def test_token_ids_constants(self):
        """Verify token ID constants are correct."""
        assert THINK_START_TOKEN_ID == 151667
        assert THINK_END_TOKEN_ID == 151668

    def test_no_thinking_tokens(self):
        """Regular tokens should be emitted as content."""
        parser = TokenLevelThinkingParser()
        results = []

        # Normal tokens (not thinking-related)
        results.extend(parser.process_token(1234, "Hello"))
        results.extend(parser.process_token(5678, " world"))
        results.extend(parser.flush())

        assert all(t == 'content' for t, _ in results)
        full_text = ''.join(txt for _, txt in results)
        assert full_text == "Hello world"

    def test_think_start_token(self):
        """<think> token should switch to thinking mode."""
        parser = TokenLevelThinkingParser()

        # <think> token - should be suppressed
        results = parser.process_token(THINK_START_TOKEN_ID, "<think>")
        assert results == []
        assert parser.in_thinking is True

    def test_think_end_token(self):
        """</think> token should exit thinking mode."""
        parser = TokenLevelThinkingParser()
        parser.in_thinking = True  # Start in thinking mode

        # </think> token - should be suppressed
        results = parser.process_token(THINK_END_TOKEN_ID, "</think>")
        assert results == []
        assert parser.in_thinking is False
        assert parser.thinking_complete is True

    def test_thinking_content_between_tokens(self):
        """Content between <think> and </think> should be thinking."""
        parser = TokenLevelThinkingParser()
        results = []

        # <think> token
        results.extend(parser.process_token(THINK_START_TOKEN_ID, "<think>"))
        # Thinking content
        results.extend(parser.process_token(1001, "Let me"))
        results.extend(parser.process_token(1002, " reason"))
        # </think> token
        results.extend(parser.process_token(THINK_END_TOKEN_ID, "</think>"))
        # Response content
        results.extend(parser.process_token(2001, "The answer"))
        results.extend(parser.process_token(2002, " is 42"))
        results.extend(parser.flush())

        thinking_parts = [(t, txt) for t, txt in results if t == 'thinking']
        content_parts = [(t, txt) for t, txt in results if t == 'content']

        thinking_text = ''.join(txt for _, txt in thinking_parts)
        content_text = ''.join(txt for _, txt in content_parts)

        assert "Let me reason" in thinking_text
        assert "The answer is 42" in content_text

    def test_reset(self):
        """Reset should clear parser state."""
        parser = TokenLevelThinkingParser()
        parser.in_thinking = True
        parser.thinking_complete = True

        parser.reset()

        assert parser.in_thinking is False
        assert parser.thinking_complete is False


class TestHybridThinkingParser:
    """Tests for the HybridThinkingParser class."""

    def test_token_mode_with_token_ids(self):
        """Parser should use token mode when token IDs are provided."""
        parser = HybridThinkingParser()
        results = []

        # First chunk with token ID - activates token mode
        results.extend(parser.process_chunk("Hello", token_id=1234))
        assert parser._use_token_mode is True

        results.extend(parser.process_chunk(" world", token_id=5678))
        results.extend(parser.flush())

        full_text = ''.join(txt for _, txt in results)
        assert full_text == "Hello world"

    def test_text_mode_without_token_ids(self):
        """Parser should use text mode when no token IDs are provided."""
        parser = HybridThinkingParser()
        results = []

        # Chunks without token IDs - uses text mode
        results.extend(parser.process_chunk("<think>"))
        results.extend(parser.process_chunk("Thinking"))
        results.extend(parser.process_chunk("</think>"))
        results.extend(parser.process_chunk("Answer"))
        results.extend(parser.flush())

        assert parser._use_token_mode is False

        thinking_parts = [(t, txt) for t, txt in results if t == 'thinking' and txt]
        content_parts = [(t, txt) for t, txt in results if t == 'content' and txt]

        thinking_text = ''.join(txt for _, txt in thinking_parts)
        content_text = ''.join(txt for _, txt in content_parts)

        assert "Thinking" in thinking_text
        assert "Answer" in content_text

    def test_hybrid_with_thinking_tokens(self):
        """Hybrid parser should handle Qwen3 thinking tokens."""
        parser = HybridThinkingParser()
        results = []

        # Simulate Qwen3 output with thinking tokens
        results.extend(parser.process_chunk("<think>", token_id=THINK_START_TOKEN_ID))
        results.extend(parser.process_chunk("Reasoning step 1", token_id=1001))
        results.extend(parser.process_chunk("</think>", token_id=THINK_END_TOKEN_ID))
        results.extend(parser.process_chunk("Final answer", token_id=2001))
        results.extend(parser.flush())

        thinking_parts = [(t, txt) for t, txt in results if t == 'thinking' and txt]
        content_parts = [(t, txt) for t, txt in results if t == 'content' and txt]

        thinking_text = ''.join(txt for _, txt in thinking_parts)
        content_text = ''.join(txt for _, txt in content_parts)

        assert "Reasoning step 1" in thinking_text
        assert "Final answer" in content_text
        # Special tokens should not appear in output
        assert "<think>" not in thinking_text
        assert "</think>" not in thinking_text

    def test_reset(self):
        """Reset should clear both parsers."""
        parser = HybridThinkingParser()

        # Activate token mode
        parser.process_chunk("test", token_id=123)
        assert parser._use_token_mode is True

        parser.reset()

        assert parser._use_token_mode is False
        assert parser._token_parser.in_thinking is False
        assert parser._text_parser.in_thinking is False
