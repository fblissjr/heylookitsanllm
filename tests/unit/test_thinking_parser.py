# tests/unit/test_thinking_parser.py
"""
Unit tests for the <think>-block thinking parser
(StreamingThinkingParser + the HybridThinkingParser wrapper).

The plain-text, complete-block, split-tag, long-block and angle-bracket
cases are rows of test_reasoning_parser.py TestParserInvariants (CORPUS
"think" and TOKEN_FREE), which run them through the factory's
HybridThinkingParser, the adapter over StreamingThinkingParser.
"""
import pytest
from heylook_llm.thinking_parser import (
    StreamingThinkingParser,
    HybridThinkingParser,
)


class TestStreamingThinkingParser:
    """Tests for the StreamingThinkingParser class."""

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
    """Edge cases through the streaming parser (the production path)."""

    def _run(self, chunks):
        p = StreamingThinkingParser()
        out = []
        for ch in chunks:
            out += p.process_chunk(ch)
        out += p.flush()
        content = "".join(x for k, x in out if k == "content")
        thinking = "".join(x for k, x in out if k == "thinking")
        return content, thinking

    def test_unclosed_think_tag_flushes_to_thinking(self):
        # streaming semantics: an opened-never-closed block is thinking at
        # flush (abort mid-thought), unlike the old regex extractor
        content, thinking = self._run(["<think>never closed, stream aborts"])
        assert content == ""
        assert thinking == "never closed, stream aborts"


class TestHybridThinkingParser:
    """Text-based contract: token_id is accepted and IGNORED (the retired
    token-level mode hardcoded Qwen3 vocab ids and silently failed to split
    for any other <think>-family vocabulary)."""

    def _collect(self, parser, chunks, token_ids=None):
        results = []
        for i, ch in enumerate(chunks):
            tid = token_ids[i] if token_ids else None
            results.extend(parser.process_chunk(ch, token_id=tid))
        results.extend(parser.flush())
        thinking = ''.join(t for k, t in results if k == 'thinking')
        content = ''.join(t for k, t in results if k == 'content')
        return content, thinking

    def test_splits_markers_even_with_foreign_token_ids(self):
        # the regression the hardcoded-id mode caused: token ids present but
        # from a different vocab -> the split must still happen on TEXT
        parser = HybridThinkingParser()
        content, thinking = self._collect(
            parser,
            ['<think>', 'Thinking', '</think>', 'Answer'],
            token_ids=[11, 22, 33, 44],
        )
        assert thinking == 'Thinking'
        assert content == 'Answer'
