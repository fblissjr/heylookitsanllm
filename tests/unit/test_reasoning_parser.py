"""Tests for the format-aware reasoning parser.

Covers the parser implementations + factory:

  PassThroughParser     -- no reasoning structure; text -> content
  HarmonyChannelParser  -- OpenAI harmony multi-channel format:
                           <|start|>ROLE<|channel|>NAME<|message|>CONTENT<|end|>
                           analysis/commentary -> thinking; final -> content;
                           control tokens stripped.
  HybridThinkingParser  -- <think>...</think> markers (from thinking_parser.py;
                           factory returns directly, no wrapper).

Factory selects based on ``ModelTemplateInfo`` flags derived from the
model's chat template + tokenizer config.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# Reference fixture representing an analysis-then-final harmony response.
# Analysis content is wrapped in control tokens; only the final-channel
# message should reach the user-visible content stream.
_HARMONY_REPRODUCER = (
    "<|channel|>analysis<|message|>"
    "User wants to change the man's shirt to red. Need to specify keep "
    "everything else same, keep face, lighting, composition, only change "
    "shirt color to red. Provide concise instruction 30-80 words, likely "
    "around 30-40."
    "<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
    "Replace the shirt worn by the man on the left with a solid red color, "
    "preserving its original texture, folds, and fit; keep his facial "
    "features, hair, skin tone, pose, lighting, background, and all other "
    "elements of the image unchanged."
)

_HARMONY_EXPECTED_CONTENT = (
    "Replace the shirt worn by the man on the left with a solid red color, "
    "preserving its original texture, folds, and fit; keep his facial "
    "features, hair, skin tone, pose, lighting, background, and all other "
    "elements of the image unchanged."
)

_HARMONY_EXPECTED_THINKING = (
    "User wants to change the man's shirt to red. Need to specify keep "
    "everything else same, keep face, lighting, composition, only change "
    "shirt color to red. Provide concise instruction 30-80 words, likely "
    "around 30-40."
)


def _collect(parser, text_chunks):
    """Helper: feed chunks one-at-a-time + flush; return joined (content, thinking)."""
    content_parts = []
    thinking_parts = []
    for chunk in text_chunks:
        for kind, text in parser.process_chunk(chunk):
            if kind == "content":
                content_parts.append(text)
            elif kind == "thinking":
                thinking_parts.append(text)
    for kind, text in parser.flush():
        if kind == "content":
            content_parts.append(text)
        elif kind == "thinking":
            thinking_parts.append(text)
    return "".join(content_parts), "".join(thinking_parts)


class TestPassThroughParser:
    def test_text_routes_to_content(self):
        from heylook_llm.reasoning_parser import PassThroughParser

        parser = PassThroughParser()
        content, thinking = _collect(parser, ["hello ", "world"])
        assert content == "hello world"
        assert thinking == ""

    def test_empty_chunks_ignored(self):
        from heylook_llm.reasoning_parser import PassThroughParser

        parser = PassThroughParser()
        content, thinking = _collect(parser, ["", "x", ""])
        assert content == "x"
        assert thinking == ""


class TestHarmonyChannelParser:
    def test_reproducer_whole_input(self):
        """Full reproducer text fed as a single chunk."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        content, thinking = _collect(parser, [_HARMONY_REPRODUCER])

        assert "<|" not in content, f"control tokens leaked into content: {content!r}"
        assert "<|" not in thinking, f"control tokens leaked into thinking: {thinking!r}"
        assert content == _HARMONY_EXPECTED_CONTENT
        assert thinking == _HARMONY_EXPECTED_THINKING

    def test_reproducer_split_per_character(self):
        """Same input fed one character at a time -- stress-tests partial
        control-token buffering."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        content, thinking = _collect(parser, list(_HARMONY_REPRODUCER))

        assert "<|" not in content
        assert "<|" not in thinking
        assert content == _HARMONY_EXPECTED_CONTENT
        assert thinking == _HARMONY_EXPECTED_THINKING

    def test_split_mid_control_token(self):
        """Buffer must hold partial control tokens across chunk boundaries."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        # Break at every awkward point inside <|channel|>, <|message|>, <|end|>.
        chunks = [
            "<|chan", "nel|>", "final", "<|mess", "age|>",
            "hi", "<|en", "d|>",
        ]
        content, thinking = _collect(parser, chunks)
        assert content == "hi"
        assert thinking == ""

    def test_analysis_to_thinking_final_to_content(self):
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        text = (
            "<|channel|>analysis<|message|>reasoning stuff<|end|>"
            "<|start|>assistant<|channel|>final<|message|>visible answer"
        )
        content, thinking = _collect(parser, [text])
        assert content == "visible answer"
        assert thinking == "reasoning stuff"

    def test_commentary_channel_routes_to_thinking(self):
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        text = (
            "<|channel|>commentary<|message|>side note<|end|>"
            "<|channel|>final<|message|>main"
        )
        content, thinking = _collect(parser, [text])
        assert content == "main"
        assert thinking == "side note"

    def test_unknown_channel_routes_to_content(self):
        """An unexpected channel name must NOT silently vanish; route it to
        content so nothing important is lost if harmony adds new channels."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        text = "<|channel|>novel_channel<|message|>payload<|end|>"
        content, thinking = _collect(parser, [text])
        assert content == "payload"
        assert thinking == ""

    def test_preamble_before_first_control_token_passes_to_content(self):
        """If a harmony model emits free text before any <|channel|>, route
        it to content rather than dropping it. Defensive fallback."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser()
        text = "oops no tokens here"
        content, thinking = _collect(parser, [text])
        assert content == "oops no tokens here"
        assert thinking == ""


class TestThinkingMarkersViaFactory:
    """Factory returns HybridThinkingParser directly when the template has
    ``<think>...</think>`` markers -- no wrapper class. Tests target the
    factory output so the call path stays identical to production."""

    def test_basic_thinking_block(self):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = ModelTemplateInfo(has_thinking_markers=True)
        parser = select_reasoning_parser(info)
        text = "<think>internal reasoning</think>visible answer"
        content, thinking = _collect(parser, [text])
        assert content == "visible answer"
        assert thinking == "internal reasoning"


class TestGemmaChannelParser:
    """Gemma-4 canonical format: ``<|channel>NAME\\n BODY <channel|>`` inline in
    the model turn; the ``thought`` channel is reasoning, text outside channels
    is content."""

    _REPRODUCER = (
        "<|channel>thought\nTopic: sky color. Constraint: two sentences.\n"
        "<channel|>The sky appears blue because of Rayleigh scattering."
    )
    _EXPECTED_THINKING = "Topic: sky color. Constraint: two sentences.\n"
    _EXPECTED_CONTENT = "The sky appears blue because of Rayleigh scattering."

    def _parser(self):
        from heylook_llm.reasoning_parser import GemmaChannelParser
        return GemmaChannelParser()

    def test_reproducer_whole_input(self):
        content, thinking = _collect(self._parser(), [self._REPRODUCER])
        assert content == self._EXPECTED_CONTENT
        assert thinking == self._EXPECTED_THINKING

    def test_reproducer_split_per_character(self):
        content, thinking = _collect(self._parser(), list(self._REPRODUCER))
        assert content == self._EXPECTED_CONTENT
        assert thinking == self._EXPECTED_THINKING

    def test_split_mid_control_token(self):
        chunks = ["<|chan", "nel>thought\nplan", "ning\n<chan", "nel|>Answer."]
        content, thinking = _collect(self._parser(), chunks)
        assert content == "Answer."
        assert thinking == "planning\n"

    def test_unknown_channel_routes_to_content(self):
        content, thinking = _collect(
            self._parser(), ["<|channel>notes\nremember this<channel|>done"]
        )
        assert content == "remember thisdone"
        assert thinking == ""

    def test_plain_text_is_content(self):
        content, thinking = _collect(self._parser(), ["Just a plain answer."])
        assert content == "Just a plain answer."
        assert thinking == ""

    def test_unclosed_thought_flushes_to_thinking(self):
        # aborted stream mid-thought: flush routes the partial body to thinking
        content, thinking = _collect(self._parser(), ["<|channel>thought\nhalf a plan"])
        assert content == ""
        assert thinking == "half a plan"

    def test_abort_mid_close_token_drops_partial(self):
        # gemma's close token starts "<c", which harmony's partial-strip
        # doesn't know -- an abort landing mid-<channel|> must not flush
        # literal garbage like "<chan"
        content, thinking = _collect(
            self._parser(), ["<|channel>thought\nplan text\n<chan"]
        )
        assert thinking == "plan text\n"
        assert content == ""

    def test_abort_mid_open_token_drops_partial(self):
        content, thinking = _collect(self._parser(), ["Answer text<|chann"])
        assert content == "Answer text"
        assert thinking == ""


class TestImplicitThinkOpen:
    """Qwen3.5-style templates PRE-FILL `<think>\\n` into the generation
    prompt when thinking is enabled -- the model's output starts INSIDE the
    think block and never emits the opening tag. The parser must start in
    thinking state or everything routes to content and `</think>` leaks."""

    def _chunks(self):
        return ["I should ", "plan this.", "</think>", "The answer", " is 4."]

    def test_initial_thinking_splits_implicit_open(self):
        from heylook_llm.thinking_parser import HybridThinkingParser

        content, thinking = _collect(
            HybridThinkingParser(initial_thinking=True), self._chunks()
        )
        assert thinking == "I should plan this."
        assert content == "The answer is 4."

    def test_default_explicit_mode_unchanged(self):
        from heylook_llm.thinking_parser import HybridThinkingParser

        content, thinking = _collect(
            HybridThinkingParser(),
            ["<think>", "plan", "</think>", "answer"],
        )
        assert thinking == "plan"
        assert content == "answer"

    def test_reset_restores_initial_state(self):
        from heylook_llm.thinking_parser import HybridThinkingParser

        p = HybridThinkingParser(initial_thinking=True)
        _collect(p, self._chunks())
        p.reset()
        content, thinking = _collect(p, self._chunks())
        assert thinking == "I should plan this."
        assert content == "The answer is 4."

    def test_factory_arms_initial_thinking_from_template_and_request(self):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = ModelTemplateInfo(
            has_thinking_markers=True, prefills_thinking=True,
        )
        armed = select_reasoning_parser(info, thinking_enabled=True)
        content, thinking = _collect(armed, self._chunks())
        assert thinking == "I should plan this."

        # thinking off (or unknown): classic explicit mode
        off = select_reasoning_parser(info, thinking_enabled=False)
        content, thinking = _collect(off, ["plain answer"])
        assert content == "plain answer"
        assert thinking == ""


class TestReasoningParserFactory:
    def _info(self, *, has_harmony=False, has_thinking=False, has_gemma=False, specials=()):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

        return ModelTemplateInfo(
            chat_template="",  # unused by factory directly
            special_tokens=frozenset(specials),
            template_source="jinja",
            has_harmony_structure=has_harmony,
            has_thinking_markers=has_thinking,
            has_gemma_channel_structure=has_gemma,
        )

    def test_gemma_selected_when_template_has_channel_structure(self):
        from heylook_llm.reasoning_parser import (
            GemmaChannelParser, select_reasoning_parser,
        )

        parser = select_reasoning_parser(self._info(has_gemma=True))
        assert isinstance(parser, GemmaChannelParser)

    def test_harmony_wins_over_gemma_structure(self):
        from heylook_llm.reasoning_parser import (
            HarmonyChannelParser, select_reasoning_parser,
        )

        parser = select_reasoning_parser(self._info(has_harmony=True, has_gemma=True))
        assert isinstance(parser, HarmonyChannelParser)

    def test_harmony_selected_when_template_has_harmony_structure(self):
        from heylook_llm.reasoning_parser import (
            HarmonyChannelParser, select_reasoning_parser,
        )

        info = self._info(has_harmony=True, specials=["<|channel|>", "<|message|>"])
        parser = select_reasoning_parser(info)
        assert isinstance(parser, HarmonyChannelParser)

    def test_qwen3_selected_when_template_has_thinking_markers(self):
        from heylook_llm.reasoning_parser import select_reasoning_parser
        from heylook_llm.thinking_parser import HybridThinkingParser

        info = self._info(has_thinking=True, specials=["<think>", "</think>"])
        parser = select_reasoning_parser(info)
        assert isinstance(parser, HybridThinkingParser)

    def test_pass_through_when_nothing_matches(self):
        from heylook_llm.reasoning_parser import (
            PassThroughParser, select_reasoning_parser,
        )

        info = self._info()
        parser = select_reasoning_parser(info)
        assert isinstance(parser, PassThroughParser)

    def test_harmony_wins_over_thinking(self):
        """Template with both harmony channels and <think> markers (unusual)
        -- harmony is more specific, takes precedence."""
        from heylook_llm.reasoning_parser import (
            HarmonyChannelParser, select_reasoning_parser,
        )

        info = self._info(has_harmony=True, has_thinking=True)
        parser = select_reasoning_parser(info)
        assert isinstance(parser, HarmonyChannelParser)

    def test_none_template_info_falls_back_to_pass_through(self):
        from heylook_llm.reasoning_parser import (
            PassThroughParser, select_reasoning_parser,
        )

        parser = select_reasoning_parser(template_info=None)
        assert isinstance(parser, PassThroughParser)

    def test_harmony_parser_strips_non_structural_specials(self):
        """Factory threads the tokenizer-config-declared specials into the
        harmony parser so it strips ANY declared control token -- not just
        the six structural harmony tokens. Assert behaviorally: feed a
        payload with a non-structural special mid-message; it must not
        appear in the output."""
        from heylook_llm.reasoning_parser import (
            HarmonyChannelParser, select_reasoning_parser,
        )

        specials = ["<|channel|>", "<|message|>", "<|start|>", "<|end|>",
                    "<|return|>", "<|call|>", "<|reserved_200000|>"]
        info = self._info(has_harmony=True, specials=specials)
        parser = select_reasoning_parser(info)
        assert isinstance(parser, HarmonyChannelParser)

        text = "<|channel|>final<|message|>hello <|reserved_200000|> world<|return|>"
        content, _ = _collect(parser, [text])
        assert content == "hello  world"
        assert "<|reserved_200000|>" not in content



class TestParseFullText:
    """Non-streaming path: ``parse_reasoning(text, parser)`` -> (content, thinking)."""

    def test_harmony_full_text(self):
        from heylook_llm.reasoning_parser import (
            HarmonyChannelParser, parse_reasoning,
        )

        content, thinking = parse_reasoning(_HARMONY_REPRODUCER, HarmonyChannelParser())
        assert content == _HARMONY_EXPECTED_CONTENT
        assert thinking == _HARMONY_EXPECTED_THINKING

    def test_pass_through_full_text(self):
        from heylook_llm.reasoning_parser import (
            PassThroughParser, parse_reasoning,
        )

        content, thinking = parse_reasoning("just text", PassThroughParser())
        assert content == "just text"
        assert thinking is None


class TestStripTokensDefense:
    """EVERY selectable parser gets a ``strip_tokens`` set so any special
    token the detokenizer leaks (or the model emits mid-payload) gets
    cleaned out before the delta reaches the user."""

    def test_hybrid_thinking_strips_declared_specials(self):
        from heylook_llm.thinking_parser import HybridThinkingParser

        parser = HybridThinkingParser(
            strip_tokens=frozenset(["<|reserved_200000|>"])
        )
        out = []
        for ch in ["<think>", "plan <|reserved_200000|> here", "</think>", "answer <|reserved_200000|>"]:
            out += parser.process_chunk(ch)
        out += parser.flush()
        thinking = "".join(t for k, t in out if k == "thinking")
        content = "".join(t for k, t in out if k == "content")
        assert "<|reserved_200000|>" not in thinking
        assert "<|reserved_200000|>" not in content
        assert "plan" in thinking and "answer" in content

    def test_factory_threads_strip_tokens_to_hybrid(self):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = ModelTemplateInfo(
            has_thinking_markers=True,
            special_tokens=frozenset(["<|im_end|>"]),
        )
        parser = select_reasoning_parser(info)
        out = parser.process_chunk("hello <|im_end|>") + parser.flush()
        assert all("<|im_end|>" not in t for _, t in out)

    def test_pass_through_strips_declared_specials(self):
        from heylook_llm.reasoning_parser import PassThroughParser

        parser = PassThroughParser(
            strip_tokens=frozenset(["<|endoftext|>", "<|reserved_200000|>"])
        )
        out, _ = _collect(parser, ["hello <|endoftext|> world <|reserved_200000|>"])
        assert out == "hello  world "

    def test_harmony_strips_non_structural_specials_in_message_body(self):
        """A reserved token that sneaks into message payload gets stripped;
        the structural tokens are consumed by the state machine."""
        from heylook_llm.reasoning_parser import HarmonyChannelParser

        parser = HarmonyChannelParser(
            strip_tokens=frozenset(["<|reserved_200000|>"])
        )
        text = (
            "<|channel|>final<|message|>"
            "hello <|reserved_200000|> world"
            "<|return|>"
        )
        content, _ = _collect(parser, [text])
        assert content == "hello  world"
        assert "<|reserved_200000|>" not in content
