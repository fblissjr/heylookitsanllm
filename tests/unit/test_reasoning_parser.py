"""Tests for the format-aware reasoning parser.

Covers the parser implementations + factory:

  PassThroughParser     -- no reasoning structure; text -> content
  HarmonyChannelParser  -- OpenAI harmony multi-channel format:
                           <|start|>ROLE<|channel|>NAME<|message|>CONTENT<|end|>
                           analysis/commentary -> thinking; final -> content;
                           control tokens stripped.
  GemmaChannelParser    -- gemma-4 inline channels: <|channel>NAME\\n BODY <channel|>
  HybridThinkingParser  -- <think>...</think> markers (from thinking_parser.py).
  StripSpecials         -- the ONE declared-specials filter + holdback, composed
                           over whichever of the above the factory picks.

Factory selects based on ``ModelTemplateInfo`` flags derived from the
model's chat template + tokenizer config.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import pytest

from heylook_llm.reasoning_parser import (
    GemmaChannelParser,
    HarmonyChannelParser,
    PassThroughParser,
    StripSpecials,
    select_reasoning_parser,
)
from heylook_llm.thinking_parser import HybridThinkingParser


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

# Gemma-4 canonical format: ``<|channel>NAME\n BODY <channel|>`` inline in
# the model turn; the ``thought`` channel is reasoning, text outside channels
# is content.
_GEMMA_REPRODUCER = (
    "<|channel>thought\nTopic: sky color. Constraint: two sentences.\n"
    "<channel|>The sky appears blue because of Rayleigh scattering."
)
_GEMMA_EXPECTED_THINKING = "Topic: sky color. Constraint: two sentences.\n"
_GEMMA_EXPECTED_CONTENT = "The sky appears blue because of Rayleigh scattering."


def _template(*, harmony=False, gemma=False, thinking=False, specials=()):
    """ModelTemplateInfo stand-in for factory-driven tests."""
    from heylook_llm.providers.common.template_info import ModelTemplateInfo

    return ModelTemplateInfo(
        chat_template="",
        special_tokens=frozenset(specials),
        template_source="jinja",
        has_harmony_structure=harmony,
        has_gemma_channel_structure=gemma,
        has_thinking_markers=thinking,
    )


def _core(parser):
    """The routing parser under the shared declared-specials filter.

    ``select_reasoning_parser`` composes ``StripSpecials(inner, ...)`` when
    the model declares specials; structural assertions target the inner
    parser, behavioral ones target the composed object."""
    return parser.inner if isinstance(parser, StripSpecials) else parser


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


class _Pinned(NamedTuple):
    """A TestParserInvariants.CORPUS row whose split is pinned: the stream
    as these exact chunks must give (content, thinking), and so must the
    whole-input parse and every random split of the joined text."""

    chunks: tuple
    content: str
    thinking: str = ""


class TestHarmonyChannelParser:
    def test_reproducer_whole_input(self):
        """Full reproducer text fed as a single chunk."""
        parser = HarmonyChannelParser()
        content, thinking = _collect(parser, [_HARMONY_REPRODUCER])

        assert "<|" not in content, f"control tokens leaked into content: {content!r}"
        assert "<|" not in thinking, f"control tokens leaked into thinking: {thinking!r}"
        assert content == _HARMONY_EXPECTED_CONTENT
        assert thinking == _HARMONY_EXPECTED_THINKING

    def test_commentary_channel_routes_to_thinking(self):
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
        parser = HarmonyChannelParser()
        text = "<|channel|>novel_channel<|message|>payload<|end|>"
        content, thinking = _collect(parser, [text])
        assert content == "payload"
        assert thinking == ""


class TestGemmaChannelParser:
    """Gemma-4 canonical format (see ``_GEMMA_REPRODUCER``)."""

    def _parser(self):
        return GemmaChannelParser()

    def test_reproducer_whole_input(self):
        content, thinking = _collect(self._parser(), [_GEMMA_REPRODUCER])
        assert content == _GEMMA_EXPECTED_CONTENT
        assert thinking == _GEMMA_EXPECTED_THINKING

    def test_unknown_channel_routes_to_content(self):
        content, thinking = _collect(
            self._parser(), ["<|channel>notes\nremember this<channel|>done"]
        )
        assert content == "remember thisdone"
        assert thinking == ""

    def test_unclosed_thought_flushes_to_thinking(self):
        # aborted stream mid-thought: flush routes the partial body to thinking
        content, thinking = _collect(self._parser(), ["<|channel>thought\nhalf a plan"])
        assert content == ""
        assert thinking == "half a plan"


class TestAbortMidControlToken:
    """The final flush emits the WHOLE buffer, so a trailing partial control
    token must be dropped BEFORE the drain: an abort landing inside a control
    token would otherwise flush literal garbage like ``<|en`` or ``<chan``.
    Gemma's close token starts "<c", which harmony's partial-strip does not
    know, so each family needs its own rows."""

    @pytest.mark.parametrize(
        "parser_cls, chunks, expected",
        [
            (HarmonyChannelParser, ["<|channel|>final<|message|>answer text<|en"],
             ("answer text", "")),
            (HarmonyChannelParser, ["free text<|st"], ("free text", "")),
            (GemmaChannelParser, ["<|channel>thought\nplan text\n<chan"],
             ("", "plan text\n")),
            (GemmaChannelParser, ["Answer text<|chann"], ("Answer text", "")),
        ],
        ids=["harmony-end", "harmony-preamble", "gemma-close", "gemma-open"],
    )
    def test_abort_mid_control_token_drops_partial(self, parser_cls, chunks, expected):
        assert _collect(parser_cls(), chunks) == expected


class TestImplicitThinkOpen:
    """Qwen3.5-style templates PRE-FILL `<think>\\n` into the generation
    prompt when thinking is enabled -- the model's output starts INSIDE the
    think block and never emits the opening tag. The parser must start in
    thinking state or everything routes to content and `</think>` leaks."""

    def _chunks(self):
        return ["I should ", "plan this.", "</think>", "The answer", " is 4."]

    def test_initial_thinking_splits_implicit_open(self):
        content, thinking = _collect(
            HybridThinkingParser(initial_thinking=True), self._chunks()
        )
        assert thinking == "I should plan this."
        assert content == "The answer is 4."

    def test_reset_restores_initial_state(self):
        p = HybridThinkingParser(initial_thinking=True)
        _collect(p, self._chunks())
        p.reset()
        content, thinking = _collect(p, self._chunks())
        assert thinking == "I should plan this."
        assert content == "The answer is 4."

    def test_factory_arms_initial_thinking_from_template_and_request(self):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo

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
    """Template flags (and whether the model declares specials) pick the
    parser class and whether it is composed under ``StripSpecials``.

    Harmony is the most specific structure and wins over gemma channels and
    over ``<think>`` markers (a template with both is unusual). No strip set
    means no wrapper: the filter exists only for stripping. ``None`` template
    info falls back to pass-through."""

    @pytest.mark.parametrize(
        "info, parser_cls, composed",
        [
            (_template(gemma=True), GemmaChannelParser, False),
            (_template(harmony=True, gemma=True), HarmonyChannelParser, False),
            (_template(harmony=True, specials=["<|channel|>", "<|message|>"]),
             HarmonyChannelParser, True),
            (_template(thinking=True, specials=["<think>", "</think>"]),
             HybridThinkingParser, True),
            (_template(), PassThroughParser, False),
            (_template(harmony=True, thinking=True), HarmonyChannelParser, False),
            (None, PassThroughParser, False),
            (_template(harmony=True), HarmonyChannelParser, False),
        ],
        ids=[
            "gemma-channels", "harmony-over-gemma", "harmony-with-specials",
            "think-markers-with-specials", "nothing-matches",
            "harmony-over-thinking", "no-template-info",
            "no-specials-leaves-uncomposed",
        ],
    )
    def test_template_flags_pick_the_parser(self, info, parser_cls, composed):
        parser = select_reasoning_parser(template_info=info)
        assert isinstance(_core(parser), parser_cls)
        assert isinstance(parser, StripSpecials) is composed


class TestParseFullText:
    """Non-streaming path: ``parse_reasoning(text, parser)`` -> (content, thinking).
    Pass-through reports no reasoning as ``None``, not ``""``."""

    @pytest.mark.parametrize(
        "text, parser_cls, expected",
        [
            (_HARMONY_REPRODUCER, HarmonyChannelParser,
             (_HARMONY_EXPECTED_CONTENT, _HARMONY_EXPECTED_THINKING)),
            ("just text", PassThroughParser, ("just text", None)),
        ],
        ids=["harmony", "pass-through"],
    )
    def test_full_text(self, text, parser_cls, expected):
        from heylook_llm.reasoning_parser import parse_reasoning

        assert parse_reasoning(text, parser_cls()) == expected


class TestStripTokensDefense:
    """EVERY selectable parser is composed under ``StripSpecials`` when the
    model declares specials, so any special token the detokenizer leaks (or
    the model emits mid-payload) is cleaned out before the delta reaches the
    user.

    Rows: the factory threads tokenizer-config-declared specials into the
    harmony parser so it strips ANY declared control token, not just the six
    structural harmony tokens (those are consumed by the state machine); the
    factory threads them to the hybrid parser too; and a hand-composed filter
    strips over hybrid, pass-through and a harmony message body."""

    _HARMONY_SPECIALS = [
        "<|channel|>", "<|message|>", "<|start|>", "<|end|>",
        "<|return|>", "<|call|>", "<|reserved_200000|>",
    ]

    @pytest.mark.parametrize(
        "build, specials, chunks, expected",
        [
            (lambda s: select_reasoning_parser(_template(harmony=True, specials=s)),
             _HARMONY_SPECIALS,
             ["<|channel|>final<|message|>hello <|reserved_200000|> world<|return|>"],
             ("hello  world", "")),
            (lambda s: StripSpecials(HybridThinkingParser(), frozenset(s)),
             ["<|reserved_200000|>"],
             ["<think>", "plan <|reserved_200000|> here", "</think>",
              "answer <|reserved_200000|>"],
             ("answer ", "plan  here")),
            (lambda s: select_reasoning_parser(_template(thinking=True, specials=s)),
             ["<|im_end|>"], ["hello <|im_end|>"], ("hello ", "")),
            (lambda s: StripSpecials(PassThroughParser(), frozenset(s)),
             ["<|endoftext|>", "<|reserved_200000|>"],
             ["hello <|endoftext|> world <|reserved_200000|>"],
             ("hello  world ", "")),
            (lambda s: StripSpecials(HarmonyChannelParser(), frozenset(s)),
             ["<|reserved_200000|>"],
             ["<|channel|>final<|message|>hello <|reserved_200000|> world<|return|>"],
             ("hello  world", "")),
        ],
        ids=[
            "factory-harmony-non-structural", "hybrid", "factory-threads-to-hybrid",
            "pass-through", "harmony-message-body",
        ],
    )
    def test_declared_specials_never_reach_either_channel(
        self, build, specials, chunks, expected
    ):
        content, thinking = _collect(build(specials), chunks)
        assert (content, thinking) == expected
        for special in specials:
            assert special not in content and special not in thinking


class TestStripSpecialsOptOut:
    """``strip_specials=False`` composes no filter at all.

    NO PRODUCTION CALLER passes False since v2.0.38, which removed the
    `show_special_tokens` request field this served. The parser-level knob is
    kept deliberately and these pin it: it is the seam a correct version of
    that feature needs -- storing the model's output UNSTRIPPED and stripping
    at READ instead, so the toggle applies to replies that already exist
    (design + its three traps in docs/project/TODO.md). Delete the knob and
    these together if that design is ever abandoned."""

    def test_declared_specials_survive_when_not_stripping(self):
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = _template(thinking=True, specials=["<|im_end|>"])
        parser = select_reasoning_parser(info, strip_specials=False)
        out = parser.process_chunk("hello <|im_end|>") + parser.flush()
        assert "".join(t for _, t in out) == "hello <|im_end|>"

    def test_default_still_strips(self):
        """The opt-out is opt-IN: the same template with no flag strips, so a
        client that says nothing (every OpenAI-compat consumer) is unchanged."""
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = _template(thinking=True, specials=["<|im_end|>"])
        parser = select_reasoning_parser(info)
        out = parser.process_chunk("hello <|im_end|>") + parser.flush()
        assert "".join(t for _, t in out) == "hello "

    def test_routing_is_unaffected(self):
        """Structural tokens are NOT declared-specials handling: the parser
        consumes ``<think>`` to know what is thinking, and that must keep
        working with the filter off (otherwise "show specials" would silently
        dump the reasoning into the answer)."""
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = _template(thinking=True, specials=["<|im_end|>"])
        parser = select_reasoning_parser(info, strip_specials=False)
        out = []
        for ch in ["<think>", "plan <|im_end|>", "</think>", "answer <|im_end|>"]:
            out += parser.process_chunk(ch)
        out += parser.flush()
        thinking = "".join(t for k, t in out if k == "thinking")
        content = "".join(t for k, t in out if k == "content")
        assert thinking == "plan <|im_end|>"
        assert content == "answer <|im_end|>"

    def test_holdback_does_not_swallow_the_tail(self):
        """With the filter composed, a trailing prefix of a special is HELD
        pending; with it off there is no holdback at all, so a reply ending in
        a bare ``<`` must still arrive."""
        from heylook_llm.reasoning_parser import select_reasoning_parser

        info = _template(specials=["<|im_end|>"])
        parser = select_reasoning_parser(info, strip_specials=False)
        out = parser.process_chunk("done <") + parser.flush()
        assert "".join(t for _, t in out) == "done <"


class TestSharedStripHoldback:
    """One holdback for all four parsers, sized by the STRIP SET.

    Before the unification each parser sized (or skipped) its own holdback:
    harmony/gemma held back only enough for their own STRUCTURAL tokens
    (<= 10 chars), so a longer declared special straddling an emit boundary
    leaked; pass-through had no holdback at all. The shared filter holds
    back the longest tail that could still grow into a declared special.
    The straddling cases themselves are pinned rows of
    TestParserInvariants.CORPUS.
    """

    def test_held_tail_that_never_completes_is_emitted(self):
        """Holdback must not swallow text: a tail that looked like a partial
        special but never completes comes out at flush."""
        parser = select_reasoning_parser(_template(specials=["<|endoftext|>"]))
        content, _ = _collect(parser, ["price < ", "x <|end"])
        assert content == "price < x <|end"

    def test_reset_clears_held_tail(self):
        parser = select_reasoning_parser(_template(specials=["<|endoftext|>"]))
        parser.process_chunk("stale <|end")
        parser.reset()
        content, _ = _collect(parser, ["fresh"])
        assert content == "fresh"

    def test_kind_change_flushes_held_tail_in_order(self):
        parser = select_reasoning_parser(
            _template(thinking=True, specials=["<|endoftext|>"])
        )
        content, thinking = _collect(
            parser, ["<think>plan <|end", "</think>", "answer"]
        )
        assert thinking == "plan <|end"
        assert content == "answer"


class TestUnterminatedChannelHeaderIsNotSwallowed:
    """A channel opened but never named/closed must not eat the turn.

    Live find (2026-07-23): gemma-4 sometimes emits a spurious `<|channel>`
    mid-answer and then just keeps answering -- raw tokens
    ``['<|channel>', ' to', ' the', ' movies', '!', '<turn|>']``. Both channel
    parsers accumulated everything after the open token into the channel-NAME
    buffer and dropped it as structural at flush, so the user got an empty
    reply while the server reported a normal 6-token `stop`. That is the
    "immediate empty-EOS" long attributed to model behavior; it is text loss.

    Rule: at end of turn, unrouted model text goes to content. Never drop it.
    (The per-character stream of the spurious open, and the well-formed
    headers that must still be consumed, are pinned CORPUS rows.)
    """

    def test_gemma_spurious_channel_open_keeps_the_answer(self):
        content, thinking = _collect(
            GemmaChannelParser(), ["<|channel> to the movies!"]
        )
        assert content == " to the movies!"
        assert thinking == ""

    def test_gemma_abort_mid_channel_name_surfaces_the_fragment(self):
        """The trade-off, stated explicitly: an abort inside a LEGIT header
        now surfaces a short fragment instead of vanishing. Losing a whole
        answer is the worse failure."""
        content, _ = _collect(GemmaChannelParser(), ["<|channel>thou"])
        assert content == "thou"

    def test_harmony_unterminated_channel_header_keeps_the_text(self):
        content, thinking = _collect(
            HarmonyChannelParser(), ["<|channel|>analysis and then some answer"]
        )
        assert content == "analysis and then some answer"
        assert thinking == ""


class TestParserInvariants:
    """Two properties that hold for EVERY parser, checked over random inputs.

    These exist because both 2026-07-23 parser bugs were failures of an
    unstated invariant, not of a missing case. Example-based tests can only
    pin the boundaries someone thought to write down; a property says what
    must be true everywhere, and randomised splits explore the chunk
    boundaries that are exactly where streaming parsers break.

    A ``_Pinned`` row also pins one split and its exact output: the
    reproducers streamed per character, splits inside control tokens, and
    declared specials straddling an emit boundary (finding #1).
    """

    CORPUS = {
        "harmony": [
            "<|channel|>analysis<|message|>reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>the answer",
            "<|channel|>final<|message|>hi <|reserved_200000|> there<|return|>",
            "free preamble text with no tokens",
            "<|channel|>final<|message|>answer text<|en",
            "<|channel|>analysis and then it just keeps going",
            # the reproducer one character at a time: partial control tokens
            # buffered at every position
            _Pinned(tuple(_HARMONY_REPRODUCER),
                    _HARMONY_EXPECTED_CONTENT, _HARMONY_EXPECTED_THINKING),
            # broken at every awkward point inside <|channel|>, <|message|>, <|end|>
            _Pinned(("<|chan", "nel|>", "final", "<|mess", "age|>", "hi", "<|en", "d|>"),
                    "hi"),
            _Pinned(("<|channel|>analysis<|message|>reasoning stuff<|end|>"
                     "<|start|>assistant<|channel|>final<|message|>visible answer",),
                    "visible answer", "reasoning stuff"),
            # a long declared special straddling the emit boundary in a message
            _Pinned(("<|channel|>final<|message|>", "hello world padding<|reserved_2",
                     "00000|> tail"),
                    "hello world padding tail"),
            # a well-formed header is structural and must NOT leak as content
            _Pinned(("<|channel|>analysis<|message|>reasoning<|end|>"
                     "<|channel|>final<|message|>answer",),
                    "answer", "reasoning"),
        ],
        "gemma": [
            "<|channel>thought\nplanning\n<channel|>The answer.",
            "padding text<|reserved_200000|> tail",
            "<|channel> to the movies!",
            "Answer text<|chann",
            "<|channel>thought\nhalf a plan",
            _Pinned(tuple(_GEMMA_REPRODUCER),
                    _GEMMA_EXPECTED_CONTENT, _GEMMA_EXPECTED_THINKING),
            _Pinned(("<|chan", "nel>thought\nplan", "ning\n<chan", "nel|>Answer."),
                    "Answer.", "planning\n"),
            _Pinned(("padding text here<|reserved_2", "00000|> tail"),
                    "padding text here tail"),
            # the same straddle inside a thought
            _Pinned(("<|channel>thought\npadding text here<|reserved_2",
                     "00000|> more\n<channel|>Answer."),
                    "Answer.", "padding text here more\n"),
            # the 2026-07-23 spurious channel-open, streamed per character
            _Pinned(tuple("Hi<|channel> to the movies!"), "Hi to the movies!"),
            _Pinned(("<|channel>thought\nplanning<channel|>Answer.",),
                    "Answer.", "planning"),
        ],
        "think": [
            "<think>plan</think>answer",
            "<think>plan <|endoftext|> more</think>done [INST] x",
            "no markers at all",
            # the inner parser's own buffering splits a declared special
            # across deltas, so a per-delta sub() would miss both halves
            _Pinned(("<think>plan</think>done <|reserved_2", "00000|> now"),
                    "done  now", "plan"),
            # tags split across chunks
            _Pinned(("<thi", "nk>", "Thinking", "</th", "ink>", "Content"),
                    "Content", "Thinking"),
            _Pinned(("<think>internal reasoning</think>visible answer",),
                    "visible answer", "internal reasoning"),
            # explicit mode (no prefill) is the default
            _Pinned(("<think>", "plan", "</think>", "answer"), "answer", "plan"),
            _Pinned(("<think>", "Reasoning", "</think>", "Answer"), "Answer", "Reasoning"),
            _Pinned((f"<think>{'A' * 10000}</think>Short answer",),
                    "Short answer", "A" * 10000),
            # text mode: no token ids passed
            _Pinned(("<think>", "Thinking", "</think>", "Answer"), "Answer", "Thinking"),
        ],
        "plain": [
            "just some text",
            "text with <|endoftext|> a special",
            "mistral style [INST] marker",
            _Pinned(("bye <|end", "oftext|> now"), "bye  now"),
            # Mistral-family specials are [INST]-shaped: a holdback that only
            # scans for '<' leaks them across a chunk boundary
            _Pinned(("hi [IN", "ST] there"), "hi  there"),
            # empty chunks are ignored
            _Pinned(("", "x", ""), "x"),
        ],
    }

    # Text a model can legitimately produce that contains NO structural token
    # of any parser -- including the shapes that tempt a partial-token guess.
    TOKEN_FREE = [
        "The quick brown fox jumps over the lazy dog.",
        "5 < 3 is false, and 3 > 1 is true.",
        "A line\nwith newlines\nand   spacing.",
        "Trailing angle bracket <",
        "brackets [like these] and <these>",
        "generic<T> in code",
        "hello world",
        # a harmony model's free text before any <|channel|> is content
        "oops no tokens here",
        "Just a plain answer.",
        # a channel parser with no resume starts in content
        "plain text",
        "Hello world!",
        "Hello world",
        "Compare x<y and y>z in the expression",
        "<thin some random text with angle brackets",
    ]
    # Exact feeds of the tests these rows replaced, kept as-is.
    TOKEN_FREE_SPLITS = [
        ("hello ", "world"),
        ("Hello", " world", "!"),
        ("Hello", " world"),
    ]

    def _parser(self, kind):
        specials = ["<|reserved_200000|>", "<|endoftext|>", "[INST]", "<turn|>"]
        flags = {
            "harmony": dict(harmony=True), "gemma": dict(gemma=True),
            "think": dict(thinking=True), "plain": {},
        }[kind]
        return select_reasoning_parser(_template(specials=specials, **flags))

    def _splits(self, text, rng):
        """A random chopping of `text` into consecutive chunks."""
        cuts = sorted(rng.sample(range(1, len(text)), min(rng.randint(1, 6), len(text) - 1)))
        parts, prev = [], 0
        for cut in cuts:
            parts.append(text[prev:cut])
            prev = cut
        parts.append(text[prev:])
        return [p for p in parts if p]

    def test_output_is_invariant_to_chunking(self):
        """How the stream was chopped must not change what the user sees.

        This is the property finding #1 violated: harmony/gemma sized their
        holdback to their own structural tokens, so a longer declared special
        landing across an emit boundary leaked. Seeded, so a failure is
        reproducible rather than a heisenbug.
        """
        rng = random.Random(20260723)
        for kind, rows in self.CORPUS.items():
            for row in rows:
                text = row if isinstance(row, str) else "".join(row.chunks)
                whole = _collect(self._parser(kind), [text])
                if isinstance(row, _Pinned):
                    expected = (row.content, row.thinking)
                    assert whole == expected, f"[{kind}] {text[:60]!r}: {whole}"
                    assert _collect(self._parser(kind), row.chunks) == expected, (
                        f"[{kind}] pinned split changed the output: {row.chunks[:8]}"
                    )
                for _ in range(40):
                    chunks = self._splits(text, rng)
                    assert _collect(self._parser(kind), chunks) == whole, (
                        f"[{kind}] chunking changed the output for {text!r}: "
                        f"whole={whole} chunks={chunks}"
                    )

    def test_text_with_no_structural_tokens_survives_intact(self):
        """No silent loss: if the model emitted no structure, every character
        it produced reaches content.

        This is the property finding #2 violated (a spurious channel-open ate
        the turn), and the one that keeps `_strip_partial_token`'s guess
        honest -- a bare trailing '<' is content, not a truncated token.
        """
        rng = random.Random(20260724)
        for kind in self.CORPUS:
            for text in self.TOKEN_FREE:
                # whole (one chunk, as most replaced tests fed it) and chopped
                for feed in ([text], self._splits(text, rng)):
                    content, thinking = _collect(self._parser(kind), feed)
                    assert content == text, f"[{kind}] lost text: {feed!r} -> {content!r}"
                    assert thinking == ""
            for chunks in self.TOKEN_FREE_SPLITS:
                assert _collect(self._parser(kind), list(chunks)) == ("".join(chunks), ""), (
                    f"[{kind}] pinned split lost text: {chunks}")


class TestChannelParsersResumeInsideThinking:
    """v1.79.63: a mid-thought resume re-opens the channel in the PROMPT and
    appends the partial trace, so the stream's first token is reasoning.
    Both channel parsers take `initial_thinking` for it; reset() keeps it.
    (That a channel parser with no resume starts in content is a TOKEN_FREE
    row of TestParserInvariants.)"""

    def test_gemma_starts_inside_thought(self):
        parser = GemmaChannelParser(initial_thinking=True)
        content, thinking = _collect(parser, [" and so on.\n<channel|>The answer."])
        assert thinking == " and so on.\n"
        assert content == "The answer."
        parser.reset()
        content, thinking = _collect(parser, ["more<channel|>x"])
        assert (thinking, content) == ("more", "x")

    def test_harmony_starts_inside_analysis(self):
        parser = HarmonyChannelParser(initial_thinking=True)
        content, thinking = _collect(parser, [
            " keep reasoning<|end|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"])
        assert thinking == " keep reasoning"
        assert content == "Done."

    def test_factory_arms_the_channel_parsers(self):
        class Gemma:
            has_harmony_structure = False
            has_gemma_channel_structure = True
            has_thinking_markers = False
            prefills_thinking = False
            special_tokens = frozenset()

        class Harmony(Gemma):
            has_harmony_structure = True
            has_gemma_channel_structure = False

        g = select_reasoning_parser(Gemma(), thinking_enabled=True, continuing=True,
                                    resumes_thinking=True)
        assert isinstance(g, GemmaChannelParser)
        assert _collect(g, ["t<channel|>c"]) == ("c", "t")
        h = select_reasoning_parser(Harmony(), thinking_enabled=True, continuing=True,
                                    resumes_thinking=True)
        assert isinstance(h, HarmonyChannelParser)
        assert _collect(h, ["t<|end|>"]) == ("", "t")
