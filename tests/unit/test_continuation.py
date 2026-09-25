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
    """Auto = a trailing assistant message continues; an explicit flag wins
    either way (True continues any role, False never continues)."""

    @pytest.mark.parametrize(
        "messages, flag, expected",
        [
            (ASSISTANT_LAST, None, True),
            (USER_LAST, None, False),
            (USER_LAST, True, True),
            (ASSISTANT_LAST, False, False),
        ],
        ids=["auto-trailing-assistant-continues", "auto-trailing-user-does-not",
             "explicit-true-continues-any-role", "explicit-false-never-continues"],
    )
    def test_is_continuation(self, messages, flag, expected):
        assert _req(messages, flag=flag).is_continuation() is expected


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


class TestThinkingResume:
    """A continue whose final assistant message has thinking and no content
    resumes INSIDE the thinking block (v1.79.62): the turn is re-rendered as
    a generation prompt and the partial trace appended after the opener."""

    def _req(self, messages, flag=None):
        from heylook_llm.config import ChatRequest
        return ChatRequest(model="m", messages=messages, continue_final_message=flag)

    def test_resume_shape_is_thinking_and_no_content(self):
        from heylook_llm.providers.mlx_provider import _thinking_resume
        req = self._req([{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "", "thinking": "so far"}])
        assert _thinking_resume(req) == "so far"
        # content present -> the content is what continues, not the thought
        req = self._req([{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "ans", "thinking": "so far"}])
        assert _thinking_resume(req) is None
        # not a continuation at all
        req = self._req([{"role": "user", "content": "q"}])
        assert _thinking_resume(req) is None

    def test_the_request_owns_the_predicate(self):
        """ChatRequest.resumes_thinking is the ONE spelling: the MLX provider,
        the route's parser selection and the prompt preview all read it, so
        the three hand-copied versions cannot drift apart again."""
        assistant = {"role": "assistant", "content": "", "thinking": "so far"}
        assert self._req([{"role": "user", "content": "q"}, assistant]).resumes_thinking()
        # whitespace-only content is no content, as a str or as parts
        spaced = dict(assistant, content="  \n")
        assert self._req([{"role": "user", "content": "q"}, spaced]).resumes_thinking()
        parts = dict(assistant, content=[{"type": "text", "text": " "}])
        assert self._req([{"role": "user", "content": "q"}, parts]).resumes_thinking()
        # real content: the CONTENT continues, the thought is closed
        answered = dict(assistant, content=[{"type": "text", "text": "ans"}])
        assert not self._req([{"role": "user", "content": "q"}, answered]).resumes_thinking()
        # no thinking to resume
        assert not self._req([{"role": "user", "content": "q"},
                              {"role": "assistant", "content": ""}]).resumes_thinking()
        # an explicit "do not continue" wins over the trailing-assistant shape
        assert not self._req([{"role": "user", "content": "q"}, assistant],
                             flag=False).resumes_thinking()
        # a user-role continuation is never a thinking resume
        assert not self._req([{"role": "user", "content": "q"}], flag=True).resumes_thinking()

    # The resume appends the family's opener unless the prompt already ends
    # in it. Qwen3.5+: the generation prompt already ends in an open <think>;
    # Qwen3: the model emits <think> itself, so the opener is added.
    # v1.79.63: gemma re-opens <|channel>thought, harmony the analysis
    # channel; the matching parser is armed to start inside it.
    @pytest.mark.parametrize(
        "cases",
        [
            [("think", "<|im_start|>assistant\n<think>\n", "  so far",
              "<|im_start|>assistant\n<think>\nso far"),
             ("think", "<|im_start|>assistant\n", "so far",
              "<|im_start|>assistant\n<think>\nso far")],
            [("gemma", "<|turn>model\n", "so far", "<|turn>model\n<|channel>thought\nso far"),
             ("harmony", "<|start|>assistant", "so far",
              "<|start|>assistant<|channel|>analysis<|message|>so far")],
        ],
        ids=["append-after-an-already-open-block", "channel-families-reopen-their-own-channel"],
    )
    def test_resume_appends_the_familys_opener(self, cases):
        from types import SimpleNamespace
        from heylook_llm.providers.mlx_provider import _append_thinking_resume
        for family, prompt, trace, expected in cases:
            info = SimpleNamespace(has_thinking_markers=family == "think",
                                   has_harmony_structure=family == "harmony",
                                   has_gemma_channel_structure=family == "gemma")
            assert _append_thinking_resume(prompt, trace, info) == expected, family

    def test_a_template_with_no_thinking_structure_refuses_loudly(self):
        from types import SimpleNamespace
        from heylook_llm.providers.base import InvalidGenerationRequest
        from heylook_llm.providers.mlx_provider import _append_thinking_resume
        plain = SimpleNamespace(has_thinking_markers=False, has_harmony_structure=False,
                                has_gemma_channel_structure=False)
        for info in (plain, None):
            with pytest.raises(InvalidGenerationRequest, match="not supported"):
                _append_thinking_resume("<|start|>assistant", "so far", info)

    def test_parser_starts_in_thinking_state_on_a_resume(self):
        from heylook_llm.reasoning_parser import select_reasoning_parser, parse_reasoning
        info = _PrefillingTemplate()
        parser = select_reasoning_parser(info, thinking_enabled=True, continuing=True,
                                         resumes_thinking=True)
        content, thinking = parse_reasoning(" and on</think>answer", parser)
        assert thinking == " and on"
        assert content == "answer"


class TestContinuationKeepsTheSeamSpace:
    """v1.79.64: mlx-lm's streaming detokenizers drop a leading space on the
    first text they flush while their buffer is empty -- right for a fresh
    turn, wrong for a continuation, where the first token completes "First I"
    and the space in " need" is real. The context manager seeds the buffer
    so the trim never fires, and restores the factory afterwards."""

    # Driven through the real source (detokenizer_source picks the vendored
    # SPMStreamingDetokenizer from tokenizer.json's decoder), since a rename
    # of the prototype attribute would silently stop the seeding.
    @pytest.mark.parametrize(
        "continuing, raises, expected",
        [
            (False, False, ("need", " the")),
            (True, False, (" need", " the")),
            (True, True, None),
        ],
        ids=["fresh-turn-still-trims", "continuation-keeps-the-first-space",
             "restored-even-when-the-generation-raises"],
    )
    def test_seam_space(self, tmp_path, continuing, raises, expected):
        import json
        from heylook_llm.providers.common.generation_core import (
            continuation_detokenizer, detokenizer_source)
        from heylook_llm.providers.common.lm_detokenizer import SPMStreamingDetokenizer

        (tmp_path / "tokenizer.json").write_text(json.dumps({"decoder": _SPM_DECODER}))
        source = detokenizer_source(_FakeSpmTokenizer(), tmp_path)
        prototype = source._detokenizer
        assert type(prototype) is SPMStreamingDetokenizer
        if raises:
            with pytest.raises(RuntimeError):
                with continuation_detokenizer(source, continuing):
                    raise RuntimeError("mid-generation")
        else:
            with continuation_detokenizer(source, continuing):
                d = source.detokenizer   # what stream_generate does, once
                d.reset()
                d.add_token(1)
                first = d.last_segment
                d.add_token(2)
                second = d.last_segment
            assert (first, second) == expected
            assert "\x00" not in first + second
        assert source._detokenizer is prototype  # restored in finally

    def test_a_read_only_text_detokenizer_is_left_alone(self):
        """The runtime shape on the mlx-vlm path: a raw HF tokenizer wrapped
        by detokenizer_source without a model_path takes the DEFAULT,
        NaiveStreamingDetokenizer, whose `text` is a property with no setter.
        v1.79.64 seeded it and raised AttributeError inside the first next()
        of every continuation on every mlx-vlm-loaded model. It never trims
        a leading space either, so leaving it alone is also the right
        output."""
        from heylook_llm.providers.common.generation_core import (
            continuation_detokenizer, detokenizer_source)
        from heylook_llm.providers.common.lm_detokenizer import NaiveStreamingDetokenizer

        wrapper = detokenizer_source(_FakeHfForDetok())
        assert type(wrapper._detokenizer) is NaiveStreamingDetokenizer
        with continuation_detokenizer(wrapper, True):
            d = wrapper.detokenizer
            d.reset()
            d.add_token(1)
            first = d.last_segment
            d.add_token(2)
            second = d.last_segment
        assert (first, second) == (" need", " the")
        assert type(wrapper._detokenizer) is NaiveStreamingDetokenizer


_SPM_DECODER = {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
    {"type": "ByteFallback"}, {"type": "Fuse"},
    {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}


class _FakeSpmTokenizer:
    """A raw-HF-tokenizer stand-in the vendored SPM detokenizer can build its
    token map from: id 1 is "\u2581need", id 2 is "\u2581the"."""
    _tokens = ["<unk>", "\u2581need", "\u2581the"]

    def __len__(self):
        return len(self._tokens)

    def convert_ids_to_tokens(self, ids):
        return [self._tokens[i] for i in ids]


class _FakeHfForDetok:
    """A raw-HF-tokenizer stand-in the naive detokenizer can drive: decode by
    id, and the encode/decode round trip it probes at construction."""
    clean_up_tokenization_spaces = False

    def __init__(self):
        self._pieces = {1: " need", 2: " the"}

    def decode(self, ids, **_kw):
        return "".join(self._pieces.get(i, "") for i in ids)

    def encode(self, text, **_kw):
        # the naive detokenizer probes encode/decode of "a ,b" at
        # construction to learn whether decode drops spaces; round-trip it.
        ids = [k for k, v in self._pieces.items() if v == text]
        if not ids:
            ids = [100 + len(self._pieces)]
            self._pieces[ids[0]] = text
        return ids


class _FakeTokenizer:
    """A raw-HF-tokenizer stand-in every streaming detokenizer can build on:
    a vocab for the SPM/BPE token maps, and `decode` (plus the encode round
    trip the naive one probes at construction) giving the text a real
    tokenizer's decode would."""
    clean_up_tokenization_spaces = False

    def __init__(self, vocab, pieces):
        self._vocab = list(vocab)
        self._pieces = dict(pieces)

    def __len__(self):
        return len(self._vocab)

    def convert_ids_to_tokens(self, ids):
        return [self._vocab[i] for i in ids]

    def decode(self, ids, **_kw):
        return "".join(self._pieces.get(i, "") for i in ids)

    def encode(self, text, **_kw):
        ids = [k for k, v in self._pieces.items() if v == text]
        if not ids:
            ids = [1000 + len(self._pieces)]
            self._pieces[ids[0]] = text
        return ids


_SPM_VOCAB = ["<unk>", "\u2581Hello", "\u2581wor", "ld", "<0x21>"]
_NAIVE_VOCAB = ["<unk>", "need", "the"]  # what SPM/BPE maps would misread


class TestDetokenizerSourceStreamsTheDecodedText:
    """With a model_path the streaming detokenizer comes from tokenizer.json's
    decoder (the vendored mlx-lm predicates), primed at load; the per-request
    call streams with it. Whatever it picks, the streamed text is the
    tokenizer's own decode of the ids -- leading space trimmed only where the
    decoder strips it -- and an unreadable tokenizer.json still generates."""

    @pytest.mark.parametrize("tokenizer_json, vocab, pieces, expected", [
        ('{"decoder": {"type": "ByteLevel"}}',
         ["<s>", "Hello", "\u0120world", "!", "\u010a", "\u0120how"],
         {1: "Hello", 2: " world", 3: "!", 4: "\n", 5: " how"}, "Hello world!\n how"),
        (None, _SPM_VOCAB, {1: "Hello", 2: " wor", 3: "ld", 4: "!"}, "Hello world!"),
        ("no-strip", _SPM_VOCAB, {1: " Hello", 2: " wor", 3: "ld", 4: "!"}, " Hello world!"),
        ('{"decoder": {"type": "WordPiece"}}', _NAIVE_VOCAB, {1: " need", 2: " the"}, " need the"),
        ("absent", _NAIVE_VOCAB, {1: " need", 2: " the"}, " need the"),
        ("{not json", _NAIVE_VOCAB, {1: " need", 2: " the"}, " need the"),
    ], ids=["bpe", "spm", "spm-keeps-the-leading-space", "other-decoder",
            "no-tokenizer-json", "unreadable-tokenizer-json"])
    def test_streamed_text_is_the_decoded_text(self, tmp_path, tokenizer_json, vocab,
                                               pieces, expected):
        import json
        from heylook_llm.providers.common.generation_core import detokenizer_source

        if tokenizer_json is None:
            tokenizer_json = json.dumps({"decoder": _SPM_DECODER})
        elif tokenizer_json == "no-strip":
            tokenizer_json = json.dumps({"decoder": {
                "type": "Sequence", "decoders": _SPM_DECODER["decoders"][:3]}})
        if tokenizer_json != "absent":
            (tmp_path / "tokenizer.json").write_text(tokenizer_json)
        tok = _FakeTokenizer(vocab, pieces)
        ids = sorted(pieces)
        assert tok.decode(ids) == expected  # the fixture's own consistency

        detokenizer_source(tok, tmp_path)           # primed at load
        d = detokenizer_source(tok).detokenizer     # the per-request call
        segments = []
        for i in ids:
            d.add_token(i)
            segments.append(d.last_segment)
        d.finalize()
        segments.append(d.last_segment)
        assert "".join(segments) == expected


@pytest.mark.unit
class TestContinueFromGenerationPrompt:
    """A content continuation resumes on the prompt the reply was generated
    under (gemma-4 thinking off: an empty thought channel the history render
    drops; without it the continuation degrades -- A/B 2026-09-23)."""

    GEN = "<u>hi</u><model>\n<ch>thought\n</ch>"
    HISTORY_HEAD = "<u>hi</u><model>\n"
    MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Mars is"}]

    def _run(self, prompt, generation):
        from heylook_llm.providers.common.vlm_inputs import continue_from_generation_prompt
        seen = []

        def render(msgs):
            seen.append(msgs)
            return generation
        return continue_from_generation_prompt(prompt, self.MSGS, render), seen

    def test_dropped_generation_prefix_is_restored(self):
        out, seen = self._run(self.HISTORY_HEAD + "Mars is", self.GEN)
        assert out == self.GEN + "Mars is"
        assert seen == [self.MSGS[:-1]]          # rendered WITHOUT the partial reply

    def test_a_render_that_already_matches_is_left_alone(self):
        prompt = self.GEN + "Mars is"
        assert self._run(prompt, self.GEN)[0] == prompt

    def test_anything_that_is_not_a_clean_extension_keeps_the_continuation(self):
        # the history render carries thinking the generation prompt lacks
        prompt = self.HISTORY_HEAD + "<ch>thought\nplan</ch>Mars is"
        assert self._run(prompt, self.GEN)[0] == prompt
        # the template rewrote the reply's text
        assert self._run(self.HISTORY_HEAD + "Mars is.", self.GEN)[0] == self.HISTORY_HEAD + "Mars is."
