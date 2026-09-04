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

    def test_append_after_an_already_open_block(self):
        from types import SimpleNamespace
        from heylook_llm.providers.mlx_provider import _append_thinking_resume
        info = SimpleNamespace(has_thinking_markers=True, has_harmony_structure=False,
                               has_gemma_channel_structure=False)
        # Qwen3.5+: the generation prompt already ends in an open <think>
        out = _append_thinking_resume("<|im_start|>assistant\n<think>\n", "  so far", info)
        assert out == "<|im_start|>assistant\n<think>\nso far"
        # Qwen3: the model emits <think> itself, so the opener is added
        out = _append_thinking_resume("<|im_start|>assistant\n", "so far", info)
        assert out == "<|im_start|>assistant\n<think>\nso far"

    def test_channel_families_reopen_their_own_channel(self):
        # v1.79.63: gemma re-opens <|channel>thought, harmony the analysis
        # channel; the matching parser is armed to start inside it.
        from types import SimpleNamespace
        from heylook_llm.providers.mlx_provider import _append_thinking_resume
        gemma = SimpleNamespace(has_thinking_markers=False, has_harmony_structure=False,
                                has_gemma_channel_structure=True)
        assert _append_thinking_resume("<|turn>model\n", "so far", gemma) \
            == "<|turn>model\n<|channel>thought\nso far"
        harmony = SimpleNamespace(has_thinking_markers=False, has_harmony_structure=True,
                                  has_gemma_channel_structure=False)
        assert _append_thinking_resume("<|start|>assistant", "so far", harmony) \
            == "<|start|>assistant<|channel|>analysis<|message|>so far"

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

    class _SpmLike:
        # The shape of mlx-lm's SPMStreamingDetokenizer trim: a leading space
        # is dropped only while `text` is empty.
        def __init__(self, tok):
            self.reset()

        def reset(self):
            self.text = ""
            self.offset = 0
            self.tokens = []

        def add_token(self, piece):
            if not self.text and piece.startswith(" "):
                piece = piece[1:]
            self.text += piece

        @property
        def last_segment(self):
            seg = self.text[self.offset:]
            self.offset = len(self.text)
            return seg

    class _Wrapper:
        def __init__(self, cls):
            self._detokenizer_class = cls

        @property
        def detokenizer(self):
            return self._detokenizer_class(self)

    def _run(self, continuing):
        from heylook_llm.providers.common.generation_core import continuation_detokenizer
        tok = self._Wrapper(self._SpmLike)
        with continuation_detokenizer(tok, continuing):
            d = tok.detokenizer   # what stream_generate does, once
            d.reset()
            d.add_token(" need")
            first = d.last_segment
            d.add_token(" the")
            second = d.last_segment
        return tok, first, second

    def test_fresh_turn_still_trims(self):
        tok, first, second = self._run(continuing=False)
        assert (first, second) == ("need", " the")
        assert tok._detokenizer_class is self._SpmLike

    def test_continuation_keeps_the_first_space_and_restores_the_factory(self):
        tok, first, second = self._run(continuing=True)
        assert (first, second) == (" need", " the")
        assert "\x00" not in first + second
        assert tok._detokenizer_class is self._SpmLike  # restored in finally

    def test_restored_even_when_the_generation_raises(self):
        from heylook_llm.providers.common.generation_core import continuation_detokenizer
        tok = self._Wrapper(self._SpmLike)
        with pytest.raises(RuntimeError):
            with continuation_detokenizer(tok, True):
                raise RuntimeError("mid-generation")
        assert tok._detokenizer_class is self._SpmLike

    def test_a_read_only_text_detokenizer_is_left_alone(self):
        """The runtime shape on the mlx-vlm path: a raw HF tokenizer wrapped
        by ensure_gen_tokenizer without a model_path takes mlx-lm's DEFAULT,
        NaiveStreamingDetokenizer, whose `text` is a property with no setter.
        v1.79.64 seeded it and raised AttributeError inside the first next()
        of every continuation on every mlx-vlm-loaded model. It never trims
        a leading space either, so leaving it alone is also the right
        output."""
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper
        from heylook_llm.providers.common.generation_core import continuation_detokenizer

        wrapper = TokenizerWrapper(_FakeHfForDetok(), eos_token_ids={0})
        assert wrapper._detokenizer_class is NaiveStreamingDetokenizer
        with continuation_detokenizer(wrapper, True):
            d = wrapper.detokenizer
            d.reset()
            d.add_token(1)
            first = d.last_segment
            d.add_token(2)
            second = d.last_segment
        assert (first, second) == (" need", " the")
        assert wrapper._detokenizer_class is NaiveStreamingDetokenizer

    def test_a_partial_factory_is_unwrapped_before_the_seed_check(self):
        """mlx-lm hands gemma-family SPM tokenizers a functools.partial
        (trim_space=False); the seedability check must look through it."""
        from functools import partial
        from heylook_llm.providers.common.generation_core import _seedable
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer, NaiveStreamingDetokenizer, SPMStreamingDetokenizer)
        assert _seedable(SPMStreamingDetokenizer)
        assert _seedable(partial(SPMStreamingDetokenizer, trim_space=False))
        assert _seedable(BPEStreamingDetokenizer)
        assert not _seedable(NaiveStreamingDetokenizer)


class _FakeHfForDetok:
    """A raw-HF-tokenizer stand-in that mlx-lm's TokenizerWrapper AND its
    naive detokenizer can drive: vocab scan at construction, decode by id."""
    eos_token_id = 0
    eos_token_ids = {0}
    chat_template = None
    clean_up_tokenization_spaces = False
    _pieces = {1: " need", 2: " the"}

    def get_vocab(self):
        return {"<eos>": 0}

    def decode(self, ids, **_kw):
        return "".join(self._pieces.get(i, "") for i in ids)


class TestEnsureGenTokenizerPicksAStreamingDetokenizer:
    """v1.79.66: with a model_path, the raw-tokenizer wrapper is built by
    mlx-lm's own loader, which selects SPM/BPE streaming from tokenizer.json
    -- the same choice the mlx-lm text path gets -- instead of the naive
    default. The provider primes it at load; the per-request call from
    run_generation must then find it in the cache."""

    def test_model_path_routes_through_the_mlx_lm_loader_and_primes_the_cache(self, monkeypatch, tmp_path):
        from pathlib import Path
        from heylook_llm.providers.common import generation_core as gc

        built = object()
        seen = {}

        def fake_load(path, tokenizer_config_extra=None, eos_token_ids=None):
            seen["path"] = path
            seen["eos"] = set(eos_token_ids)
            return built

        monkeypatch.setattr(gc.lm_tokenizer_utils, "load", fake_load)
        raw = _FakeHfForDetok()
        assert gc.ensure_gen_tokenizer(raw, tmp_path) is built
        assert seen["path"] == Path(tmp_path)
        assert seen["eos"] == gc.resolve_stop_tokens(raw)   # the extended set, not the loader's own
        assert gc.ensure_gen_tokenizer(raw) is built          # run_generation's call: cache hit, no path needed

    def test_a_loader_failure_falls_back_to_the_default_wrapper(self, monkeypatch, tmp_path):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from heylook_llm.providers.common import generation_core as gc

        def broken_load(path, tokenizer_config_extra=None, eos_token_ids=None):
            raise OSError("no tokenizer.json here")

        monkeypatch.setattr(gc.lm_tokenizer_utils, "load", broken_load)
        raw = _FakeHfForDetok()
        wrapped = gc.ensure_gen_tokenizer(raw, tmp_path)
        assert isinstance(wrapped, TokenizerWrapper)        # a model that loaded still generates
        assert gc.ensure_gen_tokenizer(raw) is wrapped

    def test_no_model_path_keeps_the_old_default(self):
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper
        from heylook_llm.providers.common import generation_core as gc
        wrapped = gc.ensure_gen_tokenizer(_FakeHfForDetok())
        assert isinstance(wrapped, TokenizerWrapper)
        assert wrapped._detokenizer_class is NaiveStreamingDetokenizer
