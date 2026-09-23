# src/heylook_llm/providers/common/generation_core.py
"""
What heylook keeps as its own around mlx-vlm's engine (plan W10).

Generation itself runs in ``vlm_engine``. This module holds the pieces of
the old decode loop the engine still uses, because they are heylook's
contract rather than an engine's:

- ``ensure_gen_tokenizer`` / ``continuation_detokenizer``: the streaming
  detokenizer the engine streams through (mlx-lm's: mlx-vlm's BPE one holds
  space-free text until the end) with a continuation's first-token space
  kept;
- ``generated_only``: logits processors scoped to the tokens this reply has
  generated (owner decision 2026-09-21), whatever the engine prefilled.
"""

import copy
import logging
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from mlx_lm import tokenizer_utils as lm_tokenizer_utils
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .stop_tokens import resolve_stop_tokens


def ensure_gen_tokenizer(tokenizer, model_path=None):
    """Wrap a raw HF tokenizer for generation with the FULL stop set and a
    real streaming detokenizer.

    mlx-lm's ``stream_generate`` auto-wraps raw tokenizers as
    ``TokenizerWrapper(tokenizer)``, whose eos set defaults to the single
    ``eos_token_id`` -- dropping any extra terminators resolved from
    generation_config (gemma-4 via mlx-vlm: <turn|> and <|tool_response|>
    would be lost, and generation runs past end-of-turn). Wrapping here with
    ``resolve_stop_tokens`` preserves them. Already-wrapped tokenizers
    (mlx-lm loads) pass through untouched.

    ``model_path`` (v1.79.66) picks the streaming detokenizer the way mlx-lm's
    own loader does, from tokenizer.json's decoder. Without it the wrapper
    takes mlx-lm's DEFAULT, ``NaiveStreamingDetokenizer``: it re-decodes the
    whole current line on every token (quadratic per line) and computes
    ``text`` as a read-only property -- every mlx-vlm-loaded model streamed
    through it, and the v1.79.64 continuation seed raised on it inside the
    first next() of every continuation. The provider primes the cache at
    load, where the path is known; the per-request call then
    hits the cache. The loader wraps a SECOND tokenizer instance read from
    the same files (public API rather than mlx-lm's private class-selection
    predicates); the eos set is passed explicitly so the raw tokenizer's
    extended set (extend_eos_from_generation_config) still wins.
    """
    if isinstance(tokenizer, TokenizerWrapper):
        return tokenizer
    # Cache the wrapper ON the raw tokenizer: TokenizerWrapper.__init__ runs a
    # full-vocab get_vocab() scan (_infer_thinking), pure waste to repeat per
    # request. The attribute dies with the tokenizer/model, so a reload
    # naturally invalidates it.
    cached = getattr(tokenizer, "_heylook_gen_wrapper", None)
    if cached is not None:
        return cached
    eos = resolve_stop_tokens(tokenizer)
    wrapped = None
    if model_path is not None:
        try:
            wrapped = lm_tokenizer_utils.load(Path(model_path), eos_token_ids=eos)
        except Exception as e:
            # A model that loaded still has to generate: fall back to the
            # default wrapper (naive streaming) and say why, rather than
            # turn a detokenizer choice into a load failure.
            logging.warning(
                f"Could not build a streaming detokenizer from {model_path} ({e}); "
                f"generation falls back to mlx-lm's naive detokenizer")
    if wrapped is None:
        wrapped = TokenizerWrapper(tokenizer, eos_token_ids=eos)
    tokenizer._heylook_gen_wrapper = wrapped
    return wrapped


def _seedable(detokenizer) -> bool:
    """Whether this detokenizer lets ``reset`` seed ``text``.

    The SPM and BPE classes assign ``text`` in ``reset``, so a seed is one
    assignment. ``NaiveStreamingDetokenizer`` computes ``text`` as a property
    with no setter -- and never trims a leading space, so it needs no seed
    either: seeding it raised AttributeError inside the first next() of
    every continuation on an mlx-vlm-loaded model (v1.79.64). Takes an
    instance or a class; the question is about the CLASS either way, because
    an instance's ``text`` reads as the string it currently holds.
    """
    cls = detokenizer if isinstance(detokenizer, type) else type(detokenizer)
    attr = getattr(cls, "text", None)
    return not (isinstance(attr, property) and attr.fset is None)


@lru_cache(maxsize=None)
def _seeding_subclass(cls: type) -> type:
    """``cls`` with a ``reset`` that leaves the sentinel in place.

    A SUBCLASS rather than a patched instance, because the wrapper hands
    every caller ``copy.copy`` of its prototype: a closure bound to the
    prototype would seed the prototype while the copy -- the one actually
    streaming -- stayed empty. A class travels through the copy intact.
    """
    def reset(self):
        cls.reset(self)
        self.text = "\x00"
        self.offset = 1

    return type(f"Seeded{cls.__name__}", (cls,), {"reset": reset})


def generated_only(processors):
    """Scope logits processors to the tokens THIS reply has generated.

    mlx-lm hands a processor ``(tokens, logits)`` where ``tokens`` is every
    prompt token mlx-lm ITSELF prefilled plus what it has generated. How much
    prompt that is was an accident of the path, so until v2.0.60 a presence or
    repetition penalty meant three different things on MLX:

    - text request, cold cache: the WHOLE prompt -- system prompt, every
      earlier turn, their end-of-turn tokens;
    - text request that hit the prompt cache: only the uncached suffix, so the
      same request sampled differently depending on what ran before it;
    - image request: nothing but the reply, because the vision strategy
      prefills the prompt itself and mlx-lm only ever sees its last token.

    One rule now (owner decision 2026-09-21): generated tokens only. It is what
    vendor-documented penalty values assume, it cannot depend on cache state,
    and a long fixed system prompt no longer penalises most of the vocabulary
    the answer needs.

    No token COUNT is assumed, which is what makes it hold for the normal loop,
    the speculative loop, a cache hit and the vision path alike: at the first
    processor call nothing has been generated yet, so the history's length
    there IS the prompt part. It is recorded once and sliced off every call.
    Per generation by construction -- the closure is built inside
    ``vlm_engine.generate`` -- and an empty slice is fine: both penalty processors
    return the logits untouched for an empty history.
    """
    if not processors:
        return processors
    prompt_len: int | None = None

    def scope(processor):
        def scoped(tokens, logits):
            nonlocal prompt_len
            if prompt_len is None:
                prompt_len = len(tokens)
            return processor(tokens[prompt_len:], logits)
        return scoped

    return [scope(p) for p in processors]


@contextmanager
def continuation_detokenizer(tokenizer, continuing: bool):
    """Keep the FIRST token's leading space when ``continuing`` (v1.79.64).

    mlx-lm's streaming detokenizers drop a leading space on the first text
    they flush (SPM: ``trim_space`` while ``self.text`` is empty; BPE:
    ``_maybe_trim_space`` on an empty buffer). Right for a fresh turn, where
    the space after the role marker is an artifact -- wrong for a continuation,
    where the model's first token completes a prefilled "First I" and the
    space in " need" is real. ``TokenizerWrapper.detokenizer`` hands each
    caller a ``copy.copy`` of one prebuilt prototype and resets it, so the
    PROTOTYPE is swapped for the duration of one generation (the
    process-global gate serialises them) for one whose ``reset`` seeds a
    one-char sentinel into ``text`` and advances ``offset`` past it: every
    trim test reads a non-empty buffer, every ``last_segment`` starts after
    the sentinel, and no caller ever sees it. Restored in ``finally``
    whatever happens to the generator.

    The prototype is ``_detokenizer``; before mlx-lm's detokenizer refactor
    it was a CLASS on ``_detokenizer_class``, instantiated per access. A
    rename is SILENT here by construction -- no attribute means no seeding
    means the space goes back to being trimmed, with nothing raised -- so
    the seam-space tests drive the real wrapper, not a stand-in alone.
    """
    prototype = getattr(tokenizer, "_detokenizer", None)
    if not continuing or prototype is None or not _seedable(prototype):
        yield
        return

    seeded = copy.copy(prototype)
    seeded.__class__ = _seeding_subclass(type(prototype))
    tokenizer._detokenizer = seeded
    try:
        yield
    finally:
        tokenizer._detokenizer = prototype
