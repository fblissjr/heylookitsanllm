# src/heylook_llm/providers/common/generation_core.py
"""
Unified generation loop for text-based MLX generation.

Extracts the shared generation logic from TextOnlyStrategy and VLMTextOnlyStrategy
into a single function. This is the sole call site for lm_stream_generate in the
text path, and the natural integration point for future mx.compile, dynamic draft
tuning, and shape bucketing.

MLX alignment:
- wired_limit wraps the entire generation, not individual calls
- All counters are Python ints -- zero GPU sync in the loop
- response.token is already a Python int from mlx-lm (no .item() needed)
- generation_stream is mlx-lm's internal stream -- we don't create our own
- Logging happens in finally (outside the hot loop)

Error handling:
- Prompt-cache snapshot storage is skipped on generation errors. A failed
  prefill leaves partially-populated cache layers (some layers updated, others
  fresh). Storing this state would cascade into shape mismatches on future
  requests. See internal/bugs/radix_cache_vlm_crash.md.
"""

import logging
from contextlib import contextmanager
import threading
from collections import deque
from pathlib import Path
from typing import Any, Generator

import mlx.core as mx
from mlx_lm import tokenizer_utils as lm_tokenizer_utils
from mlx_lm.generate import stream_generate as lm_stream_generate, wired_limit
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..base import GenerationChunk, InvalidGenerationRequest
from .prompt_cache import get_global_cache_manager, process_prompt_with_cache, store_generation_cache
from .model_wrappers import MROPE_STATE_ATTRS, unwrap_language_model
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
    load, where the path is known; run_generation's per-request call then
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


def _reset_vlm_positions(model) -> None:
    """Reset mRoPE position state cached on VLM language models.

    Qwen3.5-style models cache _position_ids and _rope_deltas on the
    LanguageModel instance during forward(). These must be cleared between
    fresh generations (no pre-filled cache) to prevent broadcast shape
    mismatches in rotary embedding computation. Handles LogitsWrapper-style
    wrapping via the SHARED unwrap (model_wrappers) -- the cache-reuse gate
    inspects the same object with the same attribute list, by construction.
    """
    target = unwrap_language_model(model)
    for attr in MROPE_STATE_ATTRS:
        if hasattr(target, attr):
            try:
                # NEVER object.__setattr__ here: mlx Module routes array
                # assignments to its module dict and un-shadows an instance
                # __dict__ entry only when the key is absent -- a bypassed
                # None write creates a PERMANENT shadow, after which the
                # model's own position-state writes become invisible and
                # every warm generation recomputes positions from the
                # current input (empirically confirmed 2026-08-18 on a real
                # nn.Module). delattr any existing shadow first (heals
                # processes that already have one), then plain setattr.
                try:
                    object.__delattr__(target, attr)
                except AttributeError:
                    pass
                setattr(target, attr, None)
            except Exception:
                pass


class DraftTuner:
    """Dynamically adjusts num_draft_tokens based on acceptance rate.

    Keyed by model_id, tracks a rolling window of boolean acceptance results
    per model. Conservative policy:
    - acceptance > 80% over last 50 samples: increase by 1 (max 8)
    - acceptance < 50% over last 50 samples: decrease by 1 (min 1)
    - < 10 total samples: use configured default

    Thread-safe via Lock.
    """

    MIN_DRAFT_TOKENS = 1
    MAX_DRAFT_TOKENS = 8
    WINDOW_SIZE = 50
    MIN_SAMPLES = 10
    HIGH_THRESHOLD = 0.80
    LOW_THRESHOLD = 0.50

    def __init__(self):
        self._lock = threading.Lock()
        self._windows: dict[str, deque[bool]] = {}
        self._current: dict[str, int] = {}

    def get_num_draft_tokens(self, model_id: str, configured_default: int) -> int:
        """Return the current draft token count for model_id."""
        with self._lock:
            if model_id not in self._current:
                return configured_default
            return self._current[model_id]

    def record(self, model_id: str, accepted: int, total: int) -> None:
        """Record acceptance data from a generation and adjust if needed.

        Args:
            model_id: The model identifier.
            accepted: Number of draft tokens accepted this generation.
            total: Total draft tokens proposed this generation.
        """
        if total <= 0 or not model_id:
            return

        with self._lock:
            if model_id not in self._windows:
                self._windows[model_id] = deque(maxlen=self.WINDOW_SIZE)

            window = self._windows[model_id]

            # Add individual results to window
            for i in range(total):
                window.append(i < accepted)

            # Not enough samples to make a decision
            if len(window) < self.MIN_SAMPLES:
                return

            rate = sum(window) / len(window)
            current = self._current.get(model_id)

            if current is None:
                # First adjustment -- don't adjust, just set baseline
                return

            if rate > self.HIGH_THRESHOLD and current < self.MAX_DRAFT_TOKENS:
                self._current[model_id] = current + 1
                logging.info(
                    f"DraftTuner: {model_id} acceptance {rate:.0%} -- "
                    f"increasing draft tokens {current} -> {current + 1}"
                )
            elif rate < self.LOW_THRESHOLD and current > self.MIN_DRAFT_TOKENS:
                self._current[model_id] = current - 1
                logging.info(
                    f"DraftTuner: {model_id} acceptance {rate:.0%} -- "
                    f"decreasing draft tokens {current} -> {current - 1}"
                )

    def ensure_and_get(self, model_id: str, configured_default: int) -> int:
        """Ensure baseline exists and return current draft token count. Single lock."""
        with self._lock:
            if model_id not in self._current:
                self._current[model_id] = configured_default
            return self._current[model_id]

    def _ensure_baseline(self, model_id: str, configured_default: int) -> None:
        """Set the baseline for a model if not already tracked."""
        with self._lock:
            if model_id not in self._current:
                self._current[model_id] = configured_default


# Module-level singleton
_draft_tuner = DraftTuner()


def get_draft_tuner() -> DraftTuner:
    """Return the module-level DraftTuner singleton."""
    return _draft_tuner


# Import the module-level generation stream from mlx_provider
# This is created once at import time for the lifetime of the process
def _get_generation_stream():
    """Lazy import to avoid circular dependency with mlx_provider."""
    from ..mlx_provider import generation_stream
    return generation_stream


def _build_cache_config(effective_request: dict) -> dict:
    """Build cache configuration dict from effective_request."""
    return {
        'cache_type': effective_request.get('cache_type', 'standard'),
        'kv_bits': effective_request.get('kv_bits'),
        'kv_group_size': effective_request.get('kv_group_size', 64),
        'max_kv_size': effective_request.get('max_kv_size'),
    }


def _setup_prompt_cache(model_id, model, prompt_tokens, cache_config, cache_manager,
                        allow_reuse: bool = True):
    """Set up the prompt cache from the model's single-slot snapshot.

    Returns:
        Tuple of (prompt_cache, tokens_to_process, generation_cache).
        If model_id is None, prompt_cache and generation_cache are None.
        ``allow_reuse=False`` (spec decode) still REGISTERS the working
        cache -- telemetry and reclamation depend on that -- but skips
        cross-request lookup/store.
    """
    if not model_id:
        return None, prompt_tokens, None

    prompt_cache = cache_manager.get_or_create_cache(model_id, model, cache_config)
    tokens_to_process, updated_cache = process_prompt_with_cache(
        prompt_cache, prompt_tokens, model, cache_config, allow_reuse=allow_reuse
    )
    generation_cache = updated_cache.cache
    logging.info(f"Prompt cache: processing {len(tokens_to_process)}/{len(prompt_tokens)} tokens")
    return prompt_cache, tokens_to_process, generation_cache


def _seedable(factory) -> bool:
    """Whether the detokenizer ``factory`` builds lets ``reset`` seed ``text``.

    The SPM and BPE classes keep ``text`` in a slot, so a seed is one
    assignment. ``NaiveStreamingDetokenizer`` computes ``text`` as a property
    with no setter -- and never trims a leading space, so it needs no seed
    either: seeding it raised AttributeError inside the first next() of
    every continuation on an mlx-vlm-loaded model (v1.79.64). Unwraps a
    functools.partial: mlx-lm hands gemma an SPM partial with trim_space off.
    """
    cls = getattr(factory, "func", factory)
    attr = getattr(cls, "text", None)
    return not (isinstance(attr, property) and attr.fset is None)


@contextmanager
def continuation_detokenizer(tokenizer, continuing: bool):
    """Keep the FIRST token's leading space when ``continuing`` (v1.79.64).

    mlx-lm's streaming detokenizers drop a leading space on the first text
    they flush (SPM: ``trim_space`` while ``self.text`` is empty; BPE:
    ``_maybe_trim_space`` on an empty buffer). Right for a fresh turn, where
    the space after the role marker is an artifact -- wrong for a continuation,
    where the model's first token completes a prefilled "First I" and the
    space in " need" is real. ``TokenizerWrapper.detokenizer`` builds a fresh
    instance per access, and ``stream_generate`` accesses it exactly once
    and calls ``reset()`` on it, so the class factory is swapped for the
    duration of one generation (the process-global gate serialises them)
    with one whose ``reset`` seeds a one-char sentinel into ``text`` and
    advances ``offset`` past it: every trim test reads a non-empty buffer,
    every ``last_segment`` starts after the sentinel, and no caller ever
    sees it. Restored in ``finally`` whatever happens to the generator.
    """
    factory_attr = "_detokenizer_class"
    original = getattr(tokenizer, factory_attr, None)
    if not continuing or original is None or not _seedable(original):
        yield
        return

    def seeded(tok):
        detok = original(tok)
        base_reset = detok.reset

        def reset():
            base_reset()
            detok.text = "\x00"
            detok.offset = 1

        detok.reset = reset
        return detok

    setattr(tokenizer, factory_attr, seeded)
    try:
        yield
    finally:
        setattr(tokenizer, factory_attr, original)


def generate_text(
    model,
    tokenizer,
    prompt_tokens: list[int],
    effective_request: dict,
    model_id: str | None = None,
    draft_model=None,
    cache_manager=None,
    abort_event=None,
    continuing: bool = False,
    context_length: int | None = None,
) -> Generator:
    """High-level entry point for text-based generation.

    Builds the sampler/processors from effective_request, then delegates to
    run_generation(). Strategies should call this instead of run_generation()
    directly -- it keeps sampler construction co-located with the generation loop.

    VLMVisionStrategy builds its own sampler (different tokenizer source) and
    calls stream_generate_with_sampling directly, so it does not use this.
    """
    from .samplers import build as build_sampler
    sampler, processors = build_sampler(tokenizer, effective_request)
    yield from run_generation(
        model, tokenizer, prompt_tokens, effective_request,
        sampler, processors,
        model_id=model_id, draft_model=draft_model,
        cache_manager=cache_manager, abort_event=abort_event,
        continuing=continuing, context_length=context_length,
    )


def run_generation(
    model,
    tokenizer,
    prompt_tokens: list[int],
    effective_request: dict,
    sampler,
    processors,
    model_id: str | None = None,
    draft_model=None,
    cache_manager=None,
    abort_event=None,
    pre_filled_cache=None,
    continuing: bool = False,
    context_length: int | None = None,
) -> Generator:
    """Single generation loop for all text-based MLX generation.

    This is the only place lm_stream_generate is called for text. It handles:
    - Cache config construction from effective_request
    - Single-slot prompt cache lookup (skipped when pre_filled_cache provided)
    - lm_stream_generate call with wired_limit scope
    - Abort checking (Python bool, no GPU sync)
    - Speculative decoding acceptance tracking (Python ints only)
    - Leading space cleanup on first token (skipped for pre_filled_cache)
    - KV snapshot storage in finally block

    Args:
        model: Raw model (text-only) or LanguageModelLogitsWrapper (VLM text)
        tokenizer: The tokenizer for the model
        prompt_tokens: Tokenized prompt
        effective_request: Merged config with model defaults + request overrides
        sampler: Sampler function from build_sampler
        processors: Logits processors from build_sampler
        model_id: Model identifier for cache management
        draft_model: Draft model for speculative decoding (or None)
        cache_manager: PromptCacheManager instance (or None for default)
        abort_event: AbortEvent for cooperative cancellation
        pre_filled_cache: Pre-populated KV cache from VLM vision forward pass.
            When provided, skips prompt-cache setup and leading space cleanup.
        context_length: The model's context window (capabilities.
            model_context_length). A prompt longer than it is refused up
            front as the client's error, not generated into garbage; None =
            unknown, no guard. Text path only: the vision path arrives with
            its prompt already in the pre-filled cache.

    Yields:
        Generation response objects from lm_stream_generate.
    """
    if context_length and pre_filled_cache is None and len(prompt_tokens) > context_length:
        # gguf gets the same answer from llama-server itself (a 400 mapped to
        # InvalidGenerationRequest); MLX has no fixed allocation to refuse,
        # so this is where the ceiling is enforced for it.
        raise InvalidGenerationRequest(
            f"Prompt is {len(prompt_tokens)} tokens; {model_id or 'this model'} has a "
            f"context of {context_length} tokens. Shorten the conversation or the "
            f"system prompt.")

    # Reset VLM mRoPE position state for fresh text generations.
    # Qwen3.5-style models cache _position_ids and _rope_deltas on the
    # language_model instance. Stale values from a prior generation cause
    # broadcast shape mismatches in rotary embedding computation.
    # Skip when pre_filled_cache is set -- the vision forward pass sets
    # correct position state that must be preserved.
    if pre_filled_cache is None:
        _reset_vlm_positions(model)

    tokenizer = ensure_gen_tokenizer(tokenizer)

    if pre_filled_cache is not None and draft_model is not None:
        # Spec decode needs prompt_cache=None so mlx-lm can build its paired
        # target+draft caches -- which would DISCARD the vision prefill KV
        # and generate fluent nonsense from a single token. Correctness
        # wins: drop the draft for this request, keep the vision cache.
        logging.warning(
            "speculative decoding is not supported on the vision path -- "
            "generating without the draft model for this request"
        )
        draft_model = None

    if pre_filled_cache is not None:
        # Vision path: cache already populated by VLM forward pass.
        # Skip the prompt cache -- vision embeddings can't be snapshotted here.
        prompt_cache = None
        tokens_to_process = prompt_tokens
        generation_cache = pre_filled_cache
        logging.info(
            f"Using pre-filled cache for generation, processing {len(tokens_to_process)} token(s)"
        )
    else:
        if cache_manager is None:
            cache_manager = get_global_cache_manager()

        cache_config = _build_cache_config(effective_request)

        # Reuse eligibility (mRoPE gate + config gate) is enforced INSIDE
        # process_prompt_with_cache -- path enforcement, so any future
        # caller inherits it. model_id always flows: the working cache must
        # register even for no-reuse models, or context-usage telemetry,
        # /v1/cache info and the byte-budget/memory-pressure reclamation
        # paths all go dark for them (review finding on the first cut).
        # Spec decode: mlx-lm's speculative path slices a provided
        # prompt_cache into target+draft halves; a target-only list (ours,
        # fresh OR restored) would hand the drafter an empty cache. With a
        # draft model, mlx-lm builds its own paired caches and cross-request
        # reuse is skipped (allow_reuse below; logged by the eligibility
        # machinery).
        prompt_cache, tokens_to_process, generation_cache = _setup_prompt_cache(
            model_id, model, prompt_tokens, cache_config, cache_manager,
            allow_reuse=draft_model is None,
        )

    # mlx-lm fires this per prefill step INSIDE the first next() -- nothing
    # can be yielded from here, so progress goes up the request's signal
    # channel and the async wrapper turns each change into a frame. `total`
    # is tokens_to_process: the work this request runs, cached prefix
    # excluded, the same meaning the gguf provider reports.
    report_progress = getattr(abort_event, "set_prefill_progress", None)

    def prompt_progress_callback(processed: int, total: int):
        logging.debug(f"Prompt processing: {processed}/{total} tokens")
        if report_progress is not None:
            report_progress(processed, total)

    generation_stream = _get_generation_stream()

    # Consult DraftTuner for dynamic draft token count (single lock acquisition)
    configured_draft = effective_request.get('num_draft_tokens', 3)
    if draft_model is not None and model_id:
        tuner = get_draft_tuner()
        num_draft_tokens = tuner.ensure_and_get(model_id, configured_draft)
    else:
        num_draft_tokens = configured_draft

    # All counters are Python ints -- no mx.array overhead, no GPU sync
    generated_token_ids = []
    draft_accepted = 0
    draft_total = 0
    generation_failed = False

    # Compute how many tokens came from cache for reporting in API responses
    cached_count = len(prompt_tokens) - len(tokens_to_process)

    # Scope peak memory to this request so API can report per-request peak
    # (not carry-over from a prior request's high-water mark).
    mx.reset_peak_memory()

    # Snapshot server-wide KV cache byte total once at start; cheap enough to
    # attach to the first token so the streaming API picks it up via getattr.
    kv_cache_bytes_snapshot = cache_manager.total_cache_bytes if cache_manager is not None else 0

    # Forward prefill_step_size through to mlx-lm when the caller set one.
    # When absent, mlx-lm picks its own default (2048 at the time of writing)
    # -- passing None would suppress the default, so we only pass the kwarg
    # when we have an explicit value.
    extra_generate_kwargs: dict[str, Any] = {}
    prefill_step_size = effective_request.get('prefill_step_size')
    if prefill_step_size is not None:
        extra_generate_kwargs['prefill_step_size'] = prefill_step_size

    try:
        with wired_limit(model, [generation_stream]), \
                continuation_detokenizer(tokenizer, continuing):
            first_token = True
            for response in lm_stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=tokens_to_process,
                sampler=sampler,
                logits_processors=processors,
                max_tokens=effective_request['max_tokens'],
                draft_model=draft_model,
                num_draft_tokens=num_draft_tokens,
                prompt_progress_callback=prompt_progress_callback,
                prompt_cache=generation_cache if (generation_cache and draft_model is None) else None,
                **extra_generate_kwargs,
            ):
                # Abort check: Python bool, no GPU sync
                if abort_event and abort_event.is_set():
                    logging.info("Generation aborted")
                    break

                generated_token_ids.append(response.token)

                # Acceptance tracking: Python ints only
                if hasattr(response, 'from_draft') and draft_model is not None:
                    draft_total += 1
                    if response.from_draft:
                        draft_accepted += 1

                # Convert to the owned chunk type at the engine boundary --
                # this is the ONLY place mlx-lm's GenerationResponse shape is
                # known; everything downstream sees GenerationChunk.
                chunk = GenerationChunk.from_engine(response)
                # Spec-decode acceptance: running totals on every chunk
                # (ChunkTelemetry latches; two int writes, negligible).
                if draft_total:
                    chunk.draft_tokens = draft_total
                    chunk.draft_accepted = draft_accepted

                # Leading space cleanup (first token only; skipped for a
                # pre-filled vision cache AND for a continuation, where the
                # first token completes prefilled text and its space is real
                # -- see continuation_detokenizer) + cache stats snapshot
                # (first chunk only; ChunkTelemetry latches them).
                if first_token:
                    if pre_filled_cache is None and not continuing and chunk.text.startswith(' '):
                        chunk.text = chunk.text.lstrip()
                    chunk.cached_tokens = cached_count
                    chunk.kv_cache_bytes = kv_cache_bytes_snapshot
                    first_token = False

                yield chunk
    except Exception:
        generation_failed = True
        raise
    finally:
        # Feed acceptance data to DraftTuner for dynamic adjustment
        if draft_total > 0:
            rate = draft_accepted / draft_total
            logging.info(
                f"Speculative decoding: {draft_accepted}/{draft_total} draft tokens accepted "
                f"({rate:.0%}), {len(generated_token_ids)} total generated"
            )
            if model_id:
                get_draft_tuner().record(model_id, draft_accepted, draft_total)
        # Store the KV snapshot as the model's slot for future prefix reuse.
        # Skip on error: a failed prefill leaves partially-populated cache
        # layers that would corrupt future requests via cascade failures.
        if not generation_failed and prompt_cache and generation_cache:
            full_tokens = prompt_tokens + generated_token_ids
            store_generation_cache(prompt_cache, full_tokens, generation_cache)
            logging.debug(
                f"Stored cache: {len(prompt_tokens)} prompt + {len(generated_token_ids)} generated"
            )


