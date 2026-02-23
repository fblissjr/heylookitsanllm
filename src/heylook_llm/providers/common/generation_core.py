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
"""

import logging
import threading
from collections import deque
from typing import Generator

from mlx_lm.generate import stream_generate as lm_stream_generate, wired_limit

from .prompt_cache import get_global_cache_manager, process_prompt_with_cache, store_generation_cache


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


def _setup_prompt_cache(model_id, model, prompt_tokens, cache_config, cache_manager):
    """Set up prompt cache using radix tree lookup.

    Returns:
        Tuple of (prompt_cache, tokens_to_process, generation_cache).
        If model_id is None, prompt_cache and generation_cache are None.
    """
    if not model_id:
        return None, prompt_tokens, None

    prompt_cache = cache_manager.get_or_create_cache(model_id, model, cache_config)
    tokens_to_process, updated_cache = process_prompt_with_cache(
        prompt_cache, prompt_tokens, model, cache_config
    )
    generation_cache = updated_cache.cache
    logging.info(f"Prompt cache: processing {len(tokens_to_process)}/{len(prompt_tokens)} tokens")
    return prompt_cache, tokens_to_process, generation_cache


def generate_text(
    model,
    tokenizer,
    prompt_tokens: list[int],
    effective_request: dict,
    model_id: str | None = None,
    draft_model=None,
    cache_manager=None,
    abort_event=None,
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
) -> Generator:
    """Single generation loop for all text-based MLX generation.

    This is the only place lm_stream_generate is called for text. It handles:
    - Cache config construction from effective_request
    - Radix-tree prompt cache lookup
    - lm_stream_generate call with wired_limit scope
    - Abort checking (Python bool, no GPU sync)
    - Speculative decoding acceptance tracking (Python ints only)
    - Leading space cleanup on first token
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

    Yields:
        Generation response objects from lm_stream_generate.
    """
    if cache_manager is None:
        cache_manager = get_global_cache_manager()

    cache_config = _build_cache_config(effective_request)

    prompt_cache, tokens_to_process, generation_cache = _setup_prompt_cache(
        model_id, model, prompt_tokens, cache_config, cache_manager
    )

    def prompt_progress_callback(processed: int, total: int):
        logging.debug(f"Prompt processing: {processed}/{total} tokens")

    generation_stream = _get_generation_stream()

    # Consult DraftTuner for dynamic draft token count
    configured_draft = effective_request.get('num_draft_tokens', 3)
    if draft_model is not None and model_id:
        tuner = get_draft_tuner()
        tuner._ensure_baseline(model_id, configured_draft)
        num_draft_tokens = tuner.get_num_draft_tokens(model_id, configured_draft)
    else:
        num_draft_tokens = configured_draft

    # All counters are Python ints -- no mx.array overhead, no GPU sync
    generated_token_ids = []
    draft_accepted = 0
    draft_total = 0

    try:
        with wired_limit(model, [generation_stream]):
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
                prompt_cache=generation_cache if generation_cache else None,
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

                # Leading space cleanup (first token only)
                if first_token:
                    if response.text.startswith(' '):
                        response.text = response.text.lstrip()
                    first_token = False

                yield response
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
        # Store KV snapshot in radix tree for future prefix reuse
        if model_id and prompt_cache and generation_cache:
            full_tokens = prompt_tokens + generated_token_ids
            store_generation_cache(prompt_cache, full_tokens, generation_cache)
            logging.debug(
                f"Stored cache: {len(prompt_tokens)} prompt + {len(generated_token_ids)} generated"
            )
