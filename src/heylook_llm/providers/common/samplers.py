# src/heylook_llm/providers/common/samplers.py
"""
Why this file exists:
This module centralizes the creation of sampler and logits processor functions
for every MLX model, from mlx-vlm's ``sample_utils`` (the engine's own; for
every knob heylook sends it samples identically to mlx-lm's, checked with seeded
runs when mlx-lm was dropped in plan W10 stage 3). It acts as a single source
of truth for sampling logic.

A key absent from ``params`` falls back to the shared sampler floor
(``heylook_llm.samplers.GLOBAL_SAMPLER_FLOOR``), never to a copy of some
engine's defaults: requests arrive through the cascade with every key set, so
only direct callers (warmup, scripts) ever reach a fallback.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_vlm.sample_utils import make_sampler, make_logits_processors

from ...samplers import GLOBAL_SAMPLER_FLOOR


def _apply_presence_penalty(logits: mx.array, tokens: mx.array, penalty: float) -> mx.array:
    """Apply presence penalty via scatter-count -- zero GPU-CPU syncs.

    Instead of computing unique tokens (which requires knowing the output
    size, forcing a sync), scatter-count token occurrences into a vocab-
    sized array, clamp to binary presence, and subtract the penalty.
    All shapes are determined by vocab_size (fixed per model), so this
    is fully compilable with no recompilation across decode steps.
    """
    # Scatter along the VOCAB axis, whatever leads it. mlx-lm hands a logits
    # processor ``logits[:, -1, :]`` -- shape ``(1, vocab)`` -- and the
    # original ``zeros_like(logits).at[tokens]`` scattered along axis 0, whose
    # size is 1: every token id past 0 was an out-of-bounds GPU write. MLX
    # does not bounds-check a Metal scatter, so it corrupted memory until a
    # command buffer faulted mid-generation ("victim of GPU error/recovery")
    # and poisoned the process; gemma-4-26B on MLX reproduced it on the third
    # or fourth token of any request carrying a presence penalty (2026-09-04),
    # which is every thinking-on request through the thinking overlay. A
    # 1-D presence vector broadcasts over any leading batch axis.
    vocab = logits.shape[-1]
    counts = mx.zeros((vocab,), dtype=logits.dtype).at[tokens].add(1.0)
    # Clamp to presence: 1.0 if seen at least once, 0.0 otherwise
    present = mx.minimum(counts, 1.0)
    return logits - penalty * present


def make_presence_penalty_processor(penalty: float):
    """Create a presence penalty logits processor.

    Uses scatter-count instead of unique+gather: scatter 1s at each token
    position, clamp to binary, subtract penalty. Zero GPU-CPU syncs and
    fixed output shapes (vocab_size) for stable compilation.

    WHICH tokens it sees is not decided here: ``vlm_engine.generate`` wraps every
    processor in ``generation_core.generated_only``, so ``tokens`` is what this
    reply has generated and never the prompt. Called bare (tests, scripts) it
    penalises whatever it is handed.

    Args:
        penalty: Penalty value (0.0-2.0). Higher values discourage repetition more.

    Returns:
        A logits processor function, ``(tokens, logits) -> logits``.
    """
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if penalty <= 0.0 or len(tokens) == 0:
            return logits
        return _apply_presence_penalty(logits, tokens, penalty)

    return processor


def build(params: dict) -> tuple[callable, list[callable]]:
    """
    Builds and returns a sampler function and a list of logits processors.

    Args:
        params: A dictionary of user-provided or default generation parameters.
            Only the knobs a request or the cascade can set reach here; XTC
            and logit_bias were passed through until 2026-09-25 although no
            request field or cascade key could ever set them.

    Returns:
        A tuple containing the configured sampler function and list of logits processors.
    """
    # Set the random seed for reproducibility
    if (seed := params.get("seed")) is not None:
        mx.random.seed(seed)

    sampler = make_sampler(
        temp=params.get("temperature", GLOBAL_SAMPLER_FLOOR["temperature"]),
        top_p=params.get("top_p", GLOBAL_SAMPLER_FLOOR["top_p"]),
        min_p=params.get("min_p", GLOBAL_SAMPLER_FLOOR["min_p"]),
        top_k=params.get("top_k", GLOBAL_SAMPLER_FLOOR["top_k"]),
    )

    # An OFF knob builds NO processor. mlx-vlm's make_logits_processors adds a
    # repetition processor for any value but 0, and the floor's "off" is 1.0
    # (an identity), so every request carried one -- and any logits
    # processor makes BatchGenerator._step read the previous tokens back
    # (`inputs.tolist()`) before the next forward, a GPU sync that breaks its
    # double buffering, and it cost decode rate on every MLX request
    # (measured: internal/claude/perf/throughput_2026-09-24/).
    repetition_penalty = params.get("repetition_penalty", GLOBAL_SAMPLER_FLOOR["repetition_penalty"])
    processors = make_logits_processors(
        repetition_penalty=None if repetition_penalty == 1.0 else repetition_penalty,
        repetition_context_size=params.get("repetition_context_size", 20),
    )

    # Add presence penalty processor if specified
    presence_penalty = params.get("presence_penalty", GLOBAL_SAMPLER_FLOOR["presence_penalty"])
    if presence_penalty > 0.0:
        processors.append(make_presence_penalty_processor(presence_penalty))

    return sampler, processors
