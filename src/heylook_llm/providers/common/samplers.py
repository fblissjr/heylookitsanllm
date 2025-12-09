# src/heylook_llm/providers/common/samplers.py
"""
Why this file exists:
This module centralizes the creation of sampler and logits processor functions,
ensuring that all MLX-based models (both LLM and VLM) use the exact same,
feature-rich generation hyperparameters as the standalone mlx-lm library.
It acts as a single source of truth for sampling logic.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from transformers import PreTrainedTokenizer

# Default hyper-parameters from mlx-lm/generate.py
DEFAULT_TEMP = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1


def make_presence_penalty_processor(penalty: float):
    """
    Create a presence penalty logits processor.

    Presence penalty reduces the likelihood of tokens that have already appeared,
    regardless of how many times they appeared. This encourages the model to
    explore new topics.

    Recommended value for Qwen3 thinking mode: 1.5

    Args:
        penalty: Penalty value (0.0-2.0). Higher values discourage repetition more.

    Returns:
        A logits processor function compatible with mlx-lm.
    """
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if penalty <= 0.0 or len(tokens) == 0:
            return logits

        # Get unique tokens that have appeared
        unique_tokens = mx.unique(tokens)

        # Apply penalty to logits of tokens that have appeared
        # Subtract penalty from the logits of seen tokens
        logits = logits.at[unique_tokens].add(-penalty)

        return logits

    return processor


def _xtc_special_tokens(tokenizer: PreTrainedTokenizer | None) -> list[int]:
    """Return the newline token + all EOS ids."""
    if tokenizer is None:
        return []
    try:
        newline_token = tokenizer.encode("\n")
        eos_tokens = list(getattr(tokenizer, 'eos_token_ids', [tokenizer.eos_token_id]))
        return newline_token + eos_tokens
    except Exception:
        return []


def build(tokenizer: PreTrainedTokenizer | None, params: dict) -> tuple[callable, list[callable]]:
    """
    Builds and returns a sampler function and a list of logits processors.

    Args:
        tokenizer: The tokenizer, used to encode special tokens for XTC sampling.
        params: A dictionary of user-provided or default generation parameters.

    Returns:
        A tuple containing the configured sampler function and list of logits processors.
    """
    # Set the random seed for reproducibility
    if (seed := params.get("seed")) is not None:
        mx.random.seed(seed)

    sampler = make_sampler(
        temp=params.get("temperature", DEFAULT_TEMP),
        top_p=params.get("top_p", DEFAULT_TOP_P),
        min_p=params.get("min_p", 0.0),
        top_k=params.get("top_k", 0),
        xtc_probability=params.get("xtc_probability", 0.0),
        xtc_threshold=params.get("xtc_threshold", 0.0),
        xtc_special_tokens=_xtc_special_tokens(tokenizer),
    )

    processors = make_logits_processors(
        logit_bias=params.get("logit_bias"),
        repetition_penalty=params.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
        repetition_context_size=params.get("repetition_context_size", 20),
    )

    # Add presence penalty processor if specified
    presence_penalty = params.get("presence_penalty", 0.0)
    if presence_penalty > 0.0:
        processors.append(make_presence_penalty_processor(presence_penalty))

    return sampler, processors
