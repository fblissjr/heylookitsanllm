# src/edge_llm/providers/mlx_unified/samplers.py
"""Build the *exact* sampler + logits‑processor chain that `mlx_lm.generate`
creates from its CLI flags.

This file centralises all "generation hyper‑parameters" so both the text and
vision back‑ends behave 1‑for‑1 like the upstream mlx‑lm reference
implementation.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from transformers import PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Default hyper‑parameters (match mlx_lm/generate.py constants)
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS           = 256
DEFAULT_TEMP                 = 1.0
DEFAULT_TOP_P                = 0.95
DEFAULT_MIN_P                = 0.0
DEFAULT_TOP_K                = 0
DEFAULT_XTC_PROBABILITY      = 0.0
DEFAULT_XTC_THRESHOLD        = 0.0
DEFAULT_MIN_TOKENS_TO_KEEP   = 1
DEFAULT_SEED                 = 42
DEFAULT_QUANTIZED_KV_START   = 5000

__all__ = [
    "build",
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _xtc_special_tokens(tokenizer: PreTrainedTokenizer | None):
    """Return the newline token + all EOS ids (may be empty)."""
    if tokenizer is None:
        return []
    try:
        return tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public builder (used by both back‑ends)
# ---------------------------------------------------------------------------

def build(tokenizer: PreTrainedTokenizer | None, params: dict):
    """Return `(sampler_fn, logits_processors)`.

    `params` is expected to be a dict‑like collection of user / router kwargs.
    Missing keys fall back to the defaults above *so every call site can pass
    only what it wants to override*.
    """

    # 0. RNG seed – must be set *before* sampler construction
    seed = params.get("seed", DEFAULT_SEED)
    if seed is not None:
        mx.random.seed(seed)

    # 1. XTC special list
    xtc_special = _xtc_special_tokens(tokenizer)

    # 2. Sampler (keep positional order!)
    sampler = make_sampler(
        params.get("temp",                 DEFAULT_TEMP),
        params.get("top_p",               DEFAULT_TOP_P),
        params.get("min_p",               DEFAULT_MIN_P),
        params.get("min_tokens_to_keep",  DEFAULT_MIN_TOKENS_TO_KEEP),
        top_k             = params.get("top_k",            DEFAULT_TOP_K),
        xtc_probability   = params.get("xtc_probability",  DEFAULT_XTC_PROBABILITY),
        xtc_threshold     = params.get("xtc_threshold",    DEFAULT_XTC_THRESHOLD),
        xtc_special_tokens= xtc_special,
    )

    # 3. logits processors
    processors = make_logits_processors(
        logit_bias              = params.get("logit_bias"),
        repetition_penalty      = params.get("repetition_penalty"),
        repetition_context_size = params.get("repetition_context_size", 20),
    )

    return sampler, processors
