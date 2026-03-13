# src/heylook_llm/providers/common/stop_tokens.py
"""Shared stop-token resolution for all MLX generation paths."""
from __future__ import annotations


def resolve_stop_tokens(tokenizer) -> set[int]:
    """Resolve EOS token IDs from a tokenizer, handling None and missing attrs.

    Checks eos_token_ids (plural, set by some tokenizers) first, then falls
    back to eos_token_id (singular). Returns an empty set if neither exists.
    """
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return {eos_id}
    return set()
