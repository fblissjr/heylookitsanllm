# src/heylook_llm/providers/common/stop_tokens.py
"""Shared stop-token resolution for all MLX generation paths."""
from __future__ import annotations

import logging
from pathlib import Path


def extend_eos_from_generation_config(tokenizer, model_path) -> None:
    """Union ``generation_config.json``'s eos ids into ``tokenizer.eos_token_ids``.

    mlx-lm's TokenizerWrapper does this at load; raw HF tokenizers on the
    mlx-vlm path don't. gemma-4 is the live case: the tokenizer declares
    eos=1 (<eos>) while generation_config declares [1, 106, 50] including
    the <turn|> turn terminator -- without the union every response runs
    past end-of-turn until <eos> or the token cap. Best-effort: never raises.
    """
    try:
        from .template_info import _read_json
        cfg = _read_json(Path(model_path) / "generation_config.json")
        if not isinstance(cfg, dict):
            return
        ids = cfg.get("eos_token_id")
        ids = set(ids) if isinstance(ids, list) else ({ids} if isinstance(ids, int) else set())
        if not ids:
            return
        # Stored under a private name, NOT tokenizer.eos_token_ids:
        # transformers 5.x SpecialTokensMixin.__setattr__ intercepts
        # special-token attr assignment and raises on non-string values
        # ("Cannot set a non-string value as the eos_token"), which silently
        # dropped <turn|> from gemma-4's stop set. resolve_stop_tokens reads
        # this attr first, so every consumer sees the union.
        object.__setattr__(
            tokenizer, "_heylook_eos_ids", resolve_stop_tokens(tokenizer) | ids
        )
    except Exception as e:
        logging.warning("eos extension from generation_config failed: %s", e)


def resolve_stop_tokens(tokenizer) -> set[int]:
    """Resolve EOS token IDs from a tokenizer, handling None and missing attrs.

    Checks _heylook_eos_ids (our generation_config union, see
    extend_eos_from_generation_config) first, then eos_token_ids (plural,
    set by some tokenizers), then eos_token_id (singular). Returns an empty
    set if none exists.
    """
    own = getattr(tokenizer, "_heylook_eos_ids", None)
    if isinstance(own, (set, frozenset, list, tuple)) and own:
        return set(own)
    ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(ids, int):        # some tokenizers expose it as a scalar int
        return {ids}
    if ids:                         # non-empty collection
        return set(ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return {eos_id}
    return set()
