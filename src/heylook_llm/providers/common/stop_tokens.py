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
        tokenizer.eos_token_ids = resolve_stop_tokens(tokenizer) | ids
    except Exception as e:
        logging.warning("eos extension from generation_config failed: %s", e)


def resolve_stop_tokens(tokenizer) -> set[int]:
    """Resolve EOS token IDs from a tokenizer, handling None and missing attrs.

    Checks eos_token_ids (plural, set by some tokenizers) first, then falls
    back to eos_token_id (singular). Returns an empty set if neither exists.
    """
    ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(ids, int):        # some tokenizers expose it as a scalar int
        return {ids}
    if ids:                         # non-empty collection
        return set(ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return {eos_id}
    return set()
