"""Decode-path special-token hygiene.

``apply_special_token_hygiene(tokenizer)`` patches a tokenizer's ``.decode``
to default ``skip_special_tokens=True`` -- closes a leak where mlx-lm's
``NaiveStreamingDetokenizer`` calls ``decode(tokens)`` bare, so control
tokens that are part of structured output formats render as literal
strings in user-visible text.

Fast detokenizers (``SPMStreamingDetokenizer``, ``BPEStreamingDetokenizer``)
build their output from a pre-computed token-id-to-bytes map that drops
specials by construction; Naive falls through to HF's ``.decode()`` which
defaults to ``skip_special_tokens=False``. Patching at the wrapped-tokenizer
level catches every decode site at once and survives whichever detokenizer
variant mlx-lm picks for a given tokenizer class.

Callers that deliberately want raw specials (Token Explorer UI, some
logprobs consumers) can still pass ``skip_special_tokens=False`` --
the patched decode respects explicit overrides.

Idempotent: calling twice on the same tokenizer is a no-op via a
``_hygiene_patched`` flag.
"""

from __future__ import annotations

import logging
from typing import Any, Callable


_FLAG = "_hygiene_patched"


def apply_special_token_hygiene(tokenizer: Any) -> None:
    """Install skip-specials-by-default on every ``.decode`` we can reach.

    Patches the object passed in AND, if it looks like an mlx-lm
    ``TokenizerWrapper`` (has ``_tokenizer``), the inner HF tokenizer too.
    The detokenizer inside mlx-lm captures whichever ref was available at
    construction; patching both removes that ambiguity.
    """
    if tokenizer is None:
        return

    _patch_one(tokenizer)
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and inner is not tokenizer:
        _patch_one(inner)


def _patch_one(obj: Any) -> None:
    original = getattr(obj, "decode", None)
    if original is None or not callable(original):
        return
    if getattr(original, _FLAG, False):
        return

    def patched(*args, skip_special_tokens: bool = True, **kwargs):
        return original(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    setattr(patched, _FLAG, True)
    try:
        obj.decode = patched  # type: ignore[assignment]
    except (AttributeError, TypeError) as exc:
        logging.debug("tokenizer_hygiene: could not patch decode on %r: %s", obj, exc)
