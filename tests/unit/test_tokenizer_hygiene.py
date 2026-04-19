"""Tests for decode-path special-token hygiene.

mlx-lm's ``NaiveStreamingDetokenizer.decode(tokens)`` calls the wrapped
HF tokenizer without ``skip_special_tokens=True``, so structured-output
formats leak their control tokens as literal strings into the
user-visible stream. Fast detokenizers build from a pre-computed
token->bytes map and drop specials implicitly; Naive does not.

``apply_special_token_hygiene`` patches the tokenizer's ``.decode`` to
default ``skip_special_tokens=True``, catching both streaming and
non-streaming paths regardless of which detokenizer variant mlx-lm
selected. Callers that need raw specials (token-explorer UI, some
logprobs use cases) can still pass ``skip_special_tokens=False``.
"""

from __future__ import annotations

from types import SimpleNamespace


class _FakeHFTokenizer:
    """Minimal HF-style tokenizer stand-in.

    Returns the concatenation of token-id->text mappings; when the caller
    passes ``skip_special_tokens=True``, drops IDs whose text starts with
    ``"<|"``. Mimics the real HF decode contract closely enough for the
    hygiene test.
    """

    def __init__(self, vocab):
        self._vocab = dict(vocab)

    def decode(self, tokens, skip_special_tokens=False):
        parts = []
        for t in tokens:
            text = self._vocab.get(t, "")
            if skip_special_tokens and text.startswith("<|"):
                continue
            parts.append(text)
        return "".join(parts)


def test_patch_makes_decode_skip_specials_by_default():
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    tok = _FakeHFTokenizer({1: "<|channel|>", 2: "hello", 3: "<|end|>"})
    apply_special_token_hygiene(tok)

    out = tok.decode([1, 2, 3])
    assert out == "hello"


def test_explicit_skip_false_still_works():
    """Callers that explicitly want raw specials (e.g. token explorer)
    must still be able to opt out of the hygiene default."""
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    tok = _FakeHFTokenizer({1: "<|channel|>", 2: "hello"})
    apply_special_token_hygiene(tok)

    out = tok.decode([1, 2], skip_special_tokens=False)
    assert out == "<|channel|>hello"


def test_patch_is_idempotent():
    """Double-patching the same tokenizer must not break semantics or
    stack wrappers."""
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    tok = _FakeHFTokenizer({1: "<|x|>", 2: "a"})
    apply_special_token_hygiene(tok)
    apply_special_token_hygiene(tok)
    apply_special_token_hygiene(tok)

    assert tok.decode([1, 2]) == "a"


def test_patches_wrapped_and_inner_tokenizer():
    """mlx-lm's ``TokenizerWrapper`` captures the inner HF tokenizer at
    construction; detokenizers may hold a reference to either. Apply the
    patch to both so neither path leaks."""
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    inner = _FakeHFTokenizer({1: "<|x|>", 2: "hi"})
    wrapper = SimpleNamespace(_tokenizer=inner, decode=inner.decode)

    apply_special_token_hygiene(wrapper)

    # Inner must be patched (detokenizer typically holds this ref)
    assert inner.decode([1, 2]) == "hi"
    # Wrapper's decode is replaced with the patched version too.
    assert wrapper.decode([1, 2]) == "hi"


def test_none_tokenizer_is_noop():
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    # Must not raise.
    apply_special_token_hygiene(None)


def test_tokenizer_without_decode_is_noop():
    """Defensive: some provider paths may hand us a processor without
    ``decode`` (e.g. image-only processors). Patch should skip quietly."""
    from heylook_llm.providers.common.tokenizer_hygiene import (
        apply_special_token_hygiene,
    )

    apply_special_token_hygiene(SimpleNamespace())  # no decode
