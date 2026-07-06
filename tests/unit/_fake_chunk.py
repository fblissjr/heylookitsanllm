"""Shared fake mlx-lm GenerationResponse chunk builder.

Satisfies the non-slotted attr-bag shape ``run_generation`` /
``stream_generate`` chunks have, without depending on mlx-lm. Imported via
sibling-dir path injection (pytest adds the test file's parent to sys.path),
same pattern used by ``_mock_provider.py`` for the router tests and
``_fake_request.py`` for auth tests.
"""

from __future__ import annotations

from types import SimpleNamespace


def fake_chunk(
    text="hi",
    prompt_tokens=10,
    generation_tokens=5,
    prompt_tps=123.4,
    generation_tps=87.6,
    queue_wait_ms=0.0,
    finish_reason=None,
):
    """Fake mlx-lm GenerationResponse chunk (non-slotted attr bag)."""
    return SimpleNamespace(
        text=text,
        token=None,
        logprobs=None,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        queue_wait_ms=queue_wait_ms,
        peak_memory=0.0,
    )
