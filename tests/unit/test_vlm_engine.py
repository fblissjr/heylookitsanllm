"""vlm_engine's pure decisions (plan W10). The generation loop itself is
checked live (scripts/chain_probe.py, the W10 spike harness); its upstream
surface is pinned in tests/contract/test_mlxvlm_surface.py."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from heylook_llm.providers.common.vlm_engine import cache_report, prefill_progress


@pytest.mark.unit
@pytest.mark.parametrize("cached,outcome,kept", [(None, None, None), (0, "miss", 0),
                                                  (30, "reused", 30), (500, "reused", 100)])
def test_cache_report_is_whole_prompt_normalized(cached, outcome, kept):
    rep = cache_report(100, cached)
    if outcome is None:
        assert rep is None
        return
    assert (rep.outcome, rep.prompt_tokens, rep.cached_tokens) == (outcome, 100, kept)
    assert rep.processed_tokens == 100 - kept


@pytest.mark.unit
def test_a_miss_says_why_when_the_engine_can_know():
    assert cache_report(100, 0, cold=True, has_media=True).cause == "cold"
    assert cache_report(100, 0, has_media=True).cause == "new_image_set"
    plain = cache_report(100, 0)
    assert plain.cause is None and plain.reason
    assert cache_report(100, 40, cold=True, has_media=True).cause is None   # a hit is a hit


@pytest.mark.unit
def test_prefill_progress_counts_against_the_prompt_not_the_trimmed_batch():
    # the batch trims its embeddings as it consumes them; the total must not
    # shrink with them (seen live: 2960 then 2704 for one prompt)
    batch = SimpleNamespace(_processed_prompt_columns=256,
                            _inputs_embeds=SimpleNamespace(shape=(1, 2960, 8)))
    assert prefill_progress(batch, 3216, cached=0) == (256, 3216)
    assert prefill_progress(batch, 3216, cached=3100) == (116, 116)
    assert prefill_progress(None, 3216, cached=0) is None
    assert prefill_progress(SimpleNamespace(), 3216, cached=0) is None


@pytest.mark.parametrize("prompt_len", [1, 3, 40])
def test_a_penalty_sees_the_reply_and_never_the_prompt(prompt_len):
    """The SAME reply must reach a processor the same way whatever the engine
    prefilled before the first call: an uncached suffix (a prefix-cache hit),
    the last chunk, or the whole prompt. Before v2.0.60 those were three
    different penalty histories."""
    from heylook_llm.providers.common.generation_core import generated_only

    seen = []
    (scoped,) = generated_only([lambda tokens, logits: seen.append(tokens.tolist()) or logits])

    prompt, reply = list(range(100, 100 + prompt_len)), [7, 8, 9]
    for n in range(len(reply) + 1):          # call n samples reply token n
        scoped(mx.array(prompt + reply[:n]), mx.zeros((1, 16)))
    assert seen == [[], [7], [7, 8], [7, 8, 9]]
    assert generated_only([]) == [] and generated_only(None) is None
