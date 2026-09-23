"""vlm_engine's pure decisions (plan W10). The generation loop itself is
checked live (scripts/chain_probe.py, the W10 spike harness); its upstream
surface is pinned in tests/contract/test_mlxvlm_surface.py."""
from types import SimpleNamespace

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
def test_prefill_progress_counts_against_the_prompt_not_the_trimmed_batch():
    # the batch trims its embeddings as it consumes them; the total must not
    # shrink with them (seen live: 2960 then 2704 for one prompt)
    batch = SimpleNamespace(_processed_prompt_columns=256,
                            _inputs_embeds=SimpleNamespace(shape=(1, 2960, 8)))
    assert prefill_progress(batch, 3216, cached=0) == (256, 3216)
    assert prefill_progress(batch, 3216, cached=3100) == (116, 116)
    assert prefill_progress(None, 3216, cached=0) is None
    assert prefill_progress(SimpleNamespace(), 3216, cached=0) is None
