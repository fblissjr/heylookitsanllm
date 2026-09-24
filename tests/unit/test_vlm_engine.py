"""vlm_engine's decisions (plan W10). The generation loop's model behaviour
is checked live (scripts/chain_probe.py, the W10 spike harness); here it runs
against a scripted fake engine for the loop's own bookkeeping. The upstream
surface is pinned in tests/contract/test_mlxvlm_surface.py."""
import importlib
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
    assert prefill_progress(batch, 3216) == (256, 3216)
    # a prefix hit: the batch's cached count is known DURING prefill, and the
    # processed columns count the uncached tail only
    hit = SimpleNamespace(_processed_prompt_columns=64, _cached_tokens_per_row=[3100])
    assert prefill_progress(hit, 3216) == (64, 116)
    assert prefill_progress(None, 3216) is None
    assert prefill_progress(SimpleNamespace(), 3216) is None


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


class _FakeDetok:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, tok):
        self.last_segment = f"<{tok}>"

    def finalize(self):
        self.last_segment = ""


class _FakeBatchGenerator:
    """The slice of mlx-vlm's BatchGenerator the engine loop drives: a
    scripted list of ``next()`` results, and a prompt batch holding prefix
    blocks until it is released."""
    script: list = []

    def __init__(self, *_a, **_k):
        self.apc = None
        self.released = 0
        self._prompt_batch = SimpleNamespace(
            _processed_prompt_columns=0, _cached_tokens_per_row=[0], _apc_meta=[{}],
            _release_apc_meta_blocks=self._release)
        self._steps = iter(self.script)

    def _release(self):
        self.released += 1

    def insert(self, *_a, **_k):
        return [0]

    def next(self):
        return next(self._steps)

    def remove(self, uid):
        self._prompt_batch = None
        return True

    def close(self):
        pass


def _run(monkeypatch, script, **kw):
    from heylook_llm.providers.common import vlm_engine

    _FakeBatchGenerator.script = script
    made = []
    # by module object: `mlx_vlm.generate` resolves to the re-exported function
    ar = importlib.import_module("mlx_vlm.generate.ar")
    monkeypatch.setattr(ar, "BatchGenerator",
                        lambda *a, **k: made.append(_FakeBatchGenerator()) or made[-1])
    model = SimpleNamespace(
        language_model=None,
        get_input_embeddings=lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
    ids = mx.array([[1, 2, 3]])
    chunks = list(vlm_engine.generate(
        model=model, processor=None, apc_manager=None, input_ids=ids,
        raw_inputs={"input_ids": ids}, sampler=None, processors=[],
        stop_tokens=kw.pop("stop_tokens", [99]), max_tokens=8,
        detokenizer=_FakeDetok(), **kw))
    return chunks, made[0]


@pytest.mark.unit
def test_a_token_the_engine_stops_on_is_never_text(monkeypatch):
    """mlx-vlm stops on its own list, which can hold an id heylook's set
    lacks (config.json eos): that token ends the reply and is not content."""
    resp = lambda tok, fin=None: SimpleNamespace(uid=0, token=tok, finish_reason=fin)
    chunks, _ = _run(monkeypatch, [([], [resp(5)]), ([], [resp(42, "stop")])])
    assert "".join(c.text for c in chunks) == "<5>"
    assert chunks[-1].finish_reason == "stop" and chunks[-1].generation_tokens == 1


@pytest.mark.unit
def test_a_cancel_during_prefill_gives_back_its_prefix_blocks(monkeypatch):
    """remove() during prefill drops the batch without releasing the blocks a
    prefix hit acquired; unreleased blocks are never evictable."""
    class Abort:
        def is_set(self):
            return True

    chunks, bg = _run(monkeypatch, [], abort_event=Abort())
    assert chunks == [] and bg.released == 1


@pytest.mark.unit
def test_a_checkpoint_request_also_snapshots_where_other_conversations_share_it():
    from heylook_llm.providers.common import vlm_engine

    mgr = SimpleNamespace(checkpoint_interval_tokens=64, block_size=16, exact_cache_min_tokens=16)
    # upstream's own final length (guard tokens, media), not len - 1: the
    # policy must ask the coordinator for it
    coord = SimpleNamespace(enabled=True, is_checkpoint=True, manager=mgr,
                            checkpoint_len=lambda ids, media: len(ids) - 7)
    vlm_engine.install_capture_policy(SimpleNamespace(apc=coord), [700])
    got = coord.checkpoint_lengths(list(range(1000)), set())
    # the prompt end, the near-end grid (APC_CHECKPOINT_CAPTURES in all),
    # and the shared boundary far before them
    assert got == [700, 896, 960, 993]
    assert len(got) - 1 == vlm_engine.APC_CHECKPOINT_CAPTURES

    block = SimpleNamespace(enabled=True, is_checkpoint=False, manager=mgr)
    vlm_engine.install_capture_policy(SimpleNamespace(apc=block), [700])
    assert not hasattr(block, "checkpoint_lengths")   # block caches share by hash already


@pytest.mark.unit
def test_the_system_prefix_is_where_two_renders_part():
    from heylook_llm.providers.common.vlm_inputs import system_prefix_tokens

    def render(msgs):   # a template whose user header follows the system turn
        return f"<s>{msgs[0]['content']}</s><u>{msgs[1]['content']}</u><a>"
    got = system_prefix_tokens("be brief", render, list)
    assert "".join(got) == "<s>be brief</s><u>"
    assert system_prefix_tokens("x", lambda m: m[1]["content"], list) is None


@pytest.mark.unit
def test_cold_means_both_stores_are_empty():
    from heylook_llm.providers.common.vlm_engine import apc_is_empty

    assert apc_is_empty(SimpleNamespace(_exact_cache={}, hash_table={})) is True
    assert apc_is_empty(SimpleNamespace(_exact_cache={1: 0}, hash_table={})) is False
    assert apc_is_empty(SimpleNamespace(_exact_cache={}, hash_table={1: 0})) is False
    assert apc_is_empty(SimpleNamespace()) is False    # unknown stores never read as cold
    assert apc_is_empty(None) is False
