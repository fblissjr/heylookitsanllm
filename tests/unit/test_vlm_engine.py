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
    assert cache_report(100, 0, cold=True, has_media=True, memory="no room").cause == "memory"
    assert cache_report(100, 40, memory="no room").outcome == "reused"   # a hit is a hit
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

    def __init__(self, *_a, **k):
        self.kwargs = k
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
                        lambda *a, **k: made.append(_FakeBatchGenerator(**k)) or made[-1])
    model = SimpleNamespace(
        language_model=None,
        get_input_embeddings=lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
    ids = mx.array([[1, 2, 3]])
    chunks = list(vlm_engine.generate(
        model=model, processor=None, apc_manager=None, input_ids=ids,
        raw_inputs={"input_ids": ids}, sampler=kw.pop("sampler", None),
        processors=kw.pop("processors", []),
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
def test_a_drafting_round_ends_at_a_stop_and_counts_its_kept_drafts(monkeypatch):
    """With a drafter, next() delivers a whole round. Reply text ends at the
    first stop, one only heylook's set holds included, and the report counts
    each round's kept drafts (its tokens less the target's own). Untagged
    processors (structured output) never draft; thinking budgets do."""
    from heylook_llm.providers.common.vlm_engine import drafting_blocker

    resp = lambda tok, fin=None: SimpleNamespace(uid=0, token=tok, finish_reason=fin)
    script = [([], [resp(5)]), ([], [resp(6), resp(7), resp(8)]),
              ([], [resp(9), resp(99), resp(10)])]
    chunks, bg = _run(monkeypatch, script, draft=("drafter", "mtp"))
    assert (bg.kwargs["draft_model"], bg.kwargs["draft_kind"]) == ("drafter", "mtp")
    assert [c.token for c in chunks] == [5, 6, 7, 8, 9]   # one chunk per token
    assert "".join(c.text for c in chunks) == "<5><6><7><8><9>"
    assert (chunks[-1].spec.accepted, chunks[-1].spec.emitted) == (3, 5)

    _, bg = _run(monkeypatch, [([], [resp(5, "stop")])], draft=("drafter", "mtp"),
                 processors=[lambda tokens, logits: logits])
    assert "draft_model" not in bg.kwargs
    assert drafting_blocker([lambda tokens, logits: logits], draft_kind="mtp")
    assert drafting_blocker([], draft_kind="mtp") is None


@pytest.mark.unit
def test_mtp_safe_penalties_draft_on_mtp_and_track_generated_history(monkeypatch):
    """MTP-safe stateless penalties stay drafted on draft_kind='mtp', force the
    MTP walk path, record generated tokens onto _heylook_mtp_penalties, and
    clean up on exit while still blocking dflash or mixed structured output."""
    from heylook_llm.providers.common import samplers
    from heylook_llm.providers.common.generation_core import generated_only
    from heylook_llm.providers.common.vlm_engine import drafting_blocker

    sampler, procs = samplers.build({
        "temperature": 0.0,
        "presence_penalty": 1.2,
        "repetition_penalty": 1.15,
    })
    scoped = generated_only(procs)
    assert drafting_blocker(scoped, draft_kind="mtp") is None
    assert drafting_blocker(scoped, draft_kind="dflash") is not None
    assert drafting_blocker(scoped + [lambda tokens, logits: logits], draft_kind="mtp") is not None

    recorded = []

    class CapturingBG(_FakeBatchGenerator):
        def next(self):
            recorded.append(
                (getattr(sampler, "_heylook_mtp_penalties", None)[1][:]
                 if getattr(sampler, "_heylook_mtp_penalties", None) else None)
            )
            return super().next()

    ar = importlib.import_module("mlx_vlm.generate.ar")
    made = []
    _FakeBatchGenerator.script = [
        ([], [SimpleNamespace(uid=0, token=5, finish_reason=None)]),
        ([], [SimpleNamespace(uid=0, token=6, finish_reason=None),
              SimpleNamespace(uid=0, token=7, finish_reason=None),
              SimpleNamespace(uid=0, token=99, finish_reason="stop")]),
    ]
    monkeypatch.setattr(ar, "BatchGenerator",
                        lambda *a, **k: made.append(CapturingBG(**k)) or made[-1])
    model = SimpleNamespace(
        language_model=None,
        get_input_embeddings=lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
    ids = mx.array([[1, 2, 3]])

    from heylook_llm.providers.common import vlm_engine

    chunks = list(vlm_engine.generate(
        model=model, processor=None, apc_manager=None, input_ids=ids,
        raw_inputs={"input_ids": ids}, sampler=sampler, processors=procs,
        stop_tokens=[99], max_tokens=8, greedy=True,
        detokenizer=_FakeDetok(), draft=("drafter", "mtp")))

    assert (made[0].kwargs["draft_model"], made[0].kwargs["draft_kind"]) == ("drafter", "mtp")
    assert made[0].kwargs["greedy_sampling"] is False
    assert [c.token for c in chunks] == [5, 6, 7]
    assert recorded == [[], [5]]
    assert getattr(sampler, "_heylook_mtp_penalties", None) is None


@pytest.mark.unit
def test_draft_max_context_gates_speculative_per_request(monkeypatch):
    """draft_max_context skips speculative drafting per request when prompt tokens
    exceed the cap, without unloading or blocking prompts within the cap."""
    from heylook_llm.providers.common.vlm_engine import drafting_blocker

    assert drafting_blocker([], prompt_tokens=4096, draft_max_context=4096) is None
    reason = drafting_blocker([], prompt_tokens=4097, draft_max_context=4096)
    assert reason and "4097" in reason and "4096" in reason

    resp = lambda tok, fin=None: SimpleNamespace(uid=0, token=tok, finish_reason=fin)
    # _run sends a 3-token input_ids prompt ([1, 2, 3])
    _, bg_blocked = _run(monkeypatch, [([], [resp(5, "stop")])], draft=("drafter", "mtp"),
                         draft_max_context=2)
    assert "draft_model" not in bg_blocked.kwargs

    _, bg_allowed = _run(monkeypatch, [([], [resp(5, "stop")])], draft=("drafter", "mtp"),
                         draft_max_context=3)
    assert (bg_allowed.kwargs["draft_model"], bg_allowed.kwargs["draft_kind"]) == ("drafter", "mtp")


@pytest.mark.unit
def test_a_cancel_during_prefill_gives_back_its_prefix_blocks(monkeypatch):
    """remove() during prefill drops the batch without releasing the blocks a
    prefix hit acquired; unreleased blocks are never evictable."""
    class Abort:
        def is_set(self):
            return True

    chunks, bg = _run(monkeypatch, [], abort_event=Abort())
    assert chunks == [] and bg.released == 1


def test_memory_pressure_reads_evictions_since_last_request_and_low_headroom():
    """APC holds back when something else holds the memory; the report must
    say so, or a measurement beside a big gguf model reads as ordinary misses."""
    from types import SimpleNamespace

    from heylook_llm.providers.common.vlm_engine import apc_memory_pressure

    stats = {"memory_evictions": 0}
    mgr = SimpleNamespace(stats_snapshot=lambda: dict(stats), memory_reserve_bytes=2 << 30,
                          _memory_headroom=lambda: 40 << 30)
    assert apc_memory_pressure(mgr) is None                  # room, nothing evicted
    stats["memory_evictions"] = 3
    assert "evicted 3 entries" in apc_memory_pressure(mgr)
    assert apc_memory_pressure(mgr) is None                  # already accounted for
    mgr._memory_headroom = lambda: 1 << 30
    assert "stores nothing" in apc_memory_pressure(mgr)      # empty and no room: evicts nothing
    assert apc_memory_pressure(SimpleNamespace()) is None    # never raises




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


@pytest.mark.unit
def test_a_shared_snapshot_is_refreshed_while_its_prefix_is_in_use():
    # The system-prompt snapshot is stored first and never restored from once
    # a conversation has later snapshots, so without a refresh it is the
    # store's oldest entry and ages out. The key is mlx-vlm's own.
    import threading
    from collections import OrderedDict

    from mlx_vlm.apc import _sequence_hash

    from heylook_llm.providers.common.vlm_engine import refresh_snapshots

    prompt = list(range(500))
    shared = _sequence_hash(tuple(prompt[:120]), 7, 16)
    mgr = SimpleNamespace(block_size=16, lock=threading.RLock(),
                          _exact_cache=OrderedDict([(shared, "sys"), ("later", "turn")]))
    refresh_snapshots(mgr, prompt, [120], 7)
    assert list(mgr._exact_cache) == ["later", shared]
    refresh_snapshots(mgr, prompt, [120], 8)            # another salt: not this snapshot
    refresh_snapshots(mgr, prompt, [64], 7)             # not stored: left alone
    assert list(mgr._exact_cache) == ["later", shared]


@pytest.mark.unit
def test_abort_event_end_reasoning_signals_and_pops():
    """AbortEvent.end_reasoning() triggers an attached callback or latches a
    one-shot flag consumed by pop_end_reasoning() for the MLX decode loop."""
    from heylook_llm.providers.abort import AbortEvent

    abort = AbortEvent()
    assert abort.pop_end_reasoning() is False
    assert abort.end_reasoning() is True
    assert abort.pop_end_reasoning() is True
    assert abort.pop_end_reasoning() is False

    called = []
    abort.set_end_reasoning_callback(lambda: (called.append(True), True)[1])
    assert abort.end_reasoning() is True
    assert called == [True]
    abort.clear()
    assert abort.pop_end_reasoning() is False


@pytest.mark.unit
def test_capture_lengths_coarse_interior_checkpoints():
    """Coarse interior checkpoints cover user_only prefixes on request 2 while
    doubling stride to cap snapshot count and skipping points near boundaries."""
    from heylook_llm.providers.common.vlm_engine import capture_lengths

    base = dict(interval=64, block_size=16, captures=3, min_tokens=16,
                interior_stride=1024, interior_max=4)

    # Short prompt: no interior points below 1024.
    assert capture_lengths(500, **base) == [384, 448, 500]
    # ~2.2K user_only prompt: adds 1024 interior before near-end [2048, 2112, 2174].
    assert capture_lengths(2174, **base) == [1024, 2048, 2112, 2174]
    # ~3.5K prompt: adds 1024, 2048, 3072.
    assert capture_lengths(3500, **base) == [1024, 2048, 3072, 3392, 3456, 3500]
    # ~5.5K prompt: span 5376 > 4*1024 -> stride doubles to 2048 -> adds 2048, 4096.
    assert capture_lengths(5492, **base) == [2048, 4096, 5376, 5440, 5492]
    # ~16K prompt: stride doubles twice to 4096 -> adds 4096, 8192, 12288.
    assert capture_lengths(16000, **base) == [4096, 8192, 12288, 15872, 15936, 16000]
    # Skips interior point within < interval of an explicit boundary.
    got = capture_lengths(3500, boundaries=[1000], **base)
    assert 1000 in got and 1024 not in got and 2048 in got and 3072 in got


@pytest.mark.unit
def test_install_capture_policy_directional_micro_delta_guard():
    """Observed divergence `stable` is dropped when within <64 tokens AFTER a
    covered prefix, while an EARLIER `stable` or one >=64 tokens ahead is kept."""
    from collections import OrderedDict
    import threading
    from types import SimpleNamespace

    from heylook_llm.providers.common.vlm_engine import install_capture_policy

    def run_policy(*, final, bounds, stable, cached_lengths=()):
        store = OrderedDict(
            (i, SimpleNamespace(token_ids=tuple(range(length)), extra_hash=7))
            for i, length in enumerate(cached_lengths)
        )
        mgr = SimpleNamespace(
            checkpoint_interval_tokens=64,
            _exact_cache_max=32,
            disk=None,
            block_size=16,
            exact_cache_min_tokens=16,
            _exact_cache=store,
            lock=threading.RLock(),
            checkpoint_prefix_length=lambda *_a, **_kw: stable,
        )
        coord = SimpleNamespace(
            enabled=True,
            is_checkpoint=True,
            manager=mgr,
            checkpoint_len=lambda ids, media: final,
        )
        bg = SimpleNamespace(apc=coord)
        install_capture_policy(bg, bounds)
        return coord.checkpoint_lengths(list(range(final + 1)), set(), extra_hash=7)

    # +2 micro-delta past system prompt 2234 is suppressed.
    got = run_policy(final=5492, bounds=[2234], stable=2236)
    assert 2234 in got and 2236 not in got

    # +13 micro-delta past 1469 is suppressed; +401 real shared prefix is kept.
    got_13 = run_policy(final=2500, bounds=[1469], stable=1482)
    assert 1469 in got_13 and 1482 not in got_13
    got_401 = run_policy(final=2500, bounds=[1469], stable=1870)
    assert 1469 in got_401 and 1870 in got_401

    # Directional check: stable=100 is <64 before near-end 128, with nothing <= 100,
    # so 100 MUST still be kept (a later checkpoint cannot restore an earlier divergence).
    got_earlier = run_policy(final=250, bounds=[], stable=100)
    assert 100 in got_earlier and 128 in got_earlier

    # Micro-delta (+48) past an already-cached prefix in _exact_cache is suppressed.
    got_cached = run_policy(final=5600, bounds=[], stable=5488, cached_lengths=(5440,))
    assert 5488 not in got_cached
