"""Unit tests for Heylook's CoW prefix cache and thin B=1 MLX runner."""

from types import SimpleNamespace
import mlx.core as mx
import pytest

from heylook_llm.providers.common import b1_runner, gemma4_vision, vlm_engine
from heylook_llm.providers.common.prefix_cache import (
    PrefixCacheStore,
    cache_has_recurrent,
    cache_layout_supported,
    capture_rotating_fallbacks_before_evict,
    snapshot_rotating_fallbacks,
)


class _DummyDetok:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, tok: int):
        self.last_segment = f"<{tok}>"

    def finalize(self):
        self.last_segment = ""


class _AbortSignal:
    def __init__(self, *, cancel_after=None, end_reasoning_once=False):
        self.cancel_after = cancel_after
        self.end_reasoning_once = end_reasoning_once
        self.progress: list[tuple[int, int]] = []
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.cancel_after is not None and self.calls > self.cancel_after

    def set_prefill_progress(self, done: int, total: int):
        self.progress.append((done, total))

    def pop_end_reasoning(self) -> bool:
        if self.end_reasoning_once:
            self.end_reasoning_once = False
            return True
        return False


def _make_fake_apc():
    return SimpleNamespace(
        memory_max_bytes=64 * 1024 * 1024,
        checkpoint_entries=8,
        memory_reserve_bytes=0,
        _exact_cache={},
        hash_table={},
        stats_snapshot=lambda: {"memory_evictions": 0},
        _memory_headroom=lambda: 1024 * 1024 * 1024,
    )


def _make_toy_vlm(*, vocab_size=32, window_size=8, with_recurrent=False):
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    calls = []

    class _LM:
        def __init__(self):
            self._rope_deltas = None

        def make_cache(self):
            layers = [KVCache(), RotatingKVCache(max_size=window_size, keep=0)]
            if with_recurrent:
                layers.append(ArraysCache(1))
            return layers

        def __call__(
            self,
            input_ids,
            cache=None,
            inputs_embeds=None,
            logits_to_keep=None,
            **kwargs,
        ):
            seq_len = int(
                inputs_embeds.shape[1]
                if inputs_embeds is not None
                else input_ids.shape[1]
            )
            calls.append((seq_len, dict(kwargs)))
            base = mx.arange(1, seq_len + 1, dtype=mx.float32).reshape(1, 1, seq_len, 1)
            k = mx.broadcast_to(base, (1, 1, seq_len, 2))
            v = k * 2.0
            for c in cache:
                if type(c) is ArraysCache:
                    prev = c.cache[0] if c.cache[0] is not None else mx.zeros((1, 2))
                    c.cache[0] = prev + float(seq_len)
                else:
                    c.update_and_fetch(k, v)

            last_pos = int(cache[0].offset)
            logits = mx.zeros((1, 1, vocab_size), dtype=mx.float32)
            # Pick deterministic non-stop tokens cycling within [4, vocab_size - 2].
            target_tok = 4 + (last_pos % (vocab_size - 6))
            one_hot = (mx.arange(vocab_size) == target_tok).astype(mx.float32) * 20.0
            logits = one_hot.reshape(1, 1, vocab_size)
            return SimpleNamespace(logits=logits)

    class _OuterModel:
        def __init__(self):
            self.language_model = _LM()

        def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kw):
            n = int(input_ids.shape[1])
            emb = mx.zeros((1, n, 4), dtype=mx.float32)
            return SimpleNamespace(to_dict=lambda: {"inputs_embeds": emb})

    return _OuterModel(), calls


@pytest.mark.unit
def test_prefix_cache_cow_isolation_across_supported_cache_types():
    from mlx_vlm.models.cache import (
        ArraysCache,
        BufferedRotatingKVCache,
        CacheList,
        KVCache,
        RotatingKVCache,
    )

    kv = KVCache()
    rot = RotatingKVCache(max_size=4, keep=0)
    buf = BufferedRotatingKVCache(max_size=4, keep=0, buffer_size=4)
    arr = ArraysCache(1)
    clist = CacheList(KVCache(), RotatingKVCache(max_size=4, keep=0))

    assert cache_layout_supported([kv, rot, buf, arr, clist]) is True
    assert cache_has_recurrent([kv, rot]) is False
    assert cache_has_recurrent([kv, arr]) is True
    assert cache_has_recurrent([kv, clist]) is False

    # Populate 6 tokens so rotating layers are past max_size=4.
    for t in range(1, 7):
        tk = mx.full((1, 1, 1, 2), float(t), dtype=mx.float32)
        tv = tk * 10.0
        kv.update_and_fetch(tk, tv)
        rot.update_and_fetch(tk, tv)
        buf.update_and_fetch(tk, tv)
        clist.caches[0].update_and_fetch(tk, tv)
        clist.caches[1].update_and_fetch(tk, tv)
    arr.cache[0] = mx.array([[100.0, 200.0]], dtype=mx.float32)
    mx.eval(kv.keys, rot.keys, buf.keys, arr.cache[0])

    store = PrefixCacheStore(memory_max_bytes=64 * 1024 * 1024, max_entries=8)
    rd = mx.array([[3]], dtype=mx.int32)
    tokens = [10, 20, 30, 40, 50, 60]
    assert store.store(tokens, [kv, rot, arr, clist], extra_hash=42, rope_deltas=rd)

    # Corrupt original live caches after store -> snapshot must remain unaffected.
    kv.update_and_fetch(mx.full((1, 1, 1, 2), 999.0), mx.full((1, 1, 1, 2), 999.0))
    rot.update_and_fetch(mx.full((1, 1, 1, 2), 999.0), mx.full((1, 1, 1, 2), 999.0))
    arr.cache[0] = mx.array([[888.0, 888.0]], dtype=mx.float32)
    mx.eval(kv.keys, rot.keys, arr.cache[0])

    # Lookup exact prefix (next turn extends tokens).
    restored, matched, rest_rd = store.lookup(
        [*tokens, 70],
        extra_hash=42,
        use_buffered_rotating=True,
        buffer_size=8,
    )
    assert matched == 6
    assert rest_rd is not None and int(rest_rd.item()) == 3
    assert type(restored[0]) is KVCache and restored[0].offset == 6
    assert type(restored[1]) is BufferedRotatingKVCache and restored[1].offset == 6
    assert restored[1]._idx == 4
    assert type(restored[2]) is ArraysCache
    assert type(restored[3]) is CacheList
    assert restored[3].caches[1].offset == 6

    # Mutate restored[0] & restored[1]; verify snapshot stored stays clean on second lookup.
    restored[0].update_and_fetch(mx.full((1, 1, 1, 2), 555.0), mx.full((1, 1, 1, 2), 555.0))
    restored[1].update_and_fetch(mx.full((1, 1, 1, 2), 555.0), mx.full((1, 1, 1, 2), 555.0))
    mx.eval(restored[0].keys, restored[1].keys)

    restored2, matched2, _ = store.lookup(
        [*tokens, 71],
        extra_hash=42,
        use_buffered_rotating=False,
    )
    assert matched2 == 6
    assert type(restored2[1]) is RotatingKVCache
    mx.eval(restored2[0].keys, restored2[1].keys, restored2[2].cache[0])
    assert restored2[0].keys[0, 0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert restored2[1].keys[0, 0, :, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert restored2[2].cache[0].tolist() == [[100.0, 200.0]]


@pytest.mark.unit
def test_prefix_cache_single_pass_trim_tail_compaction_and_eviction():
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    kv = KVCache()
    rot = RotatingKVCache(max_size=4, keep=0)
    # Feed 400 tokens in one chunk so rot.keys is oversized (400 >> 4 + 256).
    chunk = mx.arange(1, 401, dtype=mx.float32).reshape(1, 1, 400, 1)
    k = mx.broadcast_to(chunk, (1, 1, 400, 2))
    kv.update_and_fetch(k, k)
    rot.update_and_fetch(k, k)
    mx.eval(kv.keys, rot.keys)

    store = PrefixCacheStore(memory_max_bytes=512 * 1024, max_entries=4)
    tokens = list(range(1000, 1400))
    assert store.store(tokens, [kv, rot], extra_hash=0, trim_tail=1)
    assert store.store(tokens, [kv, rot], extra_hash=0, trim_tail=0)

    # Identical retry checks max_prefix <= N-1 -> matches 399; follow-up matches 400.
    r_nm1, m_nm1, _ = store.lookup(tokens)
    assert m_nm1 == 399 and r_nm1[1].keys[0, 0, :, 0].tolist() == [396.0, 397.0, 398.0, 399.0]
    r_n, m_n, _ = store.lookup([*tokens, 1400])
    assert m_n == 400 and r_n[1].keys[0, 0, :, 0].tolist() == [397.0, 398.0, 399.0, 400.0]

    # compact_pending shrinks oversized rotating backing slabs post-generation.
    compacted = store.compact_pending()
    assert compacted >= 2
    assert store.compact_pending() == 0

    # Trimmable entries <= max_size allow partial LCP + suffix_is_text_only clamping.
    kv_small = KVCache()
    rot_small = RotatingKVCache(max_size=8, keep=0)
    k4 = mx.arange(1, 5, dtype=mx.float32).reshape(1, 1, 4, 1)
    kv_small.update_and_fetch(k4, k4)
    rot_small.update_and_fetch(k4, k4)
    small_store = PrefixCacheStore(memory_max_bytes=128 * 1024, max_entries=2)
    assert small_store.store([1, 2, 3, 4], [kv_small, rot_small], extra_hash=7)
    _, hit_len, _ = small_store.lookup(
        [1, 2, 3, 99, 100],
        extra_hash=7,
        suffix_is_text_only=lambda m: m <= 2,
    )
    assert hit_len == 2

    # touch() keeps boundary entry MRU; byte budget evicts oldest entry.
    small_store.store([5, 6], [KVCache(), RotatingKVCache(8, keep=0)], extra_hash=0)
    small_store.touch([1, 2, 3, 4], [4], extra_hash=7)
    tiny_budget = PrefixCacheStore(memory_max_bytes=40, max_entries=8)
    big_k = mx.zeros((1, 1, 4, 4), dtype=mx.float32)  # 64B KV + 64B V = 128B > 40B
    kv_b = KVCache()
    kv_b.update_and_fetch(big_k, big_k)
    assert tiny_budget.store([1, 2, 3, 4], [kv_b]) is False


@pytest.mark.unit
def test_can_use_b1_runner_gating(monkeypatch):
    from mlx_vlm.models.cache import BatchKVCache

    good_model, _ = _make_toy_vlm()
    assert b1_runner.can_use_b1_runner(model=good_model, draft=None) is True
    assert (
        b1_runner.can_use_b1_runner(model=good_model, draft=("d", "mtp")) is True
    )
    assert (
        b1_runner.can_use_b1_runner(model=good_model, draft=("d", "dflash")) is False
    )
    assert b1_runner.can_use_b1_runner(model=good_model, kv_bits=4) is False
    assert (
        b1_runner.can_use_b1_runner(
            model=good_model,
            apc_manager=SimpleNamespace(_heylook_apc_sync_wrapped=True),
        )
        is False
    )

    monkeypatch.setenv("HEYLOOK_B1_RUNNER", "0")
    assert b1_runner.can_use_b1_runner(model=good_model) is False
    monkeypatch.delenv("HEYLOOK_B1_RUNNER")

    bad_cache_model = SimpleNamespace(
        language_model=SimpleNamespace(make_cache=lambda: [BatchKVCache([0])])
    )
    assert b1_runner.can_use_b1_runner(model=bad_cache_model) is False


@pytest.mark.unit
def test_b1_runner_prefill_reuse_stop_processors_budget_and_abort(monkeypatch):
    from mlx_vlm import apc as apc_mod

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 11)

    model, calls = _make_toy_vlm(vocab_size=32, window_size=8)
    apc = _make_fake_apc()

    # Run 1 (cold, 12 tokens with prefill_step_size=6 -> chunked + progress + cold CacheReport).
    ids1 = mx.array([[100 + i for i in range(12)]], dtype=mx.int32)
    abort1 = _AbortSignal()
    proc_seen = []

    def proc(tokens, logits):
        proc_seen.append(int(tokens.shape[-1]))
        return logits

    chunks1 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids1,
            raw_inputs={"input_ids": ids1},
            sampler=None,
            processors=[proc],
            stop_tokens=[99],
            max_tokens=4,
            greedy=True,
            abort_event=abort1,
            prefill_step_size=6,
            detokenizer=_DummyDetok(),
            shared_prefixes=[[100 + i for i in range(6)]],
        )
    )
    assert len(chunks1) == 4
    assert chunks1[0].cache.outcome == "miss"
    assert chunks1[0].cache.cause == "cold"
    assert chunks1[-1].finish_reason == "length"
    assert abort1.progress == [(6, 12), (12, 12)]
    assert proc_seen == [0, 1, 2, 3]  # generated_only isolated prompt across all 4 sampled tokens!

    # Run 2 (warm, extends prompt + generated? Test extending prompt 12 + 3 new = 15 tokens).
    calls.clear()
    ids2 = mx.array([[100 + i for i in range(15)]], dtype=mx.int32)
    chunks2 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids2,
            raw_inputs={"input_ids": ids2},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=3,
            greedy=True,
            prefill_step_size=6,
            detokenizer=_DummyDetok(),
        )
    )
    assert chunks2[0].cache.outcome == "reused"
    assert chunks2[0].cache.cached_tokens == 12
    assert calls[0][0] == 3  # single forward pass for 3 suffix tokens only!

    # Run 3: thinking_budget + pop_end_reasoning forces closing sequence cleanly.
    class _Budget:
        def __init__(self):
            self.enable_thinking = True
            self.in_thinking = True
            self.thinking_budget = 100
            self.budget_exceeded = False
            self._forced_sequence = [30, 31]
            self._forced_index = 0
            self.forced_token_id = None

        def __call__(self, tok):
            if self.budget_exceeded and self._forced_index < len(self._forced_sequence):
                f = self._forced_sequence[self._forced_index]
                self._forced_index += 1
                self.forced_token_id = f
                return f
            self.forced_token_id = None
            return None

        def pop_forced_token_id(self):
            f = self.forced_token_id
            self.forced_token_id = None
            return f

    ids3 = mx.array([[100 + i for i in range(16)]], dtype=mx.int32)
    chunks3 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids3,
            raw_inputs={"input_ids": ids3},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=4,
            greedy=True,
            abort_event=_AbortSignal(end_reasoning_once=True),
            thinking_budget=_Budget(),
            detokenizer=_DummyDetok(),
        )
    )
    assert [c.token for c in chunks3[1:3]] == [30, 31]

    # Run 4: abort_event mid-generation stops promptly and keeps finish clean.
    ids4 = mx.array([[100 + i for i in range(17)]], dtype=mx.int32)
    chunks4 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids4,
            raw_inputs={"input_ids": ids4},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=8,
            greedy=True,
            abort_event=_AbortSignal(cancel_after=2),
            detokenizer=_DummyDetok(),
        )
    )
    assert len(chunks4) == 2

    # Run 5: stop token stops without emitting stop token as text, plus recurrent ArraysCache path.
    rec_model, rec_calls = _make_toy_vlm(vocab_size=32, window_size=8, with_recurrent=True)
    rec_apc = _make_fake_apc()
    r_ids = mx.array([[10, 11, 12, 13]], dtype=mx.int32)
    r_chunks = list(
        vlm_engine.generate(
            model=rec_model,
            processor=None,
            apc_manager=rec_apc,
            input_ids=r_ids,
            raw_inputs={"input_ids": r_ids},
            sampler=None,
            processors=[],
            stop_tokens=[10],  # offset=4 -> 4 + 4 = 8; step 2 reaches offset=6 -> 4 + 6 = 10 -> stop!
            max_tokens=6,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert [c.token for c in r_chunks] == [8, 9, None]
    assert r_chunks[-1].generation_tokens == 2
    assert r_chunks[-1].finish_reason == "stop"
    assert all(c.token != 10 for c in r_chunks)
    # Recurrent model split final tail into N-1 + 1 to snapshot N-1 cleanly.
    r_retry = list(
        vlm_engine.generate(
            model=rec_model,
            processor=None,
            apc_manager=rec_apc,
            input_ids=r_ids,
            raw_inputs={"input_ids": r_ids},
            sampler=None,
            processors=[],
            stop_tokens=[10],
            max_tokens=6,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert r_retry[0].cache.outcome == "reused"
    assert r_retry[0].cache.cached_tokens == 3


@pytest.mark.unit
def test_prefix_cache_rotating_slack_past_max_size():
    from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache

    kv = KVCache()
    buf = BufferedRotatingKVCache(max_size=8, keep=0, buffer_size=32)
    seq = mx.arange(1, 21, dtype=mx.float32).reshape(1, 1, 20, 1)
    kv.update_and_fetch(seq, seq * 2.0)
    buf.update_and_fetch(seq, seq * 2.0)
    mx.eval(kv.keys, kv.values, buf.keys, buf.values)

    store = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=4)
    tokens = list(range(100, 120))
    assert store.store(tokens, [kv, buf], extra_hash=3)
    entry = next(iter(store._entries.values()))
    # offset=20, max_size=8, valid=20 -> slack=12 (<16) -> min_match_len = 8.
    assert entry.min_match_len == 8
    assert entry.trimmable_all is False

    for match_len in (16, 19):
        diverged = [*tokens[:match_len], 999, 998]
        restored, matched, _ = store.lookup(
            diverged, extra_hash=3, use_buffered_rotating=True, buffer_size=32
        )
        assert matched == match_len
        assert restored[0].offset == match_len
        assert restored[1].offset == match_len
        assert restored[1]._idx == 8
        assert restored[1].start_position == match_len - 8
        mx.eval(restored[0].keys, restored[1].keys, restored[1].values)
        assert restored[1].keys[0, 0, :, 0].tolist() == [
            float(i) for i in range(match_len - 7, match_len + 1)
        ]
        assert restored[1].values[0, 0, :, 0].tolist() == [
            float(i * 2) for i in range(match_len - 7, match_len + 1)
        ]

    # Exact prefix match (N=20) on follow-up.
    r_exact, m_exact, _ = store.lookup(
        [*tokens, 999], extra_hash=3, use_buffered_rotating=True, buffer_size=32
    )
    assert m_exact == 20
    mx.eval(r_exact[1].keys)
    assert r_exact[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(13, 21)]
    assert r_exact[1].start_position == 12

    # Divergence below min_match_len (< 8) must miss rather than restore stale window.
    missed_restored, missed_len, _ = store.lookup([*tokens[:7], 999], extra_hash=3)
    assert missed_restored is None and missed_len == 0


@pytest.mark.unit
def test_deferred_postgen_storage_and_rotating_fallbacks():
    from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache

    kv = KVCache()
    buf = BufferedRotatingKVCache(max_size=8, keep=0, buffer_size=32)
    prompt_seq = mx.arange(1, 21, dtype=mx.float32).reshape(1, 1, 20, 1)
    kv.update_and_fetch(prompt_seq, prompt_seq)
    buf.update_and_fetch(prompt_seq, prompt_seq)
    mx.eval(kv.keys, buf.keys)

    # Short decode fits comfortably inside buffer_size=32 -> zero pre-gen fallbacks!
    assert snapshot_rotating_fallbacks([kv, buf], target_len=20, max_tokens=8) == {}

    # Simulate 8 generated tokens without compacting buf; store N=20 and N+8=28 from one buffer.
    gen_seq = mx.arange(21, 29, dtype=mx.float32).reshape(1, 1, 8, 1)
    kv.update_and_fetch(gen_seq, gen_seq)
    buf.update_and_fetch(gen_seq, gen_seq)
    mx.eval(kv.keys, buf.keys)

    store = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=8)
    prompt_tokens = list(range(200, 220))
    all_tokens = prompt_tokens + list(range(220, 228))
    assert store.store(prompt_tokens, [kv, buf], trim_tail=0, cache_trim_tail=8)
    assert store.store(all_tokens, [kv, buf], trim_tail=0)

    r_p, m_p, _ = store.lookup([*prompt_tokens[:16], 999, 998])
    assert m_p == 16
    mx.eval(r_p[1].keys)
    assert r_p[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(9, 17)]

    # Long decode exceeding buffer_size captures sliding fallback only (never KVCache).
    kv2 = KVCache()
    buf2 = BufferedRotatingKVCache(max_size=8, keep=0, buffer_size=32)
    kv2.update_and_fetch(prompt_seq, prompt_seq)
    buf2.update_and_fetch(prompt_seq, prompt_seq)
    mx.eval(kv2.keys, buf2.keys)

    fallbacks = snapshot_rotating_fallbacks([kv2, buf2], target_len=20, max_tokens=28)
    assert set(fallbacks) == {1}

    # Drive 28 tokens so buf2 compacts past target_len=20's window.
    long_gen = mx.arange(21, 49, dtype=mx.float32).reshape(1, 1, 28, 1)
    kv2.update_and_fetch(long_gen, long_gen)
    buf2.update_and_fetch(long_gen, long_gen)
    mx.eval(kv2.keys, buf2.keys)

    store2 = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=8)
    assert store2.store(
        prompt_tokens, [kv2, buf2], trim_tail=0, cache_trim_tail=28, layer_fallbacks=fallbacks
    )
    r_fb, m_fb, _ = store2.lookup([*prompt_tokens, 999])
    assert m_fb == 20 and r_fb[0].offset == 20 and r_fb[1].offset == 20
    mx.eval(r_fb[0].keys, r_fb[1].keys)
    assert r_fb[0].keys[0, 0, :, 0].tolist() == [float(i) for i in range(1, 21)]
    assert r_fb[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(13, 21)]


@pytest.mark.unit
def test_lazy_rotating_fallback_skips_pregen_capture_under_large_max_tokens_and_captures_on_compaction():
    from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache

    assert b1_runner._buffer_rotating_size(None, None) == 256
    assert (
        b1_runner._buffer_rotating_size(
            SimpleNamespace(config=SimpleNamespace(block_size=5)), None
        )
        == 256
    )
    assert (
        b1_runner._buffer_rotating_size(
            SimpleNamespace(config=SimpleNamespace(block_size=33)), None
        )
        == 512
    )

    kv = KVCache()
    buf = BufferedRotatingKVCache(max_size=8, keep=0, buffer_size=4)
    prompt_seq = mx.arange(1, 11, dtype=mx.float32).reshape(1, 1, 10, 1)
    kv.update_and_fetch(prompt_seq, prompt_seq)
    buf.update_and_fetch(prompt_seq, prompt_seq)
    mx.eval(kv.keys, buf.keys)

    # Case A: 2 decode tokens stay within max_size + buffer_size (12) -> ZERO fallback capture.
    captured: dict = {}
    for tok_val in (11, 12):
        still_monitoring = capture_rotating_fallbacks_before_evict(
            [kv, buf], target_len=10, incoming=1, captured=captured
        )
        assert still_monitoring is True
        assert captured == {}
        step = mx.full((1, 1, 1, 1), float(tok_val), dtype=mx.float32)
        kv.update_and_fetch(step, step)
        buf.update_and_fetch(step, step)
        mx.eval(kv.keys, buf.keys)

    prompt_tokens = list(range(100, 110))
    store_short = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=8)
    assert store_short.store(
        prompt_tokens, [kv, buf], trim_tail=0, cache_trim_tail=2, layer_fallbacks=captured
    )
    r_short, m_short, _ = store_short.lookup([*prompt_tokens, 999])
    assert m_short == 10
    mx.eval(r_short[0].keys, r_short[1].keys)
    assert r_short[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(3, 11)]

    # Case B: Next decode step pushes past target_size=12 -> captures pre-compaction prompt tail once.
    still_monitoring = capture_rotating_fallbacks_before_evict(
        [kv, buf], target_len=10, incoming=1, captured=captured
    )
    assert still_monitoring is False
    assert set(captured) == {1}

    for tok_val in range(13, 19):
        step = mx.full((1, 1, 1, 1), float(tok_val), dtype=mx.float32)
        kv.update_and_fetch(step, step)
        buf.update_and_fetch(step, step)
        mx.eval(kv.keys, buf.keys)
    assert buf.start_position > 2

    all_tokens = prompt_tokens + list(range(110, 118))
    store_long = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=8)
    assert store_long.store(
        prompt_tokens, [kv, buf], trim_tail=0, cache_trim_tail=8, layer_fallbacks=captured
    )
    assert store_long.store(all_tokens, [kv, buf], trim_tail=0)

    r_p, m_p, _ = store_long.lookup([*prompt_tokens, 999])
    assert m_p == 10 and r_p[0].offset == 10 and r_p[1].offset == 10
    mx.eval(r_p[0].keys, r_p[1].keys)
    assert r_p[0].keys[0, 0, :, 0].tolist() == [float(i) for i in range(1, 11)]
    assert r_p[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(3, 11)]

    r_full, m_full, _ = store_long.lookup([*all_tokens, 999])
    assert m_full == 18
    mx.eval(r_full[0].keys, r_full[1].keys)
    assert r_full[0].keys[0, 0, :, 0].tolist() == [float(i) for i in range(1, 19)]
    assert r_full[1].keys[0, 0, :, 0].tolist() == [float(i) for i in range(11, 19)]


@pytest.mark.unit
def test_b1_runner_suffix_only_embeddings_and_mrope_fallback(monkeypatch):
    from mlx_vlm import apc as apc_mod

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 21)

    model, calls = _make_toy_vlm(vocab_size=32, window_size=8)
    embed_lens: list[int] = []
    orig_embed = model.get_input_embeddings

    def tracked_embed(input_ids, pixel_values=None, mask=None, **kw):
        embed_lens.append(int(input_ids.shape[1]))
        return orig_embed(input_ids, pixel_values, mask=mask, **kw)

    model.get_input_embeddings = tracked_embed
    apc = _make_fake_apc()

    ids1 = mx.array([[100 + i for i in range(12)]], dtype=mx.int32)
    list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids1,
            raw_inputs={"input_ids": ids1, "attention_mask": mx.ones_like(ids1)},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert embed_lens == [12]

    calls.clear()
    ids2 = mx.array([[100 + i for i in range(15)]], dtype=mx.int32)
    chunks2 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids2,
            raw_inputs={"input_ids": ids2, "attention_mask": mx.ones_like(ids2)},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert chunks2[0].cache.outcome == "reused"
    assert chunks2[0].cache.cached_tokens == 12
    assert embed_lens == [12, 3]
    assert [c[0] for c in calls] == [3, 1]  # 1 suffix prefill step (3) + 1 decode step (1)

    # Extra dict key in embed output (e.g. position_ids) -> full input fallback cleanly.
    def embed_with_pos(input_ids, pixel_values=None, mask=None, **kw):
        embed_lens.append(int(input_ids.shape[1]))
        n = int(input_ids.shape[1])
        emb = mx.zeros((1, n, 4), dtype=mx.float32)
        pos = mx.broadcast_to(mx.arange(n).reshape(1, n), (3, 1, n))
        return SimpleNamespace(to_dict=lambda: {"inputs_embeds": emb, "position_ids": pos})

    model.get_input_embeddings = embed_with_pos
    ids3 = mx.array([[100 + i for i in range(16)]], dtype=mx.int32)
    chunks3 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids3,
            raw_inputs={"input_ids": ids3, "attention_mask": mx.ones_like(ids3)},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert chunks3[0].cache.cached_tokens == 15
    assert embed_lens[-2:] == [1, 16]

    # Presence of get_rope_index on language_model avoids even a speculative suffix call.
    model.get_input_embeddings = tracked_embed
    model.language_model.get_rope_index = lambda *a, **kw: (None, None)
    ids4 = mx.array([[100 + i for i in range(17)]], dtype=mx.int32)
    chunks4 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids4,
            raw_inputs={"input_ids": ids4, "attention_mask": mx.ones_like(ids4)},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert chunks4[0].cache.cached_tokens == 16
    assert embed_lens[-1] == 17


@pytest.mark.unit
def test_should_use_buffered_rotating_gating(monkeypatch):
    import types

    class _Inner:
        def _make_masks(self, h, cache, mm_token_type_ids=None):
            return []

    gemma_lm = SimpleNamespace(model=_Inner())
    gemma_lm.model._make_masks = types.MethodType(
        gemma4_vision.gemma4_make_masks, gemma_lm.model
    )
    other_lm = SimpleNamespace(model=_Inner())

    assert b1_runner._should_use_buffered_rotating(other_lm, drafting=True) is True

    monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda: True)
    assert b1_runner._should_use_buffered_rotating(gemma_lm, drafting=False) is True
    assert b1_runner._should_use_buffered_rotating(other_lm, drafting=False) is False

    monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda: False)
    assert b1_runner._should_use_buffered_rotating(gemma_lm, drafting=False) is False


@pytest.mark.unit
def test_b1_runner_recurrent_uses_capture_lengths_near_end_checkpoints(monkeypatch):
    from mlx_vlm import apc as apc_mod

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 55)

    assert b1_runner._plan_chunk_stops(
        cached=0,
        total_len=85,
        step_size=1024,
        boundaries=(),
        adjust_fn=None,
        has_recurrent=False,
    ) == ([], set())
    assert b1_runner._plan_chunk_stops(
        cached=0,
        total_len=85,
        step_size=1024,
        boundaries=(),
        adjust_fn=None,
        has_recurrent=True,
    ) == ([64], {64})

    model, calls = _make_toy_vlm(vocab_size=32, window_size=8, with_recurrent=True)
    apc = _make_fake_apc()
    apc.checkpoint_interval_tokens = 64
    apc.block_size = 16
    apc.exact_cache_min_tokens = 16

    ids1 = mx.array([[100 + i for i in range(85)]], dtype=mx.int32)
    c1 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids1,
            raw_inputs={"input_ids": ids1},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert c1[0].cache.outcome == "miss"

    # Turn 2 diverges at index 81 (simulating stripped 4-token open-turn suffix) -> hits 64.
    calls.clear()
    diverged = [100 + i for i in range(81)] + [900 + i for i in range(19)]
    ids2 = mx.array([diverged], dtype=mx.int32)
    c2 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids2,
            raw_inputs={"input_ids": ids2},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=2,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert c2[0].cache.outcome == "reused"
    assert c2[0].cache.cached_tokens == 64


@pytest.mark.unit
def test_b1_runner_mtp_natural_eos_caches_postgen_on_bonus_and_draft_stop(monkeypatch):
    """MTP replies ending naturally on EOS (bonus or accepted draft) must cache [prompt + gen]."""
    from mlx_vlm import apc as apc_mod
    from mlx_vlm.speculative import utils as spec_utils
    from heylook_llm.providers.common.speculative_session import (
        SpeculativeStepResult,
        _apply_thinking_budget_single,
    )

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 123)
    monkeypatch.setattr(spec_utils, "speculative_prefill_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(spec_utils, "speculative_hidden_state", lambda *_a, **_k: mx.zeros((1, 1, 1)))

    # Case 1: bonus token is EOS (raw_toks = [11, 12, 99], stop=99 at index 2).
    # Prefill emits first_tok=14, SpeculativeSession commits [14, 15, 16] -> offset = 10 + 3 = 13.
    class _FakeSpeculativeSession:
        planned_rounds: list[list[int]] = []

        def __init__(self, model, draft, prompt_cache, hidden, first_token, block_size, *_a, **_k):
            self.cache = prompt_cache
            self.bonus = int(first_token)
            self.position = int(prompt_cache[0].offset)
            self._rounds = list(self.planned_rounds)

        def step(self, budget, position, *, stop_tokens, thinking_budget_criteria=None):
            raw = list(self._rounds.pop(0))
            accepted = max(0, len(raw) - 1)
            accepted, raw = _apply_thinking_budget_single(accepted, raw, thinking_budget_criteria)
            stop_idx = next((i for i, t in enumerate(raw) if t in stop_tokens), None)
            prev_bonus = self.bonus
            if stop_idx is not None:
                emitted = raw[:stop_idx]
                commit_acc = stop_idx
                committed = [prev_bonus, *emitted]
                finish_reason = "stop"
            else:
                emitted = raw
                commit_acc = accepted
                committed = [prev_bonus, *emitted[:-1]] if emitted else [prev_bonus]
                finish_reason = "length" if len(emitted) >= budget else None
            k = mx.arange(len(committed), dtype=mx.float32).reshape(1, 1, len(committed), 1)
            for c in self.cache:
                c.update_and_fetch(k, k)
            self.position += len(committed)
            self.bonus = raw[-1]
            return SpeculativeStepResult(
                emitted_tokens=emitted,
                committed_tokens=committed,
                finish_reason=finish_reason,
                accepted_count=commit_acc,
                stopped_on_eos=stop_idx is not None,
            )

        def drafter_cache_for_store(self, target_token_len: int):
            return {}

    monkeypatch.setattr(b1_runner, "SpeculativeSession", _FakeSpeculativeSession)
    draft_stub = (SimpleNamespace(config=SimpleNamespace(block_size=4)), "mtp")

    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    apc = _make_fake_apc()
    prompt1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ids1 = mx.array([prompt1], dtype=mx.int32)

    # first_tok = (offset=10 + 4) % 32 = 14; round commits [14, 15, 16], bonus EOS=99.
    _FakeSpeculativeSession.planned_rounds = [[15, 16, 99]]
    c1 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids1,
            raw_inputs={"input_ids": ids1},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=16,
            greedy=True,
            detokenizer=_DummyDetok(),
            draft=draft_stub,
        )
    )
    assert [c.token for c in c1] == [14, 15, 16]
    assert c1[-1].finish_reason == "stop"

    # Turn 2 continues after the Turn 1 assistant reply [14, 15, 16] -> must reuse all 13 tokens!
    prompt2 = [*prompt1, 14, 15, 16, 20, 21]
    ids2 = mx.array([prompt2], dtype=mx.int32)
    _FakeSpeculativeSession.planned_rounds = [[99]]
    c2 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids2,
            raw_inputs={"input_ids": ids2},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=16,
            greedy=True,
            detokenizer=_DummyDetok(),
            draft=draft_stub,
        )
    )
    assert c2[0].cache.outcome == "reused"
    assert c2[0].cache.cached_tokens == 13

    # Case 2: EOS lands on an accepted draft token mid-round (raw_toks = [25, 99, 27]).
    # SpeculativeSession stops before commit so only [first_tok, 25] (12 tokens total) enter cache.
    apc2 = _make_fake_apc()
    _FakeSpeculativeSession.planned_rounds = [[25, 99, 27]]
    c3 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc2,
            input_ids=ids1,
            raw_inputs={"input_ids": ids1},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=16,
            greedy=True,
            detokenizer=_DummyDetok(),
            draft=draft_stub,
        )
    )
    assert [c.token for c in c3] == [14, 25]
    assert c3[-1].finish_reason == "stop"

    ids4 = mx.array([[*prompt1, 14, 25, 30, 31]], dtype=mx.int32)
    _FakeSpeculativeSession.planned_rounds = [[99]]
    c4 = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc2,
            input_ids=ids4,
            raw_inputs={"input_ids": ids4},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=16,
            greedy=True,
            detokenizer=_DummyDetok(),
            draft=draft_stub,
        )
    )
    assert c4[0].cache.outcome == "reused"
    assert c4[0].cache.cached_tokens == 12


@pytest.mark.unit
def test_prefix_cache_enforces_memory_reserve_bytes():
    from mlx_vlm.models.cache import KVCache

    headroom_state = {"bytes": 10_000}
    store = PrefixCacheStore(
        memory_max_bytes=10 * 1024 * 1024,
        max_entries=16,
        memory_reserve_bytes=8_000,
        headroom_fn=lambda: headroom_state["bytes"],
    )

    def _kv_with(n: int) -> list[KVCache]:
        c = KVCache()
        x = mx.ones((1, 1, n, 4), dtype=mx.float32)
        c.update_and_fetch(x, x)
        mx.eval(c.keys, c.values)
        return [c]

    assert store.store(list(range(1, 33)), _kv_with(32), extra_hash=1)
    assert store.store(list(range(101, 133)), _kv_with(32), extra_hash=1)
    assert store.stats_snapshot()["entries"] == 2
    assert store.stats_snapshot()["memory_evictions"] == 0

    # Drop headroom below reserve -> storing next entry must evict oldest entries.
    headroom_state["bytes"] = 7_500
    assert store.store(list(range(201, 233)), _kv_with(32), extra_hash=1)
    snap = store.stats_snapshot()
    assert snap["memory_evictions"] >= 1
    assert snap["entries"] < 3


@pytest.mark.unit
def test_prefix_cache_compacts_oversized_kv_slices_when_allowed():
    from mlx_vlm.models.cache import KVCache

    kv = KVCache()
    big = mx.arange(600, dtype=mx.float32).reshape(1, 1, 600, 1)
    kv.update_and_fetch(big, big * 2.0)
    mx.eval(kv.keys, kv.values)
    assert kv.keys.shape[2] >= 600

    # Shared-sibling store (allow_kv_compact=False) leaves needs_compact=False.
    store_shared = PrefixCacheStore(memory_max_bytes=10 * 1024 * 1024, max_entries=4)
    assert store_shared.store(
        list(range(100)),
        [kv],
        extra_hash=1,
        cache_trim_tail=500,
        allow_kv_compact=False,
    )
    assert store_shared.compact_pending() == 0

    # Standalone store (allow_kv_compact=True, e.g. aborted gen) compacts oversized KV slice.
    store_aborted = PrefixCacheStore(memory_max_bytes=10 * 1024 * 1024, max_entries=4)
    assert store_aborted.store(
        list(range(100)),
        [kv],
        extra_hash=1,
        cache_trim_tail=500,
        allow_kv_compact=True,
    )
    assert store_aborted.compact_pending() == 1
    restored, matched, _ = store_aborted.lookup(list(range(100)) + [999], extra_hash=1)
    assert matched == 100
    assert restored[0].keys.shape[2] == 100


@pytest.mark.unit
def test_b1_runner_config_eos_union_and_zero_max_tokens(monkeypatch):
    from mlx_vlm import apc as apc_mod

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 7)

    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    # Prompt len 6 -> first_tok = (6 + 4) = 10, next = 11. Put 11 ONLY in model.config.eos_token_id.
    model.config = SimpleNamespace(eos_token_id=[11])
    apc = _make_fake_apc()
    ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    chunks = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids,
            raw_inputs={"input_ids": ids},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=8,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert [c.token for c in chunks] == [10, None]
    assert chunks[-1].finish_reason == "stop"

    # max_tokens=0 emits no generated tokens.
    zero_chunks = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=apc,
            input_ids=ids,
            raw_inputs={"input_ids": ids},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=0,
            greedy=True,
            detokenizer=_DummyDetok(),
        )
    )
    assert len(zero_chunks) == 1
    assert zero_chunks[0].token is None
    assert zero_chunks[0].generation_tokens == 0
    assert zero_chunks[0].finish_reason == "length"


@pytest.mark.unit
def test_mlx_provider_describe_observed_reports_b1_runner_and_cow_prefix(monkeypatch):
    from heylook_llm.providers.mlx_provider import MLXProvider

    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    rec_model, _ = _make_toy_vlm(vocab_size=32, window_size=8, with_recurrent=True)
    apc = _make_fake_apc()

    provider = MLXProvider.__new__(MLXProvider)
    provider.unload = lambda **_k: None
    provider.is_diffusion = False
    provider.model = model
    provider._apc = apc
    provider._apc_mode = "blocks"
    provider._draft = None
    provider.drafter_skipped = None
    provider.loaded_chat_template = None
    provider.config = {"model_path": "/tmp/demo"}

    monkeypatch.delenv("HEYLOOK_B1_RUNNER", raising=False)
    obs = provider.describe_observed()
    assert obs.decoding["runner"].value == "b1_runner"
    assert obs.cache["reuse_mode"].value == "cow_prefix"
    assert obs.cache["image_reuse"].value is True

    provider.model = rec_model
    obs_rec = provider.describe_observed()
    assert obs_rec.decoding["runner"].value == "b1_runner"
    assert obs_rec.cache["reuse_mode"].value == "cow_prefix_recurrent"

    monkeypatch.setenv("HEYLOOK_B1_RUNNER", "0")
    provider.model = model
    obs_fb = provider.describe_observed()
    assert obs_fb.decoding["runner"].value == "batch_generator"
    assert "HEYLOOK_B1_RUNNER=0" in obs_fb.decoding["runner"].source
    assert obs_fb.cache["reuse_mode"].value == "blocks"
    assert obs_fb.cache["image_reuse"].value is False


@pytest.mark.unit
def test_b1_runner_standby_end_thinking_criteria_not_enforced_until_triggered(monkeypatch):
    """Standby end_thinking_criteria (budget=0) must stay inactive until pop_end_reasoning()."""
    from mlx_vlm import apc as apc_mod
    from mlx_vlm.speculative import utils as spec_utils
    from heylook_llm.providers.common.speculative_session import (
        SpeculativeStepResult,
        _apply_thinking_budget_single,
    )

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 123)
    monkeypatch.setattr(spec_utils, "speculative_prefill_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(spec_utils, "speculative_hidden_state", lambda *_a, **_k: mx.zeros((1, 1, 1)))

    class _StandbyCriteria:
        def __init__(self, *, preopened: bool = True):
            self.thinking_budget = 0
            self.thinking_start_token_id = 15
            self.thinking_end_token_id = 29
            self.in_thinking = preopened
            self.budget_exceeded = False
            self._forced_sequence = [28, 29]
            self._forced_index = 0
            self.calls = 0

        def __call__(self, tok: int):
            self.calls += 1
            if not self.in_thinking and tok == self.thinking_start_token_id:
                self.in_thinking = True
            elif self.in_thinking and tok == self.thinking_end_token_id:
                self.in_thinking = False
            elif self.in_thinking:
                self.budget_exceeded = True

        def pop_forced_token_id(self):
            if self.budget_exceeded and self._forced_index < len(self._forced_sequence):
                tok = self._forced_sequence[self._forced_index]
                self._forced_index += 1
                return tok
            return None

    # 1. MTP drafting: thinking_budget=None + standby end_thinking_criteria must pass
    #    thinking_budget_criteria=None until pop_end_reasoning() fires.
    seen_crit_args: list[Any] = []

    class _FakeSpeculativeSession:
        planned_rounds: list[list[int]] = []

        def __init__(self, model, draft, prompt_cache, hidden, first_token, block_size, *_a, **_k):
            self.cache = prompt_cache
            self.bonus = int(first_token)
            self.position = int(prompt_cache[0].offset)
            self._rounds = list(self.planned_rounds)

        def step(self, budget, position, *, stop_tokens, thinking_budget_criteria=None):
            seen_crit_args.append(thinking_budget_criteria)
            raw = list(self._rounds.pop(0))
            accepted = max(0, len(raw) - 1)
            accepted, raw = _apply_thinking_budget_single(accepted, raw, thinking_budget_criteria)
            stop_idx = next((i for i, t in enumerate(raw) if t in stop_tokens), None)
            prev_bonus = self.bonus
            if stop_idx is not None:
                emitted = raw[:stop_idx]
                commit_acc = stop_idx
                committed = [prev_bonus, *emitted]
                finish_reason = "stop"
            else:
                emitted = raw
                commit_acc = accepted
                committed = [prev_bonus, *emitted[:-1]] if emitted else [prev_bonus]
                finish_reason = "length" if len(emitted) >= budget else None
            k = mx.arange(len(committed), dtype=mx.float32).reshape(1, 1, len(committed), 1)
            for c in self.cache:
                c.update_and_fetch(k, k)
            self.position += len(committed)
            self.bonus = raw[-1]
            return SpeculativeStepResult(
                emitted_tokens=emitted,
                committed_tokens=committed,
                finish_reason=finish_reason,
                accepted_count=commit_acc,
                stopped_on_eos=stop_idx is not None,
            )

        def drafter_cache_for_store(self, target_token_len: int):
            return {}

    monkeypatch.setattr(b1_runner, "SpeculativeSession", _FakeSpeculativeSession)
    draft_stub = (SimpleNamespace(config=SimpleNamespace(block_size=4)), "mtp")

    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    # Round 1: no end_reasoning -> crit_arg is None, standby criteria untouched.
    standby_mtp = _StandbyCriteria(preopened=True)
    _FakeSpeculativeSession.planned_rounds = [[11, 12], [13, 99]]
    chunks = list(
        vlm_engine.generate(
            model=model,
            processor=None,
            apc_manager=_make_fake_apc(),
            input_ids=ids,
            raw_inputs={"input_ids": ids},
            sampler=None,
            processors=[],
            stop_tokens=[99],
            max_tokens=16,
            greedy=True,
            detokenizer=_DummyDetok(),
            thinking_budget=None,
            end_thinking_criteria=standby_mtp,
            draft=draft_stub,
        )
    )
    assert seen_crit_args == [None, None]
    assert standby_mtp.calls == 0
    assert standby_mtp.budget_exceeded is False
    assert [c.token for c in chunks] == [10, 11, 12, 13]

    # Round 2: standby starts outside thinking, enters via token 15 in round 1,
    # then pop_end_reasoning fires before round 2 -> crit_arg activates standby_mtp2.
    seen_crit_args.clear()
    standby_mtp2 = _StandbyCriteria(preopened=False)
    abort = _AbortSignal()
    _FakeSpeculativeSession.planned_rounds = [[15, 16], [28, 29, 99]]
    gen = vlm_engine.generate(
        model=model,
        processor=None,
        apc_manager=_make_fake_apc(),
        input_ids=ids,
        raw_inputs={"input_ids": ids},
        sampler=None,
        processors=[],
        stop_tokens=[99],
        max_tokens=16,
        greedy=True,
        abort_event=abort,
        detokenizer=_DummyDetok(),
        thinking_budget=None,
        end_thinking_criteria=standby_mtp2,
        draft=draft_stub,
    )
    out = [next(gen), next(gen), next(gen)]  # first_tok + round 1 ([15, 16])
    assert standby_mtp2.in_thinking is True
    abort.end_reasoning_once = True
    out.extend(list(gen))
    assert seen_crit_args == [None, standby_mtp2]
    assert standby_mtp2.budget_exceeded is True

    # 2. Non-drafting: pop_end_reasoning with thinking_budget=None and standby
    #    end_thinking_criteria must force ALL tokens of _forced_sequence ([28, 29]).
    standby_ar = _StandbyCriteria(preopened=True)
    abort_ar = _AbortSignal()
    gen_ar = vlm_engine.generate(
        model=model,
        processor=None,
        apc_manager=_make_fake_apc(),
        input_ids=ids,
        raw_inputs={"input_ids": ids},
        sampler=None,
        processors=[],
        stop_tokens=[99],
        max_tokens=5,
        greedy=True,
        abort_event=abort_ar,
        detokenizer=_DummyDetok(),
        thinking_budget=None,
        end_thinking_criteria=standby_ar,
    )
    ar_toks = [next(gen_ar).token]
    assert standby_ar.budget_exceeded is False
    abort_ar.end_reasoning_once = True
    ar_toks.extend(c.token for c in gen_ar)
    assert ar_toks[1:3] == [28, 29]
    assert standby_ar.in_thinking is False


@pytest.mark.unit
def test_b1_runner_and_caches_survive_quarantined_worker_thread_switch(monkeypatch):
    """Regression for 'RuntimeError: There is no Stream(gpu, N) in current thread.'
    when Worker A is quarantined mid-stream (staying alive and retaining its
    thread-local generation_stream) and Worker B reuses PrefixCacheStore,
    VisionFeatureCache, and Gemma 4 Router state."""
    import queue
    import threading
    from heylook_llm.providers.common.vision_feature_cache import VisionFeatureCache

    new_tls = getattr(mx, "new_thread_local_stream", None)
    if callable(new_tls):
        tls_stream = new_tls(mx.default_device())
        monkeypatch.setattr(b1_runner, "_generation_stream", lambda: tls_stream)

    apc = _make_fake_apc()
    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    vcache = VisionFeatureCache(max_entries=4)
    ids1 = mx.array([[10, 11, 12, 13, 14, 15]], dtype=mx.int32)
    ids2 = mx.array([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=mx.int32)
    pix = mx.ones((1, 2, 2, 3), dtype=mx.float32)

    # Simulate model setting _rope_deltas lazily during forward
    orig_call = model.language_model.__call__

    def _call_with_rope_deltas(input_ids, cache=None, **kw):
        model.language_model._rope_deltas = mx.zeros((1, 1), dtype=mx.int32) + 1
        return orig_call(input_ids, cache=cache, **kw)

    model.language_model.__call__ = _call_with_rope_deltas

    ready_q: queue.Queue = queue.Queue()
    release_q: queue.Queue = queue.Queue()
    errors: list[Exception] = []

    def worker_a():
        try:
            with mx.stream(b1_runner._generation_stream()):
                # Store an unevaluated op into VisionFeatureCache on Worker A's stream
                vcache.put("same-img", mx.ones((1, 4, 4), dtype=mx.float32) + 2.0)
                gen1 = vlm_engine.generate(
                    model=model,
                    processor=None,
                    apc_manager=apc,
                    input_ids=ids1,
                    raw_inputs={"input_ids": ids1, "pixel_values": pix},
                    sampler=None,
                    processors=[],
                    stop_tokens=[99],
                    max_tokens=8,
                    greedy=True,
                    detokenizer=_DummyDetok(),
                )
                first = next(gen1)
                assert first.token is not None
                # Close mid-stream on Worker A (just like a cancelled/quarantined request)
                gen1.close()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            ready_q.put(True)
            # Stay alive like a quarantined _executor_pool worker!
            release_q.get(timeout=10)

    t_a = threading.Thread(target=worker_a, daemon=True)
    t_a.start()
    ready_q.get(timeout=10)
    assert not errors

    out_chunks = []

    def worker_b():
        try:
            with mx.stream(b1_runner._generation_stream()):
                cached_feat = vcache.get("same-img")
                assert cached_feat is not None
                mx.eval(cached_feat + 1.0)
                gen2 = vlm_engine.generate(
                    model=model,
                    processor=None,
                    apc_manager=apc,
                    input_ids=ids2,
                    raw_inputs={"input_ids": ids2, "pixel_values": pix},
                    sampler=None,
                    processors=[],
                    stop_tokens=[99],
                    max_tokens=4,
                    greedy=True,
                    detokenizer=_DummyDetok(),
                )
                out_chunks.extend(list(gen2))
        except Exception as exc:
            errors.append(exc)

    try:
        t_b = threading.Thread(target=worker_b, daemon=True)
        t_b.start()
        t_b.join(timeout=10)
    finally:
        release_q.put(True)
        t_a.join(timeout=5)

    assert not errors, f"Worker B crashed with cross-thread stream error: {errors}"
    assert len(out_chunks) >= 2
    assert out_chunks[0].cache is not None
    assert out_chunks[0].cache.outcome == "reused"
    assert out_chunks[0].cache.cached_tokens == 6


@pytest.mark.unit
def test_detach_for_cache_handles_nested_and_snapshots():
    import queue
    import threading
    from heylook_llm.providers.common.prefix_cache import (
        _LayerSnapshot,
        detach_for_cache,
    )

    new_tls = getattr(mx, "new_thread_local_stream", None)
    tls_a = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())
    tls_b = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())

    box: dict = {}
    ready_q: queue.Queue = queue.Queue()
    release_q: queue.Queue = queue.Queue()
    errors: list[Exception] = []

    def worker_a():
        try:
            with mx.stream(tls_a):
                base = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
                snap = _LayerSnapshot(
                    kind="cache_list",
                    children=(
                        _LayerSnapshot(
                            kind="kv",
                            keys=base[..., :4, :] + 1.0,
                            values=base[..., :4, :] * 2.0,
                            offset=4,
                            idx=4,
                        ),
                        _LayerSnapshot(
                            kind="arrays",
                            arrays=[None, base[..., 2:5, :] + 10.0],
                        ),
                    ),
                )
                payload = {
                    "snap": snap,
                    "list": [None, "ignored", (base[..., :2, :] + 3.0,)],
                }
                detach_for_cache(None, payload)
                box["payload"] = payload
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            ready_q.put(True)
            release_q.get(timeout=10)

    t_a = threading.Thread(target=worker_a, daemon=True)
    t_a.start()
    ready_q.get(timeout=10)
    assert not errors

    def worker_b():
        try:
            with mx.stream(tls_b):
                p = box["payload"]
                kv_snap = p["snap"].children[0]
                arr_snap = p["snap"].children[1]
                assert (kv_snap.keys + 0.0).reshape(-1).tolist() == [1.0, 2.0, 3.0, 4.0]
                assert (arr_snap.arrays[1] + 0.0).reshape(-1).tolist() == [12.0, 13.0, 14.0]
                assert (p["list"][2][0] + 0.0).reshape(-1).tolist() == [3.0, 4.0]
        except Exception as exc:
            errors.append(exc)

    try:
        t_b = threading.Thread(target=worker_b, daemon=True)
        t_b.start()
        t_b.join(timeout=10)
    finally:
        release_q.put(True)
        t_a.join(timeout=5)

    assert not errors, f"Worker B hit stream error after detach_for_cache: {errors}"


@pytest.mark.unit
def test_prefix_cache_cow_physical_bytes_and_sibling_compaction():
    from mlx_vlm.models.cache import KVCache

    # Case A: [prompt] (100 tok, allow_kv_compact=False) + [prompt+gen] (150 tok,
    # allow_kv_compact=True) sharing one oversized backing slab (backing_len >= 600).
    kv = KVCache()
    chunk = mx.arange(1, 151, dtype=mx.float32).reshape(1, 1, 150, 1)
    k = mx.broadcast_to(chunk, (1, 1, 150, 2))
    kv.update_and_fetch(k, k * 2.0)
    mx.eval(kv.keys, kv.values)
    mx.eval(*kv.prefix_cache_reserve(600))
    assert kv.keys.shape[2] >= 600
    assert kv.offset == 150

    bytes_per_tok = 2 * 4 * 2  # K + V, 2 dims, float32
    bytes_150 = 150 * bytes_per_tok  # 2400 bytes

    # Budget fits 150 tok (2600B) cleanly, but would evict [prompt] if 100+150 (4000B) double-counted.
    store = PrefixCacheStore(memory_max_bytes=bytes_150 + 200, max_entries=8)
    tokens = list(range(1000, 1150))

    assert store.store(
        tokens,
        [kv],
        extra_hash=5,
        cache_trim_tail=50,
        trim_tail=50,
        allow_kv_compact=False,
    )
    assert store.total_bytes() == 100 * bytes_per_tok

    assert store.store(
        tokens,
        [kv],
        extra_hash=5,
        cache_trim_tail=0,
        trim_tail=0,
        allow_kv_compact=True,
    )
    snap_stats = store.stats_snapshot()
    assert snap_stats["entries"] == 2
    assert snap_stats["memory_evictions"] == 0
    assert store.total_bytes() == bytes_150

    entries = list(store._entries.values())
    assert entries[0].layers[0].backing_token is entries[1].layers[0].backing_token

    # Sibling-aware compact_pending compacts both siblings onto ONE 150-token buffer.
    assert store.compact_pending() == 1
    assert store.compact_pending() == 0
    entries = list(store._entries.values())
    assert entries[0].layers[0].backing_token is entries[1].layers[0].backing_token
    assert entries[0].layers[0].backing_len == 150
    assert entries[1].layers[0].backing_len == 150
    assert store.total_bytes() == bytes_150

    r_prompt, m_prompt, _ = store.lookup(tokens[:100] + [9999], extra_hash=5)
    assert m_prompt == 100
    assert r_prompt[0].keys.shape[2] == 100
    assert r_prompt[0].keys[0, 0, :3, 0].tolist() == [1.0, 2.0, 3.0]
    assert r_prompt[0].keys[0, 0, -2:, 0].tolist() == [99.0, 100.0]

    r_full, m_full, _ = store.lookup(tokens + [9999], extra_hash=5)
    assert m_full == 150
    assert r_full[0].keys.shape[2] == 150
    assert r_full[0].keys[0, 0, -2:, 0].tolist() == [149.0, 150.0]

    # Case B: widest sibling (500 tok on 512-tok backing slab) already fits within +256 slack,
    # so shorter sibling (100 tok, allow_kv_compact=True) must NOT copy a duplicate buffer.
    kv2 = KVCache()
    chunk2 = mx.arange(1, 501, dtype=mx.float32).reshape(1, 1, 500, 1)
    k2 = mx.broadcast_to(chunk2, (1, 1, 500, 2))
    kv2.update_and_fetch(k2, k2 * 2.0)
    mx.eval(kv2.keys, kv2.values)
    assert kv2.keys.shape[2] == 512

    store2 = PrefixCacheStore(memory_max_bytes=1024 * 1024, max_entries=8)
    toks2 = list(range(2000, 2500))
    assert store2.store(toks2, [kv2], extra_hash=1, trim_tail=0, allow_kv_compact=False)
    assert store2.store(
        toks2, [kv2], extra_hash=1, trim_tail=400, cache_trim_tail=400, allow_kv_compact=True
    )
    assert store2.compact_pending() == 0


@pytest.mark.unit
def test_prefix_cache_drafter_cache_store_restore_and_cross_worker_detach():
    import queue
    import threading
    from mlx_vlm.models.cache import ArraysCache, KVCache

    new_tls = getattr(mx, "new_thread_local_stream", None)
    tls_a = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())
    tls_b = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())

    store = PrefixCacheStore(memory_max_bytes=64 * 1024 * 1024, max_entries=8)
    tokens = list(range(500, 600))
    ready_q: queue.Queue = queue.Queue()
    release_q: queue.Queue = queue.Queue()
    errors: list[Exception] = []

    def worker_a():
        try:
            with mx.stream(tls_a):
                t_kv = KVCache()
                tk = mx.arange(1, 101, dtype=mx.float32).reshape(1, 1, 100, 1)
                t_kv.update_and_fetch(tk, tk * 2.0)
                t_arr = ArraysCache(1)
                t_arr.cache[0] = mx.array([[42.0, 43.0]], dtype=mx.float32)

                d_kv = KVCache()
                dk = mx.arange(1, 100, dtype=mx.float32).reshape(1, 1, 99, 1)
                d_kv.update_and_fetch(dk, dk * 3.0)
                mx.eval(t_kv.keys, t_kv.values, t_arr.cache[0], d_kv.keys, d_kv.values)
                mx.eval(*d_kv.prefix_cache_reserve(512))

                assert store.store(
                    tokens,
                    [t_kv, t_arr],
                    extra_hash=9,
                    drafter_cache=[d_kv],
                )
                assert store.compact_pending() >= 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            ready_q.put(True)
            release_q.get(timeout=10)

    t_a = threading.Thread(target=worker_a, daemon=True)
    t_a.start()
    ready_q.get(timeout=10)
    assert not errors

    def worker_b():
        try:
            with mx.stream(tls_b):
                res = store.lookup_full(tokens + [999], extra_hash=9)
                assert res.matched_tokens == 100
                assert res.layers is not None and res.layers[0].offset == 100
                assert res.drafter_layers is not None
                assert res.drafter_layers[0].offset == 99
                assert res.drafter_layers[0].keys.shape[2] == 99

                ext = mx.array([[[[777.0]]]], dtype=mx.float32)
                res.layers[0].update_and_fetch(ext, ext)
                res.drafter_layers[0].update_and_fetch(ext, ext)
                mx.eval(
                    res.layers[0].keys,
                    res.layers[1].cache[0],
                    res.drafter_layers[0].keys,
                )
                assert res.drafter_layers[0].offset == 100
                assert res.drafter_layers[0].state[0][0, 0, 0, 0].item() == 1.0
                assert res.drafter_layers[0].state[0][0, 0, -1, 0].item() == 777.0
        except Exception as exc:
            errors.append(exc)

    try:
        t_b = threading.Thread(target=worker_b, daemon=True)
        t_b.start()
        t_b.join(timeout=10)
    finally:
        release_q.put(True)
        t_a.join(timeout=5)

    assert not errors, f"Worker B failed restoring drafter cache: {errors}"

    # Pure trimmable target verifies partial prefix trim & drafter_token_len - drop_tail <= 0.
    store_trim = PrefixCacheStore(memory_max_bytes=64 * 1024 * 1024, max_entries=8)
    t_only = KVCache()
    tk = mx.arange(1, 101, dtype=mx.float32).reshape(1, 1, 100, 1)
    t_only.update_and_fetch(tk, tk)
    d_only = KVCache()
    dk = mx.arange(1, 100, dtype=mx.float32).reshape(1, 1, 99, 1)
    d_only.update_and_fetch(dk, dk)
    mx.eval(t_only.keys, t_only.values, d_only.keys, d_only.values)
    assert store_trim.store(tokens, [t_only], extra_hash=1, drafter_cache=[d_only])

    partial = store_trim.lookup_full(tokens[:80] + [999], extra_hash=1)
    assert partial.matched_tokens == 80
    assert partial.drafter_layers is not None
    assert partial.drafter_layers[0].offset == 79
    assert partial.drafter_layers[0].keys[0, 0, -1, 0].item() == 79.0

    too_short = store_trim.lookup_full(tokens[:1] + [999], extra_hash=1)
    assert too_short.matched_tokens == 1
    assert too_short.drafter_layers is None


@pytest.mark.unit
def test_cross_worker_torture_all_cache_classes_rotating_eviction_and_cancel(monkeypatch):
    import queue
    import threading
    from mlx_vlm import apc as apc_mod
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 31)

    new_tls = getattr(mx, "new_thread_local_stream", None)
    tls_a = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())
    tls_b = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())

    def _run_cross_worker(model, prompt_1, prompt_2, *, use_buffered_rot: bool, cancel_after_tokens: int | None):
        apc_main = _make_fake_apc()
        apc_cold = _make_fake_apc()

        if use_buffered_rot:
            monkeypatch.setattr(b1_runner, "_should_use_buffered_rotating", lambda *a, **k: True)
            monkeypatch.setattr(b1_runner, "_buffer_rotating_size", lambda *a, **k: 4)
        else:
            monkeypatch.setattr(b1_runner, "_should_use_buffered_rotating", lambda *a, **k: False)

        # Reference cold run of prompt_2 for greedy token parity.
        cold_tokens = [
            c.token
            for c in vlm_engine.generate(
                model=model,
                processor=None,
                apc_manager=apc_cold,
                input_ids=prompt_2,
                raw_inputs={"input_ids": prompt_2},
                sampler=None,
                processors=[],
                stop_tokens=[99],
                max_tokens=4,
                greedy=True,
                detokenizer=_DummyDetok(),
            )
            if c.token is not None
        ]

        ready_q: queue.Queue = queue.Queue()
        release_q: queue.Queue = queue.Queue()
        errors: list[Exception] = []
        worker_b_chunks = []

        def worker_a():
            try:
                monkeypatch.setattr(b1_runner, "_generation_stream", lambda: tls_a)
                with mx.stream(tls_a):
                    gen1 = vlm_engine.generate(
                        model=model,
                        processor=None,
                        apc_manager=apc_main,
                        input_ids=prompt_1,
                        raw_inputs={"input_ids": prompt_1},
                        sampler=None,
                        processors=[],
                        stop_tokens=[99],
                        max_tokens=8,
                        greedy=True,
                        detokenizer=_DummyDetok(),
                    )
                    if cancel_after_tokens is not None:
                        for _ in range(cancel_after_tokens):
                            next(gen1)
                        gen1.close()
                    else:
                        list(gen1)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                ready_q.put(True)
                release_q.get(timeout=10)

        t_a = threading.Thread(target=worker_a, daemon=True)
        t_a.start()
        ready_q.get(timeout=10)
        assert not errors

        def worker_b():
            try:
                monkeypatch.setattr(b1_runner, "_generation_stream", lambda: tls_b)
                with mx.stream(tls_b):
                    worker_b_chunks.extend(
                        vlm_engine.generate(
                            model=model,
                            processor=None,
                            apc_manager=apc_main,
                            input_ids=prompt_2,
                            raw_inputs={"input_ids": prompt_2},
                            sampler=None,
                            processors=[],
                            stop_tokens=[99],
                            max_tokens=4,
                            greedy=True,
                            detokenizer=_DummyDetok(),
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        try:
            t_b = threading.Thread(target=worker_b, daemon=True)
            t_b.start()
            t_b.join(timeout=10)
        finally:
            release_q.put(True)
            t_a.join(timeout=5)

        assert not errors, f"Worker B hit cross-worker error: {errors}"
        assert worker_b_chunks and worker_b_chunks[0].cache is not None
        assert worker_b_chunks[0].cache.outcome == "reused"
        warm_tokens = [c.token for c in worker_b_chunks if c.token is not None]
        assert warm_tokens == cold_tokens

    # 1. Pure KVCache model.
    kv_only_model, _ = _make_toy_vlm(vocab_size=32, window_size=8, with_recurrent=False)
    kv_only_model.language_model.make_cache = lambda: [KVCache()]
    p1 = mx.array([[10 + i for i in range(6)]], dtype=mx.int32)
    p2 = mx.array([[10 + i for i in range(8)]], dtype=mx.int32)
    _run_cross_worker(kv_only_model, p1, p2, use_buffered_rot=False, cancel_after_tokens=None)

    # 2. BufferedRotatingKVCache crossing max_size=4 eviction + mid-stream gen.close() cancel.
    rot_model, _ = _make_toy_vlm(vocab_size=32, window_size=4, with_recurrent=False)
    _run_cross_worker(rot_model, p1, p2, use_buffered_rot=True, cancel_after_tokens=6)

    # 3. Hybrid ArraysCache recurrent model (completes on Worker A -> reused on Worker B).
    rec_model, _ = _make_toy_vlm(vocab_size=32, window_size=8, with_recurrent=True)
    _run_cross_worker(rec_model, p1, p2, use_buffered_rot=False, cancel_after_tokens=None)


def _make_toy_stateful_mtp_pair(*, vocab_size: int = 32, hidden_dim: int = 8):
    """Toy recurrent target (KVCache + ArraysCache) paired with a stateful KVCache drafter."""
    from mlx_vlm.models.cache import ArraysCache, KVCache
    from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction

    mx.random.seed(19)
    embed_w = mx.random.normal((vocab_size, hidden_dim)).astype(mx.float32) * 0.25
    head_w = mx.random.normal((vocab_size, hidden_dim)).astype(mx.float32) * 0.25
    draft_fc = mx.random.normal((hidden_dim, 2 * hidden_dim)).astype(mx.float32) * 0.25
    draft_head = mx.random.normal((vocab_size, hidden_dim)).astype(mx.float32) * 0.25

    class _RecurrentLM:
        def __init__(self):
            self.config = SimpleNamespace(eos_token_id=vocab_size - 1)
            self.planned_targets: list[list[int]] = []

        def make_cache(self):
            return [KVCache(), ArraysCache(size=1)]

        def _hidden_for_token(self, tok: int, pos: int) -> mx.array:
            base = embed_w[int(tok) % vocab_size]
            return base + float(pos + 1) * 0.05

        def speculative_verify_hidden(self, inputs: mx.array, cache):
            seq_len = int(inputs.shape[1])
            start_pos = int(cache[0].offset)
            toks = [int(t) for t in inputs[0].tolist()]

            # Speculative transaction across KVCache + ArraysCache(size=1).
            arr_cache = cache[1]
            arr_cache.start_speculation(seq_len)
            state_steps = []
            prev = arr_cache.cache[0] if arr_cache.cache[0] is not None else mx.zeros((1, 2))
            cur = prev
            hidden_steps = []
            k_list = []
            v_list = []
            for idx, tok in enumerate(toks):
                pos = start_pos + idx
                h_t = self._hidden_for_token(tok, pos)
                hidden_steps.append(h_t)
                k_t = h_t.reshape(1, 1, 1, hidden_dim)
                k_list.append(k_t)
                v_list.append(k_t * 1.5)
                cur = cur + mx.array([[float(tok), float(pos + 1)]], dtype=mx.float32)
                state_steps.append(cur)

            ks = mx.concatenate(k_list, axis=2)
            vs = mx.concatenate(v_list, axis=2)
            cache[0].update_and_fetch(ks, vs)
            inter = mx.stack(state_steps[:-1], axis=1) if len(state_steps) > 1 else None
            arr_cache.record_speculative_states(0, inter, state_steps[-1])
            arr_cache.cache[0] = state_steps[-1]

            tx = SpeculativeCacheTransaction([(arr_cache, arr_cache._speculation["generation"])], {}, (cache[0], arr_cache), seq_len)
            return mx.stack(hidden_steps, axis=0)[None, :, :], {}, tx

        def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
            if self.planned_targets:
                row = self.planned_targets.pop(0)
                t_len = int(hidden.shape[1])
                rows = []
                for idx in range(t_len):
                    tok = row[idx] if idx < len(row) else 1
                    rows.append((mx.arange(vocab_size) == tok).astype(mx.float32) * 20.0)
                return mx.stack(rows, axis=0)[None, :, :]
            return hidden @ head_w.T

        def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None, logits_to_keep=1, **kwargs):
            seq_len = int(inputs.shape[1])
            start_pos = int(cache[0].offset)
            toks = [int(t) for t in inputs[0].tolist()]
            h_list = [self._hidden_for_token(tok, start_pos + i) for i, tok in enumerate(toks)]
            hs = mx.stack(h_list, axis=0)[None, :, :]
            ks = hs.reshape(1, 1, seq_len, hidden_dim)
            cache[0].update_and_fetch(ks, ks * 1.5)
            prev = cache[1].cache[0] if cache[1].cache[0] is not None else mx.zeros((1, 2))
            for idx, tok in enumerate(toks):
                prev = prev + mx.array([[float(tok), float(start_pos + idx + 1)]], dtype=mx.float32)
            cache[1].cache[0] = prev
            logits = hs[:, -1:, :] @ head_w.T
            if kwargs.get("return_hidden", False):
                return SimpleNamespace(logits=logits, hidden_states=[hs], shared_kv_states={})
            return SimpleNamespace(logits=logits)

    class _OuterTarget:
        def __init__(self):
            self.language_model = _RecurrentLM()
            self.config = self.language_model.config

        def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kw):
            emb = embed_w[input_ids % vocab_size]
            return SimpleNamespace(to_dict=lambda: {"inputs_embeds": emb})

    class _StatefulDrafter:
        def __init__(self):
            self.config = SimpleNamespace(block_size=4)
            self._cache: list[Any] = []
            self._seed_token = None
            self._seed_hidden = None
            self._next_position = 0
            self._round_appended = 0
            self.accept_lens: list[int] = []
            self.draft_lens: list[int] = []
            self.planned_drafts: list[list[int]] = []

        def make_cache(self):
            return [KVCache()]

        def reset(self, target_model):
            self._cache = self.make_cache()
            self._seed_token = None
            self._seed_hidden = None
            self._next_position = 0
            self._round_appended = 0
            self.accept_lens = []
            self.draft_lens = []
            return self._cache

        def set_shared_kv(self, shared_kv_states, kv_offset, position=None, kv_valid_len=None, left_padding=None):
            del shared_kv_states, position, left_padding
            if kv_valid_len is None:
                kv_valid_len = kv_offset
            if not self._cache or self._cache[0].empty():
                self._next_position = kv_valid_len

        def _forward_tokens(self, tokens: mx.array, hidden: mx.array, token_dtype):
            del token_dtype
            steps = int(tokens.shape[1])
            t_emb = embed_w[tokens % vocab_size]
            combined = mx.concatenate([t_emb, hidden[:, :steps, :]], axis=-1)
            proj = combined @ draft_fc.T
            pos_bias = (mx.arange(steps, dtype=mx.float32) + int(self._next_position)).reshape(1, steps, 1) * 0.01
            out_h = proj + pos_bias
            k_in = out_h.reshape(1, 1, steps, hidden_dim)
            self._cache[0].update_and_fetch(k_in, k_in * 0.5)
            self._next_position += steps
            return out_h

        def _set_seed_from_hidden(self, hidden: mx.array, sampler, greedy: bool):
            logits = hidden @ draft_head.T
            self._seed_token = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            self._seed_hidden = hidden

        def prefill_from_target_hidden(self, input_ids, hidden, bonus_token, sampler, token_dtype=mx.int32, greedy=False):
            if int(input_ids.shape[1]) == 0:
                return
            bonus = mx.array([[int(bonus_token)]], dtype=token_dtype)
            shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
            self._next_position = 0
            h = self._forward_tokens(shifted, hidden[:, : shifted.shape[1], :], token_dtype)
            self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)

        def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32, greedy=False):
            del cache, hidden
            if self.planned_drafts:
                row = self.planned_drafts.pop(0)
                toks = mx.array([row[: block_size - 1]], dtype=token_dtype)
                self._round_appended = int(toks.shape[1])
                dummy_h = mx.zeros((1, self._round_appended, hidden_dim), dtype=mx.float32)
                self._forward_tokens(toks, dummy_h, token_dtype)
                return toks
            h_prev = self._seed_hidden
            tok = self._seed_token.astype(token_dtype)
            self._seed_token = None
            self._seed_hidden = None
            out = [tok]
            self._round_appended = 0
            while len(out) < block_size - 1:
                h_prev = self._forward_tokens(tok, h_prev, token_dtype)
                self._round_appended += 1
                logits = h_prev @ draft_head.T
                tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
                out.append(tok)
            return mx.concatenate(out, axis=1)

        def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted, new_tokens, sampler, token_dtype=mx.int32, greedy=False):
            keep = min(int(accepted), self._round_appended)
            trim = self._round_appended - keep
            if trim > 0:
                for c in self._cache:
                    c.trim(trim)
                self._next_position -= trim
            t_chunks = []
            h_chunks = []
            for d_i in range(keep, int(accepted)):
                t_chunks.append(draft_tokens[:, d_i : d_i + 1])
                h_chunks.append(verify_hidden[:, d_i : d_i + 1, :])
            if new_tokens:
                t_chunks.append(mx.array([[int(new_tokens[-1])]], dtype=token_dtype))
                h_chunks.append(verify_hidden[:, int(accepted) : int(accepted) + 1, :])
            if t_chunks:
                tokens = mx.concatenate(t_chunks, axis=1).astype(token_dtype)
                hiddens = mx.concatenate(h_chunks, axis=1)
                h = self._forward_tokens(tokens, hiddens, token_dtype)
                self._set_seed_from_hidden(h[:, -1:, :], sampler, greedy)
            self._round_appended = 0

    return _OuterTarget(), _StatefulDrafter()


@pytest.mark.unit
@pytest.mark.parametrize("stop_idx", [0, 1, 2])
def test_speculative_session_stop_before_commit_arrays_cache_and_drafter_store(stop_idx):
    """Stop tokens clamp target/drafter commits BEFORE verify.commit so ArraysCache never sees EOS."""
    from heylook_llm.providers.common.speculative_session import (
        SpeculativeSession,
        StatefulDrafterPrefiller,
        _SessionSamplerProxy,
    )

    outer, drafter = _make_toy_stateful_mtp_pair(vocab_size=32, hidden_dim=8)
    lm = outer.language_model
    stop_id = 31

    prompt = [1, 2, 3, 4]
    prompt_ids = mx.array([prompt], dtype=mx.int32)
    cache = lm.make_cache()
    out = lm(prompt_ids, cache=cache, return_hidden=True)
    first_tok = 10

    prefiller = StatefulDrafterPrefiller(
        outer,
        drafter,
        token_dtype=mx.int32,
        greedy=True,
        proxy_sampler=_SessionSamplerProxy(None, greedy=True),
        cached_len=0,
    )
    prefiller.feed_chunk(prompt_ids, out.hidden_states[-1])

    session = SpeculativeSession(
        outer,
        drafter,
        cache,
        out.hidden_states[-1],
        first_tok,
        4,
        None,
        mx.int32,
        True,
        prompt_tokens=prompt_ids,
        prefiller=prefiller,
    )

    draft_row = [11, 12, 13]
    target_row = list(draft_row) + [14]
    target_row[stop_idx] = stop_id
    drafter.planned_drafts = [draft_row]
    session.draft.planned_drafts = [draft_row]
    lm.planned_targets = [target_row]

    res = session.step(16, 1, stop_tokens={stop_id})
    assert res.stopped_on_eos is True
    assert res.finish_reason == "stop"
    assert res.accepted_count == stop_idx
    assert res.emitted_tokens == draft_row[:stop_idx]
    assert res.committed_tokens == [first_tok, *draft_row[:stop_idx]]

    # Prompt cache + ArraysCache must match a clean sequential forward of ONLY committed tokens.
    full_committed = prompt + res.committed_tokens
    assert session.position == len(full_committed)
    assert cache[0].offset == len(full_committed)

    ref_cache = lm.make_cache()
    lm(mx.array([full_committed], dtype=mx.int32), cache=ref_cache)
    mx.eval(cache[0].state, cache[1].cache[0], ref_cache[0].state, ref_cache[1].cache[0])
    assert mx.array_equal(cache[0].state[0], ref_cache[0].state[0]).item()
    assert mx.allclose(cache[1].cache[0], ref_cache[1].cache[0], atol=1e-5).item()

    # Store [prompt + gen] in PrefixCacheStore and verify drafter_layers + drafter_last_hidden.
    store = PrefixCacheStore(memory_max_bytes=64 * 1024 * 1024, max_entries=8)
    d_kw = session.drafter_cache_for_store(len(full_committed))
    assert d_kw["drafter_target_len"] == len(full_committed) - 1
    assert d_kw["drafter_trim_tail"] == 0
    assert tuple(d_kw["drafter_last_hidden"].shape) == (1, 1, 8)
    assert store.store(full_committed, cache, extra_hash=5, **d_kw)

    lookup = store.lookup_full([*full_committed, 19], extra_hash=5)
    assert lookup.matched_tokens == len(full_committed)
    assert lookup.drafter_layers is not None
    assert lookup.drafter_layers[0].offset == len(full_committed) - 1
    assert lookup.drafter_last_hidden is not None
    assert tuple(lookup.drafter_last_hidden.shape) == (1, 1, 8)


@pytest.mark.unit
def test_stateful_drafter_prefiller_chunked_n_minus_1_and_cross_worker_warm_restore(monkeypatch):
    """Chunked + N-1+1 prefill and cross-worker warm restore match single-shot drafter state."""
    import queue
    import threading
    from heylook_llm.providers.common.speculative_session import (
        StatefulDrafterPrefiller,
        _SessionSamplerProxy,
    )
    from mlx_vlm.speculative.common import _SpeculativeSamplerRNG

    outer, drafter = _make_toy_stateful_mtp_pair(vocab_size=32, hidden_dim=8)
    lm = outer.language_model
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    first_tok = 14
    prompt_ids = mx.array([tokens], dtype=mx.int32)

    # 1. Single-shot reference.
    ref_t_cache = lm.make_cache()
    ref_out = lm(prompt_ids, cache=ref_t_cache, return_hidden=True)
    ref_draft = drafter
    ref_draft.reset(outer)
    ref_draft.prefill_from_target_hidden(
        prompt_ids, ref_out.hidden_states[-1], first_tok, None, mx.int32, greedy=True
    )
    mx.eval(ref_draft._cache[0].keys, ref_draft._cache[0].values, ref_draft._seed_token)

    # 2. Chunked [0:5], pre-tail [5:9] (N-1 store), last [9:10] + store at N=10.
    store = PrefixCacheStore(memory_max_bytes=64 * 1024 * 1024, max_entries=8)
    c_cache = lm.make_cache()
    pref = StatefulDrafterPrefiller(
        outer,
        drafter,
        token_dtype=mx.int32,
        greedy=True,
        proxy_sampler=_SessionSamplerProxy(None, greedy=True),
        cached_len=0,
    )
    for sl in (slice(0, 5), slice(5, 9), slice(9, 10)):
        c_ids = prompt_ids[:, sl]
        c_out = lm(c_ids, cache=c_cache, return_hidden=True)
        pref.feed_chunk(c_ids, c_out.hidden_states[-1])
        if sl.stop == 9:
            mx.eval(c_cache[0].state, c_cache[1].cache[0], pref.draft._cache[0].state, pref.pending_prev_hidden)
            assert store.store(tokens[:9], c_cache, extra_hash=9, **pref.cache_snapshot_args())
        elif sl.stop == 10:
            mx.eval(c_cache[0].state, c_cache[1].cache[0], pref.draft._cache[0].state, pref.pending_prev_hidden)
            assert store.store(tokens, c_cache, extra_hash=9, **pref.cache_snapshot_args())

    rng = _SpeculativeSamplerRNG(pref.draft, enabled=False)
    pref.finish_with_first_token(first_tok, rng)
    mx.eval(pref.draft._cache[0].keys, pref.draft._cache[0].values, pref.draft._seed_token)

    assert pref.draft._cache[0].offset == ref_draft._cache[0].offset
    assert mx.allclose(pref.draft._cache[0].keys, ref_draft._cache[0].keys, atol=1e-5).item()
    assert mx.allclose(pref.draft._cache[0].values, ref_draft._cache[0].values, atol=1e-5).item()
    assert int(pref.draft._seed_token.item()) == int(ref_draft._seed_token.item())

    # 3. Turn 2 extension [tokens + [11, 12]] on a separate worker thread with quarantined thread A.
    follow = [11, 12]
    full2 = [*tokens, *follow]
    full2_ids = mx.array([full2], dtype=mx.int32)
    ref2_cache = lm.make_cache()
    ref2_out = lm(full2_ids, cache=ref2_cache, return_hidden=True)
    _, ref2_draft = _make_toy_stateful_mtp_pair(vocab_size=32, hidden_dim=8)
    ref2_draft.reset(outer)
    ref2_draft.prefill_from_target_hidden(
        full2_ids, ref2_out.hidden_states[-1], 15, None, mx.int32, greedy=True
    )
    mx.eval(ref2_draft._cache[0].keys, ref2_draft._seed_token)

    new_tls = getattr(mx, "new_thread_local_stream", None)
    tls_b = new_tls(mx.default_device()) if callable(new_tls) else mx.default_stream(mx.default_device())
    result_q: queue.Queue = queue.Queue()

    def worker_b():
        try:
            with mx.stream(tls_b):
                res = store.lookup_full(full2, extra_hash=9)
                assert res.matched_tokens == 10
                assert res.drafter_layers is not None
                assert res.drafter_last_hidden is not None
                w_pref = StatefulDrafterPrefiller(
                    outer,
                    drafter,
                    token_dtype=mx.int32,
                    greedy=True,
                    proxy_sampler=_SessionSamplerProxy(None, greedy=True),
                    cached_len=res.matched_tokens,
                    restored_drafter_layers=res.drafter_layers,
                    restored_last_hidden=res.drafter_last_hidden,
                )
                assert w_pref.enabled is True
                suf_ids = mx.array([follow], dtype=mx.int32)
                suf_out = lm(suf_ids, cache=res.layers, return_hidden=True)
                w_pref.feed_chunk(suf_ids, suf_out.hidden_states[-1])
                w_pref.finish_with_first_token(15, _SpeculativeSamplerRNG(w_pref.draft, enabled=False))
                got_state_k = w_pref.draft._cache[0].state[0]
                mx.eval(got_state_k, w_pref.draft._seed_token)
                result_q.put((got_state_k.tolist(), int(w_pref.draft._seed_token.item()), None))
        except Exception as exc:  # pragma: no cover
            result_q.put((None, None, exc))

    t_b = threading.Thread(target=worker_b, daemon=True)
    t_b.start()
    got_keys, got_seed, err = result_q.get(timeout=10)
    t_b.join(timeout=5)
    assert err is None
    assert mx.allclose(mx.array(got_keys, dtype=mx.float32), ref2_draft._cache[0].state[0], atol=1e-5).item()
    assert got_seed == int(ref2_draft._seed_token.item())


@pytest.mark.unit
def test_speculative_session_greedy_no_logsumexp_and_non_greedy_sampler_isolation(monkeypatch):
    """Greedy B=1 verify skips mx.logsumexp; non-greedy Leviathan leaves caller sampler unmutated."""
    from heylook_llm.providers.common import samplers as heylook_samplers
    from heylook_llm.providers.common.speculative_session import (
        SpeculativeSession,
        _SessionSamplerProxy,
    )

    outer, drafter = _make_toy_stateful_mtp_pair(vocab_size=32, hidden_dim=8)
    lm = outer.language_model
    prompt = [1, 2, 3, 4]
    prompt_ids = mx.array([prompt], dtype=mx.int32)

    # 1. Greedy fast path (target lacks speculative_argmax_from_hidden -> uses _walk greedy branch).
    cache_g = lm.make_cache()
    out_g = lm(prompt_ids, cache=cache_g, return_hidden=True)
    drafter.planned_drafts = [[5, 6, 7]]
    lm.planned_targets = [[5, 6, 9, 10]]

    sess_g = SpeculativeSession(
        outer, drafter, cache_g, out_g.hidden_states[-1], 8, 4, None, mx.int32, True
    )
    logsumexp_calls = 0
    orig_lse = mx.logsumexp

    def spy_lse(*a, **kw):
        nonlocal logsumexp_calls
        logsumexp_calls += 1
        return orig_lse(*a, **kw)

    monkeypatch.setattr(mx, "logsumexp", spy_lse)
    res_g = sess_g.step(8, 1, stop_tokens={31})
    assert logsumexp_calls == 0
    assert res_g.emitted_tokens == [5, 6, 9]
    assert res_g.accepted_count == 2
    monkeypatch.setattr(mx, "logsumexp", orig_lse)

    # 2. Non-greedy Leviathan path with penalties: caller sampler must NEVER be mutated.
    sampler, procs = heylook_samplers.build(
        {"temperature": 0.8, "top_p": 0.9, "min_p": 0.0, "top_k": 0, "presence_penalty": 0.5, "seed": 7}
    )
    assert hasattr(sampler, "_heylook_spec_probs")
    assert getattr(sampler, "_heylook_pending_draft", None) is None

    # Emit non-greedy drafts + capture q_probs through proxy.
    def _sampling_draft_block(last_bonus, hidden, cache, block_size, s, token_dtype=mx.int32, greedy=False):
        del last_bonus, hidden, cache
        q_rows = []
        toks = []
        buf: list[mx.array] = []
        s._heylook_captured_q = buf
        try:
            for step_i in range(block_size - 1):
                row_logits = mx.arange(32, dtype=mx.float32).reshape(1, 32) * 0.05 + step_i
                tok = s(row_logits[:, None, :])
                toks.append(tok.reshape(1, 1).astype(token_dtype))
        finally:
            s._heylook_captured_q = None
        if buf:
            q_mat = mx.concatenate(buf, axis=0)
            mx.async_eval(q_mat)
            s._heylook_pending_draft = (mx.concatenate(toks, axis=1), q_mat)
        return mx.concatenate(toks, axis=1)

    drafter.draft_block = _sampling_draft_block
    lm.planned_targets = []
    cache_s = lm.make_cache()
    out_s = lm(prompt_ids, cache=cache_s, return_hidden=True)
    gen_hist = [8]
    sess_s = SpeculativeSession(
        outer,
        drafter,
        cache_s,
        out_s.hidden_states[-1],
        8,
        4,
        sampler,
        mx.int32,
        False,
        penalty_processors=procs,
        gen_history=gen_hist,
    )
    res_s = sess_s.step(8, 1, stop_tokens={31})
    assert len(res_s.emitted_tokens) >= 1
    assert getattr(sampler, "_heylook_pending_draft", None) is None
    assert getattr(sampler, "_heylook_mtp_penalties", None) is None
    assert getattr(sampler, "_heylook_captured_q", None) is None


@pytest.mark.unit
def test_gemma4_shared_kv_b1_caches_and_1023_maskless_swa():
    """Gemma 4 B=1 shared_kv matches upstream and slices SWA to window-1 for maskless SDPA."""
    import copy
    from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache, RotatingKVCache
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import LanguageModel
    from mlx_vlm.speculative.drafters.gemma4_assistant.masks import make_drafter_masks
    from heylook_llm.providers.common.gemma4_runtime import (
        bind_gemma4_runtime,
        slice_gemma4_maskless_swa,
    )

    mx.random.seed(11)
    cfg = TextConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        vocab_size=64,
        hidden_size_per_layer_input=0,
        sliding_window=8,
        sliding_window_pattern=2,
        num_kv_shared_layers=0,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 10000.0},
        },
        final_logit_softcapping=30.0,
    )
    lm_up = LanguageModel(cfg)
    lm_bd = copy.deepcopy(lm_up)
    wrapper = SimpleNamespace(language_model=lm_bd)
    assert bind_gemma4_runtime(wrapper) is True

    for cache_cls in (RotatingKVCache, BufferedRotatingKVCache):
        for step_len in (4, 7, 8, 9, 15):
            if cache_cls is BufferedRotatingKVCache:
                c0 = BufferedRotatingKVCache(8, keep=0, buffer_size=4)
            else:
                c0 = RotatingKVCache(8, keep=0)
            c1 = KVCache()
            k0 = mx.random.normal((1, 2, step_len, 32))
            v0 = mx.random.normal((1, 2, step_len, 32))
            k1 = mx.random.normal((1, 1, step_len, 32))
            v1 = mx.random.normal((1, 1, step_len, 32))
            c0.update_and_fetch(k0, v0)
            c1.update_and_fetch(k1, v1)
            mx.eval(c0.keys, c0.values, c1.keys, c1.values)

            su = lm_up.shared_kv_from_cache([c0, c1])
            sb = lm_bd.shared_kv_from_cache([c0, c1])
            assert set(su) == set(sb) == {"sliding_attention", "full_attention"}
            for lt in su:
                assert mx.array_equal(su[lt][0], sb[lt][0]).item()
                assert mx.array_equal(su[lt][1], sb[lt][1]).item()

            # Maskless SWA slice across offset<=window (mask=None already) and offset>window (7 keys).
            sliced = slice_gemma4_maskless_swa(sb, position=step_len, window=8)
            masks_fast = make_drafter_masks(
                sliced,
                query_len=1,
                query_offset=step_len - 1,
                sliding_window=8,
                dtype=mx.float32,
                kv_valid_len=step_len,
            )
            assert masks_fast["sliding_attention"] is None
            if step_len > 8:
                assert int(sliced["sliding_attention"][0].shape[2]) == 7

                # Verify SDPA output parity against upstream masked 8-key attention.
                masks_up = make_drafter_masks(
                    su,
                    query_len=1,
                    query_offset=step_len - 1,
                    sliding_window=8,
                    dtype=mx.float32,
                    kv_valid_len=step_len,
                )
                assert masks_up["sliding_attention"] is not None
                q = mx.random.normal((1, 4, 1, 32))
                uk, uv = su["sliding_attention"]
                fk, fv = sliced["sliding_attention"]
                out_up = mx.fast.scaled_dot_product_attention(
                    q, uk, uv, scale=1.0, mask=masks_up["sliding_attention"]
                )
                out_fast = mx.fast.scaled_dot_product_attention(
                    q, fk, fv, scale=1.0, mask=masks_fast["sliding_attention"]
                )
                mx.eval(out_up, out_fast)
                assert mx.allclose(out_up, out_fast, atol=1e-5).item()

    # Exact 1024 -> 1023 Gemma 4 window boundary check.
    for pos, expect_len in ((512, 512), (1023, 1023), (1024, 1024), (1025, 1023), (1500, 1023)):
        buf_len = min(pos, 1024)
        k_sw = mx.zeros((1, 2, buf_len, 32), dtype=mx.float32)
        v_sw = mx.zeros((1, 2, buf_len, 32), dtype=mx.float32)
        sliced_1024 = slice_gemma4_maskless_swa({"sliding_attention": (k_sw, v_sw)}, position=pos, window=1024)
        assert int(sliced_1024["sliding_attention"][0].shape[2]) == expect_len
        masks_1024 = make_drafter_masks(
            sliced_1024,
            query_len=1,
            query_offset=pos - 1,
            sliding_window=1024,
            dtype=mx.float32,
            kv_valid_len=pos,
        )
        assert masks_1024["sliding_attention"] is None


@pytest.mark.unit
def test_b1_runner_and_speculative_session_share_generate_common_stream(monkeypatch):
    """b1_runner and SpeculativeSession must both use mlx_vlm.generate.common.generation_stream."""
    import mlx_vlm.speculative.common as spec_common
    from mlx_vlm.generate.common import generation_stream
    from heylook_llm.providers.common.speculative_session import SpeculativeSession

    assert b1_runner._generation_stream() is generation_stream

    # Poison speculative.common.generation_stream (which is a separate stream on PyPI 0.8.0)
    # so SpeculativeSession._verify_tokens fails immediately if it ever imports from there.
    monkeypatch.setattr(spec_common, "generation_stream", object())

    outer, drafter = _make_toy_stateful_mtp_pair(vocab_size=32, hidden_dim=8)
    lm = outer.language_model
    cache = lm.make_cache()
    out = lm(mx.array([[1, 2, 3, 4]], dtype=mx.int32), cache=cache, return_hidden=True)
    drafter.planned_drafts = [[5, 6, 7]]
    lm.planned_targets = [[5, 6, 9, 10]]

    sess = SpeculativeSession(
        outer, drafter, cache, out.hidden_states[-1], 8, 4, None, mx.int32, True
    )
    res = sess.step(8, 1, stop_tokens={31})
    assert res.emitted_tokens == [5, 6, 9]

