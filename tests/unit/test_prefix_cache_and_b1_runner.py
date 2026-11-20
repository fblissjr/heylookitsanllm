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
    from mlx_vlm.speculative import mtp as mtp_mod
    from mlx_vlm.speculative import utils as spec_utils

    monkeypatch.setattr(apc_mod, "hash_image_payload", lambda **_k: 0)
    monkeypatch.setattr(apc_mod, "semantic_extra_hash", lambda **_k: 123)
    monkeypatch.setattr(spec_utils, "speculative_prefill_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(spec_utils, "speculative_hidden_state", lambda *_a, **_k: mx.zeros((1, 1, 1)))

    # Case 1: bonus token is EOS (raw_toks = [11, 12, 99], stop=99 at index 2).
    # Prefill emits first_tok=8, MTPBatch commits [8, 11, 12] -> offset = 10 + 3 = 13.
    class _FakeMTPBatch:
        planned_rounds: list[list[int]] = []

        def __init__(self, lm, draft, prompt_cache, hidden, first_y, block_size, *_a, **_k):
            self.cache = prompt_cache
            first_tok = int(first_y.item())
            start_pos = int(prompt_cache[0].offset)
            self.rows = [SimpleNamespace(bonus=first_tok, position=start_pos)]
            self._rounds = list(self.planned_rounds)

        def step(self, budgets, positions, thinking_budget_criteria=None):
            raw = self._rounds.pop(0)
            committed = [int(self.rows[0].bonus), *raw[:-1]]
            k = mx.arange(len(committed), dtype=mx.float32).reshape(1, 1, len(committed), 1)
            for c in self.cache:
                c.update_and_fetch(k, k)
            self.rows[0].position += len(committed)
            self.rows[0].bonus = raw[-1]
            return [raw]

    monkeypatch.setattr(mtp_mod, "MTPBatch", _FakeMTPBatch)
    draft_stub = (SimpleNamespace(config=SimpleNamespace(block_size=4)), "mtp")

    model, _ = _make_toy_vlm(vocab_size=32, window_size=8)
    apc = _make_fake_apc()
    prompt1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ids1 = mx.array([prompt1], dtype=mx.int32)

    # first_tok = (offset=10 + 4) % 32 = 14; round commits [14, 15, 16], bonus EOS=99.
    _FakeMTPBatch.planned_rounds = [[15, 16, 99]]
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
    _FakeMTPBatch.planned_rounds = [[99]]
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
    # MTPBatch commits [first_tok, 25, 99] (3 tokens); cache_trim_tail=1 trims 99, keeping [prompt3, first_tok, 25].
    apc2 = _make_fake_apc()
    _FakeMTPBatch.planned_rounds = [[25, 99, 27]]
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
    _FakeMTPBatch.planned_rounds = [[99]]
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
