# tests/unit/test_prompt_cache_slot.py
"""The Q7 single-slot prompt cache: behavior contract.

One slot per model: immutable per-layer state snapshots, registered with
the token sequence the cache actually contains. Reuse comes in exactly two safe shapes: EXTENSION (continue the
cache -- valid for every cache type, hybrids included) and TRIM to the
common prefix via mlx-lm's own trim_prompt_cache (refused per layer-type by
can_trim_prompt_cache, which is what closes the hybrid/ArraysCache
silent-wrong-output hole the radix shipped with).

Real KVCache layers + real mx arrays throughout: the trim path goes through
mlx-lm's cache classes, which a MagicMock would fake straight through.
"""

from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from heylook_llm.providers.common.prompt_cache import (
    PromptCacheManager,
    get_global_cache_manager,
    process_prompt_with_cache,
    store_generation_cache,
)


class _TinyModel:
    """Just enough for make_cache: layers with head dims."""
    class _Layer:
        pass
    layers = [_Layer()]


class _RecurrentLayer:
    """A non-trimmable cache layer (the ArraysCache shape): running state,
    no positional offset, refuses trims."""
    def __init__(self):
        self.state = [mx.zeros((1, 4))]

    def is_trimmable(self):
        return False


STD = {"cache_type": "standard"}


def _kv_of_len(n):
    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 2, n, 4)), mx.zeros((1, 2, n, 4)))
    return kv


def _fresh_kv():
    return [KVCache()]


def _generate(manager, model_id, prompt, generated, cache_layers=None,
              fresh=_fresh_kv):
    """Simulate one full generation: lookup, then store prompt+generated.

    By default the stored cache covers the WHOLE sequence (offset ==
    len(prompt+generated)); pass cache_layers to model other shapes.
    ``fresh`` is what make_cache would build for this architecture -- the
    restore path reconstructs INTO it, so hybrid tests must hand a factory
    matching their stored layer shape (as the real model's make_cache does).
    """
    pc = manager.get_or_create_cache(model_id, _TinyModel(), STD)
    with patch("heylook_llm.providers.common.prompt_cache.make_cache",
               side_effect=lambda *a, **k: fresh()):
        to_process, pc = process_prompt_with_cache(pc, prompt, _TinyModel(), STD)
    full = prompt + generated
    layers = cache_layers if cache_layers is not None else [_kv_of_len(len(full))]
    store_generation_cache(pc, full, layers)
    return to_process, pc


def _lookup(manager, model_id, prompt, fresh=_fresh_kv):
    pc = manager.get_or_create_cache(model_id, _TinyModel(), STD)
    with patch("heylook_llm.providers.common.prompt_cache.make_cache",
               side_effect=lambda *a, **k: fresh()):
        return process_prompt_with_cache(pc, prompt, _TinyModel(), STD)


@pytest.mark.unit
class TestSlotReuse:
    def test_next_turn_extends_the_stored_sequence(self):
        # The conversation flow: new prompt = old prompt + old reply + new
        # turn. The full stored sequence is the common prefix; nothing trims.
        mgr = get_global_cache_manager()
        prompt, reply = list(range(100)), list(range(1000, 1010))
        _generate(mgr, "slot-turn", prompt, reply)
        next_prompt = prompt + reply + list(range(2000, 2020))
        to_process, pc = _lookup(mgr, "slot-turn", next_prompt)
        assert pc._radix_matched_len == len(prompt) + len(reply)
        assert to_process == next_prompt[len(prompt) + len(reply):]

    def test_slot_registers_what_the_cache_holds(self):
        # The final sampled token is never fed through the model, so the
        # real cache is one token short of full_tokens. The slot must
        # register the cache's ACTUAL coverage (read off the KV offset) --
        # an extension then reprocesses that last reply token instead of
        # skipping past KV that does not exist.
        mgr = get_global_cache_manager()
        prompt, reply = list(range(50)), list(range(1000, 1008))
        full = prompt + reply
        _generate(mgr, "slot-short", prompt, reply,
                  cache_layers=[_kv_of_len(len(full) - 1)])
        next_prompt = full + list(range(2000, 2005))
        to_process, pc = _lookup(mgr, "slot-short", next_prompt)
        assert pc._radix_matched_len == len(full) - 1
        assert to_process[0] == full[-1]  # the uncached reply token re-enters

    def test_edit_trims_to_the_common_prefix(self):
        # Trim-to-common-prefix, the property Q7 requires kept: an edit
        # mid-thread reuses everything before the divergence point.
        mgr = get_global_cache_manager()
        prompt = list(range(100))
        _generate(mgr, "slot-edit", prompt, list(range(1000, 1010)))
        edited = prompt[:60] + list(range(3000, 3040))
        to_process, pc = _lookup(mgr, "slot-edit", edited)
        assert pc._radix_matched_len == 60
        assert to_process == edited[60:]

    def test_exact_same_prompt_reprocesses_one_token(self):
        # mlx-lm needs >= 1 token; a full-length repeat steps back one
        # (a 1-token trim) so the boundary sits before the reprocessed token.
        mgr = get_global_cache_manager()
        prompt = list(range(50))
        _generate(mgr, "slot-same", prompt, [])
        to_process, pc = _lookup(mgr, "slot-same", prompt)
        assert to_process == prompt[-1:]
        assert pc._radix_matched_len == len(prompt) - 1

    def test_no_overlap_is_a_miss_and_the_slot_survives(self):
        mgr = get_global_cache_manager()
        _generate(mgr, "slot-miss", list(range(100, 150)), [])
        to_process, pc = _lookup(mgr, "slot-miss", list(range(500, 520)))
        assert pc._radix_matched_len == 0
        assert to_process == list(range(500, 520))
        # the untouched slot is still there for a future extension
        to_process, pc = _lookup(mgr, "slot-miss", list(range(100, 160)))
        assert pc._radix_matched_len == 50

    def test_single_slot_replacement(self):
        # The slot holds the LAST generation only: a second generation with
        # a different prefix supersedes the first wholesale.
        mgr = get_global_cache_manager()
        first = list(range(100))
        _generate(mgr, "slot-repl", first, [])
        second = list(range(5000, 5080))
        _generate(mgr, "slot-repl", second, [])
        _, pc = _lookup(mgr, "slot-repl", first)
        assert pc._radix_matched_len == 0  # the old sequence is gone


@pytest.mark.unit
class TestTrimInvariant:
    def test_trimmed_cache_state_equals_matched_boundary(self):
        """The postmortem pin, restated for live caches: after a divergence
        trim, the cache's state covers exactly the matched boundary (mRoPE
        positions derive from cache.offset)."""
        mgr = get_global_cache_manager()
        prompt, reply = list(range(80)), list(range(1000, 1020))
        _generate(mgr, "slot-trim", prompt, reply)  # cache holds 100 tokens
        edited = prompt[:40] + list(range(7000, 7010))
        _, pc = _lookup(mgr, "slot-trim", edited)
        layer = pc.cache[0]
        assert layer.offset == 40
        keys, _values = layer.state  # state is the offset-trimmed view
        assert keys.shape[2] == 40

    def test_full_length_match_trims_before_the_reprocessed_token(self):
        mgr = get_global_cache_manager()
        prompt = list(range(30))
        _generate(mgr, "slot-trim2", prompt, [])
        _, pc = _lookup(mgr, "slot-trim2", prompt)
        assert pc.cache[0].offset == len(prompt) - 1


@pytest.mark.unit
class TestHybridSafety:
    """The hole the radix shipped with, closed: non-trimmable layers
    (ArraysCache-shaped) may EXTEND but never partially restore."""

    def _hybrid_layers(self, n):
        return [_kv_of_len(n), _RecurrentLayer()]

    def _fresh_hybrid(self):
        return [KVCache(), _RecurrentLayer()]

    def test_extension_still_hits(self):
        mgr = get_global_cache_manager()
        prompt = list(range(60))
        _generate(mgr, "hyb-ext", prompt, [], cache_layers=self._hybrid_layers(60),
                  fresh=self._fresh_hybrid)
        to_process, pc = _lookup(mgr, "hyb-ext", prompt + list(range(9000, 9010)),
                                 fresh=self._fresh_hybrid)
        assert pc._radix_matched_len == 60
        assert to_process == list(range(9000, 9010))

    def test_divergence_is_a_miss_not_a_slice(self):
        mgr = get_global_cache_manager()
        prompt = list(range(60))
        _generate(mgr, "hyb-div", prompt, [], cache_layers=self._hybrid_layers(60),
                  fresh=self._fresh_hybrid)
        edited = prompt[:30] + list(range(9000, 9010))
        to_process, pc = _lookup(mgr, "hyb-div", edited, fresh=self._fresh_hybrid)
        assert pc._radix_matched_len == 0      # refused, not sliced
        assert to_process == edited            # full re-prefill
        # and the slot survived for a future extension
        _, pc = _lookup(mgr, "hyb-div", prompt + [1], fresh=self._fresh_hybrid)
        assert pc._radix_matched_len == 60

    def test_exact_repeat_on_hybrid_is_a_miss(self):
        # The step-back is a 1-token trim, which a hybrid refuses.
        mgr = get_global_cache_manager()
        prompt = list(range(40))
        _generate(mgr, "hyb-same", prompt, [], cache_layers=self._hybrid_layers(40),
                  fresh=self._fresh_hybrid)
        to_process, pc = _lookup(mgr, "hyb-same", prompt, fresh=self._fresh_hybrid)
        assert pc._radix_matched_len == 0
        assert to_process == prompt


@pytest.mark.unit
class TestManagerLifecycle:
    def test_byte_budget_drops_lru_slots(self):
        mgr = PromptCacheManager(max_cache_bytes=None)
        model = _TinyModel()
        for mid, base in (("bb-old", 0), ("bb-new", 10000)):
            pc = mgr.get_or_create_cache(mid, model, STD)
            with patch("heylook_llm.providers.common.prompt_cache.make_cache",
                       return_value=[KVCache()]):
                process_prompt_with_cache(pc, list(range(base, base + 64)), model, STD)
            with patch("heylook_llm.providers.common.prompt_cache.get_global_cache_manager",
                       return_value=mgr):
                store_generation_cache(pc, list(range(base, base + 64)), [_kv_of_len(64)])
        assert mgr.total_cache_bytes > 0
        one_slot = mgr.total_cache_bytes // 2
        mgr.set_byte_budget(one_slot)
        mgr.enforce_byte_budget()
        info = mgr.get_cache_info()
        # the LRU slot went; the recent one survived
        assert info["bb-old"]["slot_bytes"] == 0
        assert info["bb-new"]["slot_bytes"] > 0

    def test_invalidate_drops_the_slot(self):
        mgr = get_global_cache_manager()
        _generate(mgr, "slot-inv", list(range(64)), [])
        mgr.invalidate_cache("slot-inv")
        _, pc = _lookup(mgr, "slot-inv", list(range(64)))
        assert pc._radix_matched_len == 0

    def test_max_entries_evicts_lru_model(self):
        mgr = PromptCacheManager(max_cache_entries=2)
        model = _TinyModel()
        for mid in ("lru-a", "lru-b", "lru-c"):
            pc = mgr.get_or_create_cache(mid, model, STD)
            with patch("heylook_llm.providers.common.prompt_cache.make_cache",
                       return_value=[KVCache()]):
                process_prompt_with_cache(pc, list(range(32)), model, STD)
            with patch("heylook_llm.providers.common.prompt_cache.get_global_cache_manager",
                       return_value=mgr):
                store_generation_cache(pc, list(range(32)), [_kv_of_len(32)])
        info = mgr.get_cache_info()
        assert "lru-a" not in info          # evicted wholesale
        assert info["lru-c"]["slot_bytes"] > 0


import threading

import pytest as _pytest

_metal = False
try:
    _metal = mx.metal.is_available()
except Exception:
    pass


@_pytest.mark.unit
@_pytest.mark.skipif(not _metal, reason="needs Metal GPU thread-local streams")
class TestSlotThreadAffinity:
    """The store path must eval the EXACT arrays it registers. Each .state
    access builds fresh lazy slice objects, so an implementation that evals
    one capture and stores another ships lazy state -- restored on the next
    request's thread, that is the "There is no Stream(gpu, N)" crash
    (postmortems/radix_thread_affinity.md). This exact split shipped in the
    v1.75.0 rewrite and was caught by the live chat e2e; this pins it at
    unit level, where the two-thread shape can go red without a model."""

    def test_stored_slot_state_evals_after_its_thread_dies(self):
        mgr = get_global_cache_manager()

        def generate_on_worker():
            local = mx.new_thread_local_stream(mx.gpu)
            with mx.stream(local):
                kv = KVCache()
                kv.update_and_fetch(mx.random.normal((1, 2, 8, 4)),
                                    mx.random.normal((1, 2, 8, 4)))
                pc = mgr.get_or_create_cache("affinity", _TinyModel(), STD)
                pc._radix_eligible = True
                store_generation_cache(pc, list(range(8)), [kv])

        worker = threading.Thread(target=generate_on_worker)
        worker.start()
        worker.join()
        assert not worker.is_alive()

        slot = mgr._get_slot("affinity")
        assert slot is not None
        arrays = []
        for state, _meta in slot.layers:
            from heylook_llm.providers.common.prompt_cache import _flat_arrays
            _flat_arrays(state, arrays)
        assert arrays
        # What the next request's thread does with restored state. Must not
        # raise "There is no Stream(gpu, N) in current thread".
        mx.eval(arrays)
