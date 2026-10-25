"""Unit tests for asymmetric MLX pipeline sharding and fast Metal sync."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlx.nn as nn
import pytest

import heylook_llm.providers.common.mlx_pipeline as mp
from heylook_llm.providers.common.mlx_pipeline import (
    PipelineAssignment,
    _BoundaryRecvWrapper,
    _BoundarySendWrapper,
    _FinalStageRecvWrapper,
    _OffsetStubCache,
    _PassThroughLayer,
    apply_asymmetric_pipeline,
    attach_send_dependency,
    compute_asymmetric_layer_split,
    enable_fast_metal_synch,
    filter_safetensors_for_rank,
)


class _FakeGroup:
    def __init__(self, rank: int, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeLayer(nn.Module):
    def __init__(self, idx: int, delta: int) -> None:
        super().__init__()
        self.idx = idx
        self.delta = delta

    def __call__(self, h: Any, mask: Any = None, cache: Any = None) -> Any:
        if cache is not None and hasattr(cache, "advance"):
            cache.advance(self._seq_len(h))
        return h + self.delta

    @staticmethod
    def _seq_len(h: Any) -> int:
        shape = getattr(h, "shape", None)
        if shape and len(shape) >= 2:
            return int(shape[1])
        return 1


class _FakeKVCache:
    def __init__(self) -> None:
        self.offset = 0
        self.keys: Any = "kv_keys"
        self.values: Any = "kv_values"

    def advance(self, n: int) -> None:
        self.offset += n


class _FakeArraysCache:
    def __init__(self) -> None:
        self.state: List[Any] = ["conv_state", "recurrent_state"]


class _FakeLanguageModel:
    def __init__(self, num_layers: int = 10) -> None:
        self.layers: List[Any] = [_FakeLayer(i, delta=i + 1) for i in range(num_layers)]

    def make_cache(self) -> List[Any]:
        return [_FakeKVCache() for _ in range(len(self.layers))]

    def __call__(self, inputs: Any, cache: Optional[List[Any]] = None, mask: Any = None) -> Any:
        h = inputs
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return h


class _FakeVLM:
    def __init__(self, num_layers: int = 10) -> None:
        self.vision_tower: Optional[str] = "siglip_vision_tower"
        self.language_model = _FakeLanguageModel(num_layers=num_layers)


def test_enable_fast_metal_synch_sets_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)
    monkeypatch.delenv("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", raising=False)

    env = enable_fast_metal_synch(residency_refresh_ms=750)
    assert env == {
        "MLX_METAL_FAST_SYNCH": "1",
        "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS": "750",
    }
    assert os.environ["MLX_METAL_FAST_SYNCH"] == "1"
    assert os.environ["MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS"] == "750"


def test_compute_asymmetric_layer_split_80_20() -> None:
    spans = compute_asymmetric_layer_split(80, ratios=(0.8, 0.2))
    assert spans == [(0, 64), (64, 80)]

    r0 = PipelineAssignment(rank=0, world_size=2, start_layer=spans[0][0], end_layer=spans[0][1], total_layers=80)
    r1 = PipelineAssignment(rank=1, world_size=2, start_layer=spans[1][0], end_layer=spans[1][1], total_layers=80)
    assert r0.owns_layer(0) and r0.owns_layer(63) and not r0.owns_layer(64)
    assert not r1.owns_layer(63) and r1.owns_layer(64) and r1.owns_layer(79)


def test_compute_asymmetric_layer_split_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="total_layers"):
        compute_asymmetric_layer_split(1, ratios=(0.8, 0.2))
    with pytest.raises(ValueError, match="at least one"):
        compute_asymmetric_layer_split(10, ratios=())
    with pytest.raises(ValueError, match="positive"):
        compute_asymmetric_layer_split(10, ratios=(0.8, 0.0))


def test_filter_safetensors_for_rank_partitions_shards(tmp_path: Path) -> None:
    for i in range(1, 5):
        (tmp_path / f"model-0000{i}-of-00004.safetensors").write_bytes(b"")

    index_data = {
        "weight_map": {
            "vision_tower.patch_embed.weight": "model-00001-of-00004.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
            "language_model.model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
            "language_model.model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "language_model.model.layers.3.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
            "language_model.model.norm.weight": "model-00004-of-00004.safetensors",
            "language_model.lm_head.weight": "model-00004-of-00004.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index_data))

    r0_assign = PipelineAssignment(rank=0, world_size=2, start_layer=0, end_layer=2, total_layers=4)
    r1_assign = PipelineAssignment(rank=1, world_size=2, start_layer=2, end_layer=4, total_layers=4)

    r0_files = [p.name for p in filter_safetensors_for_rank(tmp_path, r0_assign)]
    r1_files = [p.name for p in filter_safetensors_for_rank(tmp_path, r1_assign)]

    # Rank 0 keeps shards 1, 2 (layers 0, 1 + vision/embed) and 4 (norm + lm_head), skipping shard 3!
    assert r0_files == [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    # Rank 1 loads ONLY shard 3 (layers 2, 3), skipping vision_tower, embed_tokens, norm, and lm_head!
    assert r1_files == ["model-00003-of-00004.safetensors"]


def test_attach_send_dependency_wires_kv_and_arrays_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp.mx, "depends", lambda a, b: f"dep({a},{b})")

    kv = _FakeKVCache()
    attach_send_dependency(kv, "sent_token")
    assert kv.keys == "dep(kv_keys,sent_token)"

    arr_cache = _FakeArraysCache()
    attach_send_dependency(arr_cache, "sent_token_2")
    assert arr_cache.state[0] == "dep(conv_state,sent_token_2)"


def test_apply_asymmetric_pipeline_2rank_forward_and_cache_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class _FakeTensor:
        val: int
        shape: tuple[int, int] = (1, 5)
        dtype: str = "bfloat16"

        def __add__(self, other: int) -> "_FakeTensor":
            return _FakeTensor(self.val + other, self.shape, self.dtype)

    mailbox: dict[tuple[int, int], _FakeTensor] = {}

    monkeypatch.setattr(
        mp.mx.distributed,
        "send",
        lambda x, dst, group=None: mailbox.__setitem__((group.rank(), dst), x) or x,
    )
    monkeypatch.setattr(
        mp.mx.distributed,
        "recv_like",
        lambda x, src, group=None: mailbox[(src, group.rank())],
    )
    monkeypatch.setattr(mp.mx, "depends", lambda a, b: (a, "dep_on_send"))

    baseline_model = _FakeVLM(num_layers=10)
    baseline_cache = baseline_model.language_model.make_cache()
    expected_out = baseline_model.language_model(_FakeTensor(0), cache=baseline_cache)
    assert expected_out.val == sum(range(1, 11))  # 55
    assert all(c.offset == 5 for c in baseline_cache)

    rank0_model = _FakeVLM(num_layers=10)
    rank1_model = _FakeVLM(num_layers=10)
    g0 = _FakeGroup(rank=0, size=2)
    g1 = _FakeGroup(rank=1, size=2)

    a0 = apply_asymmetric_pipeline(rank0_model, group=g0, ratios=(0.8, 0.2))
    a1 = apply_asymmetric_pipeline(rank1_model, group=g1, ratios=(0.8, 0.2))

    assert (a0.start_layer, a0.end_layer) == (0, 8)
    assert (a1.start_layer, a1.end_layer) == (8, 10)

    # Rank 0 keeps vision_tower; Rank 1 drops vision_tower
    assert rank0_model.vision_tower == "siglip_vision_tower"
    assert rank1_model.vision_tower is None

    # Verify layer wrapping and PassThrough replacement
    assert isinstance(rank0_model.language_model.layers[7], _BoundarySendWrapper)
    assert isinstance(rank0_model.language_model.layers[8], _PassThroughLayer)
    assert isinstance(rank0_model.language_model.layers[9], _FinalStageRecvWrapper)

    assert isinstance(rank1_model.language_model.layers[0], _PassThroughLayer)
    assert isinstance(rank1_model.language_model.layers[7], _PassThroughLayer)
    assert isinstance(rank1_model.language_model.layers[8], _BoundaryRecvWrapper)
    assert isinstance(rank1_model.language_model.layers[9], _BoundarySendWrapper)

    # Verify make_cache wraps off-rank layers in _OffsetStubCache
    c0 = rank0_model.language_model.make_cache()
    c1 = rank1_model.language_model.make_cache()
    assert not isinstance(c0[7], _OffsetStubCache)
    assert isinstance(c0[8], _OffsetStubCache)
    assert isinstance(c1[0], _OffsetStubCache)
    assert not isinstance(c1[8], _OffsetStubCache)

    # Execute Rank 0 up to layer 8 (layer 7 sends to Rank 1), then Rank 1 (which receives at
    # layer 8 and sends at layer 9), then finish Rank 0 (which receives at layer 9).
    h0 = _FakeTensor(0, shape=(1, 5))
    for idx in range(9):
        h0 = rank0_model.language_model.layers[idx](h0, cache=c0[idx])
    assert mailbox[(0, 1)].val == sum(range(1, 9))
    assert c0[7].keys == ("kv_keys", "dep_on_send")

    # Rank 1 runs its full layer list (layers 0..7 are PassThrough, layer 8 recvs from 0, layer 9 sends to 0)
    out1 = rank1_model.language_model(_FakeTensor(0, shape=(1, 5)), cache=c1)
    assert out1.val == 55
    assert mailbox[(1, 0)].val == 55

    # Rank 0 completes layer 9 (FinalStageRecvWrapper receives from Rank 1 and ties to x via mx.depends)
    h0 = rank0_model.language_model.layers[9](h0, cache=c0[9])
    assert isinstance(h0, tuple) and h0[1] == "dep_on_send"
    assert h0[0].val == expected_out.val == 55

    # Every cache slot (both active and _OffsetStubCache) advanced by 5 tokens on both ranks!
    assert [c.offset for c in c0] == [5] * 10
    assert [c.offset for c in c1] == [5] * 10


@pytest.mark.skipif(
    not mp.mx.metal.is_available(),
    reason="Metal GPU required for Gated DeltaNet kernel bitwise parity test",
)
def test_gated_delta_packed_kernel_bitwise_parity_with_xtree() -> None:
    import importlib.util

    gated_delta_file = (
        Path(__file__).resolve().parents[2]
        / "forks"
        / "mlx-vlm"
        / "mlx_vlm"
        / "models"
        / "gated_delta.py"
    )
    if not gated_delta_file.exists():
        pytest.skip("forks/mlx-vlm not present")
    spec = importlib.util.spec_from_file_location("_fork_gated_delta", gated_delta_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mx = mp.mx
    mx.random.seed(42)
    B, T, Hk, Hv, Dk, Dv = 1, 16, 4, 8, 128, 128
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.float32)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(mx.bfloat16)
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    y_packed, s_packed = mod.gated_delta_kernel(q, k, v, g, beta, state)
    y_xtree, s_xtree = mod.gated_delta_kernel_xtree(q, k, v, g, beta, state)
    y_unpacked, s_unpacked = mod.gated_delta_kernel_unpacked(q, k, v, g, beta, state)
    mx.eval(y_packed, s_packed, y_xtree, s_xtree, y_unpacked, s_unpacked)

    assert mx.array_equal(y_packed, y_xtree).item()
    assert mx.array_equal(s_packed, s_xtree).item()
    assert mx.allclose(
        y_packed.astype(mx.float32),
        y_unpacked.astype(mx.float32),
        atol=1e-3,
        rtol=1e-3,
    ).item()


def test_offset_stub_cache_batch_and_apc_checkpoint_contract() -> None:
    from mlx_vlm.apc_adapters import build_prefix_cache_plan_from_caches
    from mlx_vlm.models.cache import KVCache

    caches = [KVCache(), _OffsetStubCache()]
    plan = build_prefix_cache_plan_from_caches(caches)
    assert plan.restorable is True
    assert plan.strategy == "checkpoint"

    stub = _OffsetStubCache(offset=12)
    snap = stub.prefix_cache_snapshot()
    restored = _OffsetStubCache()
    restored.prefix_cache_restore(snap)
    assert restored.offset == 12

    batched = stub.to_batch([0, 2])
    assert isinstance(batched, _OffsetStubCache)
    batched.advance(5)
    assert batched.offset.tolist() == [17, 15]
    single = batched.extract(1)
    assert single.offset == 15
    batched.filter([0])
    assert batched.offset.tolist() == [17]
    batched.extend(_OffsetStubCache(offset=9))
    assert batched.offset.tolist() == [17, 9]

    merged = _OffsetStubCache.merge([_OffsetStubCache(offset=4), _OffsetStubCache(offset=8)])
    assert merged.offset.tolist() == [4, 8]
    pm = stub.prefix_cache_merge([_OffsetStubCache(offset=3), _OffsetStubCache(offset=7)], [3, 7])
    assert pm is not None and pm.offset.tolist() == [3, 7]


def test_repoint_model_probe_indices_moves_fa_and_ssm_idx_to_owned_slice() -> None:
    class _HybridLayer(nn.Module):
        def __init__(self, is_linear: bool) -> None:
            super().__init__()
            self.is_linear = is_linear

    class _HybridBackbone:
        def __init__(self) -> None:
            self.layers = [
                _HybridLayer(is_linear=(i % 4 != 3)) for i in range(8)
            ]
            self.ssm_idx = 0
            self.fa_idx = 3

    class _HybridLM:
        def __init__(self) -> None:
            self.model = _HybridBackbone()

        @property
        def layers(self):
            return self.model.layers

        def make_cache(self):
            return [_FakeKVCache() for _ in range(8)]

    lm = _HybridLM()
    apply_asymmetric_pipeline(lm, group=_FakeGroup(rank=1, size=2), ratios=(0.5, 0.5))
    # Rank 1 owns layers 4..7: first linear in 4..7 is 4, first full-attention is 7
    assert lm.model.ssm_idx == 4
    assert lm.model.fa_idx == 7


def test_selective_safetensors_load_scope_drops_off_rank_keys_and_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    s1.write_bytes(b"")
    s2.write_bytes(b"")
    index = {
        "weight_map": {
            "vision_tower.patch_embed.weight": s1.name,
            "language_model.model.embed_tokens.weight": s1.name,
            "language_model.model.layers.0.self_attn.q_proj.weight": s1.name,
            "language_model.model.layers.1.self_attn.q_proj.weight": s2.name,
            "language_model.lm_head.weight": s2.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    fake_tensors = {
        str(s1): {
            "vision_tower.patch_embed.weight": "vt",
            "language_model.model.embed_tokens.weight": "emb",
            "language_model.model.layers.0.self_attn.q_proj.weight": "l0",
        },
        str(s2): {
            "language_model.model.layers.1.self_attn.q_proj.weight": "l1",
            "language_model.lm_head.weight": "head",
        },
    }
    monkeypatch.setattr(mp.mx, "load", lambda path, *a, **kw: dict(fake_tensors[str(path)]))

    r0 = PipelineAssignment(rank=0, world_size=2, start_layer=0, end_layer=1, total_layers=2)
    r1 = PipelineAssignment(rank=1, world_size=2, start_layer=1, end_layer=2, total_layers=2)

    with mp.selective_safetensors_load_scope(tmp_path, r0):
        loaded_r0 = {}
        for f in sorted(tmp_path.glob("model*.safetensors")):
            loaded_r0.update(mp.mx.load(str(f)))
    assert set(loaded_r0.keys()) == {
        "vision_tower.patch_embed.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.lm_head.weight",
    }

    with mp.selective_safetensors_load_scope(tmp_path, r1):
        loaded_r1 = {}
        for f in sorted(tmp_path.glob("model*.safetensors")):
            loaded_r1.update(mp.mx.load(str(f)))
    assert set(loaded_r1.keys()) == {
        "language_model.model.layers.1.self_attn.q_proj.weight",
    }


def test_rank0_step_coordinator_and_worker_loop_lockstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers: list[Any] = []
    monkeypatch.setattr(
        mp.mx.distributed,
        "send",
        lambda x, dst, group=None: headers.append(x) or x,
    )
    monkeypatch.setattr(
        mp.mx.distributed,
        "recv_like",
        lambda x, src, group=None: headers.pop(0),
    )

    calls_r1: list[tuple[int, int]] = []

    class _DummyLM:
        def __init__(self, *, record: bool = False) -> None:
            self.layers = [_FakeLayer(0, 1), _FakeLayer(1, 2)]
            self._record = record

        def make_cache(self):
            return [_FakeKVCache(), _FakeKVCache()]

        def __call__(self, inputs, cache=None, **kwargs):
            b, s = int(inputs.shape[0]), int(inputs.shape[1])
            if cache is not None:
                for c in cache:
                    c.advance(s)
            if self._record:
                calls_r1.append((b, s))
            return inputs

    r0_lm = _DummyLM(record=False)
    r1_lm = _DummyLM(record=True)
    mp.install_rank0_step_coordinator(r0_lm, group=_FakeGroup(0, 2))

    # Rank 0 runs 2 forward steps (prefill 6 tokens, decode 1 token) then sends shutdown
    c0 = r0_lm.make_cache()
    r0_lm(mp.mx.zeros((1, 6), dtype=mp.mx.int32), cache=c0)
    r0_lm(mp.mx.zeros((1, 1), dtype=mp.mx.int32), cache=c0)
    mp.send_pipeline_shutdown(group=_FakeGroup(0, 2))

    steps = mp.run_pipeline_worker_loop(r1_lm, group=_FakeGroup(1, 2))
    assert steps == 2
    assert calls_r1 == [(1, 6), (1, 1)]


def test_configure_manual_ring_and_init_pipeline_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heylook_llm.config import MLXModelConfig

    cfg = MLXModelConfig(
        model_path="/tmp/fake-mlx",
        pipeline_ratios=[0.8, 0.2],
        pipeline_peers=["peer0.local", "peer1.local"],
    )
    assert cfg.pipeline_peers == ["peer0.local", "peer1.local"]

    hostfile = mp.configure_manual_ring(["peer0.local", "peer1.local:40000"], rank=1)
    try:
        assert os.environ["MLX_HOSTFILE"] == hostfile
        assert os.environ["MLX_RANK"] == "1"
        data = json.loads(Path(hostfile).read_text(encoding="utf-8"))
        assert data == [["peer0.local:32323"], ["peer1.local:40000"]]
    finally:
        Path(hostfile).unlink(missing_ok=True)
        monkeypatch.delenv("MLX_HOSTFILE", raising=False)
        monkeypatch.delenv("MLX_RANK", raising=False)

    attempts = {"count": 0}

    def _fake_init(*, strict: bool = False, backend: str = "any"):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("couldn't connect")
        return _FakeGroup(0, 2)

    monkeypatch.setattr(mp.mx.distributed, "init", _fake_init)
    monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
    grp = mp.init_pipeline_group(peers=["peer0.local", "peer1.local"], rank=0, max_wait_s=5.0)
    assert grp.size() == 2
    assert attempts["count"] == 2




