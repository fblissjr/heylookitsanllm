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
    from heylook_llm.providers import mlx_describe

    cfg = MLXModelConfig(
        model_path="/tmp/fake-mlx",
        pipeline_ratios=[0.8, 0.2],
        pipeline_peers=["peer0.local", "peer1.local"],
    )
    assert cfg.pipeline_peers == ["peer0.local", "peer1.local"]
    desc = mlx_describe.describe_static(
        "fake-mlx",
        cfg.model_dump(),
        cfg,
        written=True,
        derived={},
    )
    assert desc.decoding is not None
    assert "pipeline" in desc.decoding
    assert desc.decoding["pipeline"].value == "0.8/0.2"
    assert desc.decoding["pipeline"].provenance == "configured"

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

    monkeypatch.setattr(mp, "_CACHED_RING_GROUP", None)
    monkeypatch.setattr(mp.mx.distributed, "init", _fake_init)
    monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
    grp = mp.init_pipeline_group(peers=["peer0.local", "peer1.local"], rank=0, max_wait_s=5.0)
    assert grp.size() == 2
    assert attempts["count"] == 2
    # Subsequent call reuses the singleton ring group without calling mx.distributed.init again
    grp2 = mp.init_pipeline_group(peers=["peer0.local", "peer1.local"], rank=0, max_wait_s=5.0)
    assert grp2 is grp
    assert attempts["count"] == 2
    monkeypatch.setattr(mp, "_CACHED_RING_GROUP", None)


def test_resolve_host_ordered_peers_places_local_host_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp, "get_local_ip_addresses", lambda: {"198.51.100.2", "127.0.0.1"})
    ordered = mp.resolve_host_ordered_peers(["198.51.100.1", "198.51.100.2:40000"])
    assert ordered == ["198.51.100.2:40000", "198.51.100.1"]

    # Already first stays first
    assert mp.resolve_host_ordered_peers(["198.51.100.2", "198.51.100.1"]) == [
        "198.51.100.2",
        "198.51.100.1",
    ]


def test_control_port_streams_only_missing_rank1_shards_and_fails_fast_when_offline(
    tmp_path: Path,
) -> None:
    import socket
    import threading

    host_model = tmp_path / "host" / "org" / "TestModel"
    host_model.mkdir(parents=True)
    (host_model / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    (host_model / "tokenizer_config.json").write_text("{}")

    s1 = host_model / "model-00001-of-00002.safetensors"
    s2 = host_model / "model-00002-of-00002.safetensors"
    s1.write_bytes(b"SHARD_ONE_DATA" * 10)
    s2.write_bytes(b"SHARD_TWO_DATA" * 12)
    index = {
        "weight_map": {
            "language_model.model.layers.0.self_attn.q_proj.weight": s1.name,
            "language_model.model.layers.1.self_attn.q_proj.weight": s1.name,
            "language_model.model.layers.2.self_attn.q_proj.weight": s2.name,
            "language_model.model.layers.3.self_attn.q_proj.weight": s2.name,
        }
    }
    (host_model / "model.safetensors.index.json").write_text(json.dumps(index))

    r1_assign = PipelineAssignment(rank=1, world_size=2, start_layer=2, end_layer=4, total_layers=4)
    req_files = mp.list_required_files_for_rank(host_model, r1_assign)
    req_names = [p.name for p in req_files]
    assert "config.json" in req_names
    assert "model-00002-of-00002.safetensors" in req_names
    assert "model-00001-of-00002.safetensors" not in req_names

    # 1. Unreachable control port raises PeerUnreachableError immediately
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_unused:
        s_unused.bind(("127.0.0.1", 0))
        unused_port = s_unused.getsockname()[1]

    with pytest.raises(mp.PeerUnreachableError, match="unreachable"):
        mp.prepare_worker_via_control_port(
            host_model,
            peers=["127.0.0.1", "127.0.0.1"],
            ratios=(0.5, 0.5),
            control_port=unused_port,
            timeout_s=0.3,
        )

    # 2. Live control port streams only missing Rank-1 files into worker_root
    worker_root = tmp_path / "worker_models"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    received_meta: dict[str, Any] = {}

    def _worker_thread() -> None:
        conn, _ = srv.accept()
        with conn:
            req = mp._recv_json_msg(conn)
            received_meta.update(req)
            target_dir = worker_root / req["model_rel_path"]
            mp.receive_missing_files_on_worker(conn, target_dir, req.get("files", []))
            mp._send_json_msg(conn, {"status": "ready", "model_path": str(target_dir)})
        srv.close()

    t = threading.Thread(target=_worker_thread, daemon=True)
    t.start()
    mp.prepare_worker_via_control_port(
        host_model,
        peers=["127.0.0.1", "127.0.0.1"],
        ratios=(0.5, 0.5),
        control_port=port,
        timeout_s=2.0,
    )
    t.join(timeout=2.0)

    worker_model_dir = worker_root / received_meta["model_rel_path"]
    assert (worker_model_dir / "config.json").exists()
    assert (worker_model_dir / "model-00002-of-00002.safetensors").read_bytes() == s2.read_bytes()
    assert not (worker_model_dir / "model-00001-of-00002.safetensors").exists()


def test_bidirectional_heal_and_scan_folder_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import threading

    # 1. Verify heylook.toml [scan].folders resolution
    scan_dir_a = tmp_path / "mac_a_models"
    scan_dir_b = tmp_path / "mac_b_models"
    scan_dir_a.mkdir()
    scan_dir_b.mkdir()
    toml_a = tmp_path / "heylook_a.toml"
    toml_a.write_text(f'[scan]\nfolders = ["{scan_dir_a.as_posix()}"]\n', encoding="utf-8")

    assert mp.resolve_default_models_root(config_path=toml_a) == scan_dir_a.resolve()
    monkeypatch.setenv("HEYLOOK_CONFIG", str(toml_a))
    nested_model = scan_dir_a / "Qwen" / "TestVL"
    nested_model.mkdir(parents=True)
    assert mp.infer_model_rel_path(nested_model) == "Qwen/TestVL"

    # 2. Bidirectional heal during prepare_worker_via_control_port:
    # Host (Mac A) is missing tokenizer_config.json and shard 1 (which Worker Mac B has),
    # while Worker (Mac B) is missing shard 2 (which Host Mac A has).
    worker_model = scan_dir_b / "Qwen" / "TestVL"
    worker_model.mkdir(parents=True)

    cfg_bytes = json.dumps({"num_hidden_layers": 4}).encode("utf-8")
    idx_bytes = json.dumps(
        {
            "weight_map": {
                "vision_tower.patch_embed.weight": "model-00001-of-00002.safetensors",
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "language_model.model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "language_model.model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                "language_model.model.layers.3.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            }
        }
    ).encode("utf-8")
    (nested_model / "config.json").write_bytes(cfg_bytes)
    (nested_model / "model.safetensors.index.json").write_bytes(idx_bytes)
    (nested_model / "model-00002-of-00002.safetensors").write_bytes(b"SHARD2_FROM_HOST")

    (worker_model / "config.json").write_bytes(cfg_bytes)
    (worker_model / "tokenizer_config.json").write_bytes(b'{"tokenizer": "from_worker"}')
    (worker_model / "model-00001-of-00002.safetensors").write_bytes(b"SHARD1_VISION_FROM_WORKER")

    # Also place a GGUF model only on Worker (Mac B) and an MLX config only on Host (Mac A)
    gguf_b = scan_dir_b / "google" / "gemma-gguf"
    gguf_b.mkdir(parents=True)
    (gguf_b / "gemma.gguf").write_bytes(b"GGUF_WEIGHTS_ON_B")
    (nested_model / "model.heylook.toml").write_bytes(b"pipeline_ratios = [0.5, 0.5]\n")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    stop_flag = {"stop": False}

    def _daemon_loop() -> None:
        while not stop_flag["stop"]:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                req = mp._recv_json_msg(conn)
                if mp.handle_sync_control_request(conn, req, scan_dir_b):
                    continue
                if req.get("op") == "prepare_model":
                    target = scan_dir_b / req["model_rel_path"]
                    mp.receive_missing_files_on_worker(
                        conn,
                        target,
                        req.get("files", []),
                        host_present=req.get("host_present"),
                    )
                    mp._send_json_msg(conn, {"status": "ready", "target_dir": str(target)})

    t = threading.Thread(target=_daemon_loop, daemon=True)
    t.start()
    try:
        mp.prepare_worker_via_control_port(
            nested_model,
            peers=["127.0.0.1", "127.0.0.1"],
            ratios=(0.5, 0.5),
            control_port=port,
            timeout_s=2.0,
        )
        # Host (Mac A) automatically pulled missing tokenizer_config.json + shard 1 from Worker (Mac B)!
        assert (nested_model / "tokenizer_config.json").read_bytes() == b'{"tokenizer": "from_worker"}'
        assert (nested_model / "model-00001-of-00002.safetensors").read_bytes() == b"SHARD1_VISION_FROM_WORKER"
        # And Worker (Mac B) received missing shard 2 from Host (Mac A)!
        assert (worker_model / "model-00002-of-00002.safetensors").read_bytes() == b"SHARD2_FROM_HOST"

        # 3. Full bidirectional [scan].folders sync (both MLX and GGUF models)
        summary = mp.sync_scan_folders_with_peer(
            "127.0.0.1",
            local_root=scan_dir_a,
            control_port=port,
            include_weights=True,
        )
        assert "google/gemma-gguf/gemma.gguf" in summary["pulled"]
        assert "Qwen/TestVL/model.heylook.toml" in summary["pushed"]
        assert (scan_dir_a / "google" / "gemma-gguf" / "gemma.gguf").read_bytes() == b"GGUF_WEIGHTS_ON_B"
        assert (worker_model / "model.heylook.toml").read_bytes() == b"pipeline_ratios = [0.5, 0.5]\n"
    finally:
        stop_flag["stop"] = True
        srv.close()
        t.join(timeout=2.0)


def test_worker_loop_handles_apc_snapshot_restore_and_unload(
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

    seen_r0: list[int] = []
    seen_r1: list[int] = []

    class _DummyLM:
        def __init__(self, sink: list[int]) -> None:
            self.layers = [_FakeLayer(0, 1)]
            self._sink = sink

        def make_cache(self):
            return [_FakeKVCache()]

        def __call__(self, inputs, cache=None, **kwargs):
            s = int(inputs.shape[1])
            if cache is not None:
                self._sink.append(int(cache[0].offset))
                for c in cache:
                    c.advance(s)
            return inputs

    class _FakeAPC:
        def __init__(self) -> None:
            self._exact: dict[tuple[tuple[int, ...], int], Any] = {}

        def store_exact_cache(self, token_ids: list[int], prompt_cache: Any, *, extra_hash: int = 0) -> bool:
            self._exact[(tuple(token_ids), int(extra_hash))] = [c.offset for c in prompt_cache]
            return True

        def get_exact_cache(self, token_ids: list[int], extra_hash: int = 0):
            for n in range(len(token_ids), 0, -1):
                key = (tuple(token_ids[:n]), int(extra_hash))
                if key in self._exact:
                    saved = self._exact[key]
                    restored = [_FakeKVCache() for _ in saved]
                    for c, off in zip(restored, saved):
                        c.offset = off
                    return restored, n
            return None, 0

    r0_lm = _DummyLM(seen_r0)
    r1_lm = _DummyLM(seen_r1)
    g0 = _FakeGroup(0, 2)
    g1 = _FakeGroup(1, 2)
    mp.install_rank0_step_coordinator(r0_lm, group=g0)
    apc = _FakeAPC()
    mp.install_rank0_apc_sync(apc, group=g0)

    # Turn 1: prefill 5 tokens -> store APC snapshot at 5 tokens
    c0 = r0_lm.make_cache()
    r0_lm(mp.mx.zeros((1, 5), dtype=mp.mx.int32), cache=c0)
    apc.store_exact_cache([10, 20, 30, 40, 50], c0)
    # Decode 1 token (offset advances to 6)
    r0_lm(mp.mx.zeros((1, 1), dtype=mp.mx.int32), cache=c0)

    # Turn 2: restore APC snapshot at 5 tokens -> prefill 3 suffix tokens (starting from offset 5!)
    c0_turn2, prefix_len = apc.get_exact_cache([10, 20, 30, 40, 50, 60, 70, 80])
    assert prefix_len == 5
    r0_lm(mp.mx.zeros((1, 3), dtype=mp.mx.int32), cache=c0_turn2)

    # Unload model on Rank 0 -> Rank 1 loop exits cleanly while keeping ring alive
    mp.send_pipeline_unload(group=g0)

    steps = mp.run_pipeline_worker_loop(r1_lm, group=g1)
    assert steps == 3
    # Offsets seen before each of the 3 forward passes on both Rank 0 and Rank 1: 0, 5, and 5 (restored from snapshot!)
    assert seen_r0 == [0, 5, 5]
    assert seen_r1 == [0, 5, 5]





