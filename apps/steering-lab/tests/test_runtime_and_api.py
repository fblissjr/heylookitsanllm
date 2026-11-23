"""CPU-only MLX runtime, verify-loop parity, tangent RMSNorm, manifold & FastAPI tests.

Forces ``mx.set_default_device(mx.cpu)`` so tests exercise the real MLX tensor
pipeline at 0% Metal GPU contention against a live ``heylook`` / H3 server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from steering_lab.bank import ModelFingerprint, create_demo_bank
from steering_lab.cli import main as cli_main
from steering_lab.operators import LayerVector, VectorSpec
from steering_lab.runtime import (
    LayerSteeringController,
    SteeringLabRuntime,
    SteeringPlan,
    classify_manifold_state,
    ngram_repetition_rate,
)
from steering_lab.server import LabState, create_app

# Token IDs for the tiny synthetic CPU-only MLX language model
THINK_START = 1
THINK_END = 2
IMAGE_PAD = 3
EOS_ID = 4
VOCAB_SIZE = 24
HIDDEN_DIM = 16


class _SimpleTokenizer:
    def __init__(self) -> None:
        self.vocab = [f"t{i}" for i in range(VOCAB_SIZE)]
        self.vocab[THINK_START] = "<think>"
        self.vocab[THINK_END] = "</think>"
        self.vocab[IMAGE_PAD] = "<|image_pad|>"
        self.vocab[EOS_ID] = "<eos>"

    def encode(self, text: str) -> list[int]:
        if "[IMG]" in text:
            return [5, IMAGE_PAD, IMAGE_PAD, 6]
        return [5, 6, 7]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[int(i) % VOCAB_SIZE] for i in ids)


class _DummyTx:
    def __init__(self, cache: list[dict[str, Any]], snapshot: list[int]) -> None:
        self.cache = cache
        self.snapshot = snapshot
        self.committed: Optional[int] = None

    def commit(self, n: int) -> None:
        self.committed = int(n)
        for c, prev in zip(self.cache, self.snapshot):
            c["len"] = prev + int(n)


def _make_synthetic_mlx_lm(*, supports_verify_hook: bool = True):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(123)

    class _ZeroCenteredRMSNorm:
        """Matches Qwen 3.8's ``(1 + weight) * x / rms(x)`` final norm."""

        def __init__(self, weight_np: np.ndarray) -> None:
            self.weight = mx.array(weight_np.astype(np.float32))

        def __call__(self, x):
            rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            return (x / rms) * (1.0 + self.weight)

    class _ToyLayer:
        def __init__(self, idx: int, is_linear: bool) -> None:
            self.idx = idx
            self.is_linear = is_linear
            w = np.eye(HIDDEN_DIM, dtype=np.float32) * 0.92 + rng.normal(
                0.0, 0.03, size=(HIDDEN_DIM, HIDDEN_DIM)
            ).astype(np.float32)
            self.w = mx.array(w)

        def forward_raw(self, x, cache_slot=None):
            if cache_slot is not None:
                cache_slot["len"] = cache_slot.get("len", 0) + int(x.shape[1])
            return x @ self.w

        def __call__(self, x, mask=None, cache=None):
            return self.forward_raw(x, cache)

    class _ToyInner:
        def __init__(self) -> None:
            # 3:1 hybrid layout matching Qwen 3.8: layers 0,1,2 GDN (is_linear=True), layer 3 FA!
            self.layers = [_ToyLayer(i, is_linear=(i % 4 != 3)) for i in range(4)]
            emb = rng.normal(0.0, 0.4, size=(VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32)
            self._emb_w = mx.array(emb)
            # Strongly non-uniform zero-centered weights (1 + w) to test tangent RMSNorm!
            norm_w = np.zeros((HIDDEN_DIM,), dtype=np.float32)
            norm_w[0] = 9.0  # 1+w = 10.0 on channel 0
            norm_w[1] = -0.9  # 1+w = 0.1 on channel 1
            self.norm = _ZeroCenteredRMSNorm(norm_w)

        def embed_tokens(self, ids):
            return self._emb_w[ids]

    class _ToyLM:
        def __init__(self) -> None:
            self.model = _ToyInner()
            self._head_w = mx.array(np.eye(HIDDEN_DIM, VOCAB_SIZE, dtype=np.float32))
            self._supports_steering_lab_hook = supports_verify_hook
            self._steering_lab_hook = None
            self.verify_calls = 0

        def lm_head(self, h):
            return h @ self._head_w

        def make_cache(self):
            return [{"len": 0} for _ in self.model.layers]

        def __call__(self, ids, cache=None):
            h = self.model.embed_tokens(ids)
            for idx, layer in enumerate(self.model.layers):
                c = cache[idx] if cache is not None else None
                h = layer(h, cache=c)
            h = self.model.norm(h)
            return self.lm_head(h)

        def speculative_verify_hidden(self, ids, cache):
            """Simulates qwen3_5_runtime.py's INLINED loop bypassing layer.__call__!"""
            self.verify_calls += 1
            snapshot = [c["len"] for c in cache]
            h = self.model.embed_tokens(ids)
            for idx, layer in enumerate(self.model.layers):
                # Bypass layer.__call__ intentionally (exact qwen3_5_runtime.py behavior!)
                h = layer.forward_raw(h, cache[idx])
                if self._supports_steering_lab_hook and callable(self._steering_lab_hook):
                    h = self._steering_lab_hook(idx, h, False)
            h = self.model.norm(h)
            return h, None, _DummyTx(cache, snapshot)

        def speculative_logits_from_hidden(self, h):
            return self.lm_head(h)

    fp = ModelFingerprint(
        model_id="toy-hybrid-cpu",
        hidden_size=HIDDEN_DIM,
        num_hidden_layers=4,
        config_sha256="toyhybrid0000004",
    )
    return _ToyLM(), _SimpleTokenizer(), fp


def _make_runtime_with_vectors(*, supports_verify_hook: bool = True) -> SteeringLabRuntime:
    lm, tok, fp = _make_synthetic_mlx_lm(supports_verify_hook=supports_verify_hook)
    rt = SteeringLabRuntime(
        model=lm,
        processor=tok,
        fingerprint=fp,
        think_start_id=THINK_START,
        think_end_id=THINK_END,
        image_pad_id=IMAGE_PAD,
        eos_token_ids=(EOS_ID,),
    )
    e0 = np.eye(HIDDEN_DIM, dtype=np.float32)[0]
    e2 = np.eye(HIDDEN_DIM, dtype=np.float32)[2]
    e5 = np.eye(HIDDEN_DIM, dtype=np.float32)[5]
    by_layer_mid = {
        l: LayerVector(
            layer=l,
            mu_d=e2,
            vr=e0.reshape(1, -1),
            d_diag=np.array([1.0], dtype=np.float32),
            mu_d_norm=1.0,
            mu_clean=e5 * 0.5,
        )
        for l in (1, 2, 3)
    }
    rt.vectors["style_mid"] = VectorSpec(
        id="style_mid",
        label="Style Mid",
        operator="regime_a",
        phase="content",
        token_mask="decode",
        layers=(1, 2, 3),
        by_layer=by_layer_mid,
    )
    rt.vectors["term_only"] = VectorSpec(
        id="term_only",
        label="Terminal Only",
        operator="regime_a",
        phase="content",
        token_mask="decode",
        layers=(3,),
        by_layer={3: by_layer_mid[3]},
    )
    return rt


def test_mlx_cpu_s0_identity_and_layer_filter_cache_sig():
    rt = _make_runtime_with_vectors()
    assert rt.layer_kinds() == {0: "GDN", 1: "GDN", 2: "GDN", 3: "FA"}

    base = rt.generate_steered("hello", plan=SteeringPlan(), max_tokens=8, enable_thinking=False)
    zero = rt.generate_steered(
        "hello", plan=SteeringPlan(scales={"style_mid": 0.0}), max_tokens=8, enable_thinking=False
    )
    assert [t.token_id for t in base.tokens] == [t.token_id for t in zero.tokens]

    # FA-only vs GDN-only layer compilation
    ctrl_fa = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.0}, layer_filter="fa_only"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    ctrl_gdn = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.0}, layer_filter="gdn_only"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    assert list(ctrl_fa.compiled.keys()) == [3]
    assert list(ctrl_gdn.compiled.keys()) == [1, 2]

    # Prefix cache signature: terminal-only is "", mid-layer is non-empty deterministic digest.
    assert SteeringPlan(scales={"term_only": 1.0}).cache_signature(rt.vectors, 4) == ""
    sig1 = SteeringPlan(scales={"style_mid": 1.0}).cache_signature(rt.vectors, 4)
    sig2 = SteeringPlan(scales={"style_mid": 0.5}).cache_signature(rt.vectors, 4)
    assert sig1 != "" and sig2 != "" and sig1 != sig2


def test_mlx_cpu_phase_split_and_image_pad_mask():
    rt = _make_runtime_with_vectors()
    # In thinking mode when model never emits </think>, phase_override="content" leaves every token at g_t == 0.
    tr = rt.generate_steered(
        "hello",
        plan=SteeringPlan(scales={"style_mid": 1.2}, phase_override="content"),
        max_tokens=6,
        enable_thinking=True,
    )
    assert all(t.phase == "think" for t in tr.tokens)
    assert all(t.g_t == 0.0 and t.rel_delta == 0.0 for t in tr.tokens)

    # Verify token_mask="image_pad" steers ONLY <|image_pad|> positions and leaves prompt_text untouched!
    ctrl = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.5}, phase_override="all", token_mask_override="image_pad"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    ctrl.set_phase("prompt")
    ctrl.set_prompt_token_types(["prompt_text", "image_pad", "image_pad", "prompt_text"])
    h = np.ones((1, 4, HIDDEN_DIM), dtype=np.float32) * 0.2
    out = ctrl.apply_layer_np(3, h)
    assert np.allclose(out[0, 0], h[0, 0], atol=1e-6)
    assert np.allclose(out[0, 3], h[0, 3], atol=1e-6)
    assert float(np.linalg.norm(out[0, 1] - h[0, 1])) > 0.5
    assert float(np.linalg.norm(out[0, 2] - h[0, 2])) > 0.5


def test_mlx_cpu_unrolled_verify_bypass_guard_and_parity():
    # Part A: stock unrolled verify (without _supports_steering_lab_hook) must NOT silently bypass steering!
    rt_unsafe = _make_runtime_with_vectors(supports_verify_hook=False)
    plan = SteeringPlan(scales={"style_mid": 1.2}, phase_override="all")
    tr_safe_fallback = rt_unsafe.generate_steered(
        "hello",
        plan=plan,
        max_tokens=6,
        enable_thinking=False,
        simulated_draft_fn=lambda seq: [8, 9],
    )
    assert rt_unsafe.lm.verify_calls == 0  # safely disabled unrolled verify!
    assert tr_safe_fallback.mean_rel_delta > 0.0

    # Part B: hook-aware verify matches undrafted greedy decode bit-for-bit!
    rt_hook = _make_runtime_with_vectors(supports_verify_hook=True)
    undrafted = rt_hook.generate_steered("hello", plan=plan, max_tokens=6, enable_thinking=False)
    drafted = rt_hook.generate_steered(
        "hello",
        plan=plan,
        max_tokens=6,
        enable_thinking=False,
        simulated_draft_fn=lambda seq: [undrafted.tokens[0].token_id, 9],
    )
    assert rt_hook.lm.verify_calls > 0
    assert [t.token_id for t in drafted.tokens] == [t.token_id for t in undrafted.tokens]

    # Part C: Per-position phase mask on a verify tensor straddling </think> -> content!
    ctrl = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.5}, phase_override="content", token_mask_override="decode"),
        rt_hook.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt_hook._layer_is_linear_flags(),
    )
    ctrl.set_phase("think")
    ctrl.set_token_phases(["think", "think", "content", "content"])
    h = np.zeros((1, 4, HIDDEN_DIM), dtype=np.float32)
    out = ctrl.apply_layer_np(3, h, commit_center=False)
    assert np.allclose(out[0, 0], 0.0, atol=1e-6)
    assert np.allclose(out[0, 1], 0.0, atol=1e-6)
    assert np.isclose(float(np.linalg.norm(out[0, 2])), 0.5, atol=1e-5)  # 1.5 / 3 active layers (divide_k default)
    assert np.isclose(float(np.linalg.norm(out[0, 3])), 0.5, atol=1e-5)


def test_mlx_cpu_tangent_rmsnorm_respects_zero_centered_weights():
    rt = _make_runtime_with_vectors()
    # Channel 0 has (1+w)=10.0, Channel 1 has (1+w)=0.1. Equal perturbation along e0 + e1
    # must project 100x stronger onto token 0 than token 1 through tangent RMSNorm!
    v = np.zeros((HIDDEN_DIM,), dtype=np.float32)
    v[0] = 1.0
    v[1] = 1.0
    h_ref = np.zeros((HIDDEN_DIM,), dtype=np.float32)
    h_ref[5] = 2.0  # orthogonal reference state
    out = rt.xray_vector(v, layer=3, top_k=4, h_ref=h_ref)
    assert out["tangent_rmsnorm"] is True
    assert out["advisory"] is False
    assert out["adds"][0]["token_id"] == 0
    # Mid-layer X-ray is marked advisory=True
    mid = rt.xray_vector(v, layer=1, top_k=4, h_ref=h_ref)
    assert mid["advisory"] is True
    assert mid["layer_kind"] == "GDN"


def test_manifold_classifier_catches_high_mtp_loop_kl_and_entropy():
    # Catch #3: High MTP (0.60) + low entropy (0.55) + high 4-gram repetition => loop_collapse!
    rep_tokens = [10, 11, 12, 13] * 6
    rep_rate = ngram_repetition_rate(rep_tokens, n=4)
    assert rep_rate > 0.75
    loop_state = classify_manifold_state(
        mean_entropy=0.55,
        baseline_entropy=0.52,
        mtp_share=0.60,
        baseline_mtp_share=0.61,
        mean_rel_delta=0.05,
        ngram_repeat_4=rep_rate,
        mean_kl_vs_base=0.4,
    )
    assert loop_state["status"] == "loop_collapse"

    collapse = classify_manifold_state(
        mean_entropy=2.9,
        baseline_entropy=0.55,
        mtp_share=0.20,
        baseline_mtp_share=0.62,
        mean_rel_delta=0.2,
        ngram_repeat_4=0.0,
        mean_kl_vs_base=4.0,
    )
    assert collapse["status"] == "manifold_collapse"

    shift = classify_manifold_state(
        mean_entropy=0.62,
        baseline_entropy=0.55,
        mtp_share=0.45,
        baseline_mtp_share=0.62,
        mean_rel_delta=0.11,
        ngram_repeat_4=0.0,
        mean_kl_vs_base=0.42,
    )
    assert shift["status"] == "in_manifold_shift"


def test_mlx_cpu_sweep_forces_mtp_off_at_temp_gt_0_kl_and_token_delta():
    rt = _make_runtime_with_vectors(supports_verify_hook=True)
    arms = [
        SteeringPlan(label="base"),
        SteeringPlan(label="steered", scales={"style_mid": 1.5}, phase_override="all"),
    ]
    # Catch #5: temperature > 0 MUST force MTP off to keep PRNG aligned across arms!
    traces = rt.sweep(
        "hello",
        arms,
        max_tokens=6,
        temperature=0.7,
        seed=42,
        enable_thinking=False,
        use_mtp=True,
        simulated_draft_fn=lambda seq: [8, 9],
    )
    assert rt.lm.verify_calls == 0
    assert traces[0].mtp_share is None and traces[1].mtp_share is None
    assert traces[0].mean_kl_vs_base == 0.0
    assert traces[1].mean_kl_vs_base is not None and traces[1].mean_kl_vs_base > 0.0

    # Click-a-token causal Δ logprob ground truth
    delta = rt.token_delta_logprob(
        "hello",
        [t.token_id for t in traces[1].tokens],
        target_index=2,
        plan=arms[1],
        enable_thinking=False,
        top_k=5,
    )
    assert delta["causal"] is True
    assert delta["adds"][0]["delta_logprob"] > 0.0


def test_fastapi_server_and_cli_demo_bank_flow(tmp_path: Path):
    bank_dir = tmp_path / "demo_bank"
    vec_dir = tmp_path / "demo_vecs"

    # CLI end-to-end (0% GPU)
    assert cli_main(["demo-bank", "--out", str(bank_dir), "--dim", "64"]) == 0
    assert cli_main(["build", "--bank", str(bank_dir), "--out", str(vec_dir)]) == 0
    assert cli_main(["inspect-bank", "--bank", str(bank_dir)]) == 0
    assert (
        cli_main(
            [
                "xray",
                "--vector-dir",
                str(vec_dir / "clinical_vs_poetic"),
                "--layer",
                "47",
                "--bank",
                str(bank_dir),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "chain-lens",
                "--vector-dir",
                str(vec_dir / "clinical_vs_poetic"),
                "--layer",
                "47",
                "--bank",
                str(bank_dir),
                "--steps",
                "3",
            ]
        )
        == 0
    )

    state = LabState(bank_dir=bank_dir, vectors_dir=vec_dir)
    client = TestClient(create_app(state, enable_access_gate=False))

    r_html = client.get("/")
    assert r_html.status_code == 200 and "MLX Steering Lab" in r_html.text

    r_status = client.get("/api/status").json()
    assert r_status["model_loaded"] is False
    assert r_status["bank_loaded"] is True
    assert r_status["layer_kinds"]["47"] == "FA"
    assert r_status["layer_kinds"]["49"] == "GDN"

    r_vecs = client.get("/api/vectors").json()
    assert len(r_vecs["vectors"]) >= 5
    # Polish #2: per-layer mu_d_norms exposed directly in VectorSpec.public_meta()!
    assert "mu_d_norms" in r_vecs["vectors"][0]
    assert len(r_vecs["vectors"][0]["mu_d_norms"]) >= 1

    r_geom = client.get("/api/bank/geometry").json()
    assert r_geom["layers"]["47"]["effective_rank"] > r_geom["layers"]["63"]["effective_rank"]

    r_comp = client.post(
        "/api/compose",
        json={
            "layer": 47,
            "scales": {"clinical_vs_poetic": 1.0, "blunt_vs_hedged": 0.8},
            "use_peer_ros": True,
            "use_clamped_union": True,
            "use_holds": True,
        },
    ).json()
    assert r_comp["active"] is True
    assert r_comp["peer_ros_applied"] is True
    assert r_comp["holds_applied"] is True

    r_xray = client.post(
        "/api/xray", json={"concept_id": "clinical_vs_poetic", "layer": 47, "component": "mu_d"}
    ).json()
    assert r_xray["advisory"] is True
    assert r_xray["layer_kind"] == "FA"

    r_cl = client.post(
        "/api/chain-lens",
        json={
            "concept_id": "clinical_vs_poetic",
            "layer": 47,
            "use_jlens": True,
            "steps": 4,
            "beam_width": 4,
        },
    ).json()
    assert r_cl["mode"] == "demo_mtp"
    assert r_cl["jlens_applied"] is True
    assert r_cl["jlens_val_r2"] is not None
    assert r_cl["used_corpus_prior"] is True
    assert len(r_cl["chains"]) == 4
    assert r_cl["chains"][0]["round_trip_cos"] is not None
    assert r_cl["chains"][0]["round_trip_rank"] == 1
    assert r_cl["chains"][0]["round_trip_percentile"] is not None

    r_sweep = client.post(
        "/api/sweep",
        json={
            "prompt": "Explain unified memory.",
            "arms": [
                {"label": "Baseline", "scales": {}},
                {"label": "Clinical", "scales": {"clinical_vs_poetic": 1.0}},
            ],
        },
    ).json()
    assert r_sweep["mode"] == "demo_synthetic"
    assert len(r_sweep["arms"]) == 2
    assert r_sweep["arms"][1]["summary"]["mean_kl_vs_base"] > 0.0

    r_td = client.post(
        "/api/token-delta",
        json={
            "prompt": "Explain unified memory.",
            "token_ids": [1, 2, 3],
            "target_index": 1,
            "arm": {"label": "Clinical", "scales": {"clinical_vs_poetic": 1.0}},
        },
    ).json()
    assert r_td["mode"] == "demo_synthetic" and len(r_td["adds"]) > 0


def test_mlx_cpu_qwen35_mtp_adapter_forward_hidden_rope_kvcache_and_norm():
    import mlx.core as mx
    from steering_lab.chain_lens import Qwen35MTPAdapter, run_chain_lens

    rt = _make_runtime_with_vectors()
    lm = rt.lm

    class _MiniKVCache:
        def __init__(self) -> None:
            self.keys = None
            self.values = None
            self.offset = 0

    class _FakeQwen35MTPDraft:
        """Exercises the exact Qwen3_5MTPDraftModel._forward_hidden contract."""

        def __init__(self) -> None:
            self._input_embed = lm.model.embed_tokens
            self._input_embed_scale = 1.0
            self.seen_positions: list[int] = []
            self.seen_offsets: list[int] = []

        def make_cache(self):
            return [_MiniKVCache()]

        def _forward_hidden(self, token_embed, hidden, cache, position_ids):
            pos_val = int(position_ids[0, 0].item())
            self.seen_positions.append(pos_val)
            c = cache[0]
            self.seen_offsets.append(c.offset)
            # Append a 4D KV slice [1, 1, 1, D] to verify per-beam isolation on fork.
            k_new = token_embed[:, None, :, :]
            v_new = hidden[:, None, :, :]
            if c.keys is None or c.offset == 0:
                c.keys = k_new
                c.values = v_new
            else:
                c.keys = mx.concatenate([c.keys[:, :, : c.offset, :], k_new], axis=2)
                c.values = mx.concatenate([c.values[:, :, : c.offset, :], v_new], axis=2)
            c.offset += 1
            return 0.7 * hidden + 0.3 * token_embed + 0.01 * float(pos_val)

    draft = _FakeQwen35MTPDraft()
    adapter = Qwen35MTPAdapter(lm, draft, start_position=1)

    # Initial state must ALREADY be post-target.model.norm (matching speculative_verify_hidden).
    raw_h = np.ones((HIDDEN_DIM,), dtype=np.float32)
    st0 = adapter.initial_state(raw_h)
    expected_normed = lm.model.norm(mx.array(raw_h.reshape(1, 1, -1)))
    mx.eval(expected_normed)
    assert np.allclose(np.asarray(st0["h_mx"]), np.asarray(expected_normed), atol=1e-5)

    # Fork two independent child beams from st0 and verify per-beam KVCache + RoPE isolation.
    _, st_a1 = adapter.step(st0, token_id=5)
    _, st_b1 = adapter.step(st0, token_id=8)
    assert st_a1["pos"] == 2 and st_b1["pos"] == 2
    assert st_a1["kv_states"][0][2] == 1 and st_b1["kv_states"][0][2] == 1
    assert not np.allclose(np.asarray(st_a1["kv_states"][0][0]), np.asarray(st_b1["kv_states"][0][0]))

    _, st_a2 = adapter.step(st_a1, token_id=6)
    assert st_a2["pos"] == 3 and st_a2["kv_states"][0][2] == 2
    assert 1 in draft.seen_positions and 2 in draft.seen_positions

    # Full run_chain_lens + real residual encode_phrase_hidden + upper-layer transport
    # Probe channels 2..5 (where norm_w == 0) so multiple plausible tokens pass plausibility_beta.
    probe_h = np.zeros((HIDDEN_DIM,), dtype=np.float32)
    probe_h[2:6] = 1.0
    base_h = np.zeros((HIDDEN_DIM,), dtype=np.float32)
    base_h[6:10] = 1.0
    res = run_chain_lens(
        probe_h,
        adapter=adapter,
        decode_token=rt._decode_token,
        h_baseline=base_h,
        steps=3,
        beam_width=3,
        embed_tokens_fn=lambda tids: rt.encode_phrase_hidden(tids, layer=3),
    )
    assert len(res["chains"]) == 3
    assert all(len(c["token_ids"]) == 3 for c in res["chains"])
    assert res["chains"][0]["round_trip_cos"] is not None

    transported = rt.transport_upper_layers(raw_h, src_layer=1, dst_layer=3)
    assert transported.shape == (HIDDEN_DIM,)


def test_mlx_cpu_mixed_mask_regime_a_and_token_delta_phase_isolation():
    rt = _make_runtime_with_vectors()
    # Add a second Regime-A slider on the SAME layer (L3) with a DIFFERENT (phase, token_mask)!
    e4 = np.eye(HIDDEN_DIM, dtype=np.float32)[4]
    e6 = np.eye(HIDDEN_DIM, dtype=np.float32)[6]
    rt.vectors["vision_img"] = VectorSpec(
        id="vision_img",
        label="Vision Img",
        operator="regime_a",
        phase="all",
        token_mask="image_pad",
        layers=(3,),
        by_layer={
            3: LayerVector(
                layer=3,
                mu_d=e6,
                vr=e4.reshape(1, -1),
                d_diag=np.array([1.0], dtype=np.float32),
                mu_d_norm=2.0,
            )
        },
    )

    # Mix both sliders on L3 simultaneously: vision_img (image_pad) + style_mid (content/decode).
    ctrl = LayerSteeringController(
        SteeringPlan(scales={"vision_img": 1.0, "style_mid": 1.0}),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    assert len(ctrl.compiled[3].regime_a_groups) == 2

    # Position 0: prompt_text (neither fires), Position 1: image_pad (ONLY vision_img fires),
    # Position 2: decode/think (neither fires), Position 3: decode/content (ONLY style_mid fires)!
    ctrl.set_phase("content")
    ctrl.set_token_phases(["prompt", "prompt", "think", "content"])
    ctrl.set_prompt_token_types(["prompt_text", "image_pad", "decode", "decode"])
    h = np.zeros((1, 4, HIDDEN_DIM), dtype=np.float32)
    out = ctrl.apply_layer_np(3, h, commit_center=False)
    assert np.allclose(out[0, 0], 0.0, atol=1e-6)  # prompt_text untouched
    assert float(out[0, 1, 6]) > 1.0 and abs(float(out[0, 1, 2])) < 1e-5  # ONLY vision_img (e6)
    assert np.allclose(out[0, 2], 0.0, atol=1e-6)  # <think> untouched!
    assert float(out[0, 3, 2]) > 0.25 and abs(float(out[0, 3, 6])) < 1e-5  # ONLY style_mid (e2, 1/3 per layer)

    # Click-a-token token_delta_logprob MUST NOT steer inside <think> prefix for content-only slider,
    # and MUST match live decode starting at target_index=3 (first content token predicted from </think>)!
    gen_seq = [8, 9, THINK_END, 10, 11]
    plan_content = SteeringPlan(scales={"style_mid": 1.5}, phase_override="content")
    td_think = rt.token_delta_logprob(
        "hello", gen_seq, target_index=1, plan=plan_content, enable_thinking=True, top_k=4
    )
    td_first_content = rt.token_delta_logprob(
        "hello", gen_seq, target_index=3, plan=plan_content, enable_thinking=True, top_k=4
    )
    td_content = rt.token_delta_logprob(
        "hello", gen_seq, target_index=4, plan=plan_content, enable_thinking=True, top_k=4
    )
    assert max(abs(x["delta_logprob"]) for x in td_think["adds"]) < 1e-5
    assert max(abs(x["delta_logprob"]) for x in td_first_content["adds"]) > 0.05
    assert max(abs(x["delta_logprob"]) for x in td_content["adds"]) > 0.05


def test_mlx_cpu_capture_span_bank_vision_conditional_nonsink_and_truncation(tmp_path: Path):
    rt = _make_runtime_with_vectors()
    # Capture vision + 4-pole conditional + calibration on CPU MLX toy model.
    # Because toy model never emits </think>, enable_thinking=True tests truncation routing
    # (decoded tokens MUST land in "think" and NEVER be mislabeled into "answer"!).
    bank_think = rt.capture_span_bank(
        tmp_path / "bank_think_trunc",
        corpus_ids=["vision_optical_vs_mood", "cond_hedge_to_concrete", "jlens_calibration"],
        layers=(1, 3),
        max_tokens=6,
        max_tokens_per_span=4,
        enable_thinking=True,
    )
    # 1. Vision corpus actually attached images and captured image_pad tokens!
    assert bank_think.get_span("vision_optical_vs_mood", "a", "image_pad", 3) is not None
    assert bank_think.get_span("vision_optical_vs_mood", "b", "image_pad", 3) is not None
    # 2. 4-pole conditional corpus captured all four poles!
    for pole in ("read_neg", "read_pos", "write_neg", "write_pos"):
        assert bank_think.get_span("cond_hedge_to_concrete", pole, "think", 3) is not None
    # 3. Truncated <think> traces were NOT mislabeled into "answer"!
    assert bank_think.get_span("cond_hedge_to_concrete", "read_pos", "answer", 3) is None
    # 4. Non-sink multi-token subsampling produced >1 row per prompt for calibration!
    cal_think = bank_think.get_span("jlens_calibration", "clean", "think", 3)
    assert cal_think is not None and cal_think.shape[0] > 12

    # With enable_thinking=False, decoded tokens cleanly populate "answer" with >1 row per prompt,
    # and progress_cb emits monotonic done/total telemetry for every generated pole.
    events: list[dict[str, object]] = []
    bank_ans = rt.capture_span_bank(
        tmp_path / "bank_ans",
        corpus_ids=["cond_hedge_to_concrete"],
        layers=(1, 3),
        max_tokens=6,
        max_tokens_per_span=4,
        enable_thinking=False,
        progress_cb=events.append,
    )
    ans_rows = bank_ans.get_span("cond_hedge_to_concrete", "write_pos", "answer", 3)
    assert ans_rows is not None and ans_rows.shape[0] > 8
    assert len(events) == 32  # 8 pairs x 4 poles
    assert [int(e["done"]) for e in events] == list(range(1, 33))
    assert all(int(e["total"]) == 32 and int(e["tokens"]) > 0 for e in events)
    assert events[-1]["eta_s"] == 0.0
    assert bank_ans.paired_layer_states(1, 3) is not None


def test_mlx_cpu_capture_filters_special_ids_and_bundled_mtp_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    rt = _make_runtime_with_vectors()
    # Polish #1: Mark token 6 as a tokenizer special ID (e.g. <|im_start|>); verify prompt_text
    # subsampling excludes both index 0 and special token 6 while keeping image_pad!
    rt.processor.all_special_ids = [THINK_START, THINK_END, IMAGE_PAD, EOS_ID, 6]
    # Encode returns [5, IMAGE_PAD, IMAGE_PAD, 6] for "[IMG]" prompts:
    # index 0 is sink (5), indices 1,2 are IMAGE_PAD (kept in image_pad!), index 3 is special (6, dropped!).
    bank = rt.capture_span_bank(
        tmp_path / "bank_special",
        corpus_ids=["vision_optical_vs_mood"],
        layers=(3,),
        max_tokens=4,
        max_tokens_per_span=4,
        enable_thinking=False,
    )
    assert bank.get_span("vision_optical_vs_mood", "a", "image_pad", 3) is not None
    # Every prompt_text position in "[IMG]" was either index 0 (5) or special (6), so filtered out!
    assert bank.get_span("vision_optical_vs_mood", "a", "prompt_text", 3) is None

    # Polish #3: _try_load_mtp_drafter tries sibling `*-mtp` first, then falls back to bundled `model_dir`.
    model_dir = tmp_path / "MyBundledModel"
    model_dir.mkdir()
    attempted: list[str] = []

    class _DummyDrafter:
        def __init__(self, src: str) -> None:
            self.src = src
            self.bound = False

        def bind(self, model: Any) -> None:
            self.bound = True

    fake_drafters = types.ModuleType("mlx_vlm.speculative.drafters")

    def _fake_load_drafter(path_str: str):
        attempted.append(path_str)
        return _DummyDrafter(path_str), "mtp"

    fake_drafters.load_drafter = _fake_load_drafter
    fake_drafters.validate_drafter_compatibility = lambda m, d, k: None
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", types.ModuleType("mlx_vlm.speculative"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", fake_drafters)

    # Case A: No sibling `*-mtp` directory -> falls back cleanly to bundled `model_dir`!
    d_bundled = rt._try_load_mtp_drafter(model_dir)
    assert isinstance(d_bundled, _DummyDrafter)
    assert d_bundled.src == str(model_dir) and d_bundled.bound is True

    # Case B: Sibling `*-mtp` exists -> prefers sibling first!
    sib_dir = tmp_path / "MyBundledModel-mtp"
    sib_dir.mkdir()
    attempted.clear()
    d_sib = rt._try_load_mtp_drafter(model_dir)
    assert isinstance(d_sib, _DummyDrafter)
    assert d_sib.src == str(sib_dir) and attempted == [str(sib_dir)]


def test_mlx_cpu_layer_scale_budget_and_posthoc_mtp_share_at_temp_gt_0():
    import mlx.core as mx

    rt = _make_runtime_with_vectors(supports_verify_hook=True)
    # 1. Layer scale budget prevents 3x/4x compounding across multi-layer FA concepts!
    c_div = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.2}, layer_scale_mode="divide_k"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    c_sqrt = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.2}, layer_scale_mode="divide_sqrt_k"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    c_raw = LayerSteeringController(
        SteeringPlan(scales={"style_mid": 1.2}, layer_scale_mode="raw"),
        rt.vectors,
        num_layers=4,
        hidden_size=HIDDEN_DIM,
        layer_is_linear=rt._layer_is_linear_flags(),
    )
    n_div = float(np.linalg.norm(c_div.compiled[3].regime_a_groups[0][2].mu_push))
    n_sqrt = float(np.linalg.norm(c_sqrt.compiled[3].regime_a_groups[0][2].mu_push))
    n_raw = float(np.linalg.norm(c_raw.compiled[3].regime_a_groups[0][2].mu_push))
    assert abs(n_div - 0.4) < 1e-4  # 1.2 / 3
    assert abs(n_sqrt - (1.2 / np.sqrt(3.0))) < 1e-4
    assert abs(n_raw - 1.2) < 1e-4

    # 2. Post-hoc teacher-forced MTP agreement works EVEN AT temp > 0 with zero RNG drift!
    class _PosthocDraft:
        def __init__(self) -> None:
            self._input_embed = rt.lm.model.embed_tokens
            self._input_embed_scale = 1.0
            self.calls = 0

        def _forward_hidden(self, token_embed, hidden, cache, position_ids):
            self.calls += 1
            assert cache is None
            assert position_ids.shape[1] == token_embed.shape[1]
            return 0.7 * hidden + 0.3 * token_embed

    rt.draft_model = _PosthocDraft()
    arms = [
        SteeringPlan(label="base"),
        SteeringPlan(label="steered", scales={"style_mid": 1.2}, phase_override="all"),
    ]
    traces = rt.sweep(
        "hello",
        arms,
        max_tokens=6,
        temperature=0.7,
        seed=42,
        enable_thinking=False,
        use_mtp=True,
        simulated_draft_fn=lambda seq: [8, 9],
    )
    # Live verify stayed OFF (PRNG safe at temp > 0), yet post-hoc MTP agreement populated mtp_share!
    assert rt.lm.verify_calls == 0
    assert rt.draft_model.calls == 2
    for tr in traces:
        assert tr.mtp_share is not None and 0.0 <= tr.mtp_share <= 1.0
        assert tr.tokens[0].mtp_accepted is None
        assert all(isinstance(t.mtp_accepted, bool) for t in tr.tokens[1:])


def test_steering_lab_heylook_access_gate_and_login_flow(tmp_path: Path, monkeypatch):
    """Steering Lab shares heylook's AccessGateMiddleware, /login cookie, and CSRF/Host guards."""
    from fastapi.testclient import TestClient

    from steering_lab.cli import main as cli_main
    from steering_lab.server import LabState, create_app

    bank_dir = tmp_path / "gate_bank"
    assert cli_main(["demo-bank", "--out", str(bank_dir), "--dim", "32"]) == 0
    monkeypatch.setenv("HEYLOOK_API_KEY", "test-steering-secret-key-1234")

    client = TestClient(
        create_app(LabState(bank_dir=bank_dir), enable_access_gate=True),
        base_url="http://127.0.0.1:8111",
    )

    # Exempt /health and /login render without credentials.
    assert client.get("/health").status_code == 200
    r_login = client.get("/login")
    assert r_login.status_code == 200 and "login-form" in r_login.text
    assert client.get("/css/app.css").status_code == 200

    # Protected API routes return 401 without credentials, 200 with x-api-key.
    assert client.get("/api/status").status_code == 401
    r_ok = client.get("/api/status", headers={"x-api-key": "test-steering-secret-key-1234"})
    assert r_ok.status_code == 200 and r_ok.json()["access_gate_enabled"] is True

    # Cross-origin POST is blocked with 403 even with a valid key.
    r_csrf = client.post(
        "/api/model/unload",
        headers={
            "x-api-key": "test-steering-secret-key-1234",
            "Origin": "https://evil.example",
        },
    )
    assert r_csrf.status_code == 403

    # Browser /login POST sets heylook_session cookie granting UI + API access.
    r_post = client.post(
        "/login",
        data={"api_key": "test-steering-secret-key-1234"},
        headers={"Origin": "http://127.0.0.1:8111"},
        follow_redirects=False,
    )
    assert r_post.status_code == 303
    assert "heylook_session" in r_post.cookies
    r_cookie_status = client.get("/api/status")
    assert r_cookie_status.status_code == 200

    # Rebound Host header is rejected by HostCheckMiddleware.
    r_rebind = client.get("/health", headers={"Host": "evil.example"})
    assert r_rebind.status_code == 403
