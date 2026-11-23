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
    assert float(np.linalg.norm(out[0, 2])) > 0.5
    assert float(np.linalg.norm(out[0, 3])) > 0.5


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
    client = TestClient(create_app(state))

    r_html = client.get("/")
    assert r_html.status_code == 200 and "MLX Steering Lab" in r_html.text

    r_status = client.get("/api/status").json()
    assert r_status["model_loaded"] is False
    assert r_status["bank_loaded"] is True
    assert r_status["layer_kinds"]["47"] == "FA"
    assert r_status["layer_kinds"]["49"] == "GDN"

    r_vecs = client.get("/api/vectors").json()
    assert len(r_vecs["vectors"]) >= 5

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
    assert r_cl["used_corpus_prior"] is True
    assert len(r_cl["chains"]) == 4
    assert r_cl["chains"][0]["round_trip_cos"] is not None

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
