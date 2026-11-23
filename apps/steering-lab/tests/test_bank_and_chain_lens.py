"""CPU unit tests for SpanBank I/O, model fingerprinting, J-Lens transport, and The Chain Lens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from steering_lab.bank import (
    FingerprintMismatchError,
    ModelFingerprint,
    SpanBank,
    build_packs_from_bank,
    create_demo_bank,
    load_hold_pack,
    load_vector_spec,
)
from steering_lab.chain_lens import (
    LinearDemoMTPAdapter,
    build_probe_pair,
    fit_jlens_transport,
    run_chain_lens,
)


def test_demo_bank_save_load_and_pack_build_roundtrip(tmp_path: Path):
    bank_dir = tmp_path / "demo_bank"
    vec_dir = tmp_path / "vectors"

    bank = create_demo_bank(bank_dir, dim=64, seed=42)
    reloaded = SpanBank.load(bank_dir)
    assert reloaded.fingerprint == bank.fingerprint
    assert reloaded.layers == (15, 27, 39, 43, 47, 49, 51, 55, 62, 63)
    # Catch #1: Qwen 3.8 3:1 hybrid FA sits strictly at idx % 4 == 3; L49 & L62 are GDN!
    assert reloaded.layer_kind(47) == "FA"
    assert reloaded.layer_kind(51) == "FA"
    assert reloaded.layer_kind(49) == "GDN"
    assert reloaded.layer_kind(62) == "GDN"
    assert reloaded.layer_kind(63) == "FA"

    res = build_packs_from_bank(reloaded, vec_dir, rank=8, hold_rank=8)
    assert "clinical_vs_poetic" in res["vectors"]
    assert "anti_slop" in res["vectors"]
    assert "cond_hedge_to_concrete" in res["vectors"]

    spec, fp = load_vector_spec(vec_dir / "clinical_vs_poetic", expected_fingerprint=reloaded.fingerprint)
    assert fp == reloaded.fingerprint
    assert spec.layers == (39, 43, 47, 51)
    assert spec.by_layer[47].mu_clean is not None

    holds = load_hold_pack(vec_dir / "_holds", expected_fingerprint=reloaded.fingerprint)
    assert 47 in holds and 63 in holds


def test_fingerprint_mismatch_raises(tmp_path: Path):
    bank = create_demo_bank(tmp_path / "bank", dim=64)
    vec_dir = tmp_path / "vecs"
    build_packs_from_bank(bank, vec_dir)

    wrong_fp = ModelFingerprint(
        model_id="other-model",
        hidden_size=128,
        num_hidden_layers=64,
        config_sha256="different0000000",
    )
    with pytest.raises(FingerprintMismatchError):
        load_vector_spec(vec_dir / "clinical_vs_poetic", expected_fingerprint=wrong_fp)


def test_demo_bank_geometry_shows_mid_vs_terminal_rank_collapse(tmp_path: Path):
    bank = create_demo_bank(tmp_path / "bank", dim=64, seed=42)
    geom = bank.geometry_by_layer(span_kind="answer")
    assert geom[47]["layer_kind"] == "FA"
    assert geom[49]["layer_kind"] == "GDN"
    assert geom[47]["effective_rank"] > geom[63]["effective_rank"]
    assert geom[47]["p90_abs_cos"] < geom[63]["p90_abs_cos"]


def test_jlens_transport_recovers_layer_rotation():
    rng = np.random.default_rng(101)
    d = 32
    # Construct a known random orthogonal rotation R from L47 -> L63.
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    offset = rng.normal(0.0, 0.5, size=(d,))
    h_src = rng.normal(0.0, 1.0, size=(80, d)).astype(np.float32)
    h_dst = (h_src @ q + offset[None, :]).astype(np.float32)

    jlens = fit_jlens_transport(h_src, h_dst, src_layer=47, dst_layer=63, ridge=1e-4)
    assert jlens.train_r2 > 0.99

    v_test = rng.normal(0.0, 1.0, size=(d,)).astype(np.float32)
    v_rot_true = (v_test @ q).astype(np.float32)
    v_rot_pred = jlens.transport_vector(v_test)
    cos = float(
        np.dot(v_rot_pred, v_rot_true)
        / (np.linalg.norm(v_rot_pred) * np.linalg.norm(v_rot_true))
    )
    assert cos > 0.999


def test_chain_lens_contrastive_sign_flip_and_round_trip(tmp_path: Path):
    bank = create_demo_bank(tmp_path / "bank", dim=64, seed=42)
    vec_dir = tmp_path / "vecs"
    build_packs_from_bank(bank, vec_dir)
    spec, _ = load_vector_spec(vec_dir / "clinical_vs_poetic")
    lv = spec.by_layer[47]
    assert bank.xray_lm_head is not None and bank.xray_vocab is not None

    paired = bank.paired_layer_states(47, 63)
    assert paired is not None
    jlens = fit_jlens_transport(paired[0], paired[1], src_layer=47, dst_layer=63)
    h_prior = lv.mu_clean
    lm_head = bank.xray_lm_head
    adapter = LinearDemoMTPAdapter(lm_head)
    decode = lambda tid: bank.xray_vocab[tid % len(bank.xray_vocab)]
    embed_fn = lambda tids: lm_head[np.asarray(list(tids), dtype=np.int64) % lm_head.shape[0]].mean(axis=0)

    ht_pos, hb_pos, _ = build_probe_pair(lv.mu_d, h_prior=h_prior, probe_scale=3.0, jlens=jlens)
    ht_neg, hb_neg, _ = build_probe_pair(-lv.mu_d, h_prior=h_prior, probe_scale=3.0, jlens=jlens)

    out_pos = run_chain_lens(
        ht_pos,
        adapter=adapter,
        decode_token=decode,
        h_baseline=hb_pos,
        steps=4,
        beam_width=5,
        alpha=0.75,
        embed_tokens_fn=embed_fn,
    )
    out_neg = run_chain_lens(
        ht_neg,
        adapter=adapter,
        decode_token=decode,
        h_baseline=hb_neg,
        steps=4,
        beam_width=5,
        alpha=0.75,
        embed_tokens_fn=embed_fn,
    )

    assert len(out_pos["chains"]) == 5
    assert all(len(c["token_ids"]) == 4 for c in out_pos["chains"])
    assert out_pos["chains"][0]["text"] != out_neg["chains"][0]["text"]
    # Positive pole ("clinical") surfaces technical vocab indices 0..5 on step 0;
    # Negative pole ("poetic") surfaces poetic vocab indices 6..11 on step 0.
    assert out_pos["chains"][0]["token_ids"][0] in set(range(0, 6))
    assert out_neg["chains"][0]["token_ids"][0] in set(range(6, 12))
    assert out_pos["chains"][0]["round_trip_cos"] is not None
    assert out_pos["chains"][0]["round_trip_cos"] > 0.1
