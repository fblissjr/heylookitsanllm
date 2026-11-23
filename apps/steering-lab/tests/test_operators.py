"""Pure-NumPy CPU unit tests for non-1D steering operators and geometry diagnostics."""

from __future__ import annotations

import numpy as np

from steering_lab.operators import (
    RunningCenter,
    apply_conditional_np,
    apply_regime_a_np,
    apply_regime_b_np,
    bank_geometry_report,
    build_conditional_layer,
    build_hold_basis,
    build_regime_a_layer,
    build_regime_b_layer,
    build_s3_random_layer,
    compose_regime_a,
    effective_rank,
)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def _synth_pair(rng: np.random.Generator, d: int, dir_a: np.ndarray, dir_b: np.ndarray, n: int = 32):
    base = rng.normal(0.0, 0.25, size=(n, d))
    a = (base + 2.5 * dir_a[None, :] + rng.normal(0.0, 0.1, size=(n, d))).astype(np.float32)
    b = (base + 2.5 * dir_b[None, :] + rng.normal(0.0, 0.1, size=(n, d))).astype(np.float32)
    return a, b


def test_s0_exact_identity_all_operators():
    rng = np.random.default_rng(7)
    d = 32
    a, b = _synth_pair(rng, d, _unit(np.eye(d)[0]), _unit(np.eye(d)[1]))
    lv_a = build_regime_a_layer(47, a, b, rank=4)
    lv_b = build_regime_b_layer(63, a, b, rank=4)
    lv_c = build_conditional_layer(47, a, b, b, a)
    h = rng.normal(0.0, 1.0, size=(1, 6, d)).astype(np.float32)
    h_c = np.zeros_like(h)

    assert compose_regime_a([(lv_a, 0.0, 1.0)]) is None

    out_b, tel_b = apply_regime_b_np(h, lv_b, scale=0.0, h_center=h_c)
    assert np.array_equal(out_b, h)
    assert tel_b.g_t == 0.0 and tel_b.rel_delta == 0.0

    out_c, tel_c = apply_conditional_np(h, lv_c, scale=0.0, h_center=h_c)
    assert np.array_equal(out_c, h)
    assert tel_c.g_t == 0.0 and tel_c.rel_delta == 0.0


def test_regime_a_peer_ros_prevents_peer_cancellation():
    rng = np.random.default_rng(11)
    d = 32
    e0, e1, e2, e3 = np.eye(d, dtype=np.float32)[:4]
    # Slider 2 targets (e1 - e3); Slider 1's Pole A variance directly spans Slider 2's target pole!
    a2, b2 = _synth_pair(rng, d, e3, e1)
    lv2 = build_regime_a_layer(47, a2, b2, rank=4, use_paired_diff=False)
    amps = rng.uniform(0.5, 4.5, size=(32, 1))
    a1 = (rng.normal(0.0, 0.15, size=(32, d)) + amps * lv2.mu_d[None, :] + 1.2 * e0[None, :]).astype(np.float32)
    b1 = (rng.normal(0.0, 0.15, size=(32, d)) + 2.5 * e2[None, :]).astype(np.float32)
    lv1 = build_regime_a_layer(47, a1, b1, rank=4, use_paired_diff=False)

    # Isolate Slider 1's erasure action against Slider 2's target pole (lv2.erase_ratio=0.0):
    comp_ros_iso = compose_regime_a([(lv1, 1.0, 1.0), (lv2, 1.0, 0.0)], use_peer_ros=True)
    comp_raw_iso = compose_regime_a([(lv1, 1.0, 1.0), (lv2, 1.0, 0.0)], use_peer_ros=False)
    assert comp_ros_iso is not None and comp_raw_iso is not None

    def _erase_energy_on(comp, vec):
        proj = comp.u_erase @ vec.astype(np.float32)
        erased = (proj * comp.eig_erase) @ comp.u_erase
        return float(np.linalg.norm(erased))

    # With peer_ros, Slider 1's erasure basis is strictly orthogonal to Slider 2's target pole!
    assert _erase_energy_on(comp_ros_iso, lv2.mu_d) < 1e-5
    assert _erase_energy_on(comp_raw_iso, lv2.mu_d) > 0.8

    # With both sliders erasing, peer_ros strictly reduces cross-erasure on Slider 2's target.
    comp_ros_both = compose_regime_a([(lv1, 1.0, 1.0), (lv2, 1.0, 1.0)], use_peer_ros=True)
    comp_raw_both = compose_regime_a([(lv1, 1.0, 1.0), (lv2, 1.0, 1.0)], use_peer_ros=False)
    assert comp_ros_both is not None and comp_raw_both is not None
    assert _erase_energy_on(comp_ros_both, lv2.mu_d) < _erase_energy_on(comp_raw_both, lv2.mu_d) - 0.3


def test_regime_a_clamped_union_prevents_overshoot():
    rng = np.random.default_rng(13)
    d = 32
    e0, e1 = np.eye(d, dtype=np.float32)[:2]
    a, b = _synth_pair(rng, d, e0, e1)
    lv = build_regime_a_layer(47, a, b, rank=4)

    clamped = compose_regime_a([(lv, 1.0, 1.0), (lv, 1.0, 1.0)], use_clamped_union=True)
    unclamped = compose_regime_a([(lv, 1.0, 1.0), (lv, 1.0, 1.0)], use_clamped_union=False)
    assert clamped is not None and unclamped is not None
    assert clamped.max_raw_eig > 1.5
    assert clamped.clamped_eigs is True
    assert float(np.max(clamped.eig_erase)) <= 1.0 + 1e-6
    assert unclamped.clamped_eigs is False
    assert float(np.max(unclamped.eig_erase)) > 1.5


def test_regime_a_capability_holds_and_tanh_knee():
    rng = np.random.default_rng(19)
    d = 32
    hold_dir = _unit(np.eye(d)[10] + np.eye(d)[11])
    clean_hold = (rng.normal(0.0, 0.01, size=(24, d)) + rng.normal(0.0, 3.0, size=(24, 1)) * hold_dir[None, :]).astype(
        np.float32
    )
    hold_basis = build_hold_basis(clean_hold, rank=1)

    # Concept that intentionally overlaps the hold subspace
    a, b = _synth_pair(rng, d, _unit(np.eye(d)[0] + 0.7 * hold_dir), _unit(np.eye(d)[1] + 0.7 * hold_dir))
    lv = build_regime_a_layer(47, a, b, rank=4)
    comp = compose_regime_a([(lv, 1.5, 1.0)], hold_basis=hold_basis, knee=0.4)
    assert comp is not None and comp.holds_applied is True

    h = rng.normal(0.0, 1.0, size=(1, 5, d)).astype(np.float32)
    out, _ = apply_regime_a_np(h, comp)
    delta = out - h
    # Zero projection onto the protected hold subspace
    hold_leak = np.max(np.abs(delta @ hold_basis.T))
    assert float(hold_leak) < 1e-5
    # Smooth tanh knee caps every token's perturbation norm <= 0.4
    norms = np.linalg.norm(delta, axis=-1)
    assert float(np.max(norms)) <= 0.4 + 1e-5


def test_regime_b_deadzone_ramp_and_boost_zero_cap():
    rng = np.random.default_rng(23)
    d = 32
    slop_dir = _unit(np.eye(d)[0] + np.eye(d)[1])
    clean_dir = _unit(np.eye(d)[2] + np.eye(d)[3])
    acts_slop, acts_clean = _synth_pair(rng, d, slop_dir, clean_dir, n=32)
    lv = build_regime_b_layer(63, acts_slop, acts_clean, rank=4)
    assert lv.mu_clean is not None

    # Clean token aligned with +mu_d (opposite of -mu_d): ray_cos < 0 < m0 => exact 0.0 delta!
    h_clean = (lv.mu_clean + 1.5 * clean_dir).reshape(1, 1, d).astype(np.float32)
    out_clean, tel_clean = apply_regime_b_np(h_clean, lv, scale=1.0, h_center=lv.mu_clean.reshape(1, 1, d))
    assert tel_clean.g_t == 0.0
    assert np.allclose(out_clean, h_clean, atol=1e-6)

    # Slop token aligned with -mu_d: gate fires and reduces energy along vr (boost=0, never increases)
    h_slop = (lv.mu_clean + 2.0 * slop_dir).reshape(1, 1, d).astype(np.float32)
    out_slop, tel_slop = apply_regime_b_np(
        h_slop, lv, scale=1.0, h_center=lv.mu_clean.reshape(1, 1, d), m0=0.08, w=0.22, e_cap=0.35
    )
    assert tel_slop.g_t > 0.5
    e_before = float(np.linalg.norm((h_slop - lv.mu_clean) @ lv.vr.T))
    e_after = float(np.linalg.norm((out_slop - lv.mu_clean) @ lv.vr.T))
    assert e_after < e_before


def test_running_center_prior_ema_phase_reset_and_rollback_safety():
    d = 16
    mu_clean = np.zeros((d,), dtype=np.float32)
    tok1 = np.ones((1, 1, d), dtype=np.float32) * 3.0

    # Raw running mode collapses to h_1 - h_bar_1 == 0 on token 1 (H3 Gotcha #1).
    rc_raw = RunningCenter(d, mode="running")
    m_raw = rc_raw.preview_means_np(tok1)
    assert np.allclose(tok1 - m_raw, 0.0, atol=1e-6)

    # prior_ema anchored on mu_clean is NON-ZERO on token 1!
    rc_ema = RunningCenter(d, mode="prior_ema", mu_clean=mu_clean, prior_weight=8.0)
    m_ema = rc_ema.preview_means_np(tok1)
    assert float(np.linalg.norm(tok1 - m_ema)) > 1.0

    # Phase reset clears <think> tokens back to prior_weight so they never bleed into content.
    rc_ema.commit_np(np.ones((1, 4, d), dtype=np.float32) * 10.0)
    assert rc_ema.count == 12.0  # prior_weight (8) + 4 committed tokens
    rc_ema.reset_phase()
    assert rc_ema.count == 8.0
    assert np.allclose(rc_ema.preview_means_np(tok1), m_ema, atol=1e-6)

    # Speculative preview + partial commit NEVER leaks rejected draft tokens.
    draft_batch = np.stack([np.ones((d,)), np.ones((d,)) * 100.0, np.ones((d,)) * 200.0], axis=0)[None, :, :].astype(
        np.float32
    )
    _ = rc_ema.preview_means_np(draft_batch)
    assert rc_ema.count == 8.0  # preview is purely functional
    rc_ema.commit_np(draft_batch[:, :1, :])  # accept ONLY token 0
    assert rc_ema.count == 9.0
    assert np.allclose(rc_ema.sum_h, np.ones((d,), dtype=np.float32), atol=1e-6)


def test_conditional_relu_decoupled_read_write():
    rng = np.random.default_rng(29)
    d = 32
    e_read = _unit(np.eye(d)[4])
    e_write = _unit(np.eye(d)[12])
    r_neg, r_pos = _synth_pair(rng, d, np.zeros(d, dtype=np.float32), e_read)
    w_neg, w_pos = _synth_pair(rng, d, np.zeros(d, dtype=np.float32), e_write)
    lv = build_conditional_layer(47, r_pos, r_neg, w_pos, w_neg)

    # Unrelated token (orthogonal to w_read): exact 0.0
    h_off = (np.eye(d)[20] * 2.0).reshape(1, 1, d).astype(np.float32)
    out_off, tel_off = apply_conditional_np(h_off, lv, scale=1.0, h_center=np.zeros((1, 1, d), dtype=np.float32))
    assert tel_off.g_t == 0.0 and np.array_equal(out_off, h_off)

    # Trigger token aligned with w_read: pushes along w_write (NOT w_read!)
    h_on = (e_read * 2.0).reshape(1, 1, d).astype(np.float32)
    out_on, tel_on = apply_conditional_np(h_on, lv, scale=1.0, h_center=np.zeros((1, 1, d), dtype=np.float32))
    assert tel_on.g_t > 0.8
    delta = (out_on - h_on).reshape(-1)
    assert abs(float(np.dot(_unit(delta), lv.w_write))) > 0.99
    assert abs(float(np.dot(_unit(delta), lv.w_read))) < 0.2


def test_s3_random_control_orthogonal_and_deterministic():
    rng = np.random.default_rng(31)
    d = 48
    a, b = _synth_pair(rng, d, _unit(np.eye(d)[0]), _unit(np.eye(d)[1]))
    lv = build_regime_a_layer(47, a, b, rank=6)
    r1 = build_s3_random_layer(lv, seed=999)
    r2 = build_s3_random_layer(lv, seed=999)
    assert np.array_equal(r1.mu_d, r2.mu_d)
    assert np.array_equal(r1.vr, r2.vr)
    assert r1.vr.shape == lv.vr.shape
    assert abs(float(np.dot(r1.mu_d, lv.mu_d))) < 1e-4
    assert float(np.max(np.abs(r1.vr @ lv.vr.T))) < 1e-4


def test_geometry_report_detects_rank_collapse():
    d = 32
    ortho = {f"c{i}": _unit(np.eye(d)[i]) for i in range(5)}
    shared = _unit(np.eye(d)[0])
    funneled = {f"c{i}": _unit(shared + 0.1 * np.eye(d)[i + 1]) for i in range(5)}
    g_ortho = bank_geometry_report(ortho)
    g_fun = bank_geometry_report(funneled)
    assert g_ortho["effective_rank"] > 4.5
    assert g_fun["effective_rank"] < 1.5
    assert g_ortho["p90_abs_cos"] < 0.05
    assert g_fun["p90_abs_cos"] > 0.9
    assert effective_rank(np.eye(4)) > 3.9
