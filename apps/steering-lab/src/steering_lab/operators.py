"""Non-1D activation steering operators, FP64 SVD builders, and subspace diagnostics.

Implements the four lab operators over pure NumPy arrays (offline building,
composition, and CPU tests) and MLX arrays (live inference hooks):

1. Regime A (Manifold Replacement, ~60-80% FA band L39, L43, L47, L51):
   Local-centered Pole-A subspace erasure + target push, with multi-slider
   `peer_ros` target orthogonalization, clamped [0, 1] union SVD, optional
   capability `holds` projection, and smooth `tanh` RMS knee.
2. Regime B (Active Noise Cancellation, L63 or mid-layer):
   Local-centered half-wave normalized deadzone ramp (`m0`, `w`) over a
   multi-rank paired-shift basis (`vr_diff`) with `boost=0` and upper energy cap.
3. Conditional `IF X -> THEN Y` Neurons:
   Decoupled read/write axes gated by clipped ReLU (`ReLU(cos - m0) / w`).
4. `S3` Random-Direction Control:
   Matched-rank, matched-norm random basis orthogonal to the true direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

EPS = 1e-8


def _is_mlx_array(x: Any) -> bool:
    return type(x).__module__.startswith("mlx.")


def _unit_np(v: np.ndarray) -> tuple[np.ndarray, float]:
    v64 = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(v64))
    if norm < EPS:
        return np.zeros_like(v64, dtype=np.float32), 0.0
    return (v64 / norm).astype(np.float32), norm


def _orthonormal_rows_fp64(mat: np.ndarray, *, tol: float = 1e-7) -> np.ndarray:
    """Return orthonormal rows spanning ``mat`` via FP64 SVD."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0 or arr.shape[0] == 0:
        return np.zeros((0, arr.shape[-1]), dtype=np.float64)
    _, s, vt = np.linalg.svd(arr, full_matrices=False)
    if s.size == 0 or s[0] < tol:
        return np.zeros((0, arr.shape[-1]), dtype=np.float64)
    keep = s > max(tol, float(s[0]) * 1e-6)
    return vt[keep]


@dataclass(frozen=True)
class LayerVector:
    """Per-layer steering payload stored in float32."""

    layer: int
    mu_d: np.ndarray  # (D,) unit direction (Pole B - Pole A)
    vr: np.ndarray  # (r, D) orthonormal rows
    d_diag: np.ndarray  # (r,) nonnegative weights in [0, 1]
    mu_d_norm: float = 1.0
    mu_clean: Optional[np.ndarray] = None  # (D,) clean Pole-B centroid (avoids T=1 cold-start zero)
    w_read: Optional[np.ndarray] = None  # (D,) unit trigger axis for conditional
    w_write: Optional[np.ndarray] = None  # (D,) unit write axis for conditional
    split_half_cos: Optional[float] = None
    flip_null_cos: Optional[float] = None


@dataclass(frozen=True)
class VectorSpec:
    """Metadata and per-layer payloads for one steering concept."""

    id: str
    label: str
    operator: str  # "regime_a" | "regime_b" | "conditional"
    phase: str = "content"  # "all" | "content" | "think"
    token_mask: str = "decode"  # "all" | "decode" | "prompt_text" | "image_pad"
    layers: tuple[int, ...] = ()
    scale_min: float = -2.0
    scale_max: float = 2.0
    scale_suggested: float = 1.0
    m0: float = 0.08
    w: float = 0.22
    e_cap: float = 0.35
    knee: float = 0.0  # 0 disables smooth tanh knee
    erase_ratio: float = 1.0
    by_layer: Mapping[int, LayerVector] = field(default_factory=dict)
    notes: str = ""

    def public_meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "operator": self.operator,
            "phase": self.phase,
            "token_mask": self.token_mask,
            "layers": list(self.layers),
            "scale": {
                "min": self.scale_min,
                "max": self.scale_max,
                "suggested": self.scale_suggested,
            },
            "params": {
                "m0": self.m0,
                "w": self.w,
                "e_cap": self.e_cap,
                "knee": self.knee,
                "erase_ratio": self.erase_ratio,
            },
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ComposedRegimeA:
    """Compiled multi-slider Regime-A operator at a single layer."""

    u_erase: np.ndarray  # (K, D) orthonormal rows in union subspace
    eig_erase: np.ndarray  # (K,) eigenvalues (clamped to [0, 1] when enabled)
    mu_push: np.ndarray  # (D,) net target push vector
    knee: float = 0.0
    max_raw_eig: float = 0.0
    clamped_eigs: bool = True
    peer_ros_applied: bool = True
    holds_applied: bool = False


@dataclass(frozen=True)
class StepTelemetry:
    """Compact per-step intervention readout for the Token Microscope."""

    g_t: float = 0.0
    ray_cos: float = 0.0
    rel_delta: float = 0.0
    active_layers: int = 0


class RunningCenter:
    """Rollback-safe causal centering with ``mu_clean`` warmup prior and per-phase reset.

    Modes:
    - ``"prior_ema"`` (default): seeds ``(sum_h, count)`` with ``prior_weight * mu_clean``
      so token 1 never collapses to ``h_1 - h_bar_1 = 0`` while smoothly adapting to
      the local prompt/span over tokens 2..N. Call ``reset_phase()`` on ``<think> -> content``.
    - ``"mu_clean"``: fixed ``mu_clean`` centroid (zero cold-start lag, zero drift).
    - ``"running"``: raw running mean with zero prior (kept for lab comparison).

    During speculative verification of T draft tokens, call ``preview_means_np``
    to get functional prefix means without mutating internal state, then call
    ``commit_np`` with ONLY the accepted slice ``h[:, :commit_accepted, :]`` so
    rejected draft tokens never leak into local centering.
    """

    def __init__(
        self,
        dim: int,
        *,
        mode: str = "prior_ema",
        mu_clean: Optional[np.ndarray] = None,
        prior_weight: float = 8.0,
    ) -> None:
        self.dim = int(dim)
        self.mode = str(mode)
        self.mu_clean: Optional[np.ndarray] = (
            None if mu_clean is None else np.asarray(mu_clean, dtype=np.float32).reshape(self.dim)
        )
        self.prior_weight = float(prior_weight)
        self.sum_h: Optional[np.ndarray] = None
        self.count: float = 0.0
        self.reset()

    def reset(self) -> None:
        if self.mode == "prior_ema" and self.mu_clean is not None and self.prior_weight > 0.0:
            self.sum_h = (self.mu_clean * self.prior_weight).astype(np.float32)
            self.count = float(self.prior_weight)
        else:
            self.sum_h = None
            self.count = 0.0

    def reset_phase(self) -> None:
        """Reset centering when transitioning phases (e.g. ``<think>`` -> ``content``)."""
        self.reset()

    def clone_state(self) -> tuple[Optional[np.ndarray], float]:
        return (None if self.sum_h is None else self.sum_h.copy()), float(self.count)

    def restore_state(self, state: tuple[Optional[np.ndarray], float]) -> None:
        s, c = state
        self.sum_h = None if s is None else s.copy()
        self.count = float(c)

    def preview_means_np(self, h_seq: np.ndarray) -> np.ndarray:
        """Compute causal prefix means for ``h_seq`` of shape ``(B, T, D)`` without mutating."""
        arr = np.asarray(h_seq, dtype=np.float32)
        b, t, d = arr.shape
        if self.mode == "mu_clean" and self.mu_clean is not None:
            return np.broadcast_to(self.mu_clean.reshape(1, 1, d), (b, t, d)).copy()
        base_sum = (
            np.zeros((b, 1, d), dtype=np.float32)
            if self.sum_h is None
            else self.sum_h.reshape(1, 1, d).astype(np.float32)
        )
        cumsum = base_sum + np.cumsum(arr, axis=1)
        counts = (self.count + np.arange(1, t + 1, dtype=np.float32)).reshape(1, t, 1)
        return cumsum / np.maximum(counts, 1.0)

    def commit_np(self, h_accepted: np.ndarray) -> None:
        """Commit accepted hidden states ``(B, T_acc, D)`` (uses batch row 0)."""
        if self.mode == "mu_clean":
            return
        arr = np.asarray(h_accepted, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[0] == 0:
            return
        added = arr.sum(axis=0, dtype=np.float64).astype(np.float32)
        if self.sum_h is None:
            self.sum_h = added
        else:
            self.sum_h = self.sum_h + added
        self.count += float(arr.shape[0])


# ---------------------------------------------------------------------------
# Offline FP64 Builders & Scientific Hygiene Diagnostics
# ---------------------------------------------------------------------------


def split_half_cosine(
    acts_a: np.ndarray, acts_b: np.ndarray, *, seed: int = 42
) -> float:
    """Cosine between two independent halves' mean-diff directions."""
    a = np.asarray(acts_a, dtype=np.float64)
    b = np.asarray(acts_b, dtype=np.float64)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    pa = rng.permutation(a.shape[0])
    pb = rng.permutation(b.shape[0])
    ma1 = a[pa[: a.shape[0] // 2]].mean(axis=0)
    ma2 = a[pa[a.shape[0] // 2 :]].mean(axis=0)
    mb1 = b[pb[: b.shape[0] // 2]].mean(axis=0)
    mb2 = b[pb[b.shape[0] // 2 :]].mean(axis=0)
    d1, n1 = _unit_np(mb1 - ma1)
    d2, n2 = _unit_np(mb2 - ma2)
    if n1 < EPS or n2 < EPS:
        return 0.0
    return float(np.dot(d1.astype(np.float64), d2.astype(np.float64)))


def flip_null_cosine(
    acts_a: np.ndarray, acts_b: np.ndarray, *, seed: int = 42, trials: int = 8
) -> float:
    """Mean absolute split-half cosine under random pole sign-flips (null baseline)."""
    a = np.asarray(acts_a, dtype=np.float64)
    b = np.asarray(acts_b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    if n < 4:
        return 0.0
    diffs = b[:n] - a[:n]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    half = n // 2
    for _ in range(max(1, trials)):
        signs = rng.choice([-1.0, 1.0], size=(n, 1))
        flipped = diffs * signs
        d1, n1 = _unit_np(flipped[:half].mean(axis=0))
        d2, n2 = _unit_np(flipped[half:].mean(axis=0))
        if n1 > EPS and n2 > EPS:
            vals.append(abs(float(np.dot(d1.astype(np.float64), d2.astype(np.float64)))))
    return float(np.mean(vals)) if vals else 0.0


def _paired_token_diffs_fp64(a: np.ndarray, b: np.ndarray, *, tau: float = 0.1) -> np.ndarray:
    """Compute paired shift vectors ``B_matched - A`` in FP64.

    Uses exact elementwise ``b - a`` when row counts match, and row-softmax cosine
    barycenter matching (``softmax(a_hat @ b_hat^T / tau) @ b - a``) when Pole-A and
    Pole-B token spans have different lengths.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((0, a.shape[1]), dtype=np.float64)
    if a.shape[0] == b.shape[0]:
        return b - a
    a_hat = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), EPS)
    b_hat = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), EPS)
    logits = (a_hat @ b_hat.T) / max(float(tau), 1e-3)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), EPS)
    return (weights @ b) - a


def build_regime_a_layer(
    layer: int,
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    *,
    rank: int = 8,
    use_paired_diff: bool = True,
    seed: int = 42,
) -> LayerVector:
    """Build a Regime-A manifold replacement layer payload via FP64 SVD.

    ``acts_a`` holds Pole-A (source/unwanted) rows ``(Na, D)`` and ``acts_b``
    holds Pole-B (target) rows ``(Nb, D)``.
    """
    a = np.asarray(acts_a, dtype=np.float64)
    b = np.asarray(acts_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"Expected 2D arrays with equal hidden dim, got {a.shape} and {b.shape}")

    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    mu_d, mu_norm = _unit_np(mu_b - mu_a)

    if use_paired_diff and a.shape[0] > 0 and b.shape[0] > 0:
        raw_diffs = _paired_token_diffs_fp64(a, b)
        source = raw_diffs - raw_diffs.mean(axis=0, keepdims=True)
        # Include the mean-diff axis explicitly as row 0 if nonzero, then residual PCs.
        basis_rows: list[np.ndarray] = []
        weights: list[float] = []
        if mu_norm > EPS:
            basis_rows.append(mu_d.astype(np.float64))
            weights.append(1.0)
        if rank > len(basis_rows) and source.shape[0] > 1:
            # Project residual diffs orthogonal to mu_d before SVD.
            if mu_norm > EPS:
                proj = (source @ mu_d.astype(np.float64))[:, None] * mu_d.astype(np.float64)[None, :]
                source = source - proj
            _, s, vt = np.linalg.svd(source, full_matrices=False)
            k = min(rank - len(basis_rows), s.shape[0])
            if k > 0 and s[0] > EPS:
                e_sq = (s[:k] ** 2) / max(float(s[0] ** 2), EPS)
                for idx in range(k):
                    if s[idx] > EPS:
                        basis_rows.append(vt[idx])
                        weights.append(float(np.clip(e_sq[idx], 0.05, 1.0)))
        vr = (
            np.stack(basis_rows, axis=0).astype(np.float32)
            if basis_rows
            else np.zeros((0, a.shape[1]), dtype=np.float32)
        )
        d_diag = np.asarray(weights, dtype=np.float32)
    else:
        # Unsupervised SVD(A_c) mode kept explicitly for lab comparison.
        a_c = a - mu_a[None, :]
        _, s, vt = np.linalg.svd(a_c, full_matrices=False)
        k = min(rank, s.shape[0])
        if k > 0 and s[0] > EPS:
            vr = vt[:k].astype(np.float32)
            d_diag = np.clip((s[:k] ** 2) / max(float(s[0] ** 2), EPS), 0.05, 1.0).astype(
                np.float32
            )
        else:
            vr = np.zeros((0, a.shape[1]), dtype=np.float32)
            d_diag = np.zeros((0,), dtype=np.float32)

    return LayerVector(
        layer=int(layer),
        mu_d=mu_d,
        vr=vr,
        d_diag=d_diag,
        mu_d_norm=mu_norm,
        mu_clean=mu_b.astype(np.float32),
        split_half_cos=split_half_cosine(a, b, seed=seed),
        flip_null_cos=flip_null_cosine(a, b, seed=seed),
    )


def build_regime_b_layer(
    layer: int,
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    *,
    rank: int = 8,
    seed: int = 42,
) -> LayerVector:
    """Build a Regime-B active noise cancellation layer payload (`vr_diff`).

    ``acts_a`` is the unwanted failure-mode span (e.g. slop/hesitation);
    ``acts_b`` is the clean contrast span. Uses uncentered paired-shift SVD
    so Pole-A erasure never drags in neutral co-occurring features.
    """
    a = np.asarray(acts_a, dtype=np.float64)
    b = np.asarray(acts_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"Expected 2D arrays with equal hidden dim, got {a.shape} and {b.shape}")

    mu_b = b.mean(axis=0)
    mu_d, mu_norm = _unit_np(mu_b - a.mean(axis=0))
    diffs = _paired_token_diffs_fp64(a, b)

    if diffs.shape[0] > 0:
        _, s, vt = np.linalg.svd(diffs, full_matrices=False)
        k = min(rank, s.shape[0])
        if k > 0 and s[0] > EPS:
            vr = vt[:k].astype(np.float32)
            d_diag = np.clip((s[:k] ** 2) / max(float(s[0] ** 2), EPS), 0.1, 1.0).astype(
                np.float32
            )
        else:
            vr = mu_d[None, :] if mu_norm > EPS else np.zeros((0, a.shape[1]), dtype=np.float32)
            d_diag = np.ones((vr.shape[0],), dtype=np.float32)
    else:
        vr = np.zeros((0, a.shape[1]), dtype=np.float32)
        d_diag = np.zeros((0,), dtype=np.float32)

    return LayerVector(
        layer=int(layer),
        mu_d=mu_d,
        vr=vr,
        d_diag=d_diag,
        mu_d_norm=mu_norm,
        mu_clean=mu_b.astype(np.float32),
        split_half_cos=split_half_cosine(a, b, seed=seed),
        flip_null_cos=flip_null_cosine(a, b, seed=seed),
    )


def build_conditional_layer(
    layer: int,
    read_pos: np.ndarray,
    read_neg: np.ndarray,
    write_pos: np.ndarray,
    write_neg: np.ndarray,
    *,
    seed: int = 42,
) -> LayerVector:
    """Build a decoupled ``IF X -> THEN Y`` ReLU conditional neuron layer payload."""
    rp = np.asarray(read_pos, dtype=np.float64)
    rn = np.asarray(read_neg, dtype=np.float64)
    wp = np.asarray(write_pos, dtype=np.float64)
    wn = np.asarray(write_neg, dtype=np.float64)
    mu_rn = rn.mean(axis=0)
    w_read, _ = _unit_np(rp.mean(axis=0) - mu_rn)
    w_write, write_norm = _unit_np(wp.mean(axis=0) - wn.mean(axis=0))
    return LayerVector(
        layer=int(layer),
        mu_d=w_write,
        vr=w_read[None, :],
        d_diag=np.ones((1,), dtype=np.float32),
        mu_d_norm=write_norm,
        mu_clean=mu_rn.astype(np.float32),
        w_read=w_read,
        w_write=w_write,
        split_half_cos=split_half_cosine(rn, rp, seed=seed),
        flip_null_cos=flip_null_cosine(rn, rp, seed=seed),
    )


def build_hold_basis(clean_acts: np.ndarray, *, rank: int = 16) -> np.ndarray:
    """Top-``rank`` orthonormal capability hold basis ``(r, D)`` from clean activations."""
    arr = np.asarray(clean_acts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D clean activations, got {arr.shape}")
    centered = arr - arr.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(rank, s.shape[0])
    if k == 0 or s[0] < EPS:
        return np.zeros((0, arr.shape[1]), dtype=np.float32)
    return vt[:k].astype(np.float32)


def build_s3_random_layer(vec: LayerVector, *, seed: int = 1337) -> LayerVector:
    """Matched-rank, matched-norm random control strictly orthogonal to ``vec``."""
    rng = np.random.default_rng(seed + vec.layer * 997)
    d = int(vec.mu_d.shape[0])
    forbid_rows = [vec.mu_d.astype(np.float64)]
    if vec.vr.size > 0:
        forbid_rows.append(vec.vr.astype(np.float64))
    if vec.w_read is not None:
        forbid_rows.append(vec.w_read.astype(np.float64))
    if vec.w_write is not None:
        forbid_rows.append(vec.w_write.astype(np.float64))
    forbid = _orthonormal_rows_fp64(np.vstack([r if r.ndim == 2 else r[None, :] for r in forbid_rows]))

    def _proj_orth(mat: np.ndarray) -> np.ndarray:
        if forbid.shape[0] == 0:
            return mat
        return mat - (mat @ forbid.T) @ forbid

    r_mu = _proj_orth(rng.standard_normal((1, d)))[0]
    mu_rand, _ = _unit_np(r_mu)
    rank = int(vec.vr.shape[0])
    if rank > 0:
        raw_vr = _proj_orth(rng.standard_normal((rank, d)))
        vr_rand = _orthonormal_rows_fp64(raw_vr)[:rank].astype(np.float32)
    else:
        vr_rand = np.zeros((0, d), dtype=np.float32)

    w_read = None
    w_write = None
    if vec.w_read is not None:
        w_read, _ = _unit_np(_proj_orth(rng.standard_normal((1, d)))[0])
    if vec.w_write is not None:
        w_write, _ = _unit_np(_proj_orth(rng.standard_normal((1, d)))[0])

    return LayerVector(
        layer=vec.layer,
        mu_d=mu_rand,
        vr=vr_rand,
        d_diag=vec.d_diag[: vr_rand.shape[0]].copy(),
        mu_d_norm=vec.mu_d_norm,
        mu_clean=None if vec.mu_clean is None else vec.mu_clean.copy(),
        w_read=w_read,
        w_write=w_write,
        split_half_cos=0.0,
        flip_null_cos=0.0,
    )


# ---------------------------------------------------------------------------
# Multi-Slider Composition (`peer_ros` + Clamped Union SVD + Capability Holds)
# ---------------------------------------------------------------------------


def compose_regime_a(
    active: Sequence[tuple[LayerVector, float, float]],
    *,
    use_peer_ros: bool = True,
    use_clamped_union: bool = True,
    hold_basis: Optional[np.ndarray] = None,
    knee: float = 0.0,
) -> Optional[ComposedRegimeA]:
    """Compose 1..N active Regime-A sliders ``(layer_vec, scale, erase_ratio)``.

    Exact identity when ``active`` is empty or every ``scale == 0``.
    """
    nonzero = [(lv, float(s), float(er)) for lv, s, er in active if abs(float(s)) > EPS]
    if not nonzero:
        return None

    dim = int(nonzero[0][0].mu_d.shape[0])
    hold64: Optional[np.ndarray] = None
    if hold_basis is not None and hold_basis.size > 0:
        hold64 = _orthonormal_rows_fp64(hold_basis)

    weighted_rows: list[np.ndarray] = []
    mu_push = np.zeros((dim,), dtype=np.float64)

    for idx, (lv, scale, erase_ratio) in enumerate(nonzero):
        sign = 1.0 if scale >= 0.0 else -1.0
        mag = abs(scale)
        # Preserve calibrated contrast magnitude `mu_d_norm` so scale=1.0 moves a full
        # Pole-A -> Pole-B step at any hidden dimension (e.g. D=5120 on Qwen 3.8).
        eff_norm = float(lv.mu_d_norm) if float(lv.mu_d_norm) > EPS else 1.0
        mu_push += (scale * eff_norm) * lv.mu_d.astype(np.float64)

        if lv.vr.shape[0] > 0 and erase_ratio > EPS:
            vr = lv.vr.astype(np.float64)
            # Peer target orthogonalization (`peer_ros`): strip any component along
            # peer sliders' target poles so stacking sliders never cancels peers.
            if use_peer_ros and len(nonzero) > 1:
                peers = np.stack(
                    [nonzero[j][0].mu_d.astype(np.float64) for j in range(len(nonzero)) if j != idx],
                    axis=0,
                )
                peer_orth = _orthonormal_rows_fp64(peers)
                if peer_orth.shape[0] > 0:
                    vr = vr - (vr @ peer_orth.T) @ peer_orth

            # Capability holds: project erasure basis orthogonal to clean holds.
            if hold64 is not None and hold64.shape[0] > 0:
                vr = vr - (vr @ hold64.T) @ hold64

            # Weight each row by sqrt(mag * erase_ratio * d_diag) so sum_k r_k^T r_k = E.
            # When scale < 0, keep erasure magnitude nonnegative (user pushed toward Pole A;
            # only erase when moving toward Pole B unless explicitly inverted).
            if sign > 0.0:
                w = np.clip(lv.d_diag.astype(np.float64), 0.0, 1.0) * (mag * erase_ratio)
                for row_i in range(vr.shape[0]):
                    row_norm = float(np.linalg.norm(vr[row_i]))
                    if row_norm > 1e-6 and w[row_i] > EPS:
                        weighted_rows.append(math.sqrt(float(w[row_i])) * vr[row_i])

    if hold64 is not None and hold64.shape[0] > 0:
        mu_push = mu_push - (mu_push @ hold64.T) @ hold64

    if weighted_rows:
        m = np.stack(weighted_rows, axis=0)  # (R_tot, D)
        # Compact union SVD in O(R_tot^2 * D) instead of D x D eigendecomp.
        _, svals, vt = np.linalg.svd(m, full_matrices=False)
        eigs = svals ** 2
        max_raw_eig = float(eigs[0]) if eigs.size > 0 else 0.0
        keep = eigs > 1e-7
        u_erase = vt[keep].astype(np.float32)
        eig_erase = eigs[keep]
        if use_clamped_union:
            eig_erase = np.clip(eig_erase, 0.0, 1.0)
        eig_erase = eig_erase.astype(np.float32)
    else:
        u_erase = np.zeros((0, dim), dtype=np.float32)
        eig_erase = np.zeros((0,), dtype=np.float32)
        max_raw_eig = 0.0

    return ComposedRegimeA(
        u_erase=u_erase,
        eig_erase=eig_erase,
        mu_push=mu_push.astype(np.float32),
        knee=float(knee),
        max_raw_eig=max_raw_eig,
        clamped_eigs=bool(use_clamped_union),
        peer_ros_applied=bool(use_peer_ros and len(nonzero) > 1),
        holds_applied=bool(hold64 is not None and hold64.shape[0] > 0),
    )


# ---------------------------------------------------------------------------
# Live Application (Dual NumPy / MLX)
# ---------------------------------------------------------------------------


def _apply_tanh_knee_np(delta: np.ndarray, h: np.ndarray, knee: float) -> np.ndarray:
    """Smoothly cap relative perturbation ``||Δh|| / ||h||`` at ``knee`` via ``tanh``."""
    if knee <= 0.0:
        return delta
    h_norm = np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), EPS)
    cap = float(knee) * h_norm
    d_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = (cap * np.tanh(d_norm / np.maximum(cap, EPS))) / np.maximum(d_norm, EPS)
    return delta * scale


def apply_regime_a_np(
    h: np.ndarray,
    composed: Optional[ComposedRegimeA],
    *,
    h_center: Optional[np.ndarray] = None,
    token_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, StepTelemetry]:
    """Apply composed Regime-A operator to ``h`` of shape ``(B, T, D)`` in NumPy."""
    if composed is None:
        return h, StepTelemetry()
    arr = np.asarray(h, dtype=np.float32)
    centered = arr if h_center is None else (arr - np.asarray(h_center, dtype=np.float32))
    delta = np.zeros_like(arr)
    if composed.u_erase.shape[0] > 0:
        # ((h - h_bar) @ U^T) * eig @ U
        coeff = centered @ composed.u_erase.T
        delta -= (coeff * composed.eig_erase[None, None, :]) @ composed.u_erase
    if float(np.linalg.norm(composed.mu_push)) > EPS:
        delta += composed.mu_push[None, None, :]
    delta = _apply_tanh_knee_np(delta, arr, composed.knee)
    if token_mask is not None:
        delta = delta * np.asarray(token_mask, dtype=np.float32)

    h_norm = float(np.mean(np.linalg.norm(arr, axis=-1)))
    d_norm = float(np.mean(np.linalg.norm(delta, axis=-1)))
    rel = d_norm / max(h_norm, EPS)
    out = (arr + delta).astype(h.dtype, copy=False)
    return out, StepTelemetry(g_t=1.0 if d_norm > EPS else 0.0, ray_cos=0.0, rel_delta=rel, active_layers=1)


def apply_regime_b_np(
    h: np.ndarray,
    lv: LayerVector,
    scale: float,
    *,
    h_center: np.ndarray,
    m0: float = 0.08,
    w: float = 0.22,
    e_cap: float = 0.35,
    rank_limit: Optional[int] = None,
    token_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, StepTelemetry]:
    """Apply Regime-B local-centered half-wave deadzone ramp (`boost=0`)."""
    s_eff = float(np.clip(scale, 0.0, 1.0))
    if s_eff <= EPS or lv.vr.shape[0] == 0:
        return h, StepTelemetry()
    arr = np.asarray(h, dtype=np.float32)
    centered = arr - np.asarray(h_center, dtype=np.float32)
    c_norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    pole_a = -lv.mu_d.astype(np.float32)  # unwanted Pole-A ray (-mu_d_hat)
    ray_cos = np.sum(centered * pole_a[None, None, :], axis=-1, keepdims=True) / np.maximum(
        c_norm, EPS
    )
    g_t = np.clip((ray_cos - float(m0)) / max(float(w), EPS), 0.0, 1.0)

    vr = lv.vr if rank_limit is None else lv.vr[: max(1, int(rank_limit))]
    dd = lv.d_diag[: vr.shape[0]]
    proj = ((centered @ vr.T) * dd[None, None, :]) @ vr
    delta = -s_eff * g_t * proj

    # Upper energy cap ONLY (`boost = 0`, never amplify noise on clean inputs).
    if e_cap > 0.0:
        d_norm_tok = np.linalg.norm(delta, axis=-1, keepdims=True)
        cap_val = float(e_cap) * np.maximum(c_norm, EPS)
        cap_scale = np.minimum(1.0, cap_val / np.maximum(d_norm_tok, EPS))
        delta = delta * cap_scale

    if token_mask is not None:
        mask_f = np.asarray(token_mask, dtype=np.float32)
        delta = delta * mask_f
        g_t = g_t * mask_f

    h_norm = float(np.mean(np.linalg.norm(arr, axis=-1)))
    d_norm = float(np.mean(np.linalg.norm(delta, axis=-1)))
    out = (arr + delta).astype(h.dtype, copy=False)
    return out, StepTelemetry(
        g_t=float(np.mean(g_t)),
        ray_cos=float(np.mean(ray_cos)),
        rel_delta=d_norm / max(h_norm, EPS),
        active_layers=1,
    )


def apply_conditional_np(
    h: np.ndarray,
    lv: LayerVector,
    scale: float,
    *,
    h_center: np.ndarray,
    m0: float = 0.08,
    w: float = 0.22,
    e_cap: float = 0.35,
    token_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, StepTelemetry]:
    """Apply decoupled ``IF X -> THEN Y`` clipped-ReLU neuron."""
    if abs(float(scale)) <= EPS or lv.w_read is None or lv.w_write is None:
        return h, StepTelemetry()
    arr = np.asarray(h, dtype=np.float32)
    centered = arr - np.asarray(h_center, dtype=np.float32)
    c_norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    ray_cos = np.sum(
        centered * lv.w_read.astype(np.float32)[None, None, :], axis=-1, keepdims=True
    ) / np.maximum(c_norm, EPS)
    g_t = np.clip((ray_cos - float(m0)) / max(float(w), EPS), 0.0, 1.0)
    eff_norm = float(lv.mu_d_norm) if float(lv.mu_d_norm) > EPS else 1.0
    delta = float(scale) * eff_norm * g_t * lv.w_write.astype(np.float32)[None, None, :]

    if e_cap > 0.0:
        d_norm_tok = np.linalg.norm(delta, axis=-1, keepdims=True)
        cap_val = float(e_cap) * np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), EPS)
        delta = delta * np.minimum(1.0, cap_val / np.maximum(d_norm_tok, EPS))

    if token_mask is not None:
        mask_f = np.asarray(token_mask, dtype=np.float32)
        delta = delta * mask_f
        g_t = g_t * mask_f

    h_norm = float(np.mean(np.linalg.norm(arr, axis=-1)))
    d_norm = float(np.mean(np.linalg.norm(delta, axis=-1)))
    out = (arr + delta).astype(h.dtype, copy=False)
    return out, StepTelemetry(
        g_t=float(np.mean(g_t)),
        ray_cos=float(np.mean(ray_cos)),
        rel_delta=d_norm / max(h_norm, EPS),
        active_layers=1,
    )


def apply_delta_mlx(
    h_mx: Any,
    delta_np: np.ndarray,
) -> Any:
    """Add a float32 NumPy delta back onto an MLX array preserving its dtype."""
    import mlx.core as mx

    delta_mx = mx.array(delta_np, dtype=h_mx.dtype)
    return h_mx + delta_mx


# ---------------------------------------------------------------------------
# Subspace Geometry & Cross-Layer Rank-Collapse Diagnostics
# ---------------------------------------------------------------------------


def effective_rank(mat: np.ndarray) -> float:
    """Shannon-entropy effective rank of matrix ``mat`` ``(N, D)`` in FP64."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) == 0:
        return 0.0
    s = np.linalg.svd(arr, compute_uv=False)
    p = s ** 2
    total = float(p.sum())
    if total < EPS:
        return 0.0
    p = p / total
    p = p[p > 1e-12]
    ent = -float(np.sum(p * np.log(p)))
    return float(math.exp(ent))


def bank_geometry_report(
    vectors_by_concept: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute cross-concept cosine matrix, bank effective rank, top-5 PC energy, and p90 |cos|."""
    names = sorted(vectors_by_concept.keys())
    n = len(names)
    if n == 0:
        return {
            "concepts": [],
            "count": 0,
            "effective_rank": 0.0,
            "top5_pc_energy": 0.0,
            "mean_abs_cos": 0.0,
            "p90_abs_cos": 0.0,
            "cosine_matrix": [],
        }

    rows = []
    for name in names:
        u, _ = _unit_np(vectors_by_concept[name])
        rows.append(u.astype(np.float64))
    mat = np.stack(rows, axis=0)  # (N, D)
    cos_mat = mat @ mat.T
    np.fill_diagonal(cos_mat, 1.0)

    off_diag = [
        abs(float(cos_mat[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    s = np.linalg.svd(mat, compute_uv=False)
    e = s ** 2
    total_e = float(e.sum())
    top5 = float(e[: min(5, e.size)].sum() / total_e) if total_e > EPS else 0.0

    return {
        "concepts": names,
        "count": n,
        "effective_rank": round(effective_rank(mat), 4),
        "top5_pc_energy": round(top5, 4),
        "mean_abs_cos": round(float(np.mean(off_diag)), 4) if off_diag else 0.0,
        "p90_abs_cos": round(float(np.percentile(off_diag, 90)), 4) if off_diag else 0.0,
        "cosine_matrix": [[round(float(v), 4) for v in row] for row in cos_mat],
    }
