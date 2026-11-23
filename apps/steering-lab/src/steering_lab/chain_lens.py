"""The Chain Lens: multi-token latent readout via contrastive decoding on an MTP head.

Inspired by vgel & Raymond Douglas's "The Chain Lens [interim]" (LessWrong):
instead of projecting a hidden state `h_l` (or steering vector delta) to a
single vocabulary token via `lm_head`, expand a bag of multi-token phrases by
unrolling the model's Multi-Token Prediction (MTP) head under contrastive
decoding:

    score(t | h, prefix) = log P_MTP(t | h, prefix) - alpha * log P_baseline(t | h_base, prefix)

Works with both:
- A live Qwen 3.8 MLX MTP head (`Qwen3_5MTPDraftModel` from `*-mtp`), where
  each step feeds `(token_id, prev_hidden)` into the 1-layer MTP block and
  reads logits via the shared `lm_head`.
- Any lightweight callable/synthetic `MTPAdapter` for CPU-only unit tests and
  zero-GPU demo-bank exploration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    m = float(np.max(x))
    lse = m + math.log(max(float(np.sum(np.exp(x - m))), 1e-12))
    return (x - lse).astype(np.float32)


@dataclass(frozen=True)
class JLensTransport:
    """Residual Ridge linear transport ``L_src -> L_dst`` (Tuned-Lens / J-Lens form).

    Fits ``h_dst - mu_dst ≈ (h_src - mu_src) + (h_src - mu_src) @ W_delta`` via dual
    Woodbury ridge in FP64 (O(N^3)), so ridge shrinks toward the identity map and
    remains exact and fast even when ``N << D`` (e.g. 200 rows in ``D=5120``).
    """

    src_layer: int
    dst_layer: int
    mu_src: np.ndarray  # (D,)
    mu_dst: np.ndarray  # (D,)
    x_centered: np.ndarray  # (N, D)
    dual_weights: np.ndarray  # (N, D) = (X X^T + ridge I)^{-1} ((H_dst - mu_dst) - X)
    ridge: float
    train_r2: float
    val_r2: Optional[float] = None
    sample_count: int = 0

    def transport_vector(self, v: np.ndarray) -> np.ndarray:
        """Transport a direction/perturbation ``v`` ``(D,)`` via the Jacobian ``I + W_delta``."""
        v64 = np.asarray(v, dtype=np.float64).reshape(-1)
        out = v64 + (v64 @ self.x_centered.T) @ self.dual_weights
        return out.astype(np.float32)

    def transport_state(self, h: np.ndarray) -> np.ndarray:
        """Transport a full hidden state ``h`` ``(D,)`` into ``L_dst`` coordinates."""
        dh = np.asarray(h, dtype=np.float64).reshape(-1) - self.mu_src
        out = self.mu_dst + dh + (dh @ self.x_centered.T) @ self.dual_weights
        return out.astype(np.float32)


def _solve_residual_ridge(
    xs: np.ndarray, xd: np.ndarray, *, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mu_src = xs.mean(axis=0)
    mu_dst = xd.mean(axis=0)
    xc = xs - mu_src
    yc = xd - mu_dst
    y_res = yc - xc
    n = xc.shape[0]
    gram = xc @ xc.T
    scale = max(float(np.trace(gram)) / max(n, 1), 1e-6)
    lam = float(ridge) * scale
    dual = np.linalg.solve(gram + lam * np.eye(n, dtype=np.float64), y_res)
    pred = xc + (gram @ dual)
    ss_res = float(np.sum((yc - pred) ** 2))
    ss_tot = max(float(np.sum(yc ** 2)), 1e-12)
    return mu_src, mu_dst, xc, dual, max(0.0, 1.0 - ss_res / ss_tot)


def _select_best_ridge_cv(
    xs_tr: np.ndarray,
    xd_tr: np.ndarray,
    xs_val: np.ndarray,
    xd_val: np.ndarray,
    *,
    ridge_grid: Sequence[float],
) -> tuple[float, float]:
    """Select the ridge value maximizing held-out ``val_r2`` via one Gram eigendecomposition."""
    mu_s = xs_tr.mean(axis=0)
    mu_d = xd_tr.mean(axis=0)
    xc = xs_tr - mu_s
    yc = xd_tr - mu_d
    y_res = yc - xc
    n = xc.shape[0]
    gram = xc @ xc.T
    scale = max(float(np.trace(gram)) / max(n, 1), 1e-6)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 0.0)

    dx_val = xs_val - mu_s
    # Project validation kernels and targets into the Gram eigenbasis once.
    k_val_q = (dx_val @ xc.T) @ eigvecs  # (N_val, N_tr)
    qt_yres = eigvecs.T @ y_res  # (N_tr, D)
    y_val_c = xd_val - xd_val.mean(axis=0, keepdims=True)
    ss_tot_v = max(float(np.sum(y_val_c ** 2)), 1e-12)

    best_ridge = float(ridge_grid[0])
    best_r2 = -1.0
    for cand in ridge_grid:
        r = float(cand)
        inv = 1.0 / (eigvals + r * scale)
        pred_val = mu_d + dx_val + (k_val_q * inv[None, :]) @ qt_yres
        ss_res_v = float(np.sum((xd_val - pred_val) ** 2))
        val_r2 = max(0.0, 1.0 - ss_res_v / ss_tot_v)
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_ridge = r
    return best_ridge, best_r2


DEFAULT_JLENS_RIDGE_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def fit_jlens_transport(
    h_src: np.ndarray,
    h_dst: np.ndarray,
    *,
    src_layer: int,
    dst_layer: int,
    ridge: float | str = "auto",
    ridge_grid: Sequence[float] = DEFAULT_JLENS_RIDGE_GRID,
    val_frac: float = 0.2,
    seed: int = 42,
) -> JLensTransport:
    """Fit an FP64 residual ridge J-Lens transport with held-out ``val_r2``.

    When ``ridge="auto"`` (default) and ``N >= 10``, picks the ridge multiplier in
    ``ridge_grid`` maximizing held-out ``val_r2`` using a single Gram eigendecomposition
    (preventing small-ridge overfitting when ``N < D``, e.g. 1k-2k rows in ``D=5120``).
    """
    xs = np.asarray(h_src, dtype=np.float64)
    xd = np.asarray(h_dst, dtype=np.float64)
    if xs.ndim != 2 or xd.ndim != 2 or xs.shape != xd.shape:
        raise ValueError(f"Expected matched 2D arrays, got {xs.shape} and {xd.shape}")

    val_r2: Optional[float] = None
    chosen_ridge: float = 0.3 if isinstance(ridge, str) else float(ridge)
    n = xs.shape[0]
    n_val = int(round(n * float(val_frac)))
    if n >= 10 and 2 <= n_val < n - 2:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        if isinstance(ridge, str):
            grid = tuple(float(g) for g in ridge_grid) or DEFAULT_JLENS_RIDGE_GRID
            chosen_ridge, raw_val_r2 = _select_best_ridge_cv(
                xs[tr_idx], xd[tr_idx], xs[val_idx], xd[val_idx], ridge_grid=grid
            )
            val_r2 = round(raw_val_r2, 4)
        else:
            mu_s_tr, mu_d_tr, xc_tr, dual_tr, _ = _solve_residual_ridge(
                xs[tr_idx], xd[tr_idx], ridge=chosen_ridge
            )
            dx_val = xs[val_idx] - mu_s_tr
            pred_val = mu_d_tr + dx_val + (dx_val @ xc_tr.T) @ dual_tr
            y_val = xd[val_idx]
            ss_res_v = float(np.sum((y_val - pred_val) ** 2))
            ss_tot_v = max(float(np.sum((y_val - y_val.mean(axis=0, keepdims=True)) ** 2)), 1e-12)
            val_r2 = round(max(0.0, 1.0 - ss_res_v / ss_tot_v), 4)

    mu_src, mu_dst, xc, dual, train_r2 = _solve_residual_ridge(xs, xd, ridge=chosen_ridge)
    return JLensTransport(
        src_layer=int(src_layer),
        dst_layer=int(dst_layer),
        mu_src=mu_src,
        mu_dst=mu_dst,
        x_centered=xc,
        dual_weights=dual,
        ridge=float(chosen_ridge),
        train_r2=round(train_r2, 4),
        val_r2=val_r2,
        sample_count=int(n),
    )


def build_probe_pair(
    direction: np.ndarray,
    *,
    h_prior: Optional[np.ndarray] = None,
    probe_scale: float = 1.0,
    jlens: Optional[JLensTransport] = None,
    norm_relative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(h_target, h_baseline, v_transported)`` for contrastive Chain Lens readout.

    Adds ``v_t`` onto the corpus-mean prior ``h_prior`` (transported to terminal
    ``L63`` via ``jlens`` when probing a mid-layer) and contrasts against
    ``h_baseline = h_prior``. When ``norm_relative=True`` and ``||h_prior|| > 0``,
    scales ``v_t`` to ``probe_scale * ||h_prior||`` (per Chain Lens §6) so mid and
    terminal layers with very different RMS norms probe at consistent strength.
    """
    v = np.asarray(direction, dtype=np.float32).reshape(-1)
    if jlens is not None:
        v_t = jlens.transport_vector(v)
        base = jlens.mu_dst.astype(np.float32) if h_prior is None else jlens.transport_state(h_prior)
    else:
        v_t = v
        base = np.zeros_like(v_t) if h_prior is None else np.asarray(h_prior, dtype=np.float32).reshape(-1)

    v_norm = float(np.linalg.norm(v_t))
    base_norm = float(np.linalg.norm(base))
    if norm_relative and base_norm > 1e-6 and v_norm > 1e-8:
        eff_delta = (float(probe_scale) * base_norm / v_norm) * v_t
    else:
        eff_delta = float(probe_scale) * v_t
    h_target = (base + eff_delta).astype(np.float32)
    return h_target, base.astype(np.float32), v_t.astype(np.float32)


@dataclass(frozen=True)
class ChainCandidate:
    """One decoded multi-token latent phrase from the Chain Lens."""

    token_ids: tuple[int, ...]
    text: str
    score: float
    step_scores: tuple[float, ...]
    mean_mtp_logprob: float
    mean_base_logprob: float
    round_trip_cos: Optional[float] = None
    round_trip_rank: Optional[int] = None  # 1 = closest to target among bank directions
    round_trip_percentile: Optional[float] = None  # percentile vs calibration states (0..100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "text": self.text,
            "score": round(float(self.score), 4),
            "step_scores": [round(float(s), 4) for s in self.step_scores],
            "mean_mtp_logprob": round(float(self.mean_mtp_logprob), 4),
            "mean_base_logprob": round(float(self.mean_base_logprob), 4),
            "round_trip_cos": (
                None if self.round_trip_cos is None else round(float(self.round_trip_cos), 4)
            ),
            "round_trip_rank": self.round_trip_rank,
            "round_trip_percentile": (
                None
                if self.round_trip_percentile is None
                else round(float(self.round_trip_percentile), 1)
            ),
        }


class MTPStepAdapter(Protocol):
    """Minimal step contract for contrastive multi-token rollout on an MTP head."""

    def initial_state(self, h: np.ndarray) -> Any:
        """Build an immutable/clonable initial state for latent vector ``h`` ``(D,)``."""
        ...

    def step(self, state: Any, token_id: Optional[int]) -> tuple[np.ndarray, Any]:
        """Return ``(logits_1d, next_state)`` after feeding ``token_id`` (None for seed step)."""
        ...


class LinearDemoMTPAdapter:
    """CPU-only 1-step-recurrent synthetic MTP adapter for tests and demo-bank UI.

    Simulates an MTP head whose transition mixes the latent state ``h`` with the
    previous token's embedding (`h_next = 0.72 * h + 0.28 * W_trans @ e_prev`) so
    contrastive decoding recovers coherent multi-token phrases from a vector.
    """

    def __init__(
        self,
        lm_head: np.ndarray,
        *,
        transition: Optional[np.ndarray] = None,
    ) -> None:
        self.lm_head = np.asarray(lm_head, dtype=np.float32)  # (V, D)
        v, d = self.lm_head.shape
        if transition is not None:
            self.transition = np.asarray(transition, dtype=np.float32)
        else:
            # Mild cyclic shift matrix so sequential tokens chain deterministically across clusters.
            t = np.zeros((d, d), dtype=np.float32)
            for i in range(d):
                t[(i + 1) % d, i] = 0.85
                t[i, i] = 0.35
            self.transition = t

    def initial_state(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float32).reshape(-1).copy()

    def step(self, state: np.ndarray, token_id: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
        if token_id is None:
            logits = self.lm_head @ state
            return logits.astype(np.float32), state.copy()
        idx = int(token_id) % self.lm_head.shape[0]
        emb = self.lm_head[idx]
        nxt = (0.68 * state + 0.52 * (self.transition @ emb)).astype(np.float32)
        logits = (self.lm_head @ nxt).astype(np.float32)
        return logits, nxt


class Qwen35MTPAdapter:
    """Live MLX adapter wrapping ``Qwen3_5MTPDraftModel`` (from ``*-mtp``).

    Mirrors ``Qwen3_5MTPDraftModel._forward_hidden`` exactly:
    1. Applies ``target_lm.model.norm`` ONCE in ``initial_state`` so both seed-step
       ``lm_head`` and step-1 ``pre_fc_norm_hidden`` receive post-norm hidden states
       (matching ``speculative_verify_hidden``).
    2. Maintains a cloned per-beam 1-layer ``KVCache`` + advancing ``position_ids``
       so steps 2..N advance RoPE and attend to earlier tokens in the Chain Lens phrase
       without mutating sibling beams or the main generation cache.
    """

    def __init__(self, target_lm: Any, draft_model: Any, *, start_position: int = 1) -> None:
        self.target_lm = target_lm
        self.draft = draft_model
        self.start_position = int(start_position)

    def _logits_from_hidden(self, h_mx: Any) -> np.ndarray:
        import mlx.core as mx

        head = getattr(self.target_lm, "lm_head", None)
        if head is not None:
            logits = head(h_mx)
        else:
            logits = self.target_lm.model.embed_tokens.as_linear(h_mx)
        mx.eval(logits)
        return np.asarray(logits[0, -1, :].astype(mx.float32))

    def _clone_cache(self, kv_states: Optional[list[tuple[Any, Any, int]]]) -> Optional[list[Any]]:
        make_cache = getattr(self.draft, "make_cache", None)
        if not callable(make_cache):
            return None
        caches = make_cache()
        if kv_states is not None:
            for c, (k, v, offset) in zip(caches, kv_states):
                if k is not None and v is not None and offset > 0:
                    c.keys = k[:, :, :offset, :]
                    c.values = v[:, :, :offset, :]
                    c.offset = int(offset)
        return caches

    @staticmethod
    def _snapshot_cache(caches: Optional[list[Any]]) -> Optional[list[tuple[Any, Any, int]]]:
        if not caches:
            return None
        snaps: list[tuple[Any, Any, int]] = []
        for c in caches:
            off = int(getattr(c, "offset", 0))
            k = getattr(c, "keys", None)
            v = getattr(c, "values", None)
            if k is not None and v is not None and off > 0:
                snaps.append((k[:, :, :off, :], v[:, :, :off, :], off))
            else:
                snaps.append((None, None, 0))
        return snaps

    def initial_state(self, h: np.ndarray) -> dict[str, Any]:
        import mlx.core as mx

        raw = mx.array(np.asarray(h, dtype=np.float32).reshape(1, 1, -1))
        # Apply target final RMSNorm once so seed logits AND step-1 MTP input share exact post-norm scale.
        target_norm = getattr(getattr(self.target_lm, "model", self.target_lm), "norm", None)
        h_normed = target_norm(raw) if callable(target_norm) else raw
        mx.eval(h_normed)
        return {
            "h_mx": h_normed,
            "tokens": (),
            "pos": self.start_position,
            "kv_states": None,
        }

    def step(self, state: dict[str, Any], token_id: Optional[int]) -> tuple[np.ndarray, dict[str, Any]]:
        import mlx.core as mx

        h_mx = state["h_mx"]
        if token_id is None:
            return self._logits_from_hidden(h_mx), state

        tok_arr = mx.array([[int(token_id)]], dtype=mx.int32)
        inner_target = getattr(self.target_lm, "model", self.target_lm)
        embed_fn = (
            getattr(self.draft, "_input_embed", None)
            or getattr(self.draft, "embed_tokens", None)
            or inner_target.embed_tokens
        )
        embed_scale = float(
            getattr(
                self.draft,
                "_input_embed_scale",
                getattr(inner_target, "embed_scale", 1.0),
            )
        )
        tok_emb = embed_fn(tok_arr) * embed_scale
        h_cast = h_mx.astype(tok_emb.dtype)
        pos = int(state.get("pos", self.start_position))
        pos_ids = mx.array([[pos]], dtype=mx.int32)
        beam_cache = self._clone_cache(state.get("kv_states"))

        forward_hidden = getattr(self.draft, "_forward_hidden", None)
        if callable(forward_hidden):
            cur = forward_hidden(tok_emb, h_cast, beam_cache, pos_ids)
        else:
            if hasattr(self.draft, "pre_fc_norm_embedding") and hasattr(self.draft, "pre_fc_norm_hidden"):
                e_n = self.draft.pre_fc_norm_embedding(tok_emb)
                h_n = self.draft.pre_fc_norm_hidden(h_cast)
                cur = self.draft.fc(mx.concatenate([e_n, h_n], axis=-1))
            elif hasattr(self.draft, "fc"):
                cur = self.draft.fc(mx.concatenate([tok_emb, h_cast], axis=-1))
            else:
                cur = tok_emb + h_cast
            layers = getattr(self.draft, "layers", ())
            c_list = beam_cache if beam_cache is not None else [None] * len(layers)
            for layer, lc in zip(layers, c_list):
                try:
                    out = layer(cur, mask=None, cache=lc, position_ids=pos_ids)
                except TypeError:
                    out = layer(cur, mask=None, cache=lc)
                cur = out[0] if isinstance(out, tuple) else out
            norm_fn = getattr(self.draft, "norm", None)
            if callable(norm_fn):
                cur = norm_fn(cur)

        mx.eval(cur)
        logits = self._logits_from_hidden(cur)
        next_state = {
            "h_mx": cur,
            "tokens": (*state["tokens"], int(token_id)),
            "pos": pos + 1,
            "kv_states": self._snapshot_cache(beam_cache),
        }
        return logits, next_state


def run_chain_lens(
    h_target: np.ndarray,
    *,
    adapter: MTPStepAdapter,
    decode_token: Callable[[int], str],
    h_baseline: Optional[np.ndarray] = None,
    steps: int = 4,
    beam_width: int = 6,
    top_k_expand: int = 12,
    alpha: float = 0.75,
    plausibility_beta: float = 0.1,
    repeat_penalty: float = 1.25,
    embed_tokens_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
    bank_directions: Optional[Mapping[str, np.ndarray]] = None,
    target_direction_key: Optional[str] = None,
    calibration_states: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Decode a bag of multi-token latent phrases from ``h_target`` via contrastive MTP beam search.

    Keep ``steps`` short (3..5 tokens; hallucination climbs rapidly past ~5 steps per
    vgel & Douglas §6). When ``embed_tokens_fn`` is supplied, computes a round-trip
    cosine fidelity score between each decoded phrase and ``h_target - h_baseline``.
    """
    h_t = np.asarray(h_target, dtype=np.float32).reshape(-1)
    h_b = (
        np.zeros_like(h_t, dtype=np.float32)
        if h_baseline is None
        else np.asarray(h_baseline, dtype=np.float32).reshape(-1)
    )
    n_steps = max(1, min(int(steps), 8))
    b_width = max(1, min(int(beam_width), 16))
    k_exp = max(b_width, min(int(top_k_expand), 64))
    log_beta = math.log(max(float(plausibility_beta), 1e-6))

    # Each beam item: (cum_score, token_ids, step_scores, mtp_lps, base_lps, state_t, state_b)
    beams: list[tuple[float, tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], Any, Any]] = [
        (0.0, (), (), (), (), adapter.initial_state(h_t), adapter.initial_state(h_b))
    ]

    for step_idx in range(n_steps):
        next_beams: list[
            tuple[float, tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], Any, Any]
        ] = []
        for cum_score, tids, s_scores, m_lps, b_lps, st_t, st_b in beams:
            feed_tok = None if step_idx == 0 else tids[-1]
            logits_t, nxt_t = adapter.step(st_t, feed_tok)
            logits_b, nxt_b = adapter.step(st_b, feed_tok)
            lp_t = _log_softmax_np(logits_t)
            lp_b = _log_softmax_np(logits_b)

            # Contrastive decoding with plausibility mask (Li et al. / Chain Lens)
            cutoff = float(np.max(lp_t)) + log_beta
            contrast = lp_t - float(alpha) * lp_b
            masked = np.where(lp_t >= cutoff, contrast, -1e9)

            # Mild repetition damping inside the short phrase so chains don't stutter.
            if repeat_penalty > 1.0 and tids:
                for prev in set(tids):
                    if 0 <= prev < masked.size:
                        masked[prev] = masked[prev] - math.log(float(repeat_penalty))

            cand_k = min(k_exp, int(masked.size))
            top_ids = np.argsort(-masked)[:cand_k]
            for tid in top_ids:
                tid_i = int(tid)
                sc = float(masked[tid_i])
                if sc <= -1e8:
                    continue
                next_beams.append(
                    (
                        cum_score + sc,
                        (*tids, tid_i),
                        (*s_scores, sc),
                        (*m_lps, float(lp_t[tid_i])),
                        (*b_lps, float(lp_b[tid_i])),
                        nxt_t,
                        nxt_b,
                    )
                )

        if not next_beams:
            break
        # Deduplicate identical token tuples, keep top beam_width by cumulative score
        seen: set[tuple[int, ...]] = set()
        deduped = []
        for item in sorted(next_beams, key=lambda row: row[0], reverse=True):
            if item[1] in seen:
                continue
            seen.add(item[1])
            deduped.append(item)
            if len(deduped) >= b_width:
                break
        beams = deduped

    probe_dir = (h_t - h_b).astype(np.float64)
    probe_norm = float(np.linalg.norm(probe_dir))

    cal_cos: Optional[np.ndarray] = None
    if calibration_states is not None and probe_norm > 1e-8:
        cal = np.asarray(calibration_states, dtype=np.float64)
        if cal.ndim == 2 and cal.shape[1] == probe_dir.shape[0] and cal.shape[0] >= 4:
            # Center calibration states around h_b so anisotropy is properly subtracted.
            cal_c = cal - h_b.astype(np.float64)[None, :]
            cal_norms = np.maximum(np.linalg.norm(cal_c, axis=-1), 1e-8)
            cal_cos = (cal_c @ probe_dir) / (cal_norms * probe_norm)

    chains: list[ChainCandidate] = []
    for cum_score, tids, s_scores, m_lps, b_lps, _, _ in beams:
        if not tids:
            continue
        rt_cos: Optional[float] = None
        rt_rank: Optional[int] = None
        rt_pct: Optional[float] = None
        if embed_tokens_fn is not None and probe_norm > 1e-8:
            emb = np.asarray(embed_tokens_fn(tids), dtype=np.float64).reshape(-1)
            en = float(np.linalg.norm(emb))
            if emb.shape == probe_dir.shape and en > 1e-8:
                rt_cos = float(np.dot(emb, probe_dir) / (en * probe_norm))
                if cal_cos is not None:
                    rt_pct = float(100.0 * np.mean(cal_cos <= rt_cos))
                if bank_directions:
                    higher = 0
                    for bk, bv in bank_directions.items():
                        if target_direction_key is not None and bk == target_direction_key:
                            continue
                        b64 = np.asarray(bv, dtype=np.float64).reshape(-1)
                        bn = float(np.linalg.norm(b64))
                        if b64.shape == emb.shape and bn > 1e-8:
                            if float(np.dot(emb, b64) / (en * bn)) > rt_cos:
                                higher += 1
                    rt_rank = higher + 1
        chains.append(
            ChainCandidate(
                token_ids=tids,
                text="".join(decode_token(t) for t in tids),
                score=cum_score / max(len(tids), 1),
                step_scores=s_scores,
                mean_mtp_logprob=float(np.mean(m_lps)) if m_lps else 0.0,
                mean_base_logprob=float(np.mean(b_lps)) if b_lps else 0.0,
                round_trip_cos=rt_cos,
                round_trip_rank=rt_rank,
                round_trip_percentile=rt_pct,
            )
        )

    return {
        "steps": n_steps,
        "beam_width": b_width,
        "alpha": float(alpha),
        "plausibility_beta": float(plausibility_beta),
        "chains": [c.to_dict() for c in chains],
    }
