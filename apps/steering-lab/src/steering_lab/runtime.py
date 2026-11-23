"""White-box MLX activation steering runtime, hook controller, and telemetry microscope.

Supports both real `mlx-vlm` checkpoints (e.g. `Swift-1.5-Qwen3.8-27B-MLX-8bit`)
and lightweight synthetic MLX models in CPU-only unit tests so every hook,
inlined verify loop, phase transition, rollback-safe centering mode, and sweep
invariant is testable at 0% GPU contention.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .bank import (
    ModelFingerprint,
    SpanBank,
    logit_lens_xray,
    span_key,
)
from .corpora import default_corpora, make_synthetic_image
from .operators import (
    ComposedRegimeA,
    LayerVector,
    RunningCenter,
    StepTelemetry,
    VectorSpec,
    apply_conditional_np,
    apply_delta_mlx,
    apply_regime_a_np,
    apply_regime_b_np,
    build_s3_random_layer,
    compose_regime_a,
)

DEFAULT_QWEN38_FA_BAND = (39, 43, 47, 51)
DEFAULT_QWEN38_LAYERS = (15, 27, 39, 43, 47, 49, 51, 55, 62, 63)


@dataclass(frozen=True)
class SteeringPlan:
    """Per-request steering configuration across sliders, operators, and masks."""

    scales: Mapping[str, float] = field(default_factory=dict)
    use_peer_ros: bool = True
    use_clamped_union: bool = True
    use_holds: bool = False
    use_s3_random: bool = False
    s3_seed: int = 1337
    layer_filter: str = "all"  # "all" | "fa_only" | "gdn_only" | "mid_only" | "terminal_only"
    phase_override: Optional[str] = None  # None | "all" | "content" | "think"
    token_mask_override: Optional[str] = None  # None | "all" | "decode" | "prompt_text" | "image_pad"
    center_mode: str = "prior_ema"  # "prior_ema" | "mu_clean" | "running"
    prior_weight: float = 8.0
    rank_limit: Optional[int] = None
    m0_override: Optional[float] = None
    w_override: Optional[float] = None
    knee_override: Optional[float] = None
    label: str = ""

    def active_items(self) -> list[tuple[str, float]]:
        return sorted((k, float(v)) for k, v in self.scales.items() if abs(float(v)) > 1e-8)

    def is_noop(self) -> bool:
        return not self.active_items()

    def cache_signature(self, vectors: Mapping[str, VectorSpec], num_layers: int) -> str:
        """Canonical cache salt for PrefixCacheStore.

        Terminal-only (`L == num_layers - 1`) steering does not modify any KV or
        GDN cache layer, so it returns ``""`` (100% prefix-cache reuse across scales).
        Mid-layer steering returns a deterministic digest so scale changes never
        read stale KV/GDN states.
        """
        mid_items: list[str] = []
        terminal = num_layers - 1
        for cid, scale in self.active_items():
            spec = vectors.get(cid)
            if spec is None:
                continue
            layers = [l for l in spec.layers if l < terminal]
            if layers:
                mid_items.append(
                    f"{cid}:{scale:.6f}:{layers}:{self.layer_filter}:{self.phase_override}:"
                    f"{self.token_mask_override}:{self.use_peer_ros}:{self.use_clamped_union}:"
                    f"{self.use_holds}:{self.use_s3_random}:{self.s3_seed}"
                )
        if not mid_items:
            return ""
        return hashlib.sha256("|".join(mid_items).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TokenRecord:
    """Single emitted token with live white-box microscope telemetry."""

    index: int
    token_id: int
    text: str
    phase: str  # "think" | "content"
    g_t: float
    ray_cos: float
    rel_delta: float
    entropy: float
    kl_vs_base: Optional[float] = None
    mtp_accepted: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "token_id": self.token_id,
            "text": self.text,
            "phase": self.phase,
            "g_t": round(self.g_t, 4),
            "ray_cos": round(self.ray_cos, 4),
            "rel_delta": round(self.rel_delta, 4),
            "entropy": round(self.entropy, 4),
            "kl_vs_base": None if self.kl_vs_base is None else round(self.kl_vs_base, 4),
            "mtp_accepted": self.mtp_accepted,
        }


def ngram_repetition_rate(token_ids: Sequence[int], n: int = 4) -> float:
    """Fraction of overlapping ``n``-grams that repeat an earlier ``n``-gram (0.0..1.0)."""
    ids = [int(t) for t in token_ids]
    if len(ids) < n + 1:
        return 0.0
    total = len(ids) - n + 1
    unique = len({tuple(ids[i : i + n]) for i in range(total)})
    return float(max(0.0, 1.0 - unique / max(total, 1)))


def _kl_from_logprobs_np(lp_p: np.ndarray, lp_q: np.ndarray) -> float:
    """Compute ``KL(p || q) = sum p * (log p - log q)`` in nats."""
    p = np.exp(np.asarray(lp_p, dtype=np.float64))
    diff = np.asarray(lp_p, dtype=np.float64) - np.asarray(lp_q, dtype=np.float64)
    return float(max(0.0, np.sum(p * diff)))


def classify_manifold_state(
    *,
    mean_entropy: float,
    baseline_entropy: Optional[float],
    mtp_share: Optional[float],
    baseline_mtp_share: Optional[float],
    mean_rel_delta: float,
    ngram_repeat_4: float = 0.0,
    mean_kl_vs_base: Optional[float] = None,
) -> dict[str, Any]:
    """Combine entropy, KL(p_steered || p_s0), 4-gram repetition, and MTP agreement.

    MTP acceptance alone misses degenerate loops (which are high-probability and keep
    high MTP share — e.g. Qwen 3.8 Round 2a ``B_L63_ray_r1``). Reading all four together:
    - High 4-gram repetition => ``loop_collapse`` (even when MTP is high and entropy low).
    - Spiking entropy (or extreme KL + rising entropy) => ``manifold_collapse``.
    - Meaningful KL / shift / drafter divergence with stable entropy & low repetition
      => ``in_manifold_shift``.
    - Otherwise => ``nominal``.
    """
    ent_ratio = (
        mean_entropy / max(float(baseline_entropy), 0.15)
        if baseline_entropy is not None
        else 1.0
    )
    mtp_drop = (
        float(baseline_mtp_share) - float(mtp_share)
        if mtp_share is not None and baseline_mtp_share is not None
        else 0.0
    )
    kl_val = None if mean_kl_vs_base is None else float(mean_kl_vs_base)

    if ngram_repeat_4 >= 0.35:
        status = "loop_collapse"
        note = "High 4-gram repetition detected — degenerate loop (MTP share can still look high!)."
    elif ent_ratio >= 1.75 or mean_entropy >= 2.6 or (kl_val is not None and kl_val >= 3.5 and ent_ratio >= 1.4):
        status = "manifold_collapse"
        note = "Entropy spiked sharply — activations are likely pushed off-manifold."
    elif (
        (kl_val is not None and kl_val >= 0.15)
        or mtp_drop >= 0.08
        or mean_rel_delta >= 0.06
    ) and ent_ratio < 1.55:
        status = "in_manifold_shift"
        note = "Low entropy, low repetition, and measurable distribution shift — coherent in-manifold steering."
    else:
        status = "nominal"
        note = "Entropy, KL, repetition, and acceptance stay near unsteered baseline."

    return {
        "status": status,
        "entropy_ratio": round(ent_ratio, 3),
        "mtp_drop": round(mtp_drop, 4),
        "ngram_repeat_4": round(float(ngram_repeat_4), 4),
        "mean_kl_vs_base": None if kl_val is None else round(kl_val, 4),
        "note": note,
    }


@dataclass
class GenerationTrace:
    """Complete output and microscope telemetry from one steered generation run."""

    arm_label: str
    plan: dict[str, Any]
    prompt: str
    thinking: str
    content: str
    tokens: list[TokenRecord]
    think_tokens: int
    content_tokens: int
    gate_fire_frac: float
    mean_rel_delta: float
    mean_entropy: float
    mtp_share: Optional[float]
    wall_s: float
    manifold: dict[str, Any]
    cache_signature: str
    ngram_repeat_4: float = 0.0
    mean_kl_vs_base: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_label": self.arm_label,
            "plan": self.plan,
            "prompt": self.prompt,
            "thinking": self.thinking,
            "content": self.content,
            "tokens": [t.to_dict() for t in self.tokens],
            "summary": {
                "think_tokens": self.think_tokens,
                "content_tokens": self.content_tokens,
                "total_tokens": self.think_tokens + self.content_tokens,
                "gate_fire_frac": round(self.gate_fire_frac, 4),
                "mean_rel_delta": round(self.mean_rel_delta, 4),
                "mean_entropy": round(self.mean_entropy, 4),
                "mean_kl_vs_base": (
                    None if self.mean_kl_vs_base is None else round(self.mean_kl_vs_base, 4)
                ),
                "ngram_repeat_4": round(self.ngram_repeat_4, 4),
                "mtp_share": None if self.mtp_share is None else round(self.mtp_share, 4),
                "wall_s": round(self.wall_s, 3),
                "manifold": self.manifold,
                "cache_signature": self.cache_signature,
            },
        }


# ---------------------------------------------------------------------------
# Layer Steering Controller (Prefill + Decode + Verify Hook Parity)
# ---------------------------------------------------------------------------


@dataclass
class _CompiledLayerPlan:
    layer: int
    regime_a_groups: list[tuple[str, str, ComposedRegimeA]]  # (eff_phase, eff_mask, composed)
    regime_b: list[tuple[VectorSpec, LayerVector, float]]
    conditionals: list[tuple[VectorSpec, LayerVector, float]]


class LayerSteeringController:
    """Applies compiled non-1D steering operators across layers with phase & token masks."""

    def __init__(
        self,
        plan: SteeringPlan,
        vectors: Mapping[str, VectorSpec],
        *,
        num_layers: int,
        hidden_size: int,
        layer_is_linear: Optional[Sequence[bool]] = None,
        holds_by_layer: Optional[Mapping[int, np.ndarray]] = None,
    ) -> None:
        self.plan = plan
        self.vectors = vectors
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.layer_is_linear = (
            list(layer_is_linear) if layer_is_linear is not None else [False] * self.num_layers
        )
        self.holds_by_layer = holds_by_layer or {}
        self.current_phase: str = "prompt"
        self.current_token_phases: Optional[list[str]] = None
        self.current_token_types: Optional[list[str]] = None
        self.compiled: dict[int, _CompiledLayerPlan] = {}
        self.centers: dict[int, RunningCenter] = {}
        self._last_step_telemetry: list[StepTelemetry] = []
        self._pending_verify_hidden: dict[int, np.ndarray] = {}
        self._compile()

    def _layer_allowed(self, layer_idx: int) -> bool:
        flt = self.plan.layer_filter
        if flt == "all":
            return True
        is_lin = self.layer_is_linear[layer_idx] if layer_idx < len(self.layer_is_linear) else False
        if flt == "fa_only":
            return not is_lin
        if flt == "gdn_only":
            return is_lin
        if flt == "terminal_only":
            return layer_idx == self.num_layers - 1
        if flt == "mid_only":
            return layer_idx < self.num_layers - 1
        return True

    def _compile(self) -> None:
        active = self.plan.active_items()
        if not active:
            return
        per_layer_a: dict[int, list[tuple[VectorSpec, LayerVector, float]]] = {}
        per_layer_b: dict[int, list[tuple[VectorSpec, LayerVector, float]]] = {}
        per_layer_c: dict[int, list[tuple[VectorSpec, LayerVector, float]]] = {}

        for cid, scale in active:
            spec = self.vectors.get(cid)
            if spec is None:
                continue
            for layer_idx, lv in spec.by_layer.items():
                if not self._layer_allowed(layer_idx):
                    continue
                eff_lv = (
                    build_s3_random_layer(lv, seed=self.plan.s3_seed)
                    if self.plan.use_s3_random
                    else lv
                )
                if spec.operator == "regime_b":
                    per_layer_b.setdefault(layer_idx, []).append((spec, eff_lv, scale))
                elif spec.operator == "conditional":
                    per_layer_c.setdefault(layer_idx, []).append((spec, eff_lv, scale))
                else:
                    per_layer_a.setdefault(layer_idx, []).append((spec, eff_lv, scale))

        all_layers = sorted(set(per_layer_a) | set(per_layer_b) | set(per_layer_c))
        for layer_idx in all_layers:
            a_items = per_layer_a.get(layer_idx, [])
            b_items = per_layer_b.get(layer_idx, [])
            c_items = per_layer_c.get(layer_idx, [])

            regime_a_groups: list[tuple[str, str, ComposedRegimeA]] = []
            if a_items:
                # Group Regime-A sliders by effective (phase, token_mask) so mixing e.g.
                # `image_pad` + `decode` sliders on the same layer never clobbers masks.
                grouped: dict[tuple[str, str], list[tuple[VectorSpec, LayerVector, float]]] = {}
                for sp, lv, s in a_items:
                    key = (
                        self.plan.phase_override or sp.phase,
                        self.plan.token_mask_override or sp.token_mask,
                    )
                    grouped.setdefault(key, []).append((sp, lv, s))
                hold = self.holds_by_layer.get(layer_idx) if self.plan.use_holds else None
                for (eff_phase, eff_mask), grp_items in grouped.items():
                    knee = (
                        self.plan.knee_override
                        if self.plan.knee_override is not None
                        else max(sp.knee for sp, _, _ in grp_items)
                    )
                    composed_grp = compose_regime_a(
                        [(lv, s, sp.erase_ratio) for sp, lv, s in grp_items],
                        use_peer_ros=self.plan.use_peer_ros,
                        use_clamped_union=self.plan.use_clamped_union,
                        hold_basis=hold,
                        knee=knee,
                    )
                    if composed_grp is not None:
                        regime_a_groups.append((eff_phase, eff_mask, composed_grp))

            # Pick a clean centroid prior for RunningCenter if available.
            mu_clean = None
            for _, lv, _ in b_items + c_items + a_items:
                if lv.mu_clean is not None:
                    mu_clean = lv.mu_clean
                    break
            self.centers[layer_idx] = RunningCenter(
                self.hidden_size,
                mode=self.plan.center_mode,
                mu_clean=mu_clean,
                prior_weight=self.plan.prior_weight,
            )
            self.compiled[layer_idx] = _CompiledLayerPlan(
                layer=layer_idx,
                regime_a_groups=regime_a_groups,
                regime_b=b_items,
                conditionals=c_items,
            )

    def set_phase(self, new_phase: str) -> None:
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            # Reset centering on <think> -> content so think activations never bleed into content.
            if new_phase in ("think", "content"):
                for center in self.centers.values():
                    center.reset_phase()

    def set_prompt_token_types(self, token_types: Optional[list[str]]) -> None:
        self.current_token_types = token_types

    def set_token_phases(self, token_phases: Optional[list[str]]) -> None:
        """Set per-position phases (e.g. when a verify batch straddles ``</think> -> content``)."""
        self.current_token_phases = token_phases

    def begin_step(self) -> None:
        self._last_step_telemetry.clear()
        self._pending_verify_hidden.clear()

    def commit_verify_slice(self, accepted_len: int) -> None:
        """Commit ONLY ``[:accepted_len]`` of speculative verify hidden states into centers."""
        if accepted_len <= 0:
            self._pending_verify_hidden.clear()
            return
        for layer_idx, h_arr in self._pending_verify_hidden.items():
            center = self.centers.get(layer_idx)
            if center is not None:
                center.commit_np(h_arr[:, :accepted_len, :])
        self._pending_verify_hidden.clear()

    def pop_step_telemetry(self) -> StepTelemetry:
        if not self._last_step_telemetry:
            return StepTelemetry()
        n = len(self._last_step_telemetry)
        return StepTelemetry(
            g_t=float(np.mean([t.g_t for t in self._last_step_telemetry])),
            ray_cos=float(np.mean([t.ray_cos for t in self._last_step_telemetry])),
            rel_delta=float(np.mean([t.rel_delta for t in self._last_step_telemetry])),
            active_layers=n,
        )

    def _matches_phase(self, target_phase: str) -> bool:
        eff = self.plan.phase_override or target_phase
        if eff == "all":
            return True
        return self.current_phase == eff

    def _build_token_mask(self, seq_len: int, target_mask: str) -> Optional[np.ndarray]:
        eff = self.plan.token_mask_override or target_mask
        if eff == "all":
            return None
        if eff == "decode":
            if self.current_token_types is not None:
                flags = [1.0 if tt == "decode" else 0.0 for tt in self.current_token_types[:seq_len]]
                if len(flags) < seq_len:
                    flags.extend([1.0 if self.current_phase != "prompt" else 0.0] * (seq_len - len(flags)))
                return np.asarray(flags, dtype=np.float32).reshape(1, seq_len, 1)
            if self.current_phase == "prompt":
                return np.zeros((1, seq_len, 1), dtype=np.float32)
            return None
        if eff in ("image_pad", "prompt_text"):
            if self.current_token_types is None:
                return np.zeros((1, seq_len, 1), dtype=np.float32)
            flags = [1.0 if tt == eff else 0.0 for tt in self.current_token_types[:seq_len]]
            if len(flags) < seq_len:
                flags.extend([0.0] * (seq_len - len(flags)))
            return np.asarray(flags, dtype=np.float32).reshape(1, seq_len, 1)
        return None

    def _build_phase_token_mask(
        self,
        seq_len: int,
        target_phase: str,
        target_mask: str,
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Build a combined ``[1, seq_len, 1]`` phase + token-type mask.

        Supports multi-token verify batches that straddle ``</think> -> content`` via
        ``self.current_token_phases`` as well as VLM ``<|image_pad|>`` masks.
        """
        eff_phase = self.plan.phase_override or target_phase
        phase_mask: Optional[np.ndarray] = None
        if eff_phase != "all":
            if self.current_token_phases is not None:
                pflags = [1.0 if p == eff_phase else 0.0 for p in self.current_token_phases[:seq_len]]
                if len(pflags) < seq_len:
                    pflags.extend([1.0 if self.current_phase == eff_phase else 0.0] * (seq_len - len(pflags)))
                phase_mask = np.asarray(pflags, dtype=np.float32).reshape(1, seq_len, 1)
                if float(phase_mask.sum()) == 0.0:
                    return False, phase_mask
            elif self.current_phase != eff_phase:
                return False, None

        tmask = self._build_token_mask(seq_len, target_mask)
        if tmask is not None and float(tmask.sum()) == 0.0:
            return False, tmask
        if phase_mask is not None and tmask is not None:
            combined = phase_mask * tmask
            return float(combined.sum()) > 0.0, combined
        return True, phase_mask if phase_mask is not None else tmask

    def apply_layer_np(
        self,
        layer_idx: int,
        h_np: np.ndarray,
        *,
        commit_center: bool = True,
    ) -> np.ndarray:
        """Apply compiled layer steering in NumPy (shared by MLX and CPU paths)."""
        cplan = self.compiled.get(layer_idx)
        if cplan is None:
            return h_np
        arr = np.asarray(h_np, dtype=np.float32)
        seq_len = int(arr.shape[1])
        center = self.centers[layer_idx]
        h_means = center.preview_means_np(arr)
        out = arr
        fired_any = False

        # 1. Regime A (grouped by effective phase + token mask)
        for eff_phase, eff_mask, composed_a in cplan.regime_a_groups:
            ok, tmask = self._build_phase_token_mask(seq_len, eff_phase, eff_mask)
            if ok:
                out, tel = apply_regime_a_np(out, composed_a, h_center=h_means, token_mask=tmask)
                self._last_step_telemetry.append(tel)
                fired_any = True

        # 2. Regime B
        for spec, lv, scale in cplan.regime_b:
            ok, tmask = self._build_phase_token_mask(seq_len, spec.phase, spec.token_mask)
            if not ok:
                continue
            out, tel = apply_regime_b_np(
                out,
                lv,
                scale,
                h_center=h_means,
                m0=spec.m0 if self.plan.m0_override is None else self.plan.m0_override,
                w=spec.w if self.plan.w_override is None else self.plan.w_override,
                e_cap=spec.e_cap,
                rank_limit=self.plan.rank_limit,
                token_mask=tmask,
            )
            self._last_step_telemetry.append(tel)
            fired_any = True

        # 3. Conditional IF X -> THEN Y
        for spec, lv, scale in cplan.conditionals:
            ok, tmask = self._build_phase_token_mask(seq_len, spec.phase, spec.token_mask)
            if not ok:
                continue
            out, tel = apply_conditional_np(
                out,
                lv,
                scale,
                h_center=h_means,
                m0=spec.m0 if self.plan.m0_override is None else self.plan.m0_override,
                w=spec.w if self.plan.w_override is None else self.plan.w_override,
                e_cap=spec.e_cap,
                token_mask=tmask,
            )
            self._last_step_telemetry.append(tel)
            fired_any = True

        if commit_center:
            if self.current_phase != "prompt":
                center.commit_np(arr)
        else:
            self._pending_verify_hidden[layer_idx] = arr.copy()

        return out if fired_any else h_np

    def apply_layer_mlx(
        self,
        layer_idx: int,
        h_mx: Any,
        *,
        commit_center: bool = True,
    ) -> Any:
        if layer_idx not in self.compiled:
            return h_mx
        import mlx.core as mx

        h_np = np.asarray(h_mx.astype(mx.float32))
        steered_np = self.apply_layer_np(layer_idx, h_np, commit_center=commit_center)
        if steered_np is h_np:
            return h_mx
        delta_np = steered_np - h_np
        return apply_delta_mlx(h_mx, delta_np)

    @contextlib.contextmanager
    def attach_to_lm(self, lm: Any) -> Iterator[None]:
        """Install clean hooks on ``lm.model.layers[idx]`` and optional inlined verify hooks."""
        if not self.compiled:
            yield
            return
        model_inner = getattr(lm, "model", lm)
        layers = getattr(model_inner, "layers", ())
        originals: dict[int, Any] = {}
        prev_hook = getattr(lm, "_steering_lab_hook", None)

        class _SteeredLayerProxy:
            def __init__(self, inner: Any, idx: int, ctrl: LayerSteeringController) -> None:
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_idx", idx)
                object.__setattr__(self, "_ctrl", ctrl)

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                out = object.__getattribute__(self, "_inner")(*args, **kwargs)
                ctrl = object.__getattribute__(self, "_ctrl")
                idx = object.__getattribute__(self, "_idx")
                if isinstance(out, tuple):
                    return (ctrl.apply_layer_mlx(idx, out[0], commit_center=True), *out[1:])
                return ctrl.apply_layer_mlx(idx, out, commit_center=True)

            def __getattr__(self, name: str) -> Any:
                return getattr(object.__getattribute__(self, "_inner"), name)

        try:
            for idx in self.compiled:
                if 0 <= idx < len(layers):
                    originals[idx] = layers[idx]
                    layers[idx] = _SteeredLayerProxy(layers[idx], idx, self)
            lm._steering_lab_hook = lambda idx, h, commit=False: self.apply_layer_mlx(
                idx, h, commit_center=commit
            )
            yield
        finally:
            for idx, orig in originals.items():
                layers[idx] = orig
            lm._steering_lab_hook = prev_hook


# ---------------------------------------------------------------------------
# White-Box MLX Steering Runtime (Live Checkpoints & Synthetic CPU Models)
# ---------------------------------------------------------------------------


def _next_token_entropy_np(logits_1d: np.ndarray) -> float:
    x = np.asarray(logits_1d, dtype=np.float64)
    m = float(np.max(x))
    ex = np.exp(x - m)
    p = ex / max(float(ex.sum()), 1e-12)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


class SteeringLabRuntime:
    """Manages an optional MLX model, vector packs, sweeps, and click-a-token Δ logprob."""

    def __init__(
        self,
        *,
        model: Any = None,
        processor: Any = None,
        fingerprint: Optional[ModelFingerprint] = None,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        image_pad_id: Optional[int] = None,
        eos_token_ids: Sequence[int] = (),
    ) -> None:
        self.model = model
        self.processor = processor
        self.draft_model: Any = None
        self.fingerprint = fingerprint
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.image_pad_id = image_pad_id
        self.eos_token_ids: set[int] = {int(t) for t in eos_token_ids}
        self.vectors: dict[str, VectorSpec] = {}
        self.holds_by_layer: dict[int, np.ndarray] = {}
        self.bank: Optional[SpanBank] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def lm(self) -> Any:
        return getattr(self.model, "language_model", self.model)

    def _layer_is_linear_flags(self) -> list[bool]:
        inner = getattr(self.lm, "model", self.lm)
        layers = getattr(inner, "layers", ())
        return [bool(getattr(l, "is_linear", False)) for l in layers]

    def layer_kinds(self) -> dict[int, str]:
        """Derive per-layer ``FA`` vs ``GDN`` kinds dynamically from loaded layers or bank."""
        if self.model is not None:
            flags = self._layer_is_linear_flags()
            return {i: ("GDN" if is_lin else "FA") for i, is_lin in enumerate(flags)}
        if self.bank is not None:
            return {l: self.bank.layer_kind(l) for l in self.bank.layers}
        return {}

    def _decode_token(self, token_id: int) -> str:
        tok = getattr(self.processor, "tokenizer", self.processor)
        if tok is not None and callable(getattr(tok, "decode", None)):
            return str(tok.decode([int(token_id)]))
        if self.bank is not None and self.bank.xray_vocab and 0 <= token_id < len(self.bank.xray_vocab):
            return self.bank.xray_vocab[token_id]
        return f"<{token_id}>"

    def _encode_prompt(
        self,
        prompt: str,
        *,
        system: str = "",
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> tuple[list[int], list[str]]:
        tok = getattr(self.processor, "tokenizer", self.processor)
        if tok is not None and callable(getattr(tok, "apply_chat_template", None)):
            msgs: list[dict[str, Any]] = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            try:
                rendered = tok.apply_chat_template(msgs, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                rendered = tok.apply_chat_template(msgs, **kwargs)
            ids = [int(x) for x in tok.encode(rendered)]
        elif tok is not None and callable(getattr(tok, "encode", None)):
            text = f"{system}\n{prompt}" if system else prompt
            ids = [int(x) for x in tok.encode(text)]
        else:
            ids = [1, 2, 3]
        types = [
            "image_pad" if (self.image_pad_id is not None and tid == self.image_pad_id) else "prompt_text"
            for tid in ids
        ]
        return ids, types

    def load_model_dir(self, model_dir: Path, *, model_id: Optional[str] = None) -> ModelFingerprint:
        """Lazily load an MLX checkpoint from disk using heylook's canonical MLX overlays."""
        try:
            from heylook_llm.mlx_overlay import activate_all_overlays

            activate_all_overlays(log=False)
        except Exception:  # noqa: BLE001
            pass

        from mlx_vlm.utils import load as vlm_load

        model_dir = Path(model_dir)
        fp = ModelFingerprint.from_model_dir(model_dir, model_id=model_id)
        model, processor = vlm_load(str(model_dir), trust_remote_code=False)
        tok = getattr(processor, "tokenizer", processor)

        # Install operator chat_template.heylook.jinja if present beside weights.
        override = model_dir / "chat_template.heylook.jinja"
        if override.is_file():
            tpl_body = override.read_text(encoding="utf-8")
            if hasattr(processor, "chat_template"):
                processor.chat_template = tpl_body
            if tok is not None and hasattr(tok, "chat_template"):
                tok.chat_template = tpl_body

        def _tok_id(sym: str) -> Optional[int]:
            if tok is None or not callable(getattr(tok, "convert_tokens_to_ids", None)):
                return None
            tid = tok.convert_tokens_to_ids(sym)
            return int(tid) if isinstance(tid, int) and tid >= 0 else None

        eos: set[int] = set()
        for attr in ("eos_token_id", "eos_token_ids"):
            val = getattr(tok, attr, None)
            if isinstance(val, int):
                eos.add(val)
            elif isinstance(val, (list, tuple, set)):
                eos.update(int(x) for x in val)
        im_end = _tok_id("<|im_end|>")
        if im_end is not None:
            eos.add(im_end)

        self.model = model
        self.processor = processor
        self.draft_model = None
        self.fingerprint = fp
        self.think_start_id = _tok_id("<think>")
        self.think_end_id = _tok_id("</think>")
        self.image_pad_id = _tok_id("<|image_pad|>")
        self.eos_token_ids = eos

        # Optional sibling `*-mtp` (or bundled) draft head for live Chain Lens readout.
        mtp_sibling = model_dir.parent / f"{model_dir.name}-mtp"
        if mtp_sibling.is_dir():
            try:
                from mlx_vlm.speculative.drafters import (
                    load_drafter,
                    validate_drafter_compatibility,
                )

                drafter, kind = load_drafter(str(mtp_sibling))
                validate_drafter_compatibility(self.model, drafter, kind)
                if hasattr(drafter, "bind"):
                    drafter.bind(self.model)
                self.draft_model = drafter
            except Exception:  # noqa: BLE001
                self.draft_model = None
        return fp

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.draft_model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def _lm_head_project_np(
        self,
        v_np: np.ndarray,
        *,
        h_ref: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        """Project ``v_np`` through final RMSNorm (tangent at ``h_ref``) + ``lm_head``.

        Qwen 3.8 uses zero-centered RMSNorm (``1 + weight``) before ``lm_head``; calling
        ``lm.model.norm`` directly preserves exact ``(1 + w)`` scaling and radial tangent
        projection instead of scrambling top-k words via raw ``v @ lm_head.T``.
        """
        if self.model is not None:
            import mlx.core as mx

            v_arr = np.asarray(v_np, dtype=np.float32).reshape(1, 1, -1)
            v_norm = float(np.linalg.norm(v_arr))
            v_unit = v_arr / max(v_norm, 1e-8)
            lm = self.lm
            norm_fn = getattr(getattr(lm, "model", lm), "norm", None)
            used_tangent = False
            if callable(norm_fn):
                if h_ref is not None and float(np.linalg.norm(h_ref)) > 1e-6:
                    ref_mx = mx.array(
                        np.asarray(h_ref, dtype=np.float32).reshape(1, 1, -1),
                        dtype=mx.float32,
                    )
                    eps = 1e-3 * max(float(np.linalg.norm(h_ref)), 1.0)
                    v_mx = (norm_fn(ref_mx + eps * mx.array(v_unit, dtype=mx.float32)) - norm_fn(ref_mx)) * (
                        v_norm / eps
                    )
                    used_tangent = True
                else:
                    v_mx = norm_fn(mx.array(v_unit, dtype=mx.float32)) * v_norm
                    used_tangent = True
            else:
                v_mx = mx.array(v_arr, dtype=mx.float32)

            head = getattr(lm, "lm_head", None)
            if head is not None:
                logits = head(v_mx)
            elif callable(getattr(lm, "as_linear", None)):
                logits = lm.as_linear(v_mx)
            else:
                embed = getattr(getattr(lm, "model", lm), "embed_tokens", None)
                logits = embed.as_linear(v_mx)
            mx.eval(logits)
            return np.asarray(logits.astype(mx.float32)).reshape(-1), used_tangent
        if self.bank is not None and self.bank.xray_lm_head is not None:
            return (self.bank.xray_lm_head @ v_np.reshape(-1)).astype(np.float32), False
        raise RuntimeError("No loaded model or demo lm_head available for X-Ray projection.")

    def xray_vector(
        self,
        vector_np: np.ndarray,
        *,
        layer: int,
        top_k: int = 12,
        h_ref: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Run tangent-RMSNorm ``lm_head`` Adds/Erases projection, flagging mid-layers advisory."""
        num_layers = (
            self.fingerprint.num_hidden_layers
            if self.fingerprint is not None
            else (self.bank.fingerprint.num_hidden_layers if self.bank is not None else 64)
        )
        tangent_used = False

        def _proj(v: np.ndarray) -> np.ndarray:
            nonlocal tangent_used
            logits, tangent_used = self._lm_head_project_np(v, h_ref=h_ref)
            return logits

        res = logit_lens_xray(
            vector_np,
            project_fn=_proj,
            decode_token=self._decode_token,
            top_k=top_k,
        )
        is_terminal = int(layer) >= num_layers - 1
        res["layer"] = int(layer)
        res["layer_kind"] = self.layer_kinds().get(int(layer), "FA" if int(layer) % 4 == 3 else "GDN")
        res["tangent_rmsnorm"] = tangent_used
        res["advisory"] = not is_terminal
        res["advisory_note"] = (
            "Terminal layer right before final RMSNorm + lm_head (tangent readout)."
            if is_terminal
            else "Advisory Logit Lens only: skips downstream GDN/FA rotation. "
            "Use J-Lens Chain Lens or click-a-token causal Δ logprob for faithful readouts."
        )
        return res

    def _prepare_multimodal_inputs(
        self,
        prompt: str,
        *,
        system: str = "",
        image: Any = None,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> tuple[list[int], list[str], dict[str, Any]]:
        """Prepare text or VLM (image + text) prefill inputs and token-type labels."""
        if image is None:
            ids, types = self._encode_prompt(
                prompt,
                system=system,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            return ids, types, {}

        pil_img = image
        if isinstance(image, (str, Path)):
            from PIL import Image

            pil_img = Image.open(Path(image)).convert("RGB")

        tok = getattr(self.processor, "tokenizer", self.processor)
        rendered: Optional[str] = None
        if tok is not None and callable(getattr(tok, "apply_chat_template", None)):
            msgs: list[dict[str, Any]] = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            )
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            try:
                rendered = tok.apply_chat_template(msgs, **kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                rendered = tok.apply_chat_template(msgs, **kwargs)

        if rendered is not None and self.processor is not None:
            try:
                from mlx_vlm.utils import prepare_inputs as _vlm_prepare_inputs

                img_tok_idx = getattr(getattr(self.model, "config", None), "image_token_index", None)
                raw = _vlm_prepare_inputs(
                    self.processor,
                    images=[pil_img],
                    prompts=rendered,
                    image_token_index=img_tok_idx,
                )
                input_ids_arr = raw["input_ids"]
                ids = [int(x) for x in input_ids_arr.reshape(-1).tolist()]
                extra = {k: v for k, v in raw.items() if k not in ("input_ids", "attention_mask") and v is not None}
                types = [
                    "image_pad" if (self.image_pad_id is not None and tid == self.image_pad_id) else "prompt_text"
                    for tid in ids
                ]
                return ids, types, extra
            except Exception:  # noqa: BLE001
                pass

        # Fallback for synthetic CPU test models without a full HuggingFace VLM processor.
        ids, types = self._encode_prompt(
            f"[IMG] {prompt}",
            system=system,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        return ids, types, {"_image_attached": True}

    def encode_phrase_hidden(self, token_ids: Sequence[int], *, layer: int) -> np.ndarray:
        """Forward a short decoded Chain Lens phrase through ``lm`` and capture ``layer`` hidden state.

        Avoids the Layer-0 ``embed_tokens`` vs. ``L63`` frame mismatch on untied-head models
        such as Qwen 3.8 27B (``tie_word_embeddings=False``).
        """
        if self.model is None:
            raise RuntimeError("No MLX model loaded.")
        import mlx.core as mx

        lm = self.lm
        inner = getattr(lm, "model", lm)
        lm_layers = getattr(inner, "layers", ())
        target_l = max(0, min(int(layer), len(lm_layers) - 1))
        captured: list[np.ndarray] = []
        orig = lm_layers[target_l]

        class _SingleCap:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                h = out[0] if isinstance(out, tuple) else out
                mx.eval(h)
                captured.append(np.asarray(h[0].astype(mx.float32)))
                return out

            def __getattr__(self, name: str) -> Any:
                return getattr(orig, name)

        try:
            lm_layers[target_l] = _SingleCap()
            cache = lm.make_cache()
            ids_mx = mx.array([[int(t) for t in token_ids]], dtype=mx.int32)
            self._forward_logits(lm, ids_mx, cache)
        finally:
            lm_layers[target_l] = orig

        if not captured:
            raise RuntimeError(f"Failed to capture hidden state at layer {target_l}.")
        mat = captured[0]  # (T, D)
        # Skip position 0 attention sink when phrase has >= 2 tokens.
        rows = mat[1:] if mat.shape[0] > 1 else mat
        return rows.mean(axis=0).astype(np.float32)

    def transport_upper_layers(
        self,
        h_vec: np.ndarray,
        *,
        src_layer: int,
        dst_layer: int,
    ) -> np.ndarray:
        """Exact ground-truth forward transport of ``h_vec`` through layers ``src_layer+1..dst_layer``."""
        if self.model is None:
            raise RuntimeError("No MLX model loaded.")
        import mlx.core as mx

        lm = self.lm
        inner = getattr(lm, "model", lm)
        lm_layers = getattr(inner, "layers", ())
        start = int(src_layer) + 1
        end = min(int(dst_layer), len(lm_layers) - 1)
        if start > end:
            return np.asarray(h_vec, dtype=np.float32).reshape(-1)
        cur = mx.array(np.asarray(h_vec, dtype=np.float32).reshape(1, 1, -1))
        cache = lm.make_cache()
        for idx in range(start, end + 1):
            layer = lm_layers[idx]
            try:
                out = layer(cur, mask=None, cache=cache[idx])
            except TypeError:
                out = layer(cur, cache=cache[idx])
            cur = out[0] if isinstance(out, tuple) else out
        mx.eval(cur)
        return np.asarray(cur[0, -1, :].astype(mx.float32))

    def _forward_logits(
        self,
        lm: Any,
        token_ids_mx: Any,
        cache: Any,
        *,
        extra_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        if extra_model_kwargs and "pixel_values" in extra_model_kwargs and self.model is not None:
            model_kwargs = {k: v for k, v in extra_model_kwargs.items() if not k.startswith("_")}
            out = self.model(token_ids_mx, cache=cache, **model_kwargs)
        else:
            out = lm(token_ids_mx, cache=cache)
        return out.logits if hasattr(out, "logits") else out

    def _sample_token(self, logits_1d_mx: Any, *, temperature: float) -> int:
        import mlx.core as mx

        if temperature <= 0.0:
            tok = mx.argmax(logits_1d_mx, axis=-1)
            mx.eval(tok)
            return int(tok.item())
        scaled = logits_1d_mx.astype(mx.float32) / float(temperature)
        tok = mx.random.categorical(scaled)
        mx.eval(tok)
        return int(tok.item())

    def _teacher_forced_logprobs(
        self,
        prompt_ids: Sequence[int],
        gen_ids: Sequence[int],
        *,
        extra_model_kwargs: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Compute unsteered ``s = 0`` next-token log-probabilities at every generated position."""
        import mlx.core as mx

        lm = self.lm
        cache = lm.make_cache()
        full = list(prompt_ids) + list(gen_ids[:-1])
        logits = self._forward_logits(
            lm, mx.array([full], dtype=mx.int32), cache, extra_model_kwargs=extra_model_kwargs
        )
        # Position len(prompt_ids)-1 predicts gen_ids[0], ..., last predicts gen_ids[-1].
        start = len(prompt_ids) - 1
        sliced = logits[0, start : start + len(gen_ids), :].astype(mx.float32)
        mx.eval(sliced)
        arr = np.asarray(sliced, dtype=np.float64)
        m = np.max(arr, axis=-1, keepdims=True)
        lse = m + np.log(np.maximum(np.sum(np.exp(arr - m), axis=-1, keepdims=True), 1e-12))
        return (arr - lse).astype(np.float32)

    def generate_steered(
        self,
        prompt: str,
        *,
        plan: SteeringPlan,
        system: str = "",
        image: Any = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        baseline_entropy: Optional[float] = None,
        baseline_mtp_share: Optional[float] = None,
        simulated_draft_fn: Optional[Callable[[list[int]], list[int]]] = None,
        compute_kl_vs_base: bool = False,
    ) -> GenerationTrace:
        """Run one isolated steered generation with fresh GDN/FA cache and per-token telemetry."""
        if self.model is None or self.fingerprint is None:
            raise RuntimeError("No MLX model loaded. Call load_model_dir() first.")

        import mlx.core as mx

        t0 = time.perf_counter()
        mx.random.seed(int(seed))
        lm = self.lm
        prompt_ids, prompt_types, extra_vlm_kwargs = self._prepare_multimodal_inputs(
            prompt,
            system=system,
            image=image,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        ctrl = LayerSteeringController(
            plan,
            self.vectors,
            num_layers=self.fingerprint.num_hidden_layers,
            hidden_size=self.fingerprint.hidden_size,
            layer_is_linear=self._layer_is_linear_flags(),
            holds_by_layer=self.holds_by_layer,
        )

        # Fresh GDN/FA cache per run so recurrent state never leaks across sweep arms!
        cache = lm.make_cache()
        tokens_out: list[TokenRecord] = []
        generated_ids: list[int] = []
        steered_logprobs_rows: list[np.ndarray] = []
        mtp_accepted_count = 0
        mtp_total_count = 0

        # Guard against stock unrolled `speculative_verify_hidden` silently bypassing layer hooks.
        can_use_verify = (
            simulated_draft_fn is not None
            and callable(getattr(lm, "speculative_verify_hidden", None))
            and (plan.is_noop() or bool(getattr(lm, "_supports_steering_lab_hook", False)))
        )

        def _row_logprobs(logits_1d_np: np.ndarray) -> np.ndarray:
            x = np.asarray(logits_1d_np, dtype=np.float64)
            m = float(np.max(x))
            lse = m + math.log(max(float(np.sum(np.exp(x - m))), 1e-12))
            return (x - lse).astype(np.float32)

        # Detect initial generation phase: if prompt ends with <think> (or enable_thinking), start in think.
        in_think = bool(enable_thinking)
        if self.think_end_id is not None and prompt_ids and prompt_ids[-1] == self.think_end_id:
            in_think = False

        with ctrl.attach_to_lm(lm):
            # Prefill prompt
            ctrl.set_phase("prompt")
            ctrl.set_token_phases(None)
            ctrl.set_prompt_token_types(prompt_types)
            ctrl.begin_step()
            inp = mx.array([prompt_ids], dtype=mx.int32)
            logits = self._forward_logits(lm, inp, cache, extra_model_kwargs=extra_vlm_kwargs)
            last_logits = logits[0, -1, :]
            mx.eval(last_logits)
            ctrl.set_prompt_token_types(None)
            ctrl.set_phase("think" if in_think else "content")

            step_idx = 0
            while step_idx < max_tokens:
                # Optional speculative verify path (only when verify honors `_steering_lab_hook`)
                if can_use_verify:
                    cur_logits_np = np.asarray(last_logits.astype(mx.float32))
                    first_tok = self._sample_token(last_logits, temperature=temperature)
                    ent0 = _next_token_entropy_np(cur_logits_np)
                    draft_toks = simulated_draft_fn(prompt_ids + generated_ids + [first_tok])
                    if draft_toks and step_idx + 1 < max_tokens:
                        verify_seq = [first_tok, *draft_toks]
                        # Build per-position phase mask so verify chunks straddling </think> mask cleanly!
                        pos_phases: list[str] = []
                        sim_think = in_think
                        for tok_p in verify_seq:
                            pos_phases.append("think" if sim_think else "content")
                            if self.think_end_id is not None and tok_p == self.think_end_id:
                                sim_think = False
                            elif self.think_start_id is not None and tok_p == self.think_start_id:
                                sim_think = True

                        verify_in = mx.array([verify_seq], dtype=mx.int32)
                        ctrl.set_token_phases(pos_phases)
                        ctrl.begin_step()
                        hidden, _, tx = lm.speculative_verify_hidden(verify_in, cache)
                        ver_logits = lm.speculative_logits_from_hidden(hidden)
                        mx.eval(ver_logits)
                        ctrl.set_token_phases(None)
                        tel = ctrl.pop_step_telemetry()

                        # Emit first_tok first, then check drafted positions with phase-boundary clamp.
                        accepted_toks: list[tuple[int, float, bool, np.ndarray]] = [
                            (first_tok, ent0, False, cur_logits_np)
                        ]
                        hit_phase_or_stop = (
                            first_tok in self.eos_token_ids
                            or (self.think_end_id is not None and first_tok == self.think_end_id)
                        )
                        commit_n = 1
                        if not hit_phase_or_stop:
                            for d_i, d_tok in enumerate(draft_toks):
                                if step_idx + len(accepted_toks) >= max_tokens:
                                    break
                                v_row = ver_logits[0, d_i, :]
                                v_row_np = np.asarray(v_row.astype(mx.float32))
                                target_tok = self._sample_token(v_row, temperature=temperature)
                                ent_i = _next_token_entropy_np(v_row_np)
                                mtp_total_count += 1
                                matched = target_tok == int(d_tok)
                                if matched:
                                    mtp_accepted_count += 1
                                accepted_toks.append((target_tok, ent_i, matched, v_row_np))
                                commit_n += 1
                                if (
                                    not matched
                                    or target_tok in self.eos_token_ids
                                    or (
                                        self.think_end_id is not None
                                        and target_tok == self.think_end_id
                                    )
                                ):
                                    break
                        tx.commit(commit_n)
                        ctrl.commit_verify_slice(commit_n)

                        stopped = False
                        for tok_id, ent_v, was_acc, row_logits in accepted_toks:
                            if tok_id in self.eos_token_ids:
                                stopped = True
                                break
                            phase_now = "think" if in_think else "content"
                            generated_ids.append(tok_id)
                            if compute_kl_vs_base:
                                steered_logprobs_rows.append(_row_logprobs(row_logits))
                            tokens_out.append(
                                TokenRecord(
                                    index=step_idx,
                                    token_id=tok_id,
                                    text=self._decode_token(tok_id),
                                    phase=phase_now,
                                    g_t=tel.g_t,
                                    ray_cos=tel.ray_cos,
                                    rel_delta=tel.rel_delta,
                                    entropy=ent_v,
                                    mtp_accepted=was_acc,
                                )
                            )
                            step_idx += 1
                            if self.think_end_id is not None and tok_id == self.think_end_id:
                                in_think = False
                                ctrl.set_phase("content")
                        if stopped or step_idx >= max_tokens:
                            break
                        last_logits = ver_logits[0, commit_n - 1, :]
                        continue

                # Standard single-token decode step
                cur_logits_np = np.asarray(last_logits.astype(mx.float32))
                ent = _next_token_entropy_np(cur_logits_np)
                next_tok = self._sample_token(last_logits, temperature=temperature)
                if next_tok in self.eos_token_ids:
                    break
                phase_now = "think" if in_think else "content"
                tel = ctrl.pop_step_telemetry()
                generated_ids.append(next_tok)
                if compute_kl_vs_base:
                    steered_logprobs_rows.append(_row_logprobs(cur_logits_np))
                tokens_out.append(
                    TokenRecord(
                        index=step_idx,
                        token_id=next_tok,
                        text=self._decode_token(next_tok),
                        phase=phase_now,
                        g_t=tel.g_t,
                        ray_cos=tel.ray_cos,
                        rel_delta=tel.rel_delta,
                        entropy=ent,
                        mtp_accepted=None,
                    )
                )
                step_idx += 1
                if self.think_end_id is not None and next_tok == self.think_end_id:
                    in_think = False
                    ctrl.set_phase("content")
                elif self.think_start_id is not None and next_tok == self.think_start_id:
                    in_think = True
                    ctrl.set_phase("think")

                if step_idx >= max_tokens:
                    break
                ctrl.begin_step()
                step_in = mx.array([[next_tok]], dtype=mx.int32)
                logits = self._forward_logits(lm, step_in, cache)
                last_logits = logits[0, -1, :]
                mx.eval(last_logits)

        # Optional exact per-token KL(p_steered || p_s0) via one unsteered teacher-forced forward.
        mean_kl: Optional[float] = None
        if compute_kl_vs_base and generated_ids and steered_logprobs_rows:
            if plan.is_noop():
                kls = [0.0] * len(tokens_out)
            else:
                base_lps = self._teacher_forced_logprobs(
                    prompt_ids, generated_ids, extra_model_kwargs=extra_vlm_kwargs
                )
                kls = [
                    _kl_from_logprobs_np(steered_logprobs_rows[i], base_lps[i])
                    for i in range(len(tokens_out))
                ]
            tokens_out = [
                TokenRecord(
                    index=t.index,
                    token_id=t.token_id,
                    text=t.text,
                    phase=t.phase,
                    g_t=t.g_t,
                    ray_cos=t.ray_cos,
                    rel_delta=t.rel_delta,
                    entropy=t.entropy,
                    kl_vs_base=kls[i],
                    mtp_accepted=t.mtp_accepted,
                )
                for i, t in enumerate(tokens_out)
            ]
            mean_kl = float(np.mean(kls))

        rep4 = ngram_repetition_rate(generated_ids, n=4)
        think_text = "".join(t.text for t in tokens_out if t.phase == "think")
        content_text = "".join(t.text for t in tokens_out if t.phase == "content")
        think_n = sum(1 for t in tokens_out if t.phase == "think")
        content_n = len(tokens_out) - think_n
        gate_fire = (
            float(np.mean([1.0 if t.g_t > 1e-4 else 0.0 for t in tokens_out]))
            if tokens_out
            else 0.0
        )
        mean_rel = float(np.mean([t.rel_delta for t in tokens_out])) if tokens_out else 0.0
        mean_ent = float(np.mean([t.entropy for t in tokens_out])) if tokens_out else 0.0
        mtp_share = (
            mtp_accepted_count / mtp_total_count if mtp_total_count > 0 else None
        )
        manifold = classify_manifold_state(
            mean_entropy=mean_ent,
            baseline_entropy=baseline_entropy,
            mtp_share=mtp_share,
            baseline_mtp_share=baseline_mtp_share,
            mean_rel_delta=mean_rel,
            ngram_repeat_4=rep4,
            mean_kl_vs_base=mean_kl,
        )
        return GenerationTrace(
            arm_label=plan.label or ("baseline" if plan.is_noop() else "steered"),
            plan={
                "scales": dict(plan.scales),
                "use_peer_ros": plan.use_peer_ros,
                "use_clamped_union": plan.use_clamped_union,
                "use_holds": plan.use_holds,
                "use_s3_random": plan.use_s3_random,
                "layer_filter": plan.layer_filter,
                "phase_override": plan.phase_override,
                "token_mask_override": plan.token_mask_override,
                "center_mode": plan.center_mode,
                "rank_limit": plan.rank_limit,
            },
            prompt=prompt,
            thinking=think_text,
            content=content_text,
            tokens=tokens_out,
            think_tokens=think_n,
            content_tokens=content_n,
            gate_fire_frac=gate_fire,
            mean_rel_delta=mean_rel,
            mean_entropy=mean_ent,
            mtp_share=mtp_share,
            wall_s=time.perf_counter() - t0,
            manifold=manifold,
            cache_signature=plan.cache_signature(
                self.vectors, self.fingerprint.num_hidden_layers
            ),
            ngram_repeat_4=rep4,
            mean_kl_vs_base=mean_kl,
        )

    def sweep(
        self,
        prompt: str,
        arms: Sequence[SteeringPlan],
        *,
        system: str = "",
        image: Any = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        use_mtp: bool = False,
        simulated_draft_fn: Optional[Callable[[list[int]], list[int]]] = None,
        compute_kl_vs_base: bool = True,
    ) -> list[GenerationTrace]:
        """Run seed-locked side-by-side arms with isolated fresh caches per arm.

        Enforces RNG alignment across arms: speculative draft acceptance lengths
        desynchronize PRNG draws at ``temperature > 0``, so MTP is forced off whenever
        ``temperature > 0.0``.
        """
        eff_draft = simulated_draft_fn if (use_mtp and float(temperature) <= 0.0) else None
        traces: list[GenerationTrace] = []
        base_ent: Optional[float] = None
        base_mtp: Optional[float] = None
        for idx, arm in enumerate(arms):
            tr = self.generate_steered(
                prompt,
                plan=arm,
                system=system,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                baseline_entropy=base_ent,
                baseline_mtp_share=base_mtp,
                simulated_draft_fn=eff_draft,
                compute_kl_vs_base=compute_kl_vs_base,
            )
            if idx == 0 and arm.is_noop():
                base_ent = tr.mean_entropy
                base_mtp = tr.mtp_share
            traces.append(tr)
        return traces

    def token_delta_logprob(
        self,
        prompt: str,
        generated_token_ids: Sequence[int],
        target_index: int,
        *,
        plan: SteeringPlan,
        system: str = "",
        enable_thinking: bool = True,
        top_k: int = 12,
    ) -> dict[str, Any]:
        """Ground-truth causal Δ logprob at ``target_index`` (steered vs. unsteered forward)."""
        if self.model is None or self.fingerprint is None:
            raise RuntimeError("No MLX model loaded.")
        import mlx.core as mx

        prompt_ids, prompt_types = self._encode_prompt(
            prompt, system=system, enable_thinking=enable_thinking
        )
        prefix_gen = [int(t) for t in generated_token_ids[: max(0, int(target_index))]]
        full_ids = prompt_ids + prefix_gen
        inp = mx.array([full_ids], dtype=mx.int32)
        lm = self.lm

        def _logprobs_at_end(active_plan: SteeringPlan) -> np.ndarray:
            cache = lm.make_cache()
            ctrl = LayerSteeringController(
                active_plan,
                self.vectors,
                num_layers=self.fingerprint.num_hidden_layers,
                hidden_size=self.fingerprint.hidden_size,
                layer_is_linear=self._layer_is_linear_flags(),
                holds_by_layer=self.holds_by_layer,
            )
            # Build exact per-position phases so a `content`-only slider NEVER steers
            # `prompt` or `<think>` prefix tokens during click-a-token Δ logprob.
            pos_phases: list[str] = ["prompt"] * len(prompt_ids)
            in_think = bool(enable_thinking)
            if self.think_end_id is not None and prompt_ids and prompt_ids[-1] == self.think_end_id:
                in_think = False
            for tid in prefix_gen:
                pos_phases.append("think" if in_think else "content")
                if self.think_end_id is not None and tid == self.think_end_id:
                    in_think = False
                elif self.think_start_id is not None and tid == self.think_start_id:
                    in_think = True

            with ctrl.attach_to_lm(lm):
                ctrl.set_phase("think" if in_think else "content")
                ctrl.set_token_phases(pos_phases)
                ctrl.set_prompt_token_types(prompt_types + ["decode"] * len(prefix_gen))
                logits = self._forward_logits(lm, inp, cache)
                last = logits[0, -1, :].astype(mx.float32)
                mx.eval(last)
            arr = np.asarray(last, dtype=np.float64)
            m = float(np.max(arr))
            lse = m + math.log(max(float(np.sum(np.exp(arr - m))), 1e-12))
            return (arr - lse).astype(np.float32)

        lp_base = _logprobs_at_end(SteeringPlan())
        lp_steered = _logprobs_at_end(plan)
        delta = lp_steered - lp_base
        k = min(int(top_k), int(delta.size))
        pos = np.argsort(-delta)[:k]
        neg = np.argsort(delta)[:k]
        return {
            "target_index": int(target_index),
            "causal": True,
            "adds": [
                {
                    "token_id": int(i),
                    "token": self._decode_token(int(i)),
                    "delta_logprob": round(float(delta[i]), 4),
                    "steered_logprob": round(float(lp_steered[i]), 4),
                }
                for i in pos
            ],
            "erases": [
                {
                    "token_id": int(i),
                    "token": self._decode_token(int(i)),
                    "delta_logprob": round(float(delta[i]), 4),
                    "steered_logprob": round(float(lp_steered[i]), 4),
                }
                for i in neg
            ],
        }

    def capture_span_bank(
        self,
        out_dir: Path,
        *,
        corpus_ids: Optional[Sequence[str]] = None,
        layers: Sequence[int] = DEFAULT_QWEN38_LAYERS,
        max_tokens: int = 384,
        max_tokens_per_span: int = 24,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = "low",
        image_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> SpanBank:
        """Generate traces and teacher-force capture non-sink multi-layer spans into a SpanBank.

        Captures subsampled non-sink token rows (always skipping token position 0) per span so
        SVD, capability holds, and J-Lens ridge regression have rich rank (hundreds/thousands of
        rows rather than ``N = 6..8`` prompt means). Never mislabels truncated ``<think>`` traces
        as ``answer`` spans, and wires real or synthetic images through VLM ``pixel_values``.
        """
        if self.model is None or self.fingerprint is None:
            raise RuntimeError("No MLX model loaded. Call load_model_dir() first.")
        import mlx.core as mx

        all_corpora = default_corpora()
        selected_ids = list(all_corpora.keys()) if not corpus_ids else list(corpus_ids)
        lm = self.lm
        inner = getattr(lm, "model", lm)
        lm_layers = getattr(inner, "layers", ())
        valid_layers = tuple(sorted(l for l in layers if 0 <= int(l) < len(lm_layers)))

        real_images: list[Path] = []
        if image_dir is not None and Path(image_dir).is_dir():
            exts = {".png", ".jpg", ".jpeg", ".webp"}
            real_images = sorted(
                p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in exts
            )

        def _subsample_rows(mat: np.ndarray, indices: Sequence[int], cap: int) -> Optional[np.ndarray]:
            # Always drop position 0 attention sink unless it is the only position.
            valid = [i for i in indices if 0 < i < mat.shape[0]]
            if not valid:
                valid = [i for i in indices if 0 <= i < mat.shape[0]]
            if not valid:
                return None
            if len(valid) > cap:
                pick = np.linspace(0, len(valid) - 1, num=cap, dtype=np.int64)
                valid = [valid[int(p)] for p in pick]
            return mat[valid].astype(np.float32)

        def _capture_forward(
            token_ids: list[int],
            token_types: list[str],
            think_end_pos: Optional[int],
            started_in_think: bool,
            extra_model_kwargs: Optional[dict[str, Any]],
        ) -> dict[str, dict[int, np.ndarray]]:
            captured: dict[int, np.ndarray] = {}
            originals: dict[int, Any] = {}

            class _CapProxy:
                def __init__(self, wrapped: Any, idx: int) -> None:
                    object.__setattr__(self, "_wrapped", wrapped)
                    object.__setattr__(self, "_idx", idx)

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    out = object.__getattribute__(self, "_wrapped")(*args, **kwargs)
                    idx = object.__getattribute__(self, "_idx")
                    h = out[0] if isinstance(out, tuple) else out
                    mx.eval(h)
                    captured[idx] = np.asarray(h[0].astype(mx.float32))
                    return out

                def __getattr__(self, name: str) -> Any:
                    return getattr(object.__getattribute__(self, "_wrapped"), name)

            try:
                for lidx in valid_layers:
                    originals[lidx] = lm_layers[lidx]
                    lm_layers[lidx] = _CapProxy(lm_layers[lidx], lidx)
                cache = lm.make_cache()
                self._forward_logits(
                    lm,
                    mx.array([token_ids], dtype=mx.int32),
                    cache,
                    extra_model_kwargs=extra_model_kwargs,
                )
            finally:
                for lidx, orig in originals.items():
                    lm_layers[lidx] = orig

            out_by_span: dict[str, dict[int, np.ndarray]] = {
                "image_pad": {},
                "prompt_text": {},
                "think": {},
                "answer": {},
            }
            for lidx, mat in captured.items():
                img_idx = [i for i, tt in enumerate(token_types) if tt == "image_pad"]
                ptxt_idx = [i for i, tt in enumerate(token_types) if tt == "prompt_text"]
                dec_start = len(token_types)
                if img_idx:
                    rows = _subsample_rows(mat, img_idx, max_tokens_per_span)
                    if rows is not None:
                        out_by_span["image_pad"][lidx] = rows
                if ptxt_idx:
                    rows = _subsample_rows(mat, ptxt_idx, max_tokens_per_span)
                    if rows is not None:
                        out_by_span["prompt_text"][lidx] = rows
                if mat.shape[0] > dec_start:
                    if think_end_pos is not None and think_end_pos > dec_start:
                        t_rows = _subsample_rows(mat, range(dec_start, think_end_pos), max_tokens_per_span)
                        if t_rows is not None:
                            out_by_span["think"][lidx] = t_rows
                        if mat.shape[0] > think_end_pos + 1:
                            a_rows = _subsample_rows(
                                mat, range(think_end_pos + 1, mat.shape[0]), max_tokens_per_span
                            )
                            if a_rows is not None:
                                out_by_span["answer"][lidx] = a_rows
                    elif started_in_think and self.think_end_id is not None:
                        # Trace truncated inside <think> before emitting </think>: NEVER dump into answer!
                        t_rows = _subsample_rows(mat, range(dec_start, mat.shape[0]), max_tokens_per_span)
                        if t_rows is not None:
                            out_by_span["think"][lidx] = t_rows
                    else:
                        a_rows = _subsample_rows(mat, range(dec_start, mat.shape[0]), max_tokens_per_span)
                        if a_rows is not None:
                            out_by_span["answer"][lidx] = a_rows
            return out_by_span

        tensors: dict[str, list[np.ndarray]] = {}
        concepts_meta: dict[str, dict[str, Any]] = {}

        for cid in selected_ids:
            corpus = all_corpora[cid]
            concepts_meta[cid] = {
                "label": corpus.label,
                "operator": corpus.operator,
                "kind": corpus.kind,
                "phase": corpus.phase,
                "token_mask": corpus.token_mask,
                "default_span": corpus.default_span,
                "layers": [l for l in corpus.layers if l in valid_layers] or list(valid_layers),
                "notes": corpus.notes,
            }
            for pair_idx, pair in enumerate(corpus.pairs):
                if corpus.kind in ("hold", "calibration"):
                    poles: tuple[tuple[str, str], ...] = (("clean", pair.system_a),)
                elif corpus.operator == "conditional" and pair.write_system_a is not None and pair.write_system_b is not None:
                    poles = (
                        ("read_neg", pair.system_a),
                        ("read_pos", pair.system_b),
                        ("write_neg", pair.write_system_a),
                        ("write_pos", pair.write_system_b),
                    )
                else:
                    poles = (("a", pair.system_a), ("b", pair.system_b))

                pil_img: Any = None
                if pair.image_variant is not None:
                    if real_images:
                        pil_img = real_images[pair_idx % len(real_images)]
                    else:
                        pil_img = make_synthetic_image(pair.image_variant)

                for pole_name, sys_prompt in poles:
                    tr = self.generate_steered(
                        pair.user_prompt,
                        plan=SteeringPlan(),
                        system=sys_prompt,
                        image=pil_img,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        seed=seed,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                    )
                    p_ids, p_types, vlm_kwargs = self._prepare_multimodal_inputs(
                        pair.user_prompt,
                        system=sys_prompt,
                        image=pil_img,
                        enable_thinking=enable_thinking,
                        reasoning_effort=reasoning_effort,
                    )
                    gen_ids = [t.token_id for t in tr.tokens]
                    full_ids = p_ids + gen_ids
                    started_in_think = bool(enable_thinking) and not (
                        self.think_end_id is not None and p_ids and p_ids[-1] == self.think_end_id
                    )
                    think_end_pos = None
                    if self.think_end_id is not None:
                        for offset, tid in enumerate(gen_ids):
                            if tid == self.think_end_id:
                                think_end_pos = len(p_ids) + offset
                                break
                    cap = _capture_forward(
                        full_ids, p_types, think_end_pos, started_in_think, vlm_kwargs
                    )
                    for sk, by_l in cap.items():
                        for lidx, rows in by_l.items():
                            tensors.setdefault(span_key(cid, pole_name, sk, lidx), []).append(rows)

        stacked = {
            k: np.concatenate(rows, axis=0).astype(np.float32)
            for k, rows in tensors.items()
            if rows
        }
        bank = SpanBank(
            fingerprint=self.fingerprint,
            layers=valid_layers,
            concepts=concepts_meta,
            tensors=stacked,
            layer_kinds={l: self.layer_kinds().get(l, "FA" if l % 4 == 3 else "GDN") for l in valid_layers},
        )
        bank.save(out_dir)
        self.bank = bank
        return bank
