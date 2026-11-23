"""Multi-layer Span Bank I/O, vector pack serialization, fingerprinting, and lm_head X-Ray."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from safetensors.numpy import load_file as st_load_file
from safetensors.numpy import save_file as st_save_file

from .operators import (
    LayerVector,
    VectorSpec,
    bank_geometry_report,
    build_conditional_layer,
    build_hold_basis,
    build_regime_a_layer,
    build_regime_b_layer,
)

SPAN_KINDS = ("image_pad", "prompt_text", "think", "answer")


class FingerprintMismatchError(ValueError):
    """Raised when a span bank or vector pack does not match the loaded model."""


@dataclass(frozen=True)
class ModelFingerprint:
    """Architecture identity checked before applying any bank or vector pack."""

    model_id: str
    hidden_size: int
    num_hidden_layers: int
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hidden_size": int(self.hidden_size),
            "num_hidden_layers": int(self.num_hidden_layers),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelFingerprint":
        return cls(
            model_id=str(data["model_id"]),
            hidden_size=int(data["hidden_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            config_sha256=str(data["config_sha256"]),
        )

    @classmethod
    def from_model_dir(cls, model_dir: Path, model_id: Optional[str] = None) -> "ModelFingerprint":
        cfg_path = model_dir / "config.json"
        raw = cfg_path.read_bytes()
        cfg = json.loads(raw.decode("utf-8"))
        text_cfg = cfg.get("text_config") or cfg.get("language_config") or cfg
        hidden_size = int(text_cfg.get("hidden_size") or cfg.get("hidden_size") or 0)
        num_layers = int(text_cfg.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError(f"Could not read hidden_size/num_hidden_layers from {cfg_path}")
        return cls(
            model_id=model_id or model_dir.name,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            config_sha256=hashlib.sha256(raw).hexdigest()[:16],
        )

    def assert_compatible(self, other: "ModelFingerprint") -> None:
        if (
            self.hidden_size != other.hidden_size
            or self.num_hidden_layers != other.num_hidden_layers
            or self.config_sha256 != other.config_sha256
        ):
            raise FingerprintMismatchError(
                f"Model fingerprint mismatch: pack built for {self.model_id} "
                f"(D={self.hidden_size}, L={self.num_hidden_layers}, sha={self.config_sha256}), "
                f"but active model is {other.model_id} "
                f"(D={other.hidden_size}, L={other.num_hidden_layers}, sha={other.config_sha256})"
            )


def span_key(concept: str, pole: str, span_kind: str, layer: int) -> str:
    return f"{concept}__{pole}__{span_kind}__L{int(layer)}"


@dataclass
class SpanBank:
    """Cached multi-layer raw span activations for zero-reload math iteration."""

    fingerprint: ModelFingerprint
    layers: tuple[int, ...]
    concepts: dict[str, dict[str, Any]]
    tensors: dict[str, np.ndarray]  # span_key -> (N_rows, D) float32
    layer_kinds: dict[int, str] = field(default_factory=dict)  # layer -> "FA" | "GDN"
    xray_vocab: Optional[list[str]] = None
    xray_lm_head: Optional[np.ndarray] = None  # optional small demo (V, D)

    def get_span(self, concept: str, pole: str, span_kind: str, layer: int) -> Optional[np.ndarray]:
        return self.tensors.get(span_key(concept, pole, span_kind, layer))

    def layer_kind(self, layer: int) -> str:
        if int(layer) in self.layer_kinds:
            return self.layer_kinds[int(layer)]
        # Qwen 3.8 3:1 hybrid layout default: Full Attention sits strictly at idx % 4 == 3.
        return "FA" if int(layer) % 4 == 3 else "GDN"

    def corpus_states(self, layer: int, *, span_kind: str = "answer") -> Optional[np.ndarray]:
        """All captured rows at ``layer`` (preferring diverse calibration rows when present)."""
        rows: list[np.ndarray] = []
        for cid, cmeta in sorted(self.concepts.items()):
            sk = str(cmeta.get("default_span", span_kind))
            for pole in ("clean", "b", "a", "write_pos", "write_neg"):
                arr = self.get_span(cid, pole, sk, layer)
                if arr is not None and arr.shape[0] > 0:
                    rows.append(arr)
        if not rows:
            return None
        return np.concatenate(rows, axis=0).astype(np.float32)

    def corpus_mean(self, layer: int, *, span_kind: str = "answer") -> Optional[np.ndarray]:
        """Mean hidden state across all captured rows at ``layer`` (Chain Lens prior baseline)."""
        states = self.corpus_states(layer, span_kind=span_kind)
        return None if states is None else states.mean(axis=0).astype(np.float32)

    def paired_layer_states(
        self,
        src_layer: int,
        dst_layer: int,
        *,
        span_kind: str = "answer",
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return aligned ``(H_src, H_dst)`` rows across all concepts for J-Lens ridge transport."""
        src_rows: list[np.ndarray] = []
        dst_rows: list[np.ndarray] = []
        for cid, cmeta in sorted(self.concepts.items()):
            sk = str(cmeta.get("default_span", span_kind))
            for pole in ("a", "b", "clean", "read_pos", "read_neg", "write_pos", "write_neg"):
                s_arr = self.get_span(cid, pole, sk, src_layer)
                d_arr = self.get_span(cid, pole, sk, dst_layer)
                if s_arr is not None and d_arr is not None and s_arr.shape[0] == d_arr.shape[0] and s_arr.shape[0] > 0:
                    src_rows.append(s_arr)
                    dst_rows.append(d_arr)
        if not src_rows:
            return None
        return (
            np.concatenate(src_rows, axis=0).astype(np.float32),
            np.concatenate(dst_rows, axis=0).astype(np.float32),
        )

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        st_tensors = {k: np.asarray(v, dtype=np.float32) for k, v in self.tensors.items()}
        if self.xray_lm_head is not None:
            st_tensors["__xray_lm_head__"] = np.asarray(self.xray_lm_head, dtype=np.float32)
        st_save_file(st_tensors, str(out_dir / "spans.safetensors"))
        manifest = {
            "format": "steering_lab_span_bank_v1",
            "fingerprint": self.fingerprint.to_dict(),
            "layers": list(self.layers),
            "layer_kinds": {str(l): self.layer_kind(l) for l in self.layers},
            "concepts": self.concepts,
            "tensor_keys": sorted(self.tensors.keys()),
            "xray_vocab": self.xray_vocab,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, bank_dir: Path) -> "SpanBank":
        bank_dir = Path(bank_dir)
        manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
        raw = st_load_file(str(bank_dir / "spans.safetensors"))
        lm_head = raw.pop("__xray_lm_head__", None)
        raw_kinds = manifest.get("layer_kinds") or {}
        return cls(
            fingerprint=ModelFingerprint.from_dict(manifest["fingerprint"]),
            layers=tuple(int(x) for x in manifest["layers"]),
            concepts=dict(manifest["concepts"]),
            tensors=raw,
            layer_kinds={int(k): str(v) for k, v in raw_kinds.items()},
            xray_vocab=manifest.get("xray_vocab"),
            xray_lm_head=lm_head,
        )

    def geometry_by_layer(self, *, span_kind: str = "answer") -> dict[int, dict[str, Any]]:
        """Compute cross-concept geometry (rank, cosine matrix, p90 |cos|) per layer."""
        report: dict[int, dict[str, Any]] = {}
        for layer in self.layers:
            vecs: dict[str, np.ndarray] = {}
            for cid, cmeta in self.concepts.items():
                if cmeta.get("kind") in ("hold", "calibration"):
                    continue
                sk = cmeta.get("default_span", span_kind)
                a = self.get_span(cid, "a", sk, layer)
                b = self.get_span(cid, "b", sk, layer)
                if a is not None and b is not None and a.shape[0] > 0 and b.shape[0] > 0:
                    vecs[cid] = b.mean(axis=0) - a.mean(axis=0)
            if vecs:
                row = bank_geometry_report(vecs)
                row["layer_kind"] = self.layer_kind(layer)
                report[int(layer)] = row
        return report


# ---------------------------------------------------------------------------
# Vector Pack Serialization (`<dir>/<id>/{vector.safetensors, meta.json}`)
# ---------------------------------------------------------------------------


def save_vector_spec(
    spec: VectorSpec,
    fingerprint: ModelFingerprint,
    out_root: Path,
) -> Path:
    """Save one steering concept pack into ``<out_root>/<spec.id>/``."""
    vdir = Path(out_root) / spec.id
    vdir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    layer_stats: dict[str, dict[str, Any]] = {}
    for layer, lv in sorted(spec.by_layer.items()):
        tensors[f"L{layer}__mu_d"] = np.asarray(lv.mu_d, dtype=np.float32)
        tensors[f"L{layer}__vr"] = np.asarray(lv.vr, dtype=np.float32)
        tensors[f"L{layer}__d_diag"] = np.asarray(lv.d_diag, dtype=np.float32)
        if lv.mu_clean is not None:
            tensors[f"L{layer}__mu_clean"] = np.asarray(lv.mu_clean, dtype=np.float32)
        if lv.w_read is not None:
            tensors[f"L{layer}__w_read"] = np.asarray(lv.w_read, dtype=np.float32)
        if lv.w_write is not None:
            tensors[f"L{layer}__w_write"] = np.asarray(lv.w_write, dtype=np.float32)
        layer_stats[str(layer)] = {
            "mu_d_norm": round(float(lv.mu_d_norm), 5),
            "rank": int(lv.vr.shape[0]),
            "split_half_cos": (
                None if lv.split_half_cos is None else round(float(lv.split_half_cos), 4)
            ),
            "flip_null_cos": (
                None if lv.flip_null_cos is None else round(float(lv.flip_null_cos), 4)
            ),
        }
    st_save_file(tensors, str(vdir / "vector.safetensors"))
    meta = {
        "format": "steering_lab_vector_v1",
        "model": fingerprint.to_dict(),
        **spec.public_meta(),
        "layer_stats": layer_stats,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return vdir


def load_vector_spec(
    vdir: Path,
    *,
    expected_fingerprint: Optional[ModelFingerprint] = None,
) -> tuple[VectorSpec, ModelFingerprint]:
    """Load one steering concept pack and validate its model fingerprint."""
    vdir = Path(vdir)
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    fp = ModelFingerprint.from_dict(meta["model"])
    if expected_fingerprint is not None:
        fp.assert_compatible(expected_fingerprint)
    tensors = st_load_file(str(vdir / "vector.safetensors"))

    by_layer: dict[int, LayerVector] = {}
    layer_stats = meta.get("layer_stats") or {}
    layers = tuple(int(x) for x in meta.get("layers") or [])
    for layer in layers:
        lstat = layer_stats.get(str(layer)) or {}
        by_layer[layer] = LayerVector(
            layer=layer,
            mu_d=tensors[f"L{layer}__mu_d"],
            vr=tensors[f"L{layer}__vr"],
            d_diag=tensors[f"L{layer}__d_diag"],
            mu_d_norm=float(lstat.get("mu_d_norm", 1.0)),
            mu_clean=tensors.get(f"L{layer}__mu_clean"),
            w_read=tensors.get(f"L{layer}__w_read"),
            w_write=tensors.get(f"L{layer}__w_write"),
            split_half_cos=lstat.get("split_half_cos"),
            flip_null_cos=lstat.get("flip_null_cos"),
        )

    scale = meta.get("scale") or {}
    params = meta.get("params") or {}
    spec = VectorSpec(
        id=str(meta["id"]),
        label=str(meta.get("label") or meta["id"]),
        operator=str(meta["operator"]),
        phase=str(meta.get("phase", "content")),
        token_mask=str(meta.get("token_mask", "decode")),
        layers=layers,
        scale_min=float(scale.get("min", -2.0)),
        scale_max=float(scale.get("max", 2.0)),
        scale_suggested=float(scale.get("suggested", 1.0)),
        m0=float(params.get("m0", 0.08)),
        w=float(params.get("w", 0.22)),
        e_cap=float(params.get("e_cap", 0.35)),
        knee=float(params.get("knee", 0.0)),
        erase_ratio=float(params.get("erase_ratio", 1.0)),
        by_layer=by_layer,
        notes=str(meta.get("notes", "")),
    )
    return spec, fp


def save_hold_pack(
    holds_by_layer: Mapping[int, np.ndarray],
    fingerprint: ModelFingerprint,
    out_root: Path,
    *,
    hold_id: str = "_holds",
) -> Path:
    hdir = Path(out_root) / hold_id
    hdir.mkdir(parents=True, exist_ok=True)
    tensors = {f"L{layer}__hold": np.asarray(mat, dtype=np.float32) for layer, mat in holds_by_layer.items()}
    st_save_file(tensors, str(hdir / "holds.safetensors"))
    meta = {
        "format": "steering_lab_holds_v1",
        "model": fingerprint.to_dict(),
        "layers": sorted(int(l) for l in holds_by_layer),
    }
    (hdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return hdir


def load_hold_pack(
    hdir: Path,
    *,
    expected_fingerprint: Optional[ModelFingerprint] = None,
) -> dict[int, np.ndarray]:
    hdir = Path(hdir)
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    fp = ModelFingerprint.from_dict(meta["model"])
    if expected_fingerprint is not None:
        fp.assert_compatible(expected_fingerprint)
    tensors = st_load_file(str(hdir / "holds.safetensors"))
    return {int(l): tensors[f"L{int(l)}__hold"] for l in meta.get("layers") or []}


def build_packs_from_bank(
    bank: SpanBank,
    out_root: Path,
    *,
    rank: int = 8,
    hold_rank: int = 16,
    use_paired_diff: bool = True,
) -> dict[str, Any]:
    """Build Regime A / B / Conditional vector packs + capability holds from a cached SpanBank."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    built: list[str] = []
    holds_by_layer: dict[int, np.ndarray] = {}

    # 1. Capability holds first
    for cid, cmeta in bank.concepts.items():
        if cmeta.get("kind") != "hold":
            continue
        sk = str(cmeta.get("default_span", "answer"))
        for layer in bank.layers:
            acts = bank.get_span(cid, "clean", sk, layer)
            if acts is not None and acts.shape[0] >= 2:
                holds_by_layer[layer] = build_hold_basis(acts, rank=hold_rank)
    if holds_by_layer:
        save_hold_pack(holds_by_layer, bank.fingerprint, out_root)

    # 2. Concept vector packs
    for cid, cmeta in bank.concepts.items():
        if cmeta.get("kind") in ("hold", "calibration"):
            continue
        op = str(cmeta.get("operator", "regime_a"))
        sk = str(cmeta.get("default_span", "answer"))
        phase = str(cmeta.get("phase", "content"))
        token_mask = str(cmeta.get("token_mask", "decode"))
        target_layers = tuple(int(x) for x in cmeta.get("layers", bank.layers))
        by_layer: dict[int, LayerVector] = {}

        for layer in target_layers:
            if op == "conditional":
                rp = bank.get_span(cid, "read_pos", sk, layer)
                rn = bank.get_span(cid, "read_neg", sk, layer)
                wp = bank.get_span(cid, "write_pos", sk, layer)
                wn = bank.get_span(cid, "write_neg", sk, layer)
                if rp is not None and rn is not None and wp is not None and wn is not None:
                    by_layer[layer] = build_conditional_layer(layer, rp, rn, wp, wn)
            else:
                a = bank.get_span(cid, "a", sk, layer)
                b = bank.get_span(cid, "b", sk, layer)
                if a is None or b is None or a.shape[0] == 0 or b.shape[0] == 0:
                    continue
                if op == "regime_b":
                    by_layer[layer] = build_regime_b_layer(layer, a, b, rank=rank)
                else:
                    by_layer[layer] = build_regime_a_layer(
                        layer, a, b, rank=rank, use_paired_diff=use_paired_diff
                    )

        if not by_layer:
            continue
        spec = VectorSpec(
            id=cid,
            label=str(cmeta.get("label", cid)),
            operator=op,
            phase=phase,
            token_mask=token_mask,
            layers=tuple(sorted(by_layer.keys())),
            scale_min=float(cmeta.get("scale_min", 0.0 if op == "regime_b" else -2.0)),
            scale_max=float(cmeta.get("scale_max", 1.0 if op == "regime_b" else 2.0)),
            scale_suggested=float(cmeta.get("scale_suggested", 1.0)),
            m0=float(cmeta.get("m0", 0.08)),
            w=float(cmeta.get("w", 0.22)),
            e_cap=float(cmeta.get("e_cap", 0.35)),
            knee=float(cmeta.get("knee", 0.0)),
            erase_ratio=float(cmeta.get("erase_ratio", 1.0)),
            by_layer=by_layer,
            notes=str(cmeta.get("notes", "")),
        )
        save_vector_spec(spec, bank.fingerprint, out_root)
        built.append(cid)

    return {
        "vectors": built,
        "hold_layers": sorted(holds_by_layer.keys()),
        "out_root": str(out_root),
    }


# ---------------------------------------------------------------------------
# `lm_head` Adds / Erases Logit-Lens X-Ray
# ---------------------------------------------------------------------------


def logit_lens_xray(
    vector: np.ndarray,
    *,
    project_fn: Callable[[np.ndarray], np.ndarray],
    decode_token: Callable[[int], str],
    top_k: int = 12,
) -> dict[str, Any]:
    """Project direction ``vector`` through ``lm_head`` and return top Adds / Erases."""
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    logits = np.asarray(project_fn(v), dtype=np.float32).reshape(-1)
    if logits.size == 0:
        return {"adds": [], "erases": []}
    k = min(int(top_k), int(logits.size))
    top_pos = np.argsort(-logits)[:k]
    top_neg = np.argsort(logits)[:k]
    return {
        "adds": [
            {"token_id": int(idx), "token": decode_token(int(idx)), "score": round(float(logits[idx]), 4)}
            for idx in top_pos
        ],
        "erases": [
            {"token_id": int(idx), "token": decode_token(int(idx)), "score": round(float(logits[idx]), 4)}
            for idx in top_neg
        ],
    }


# ---------------------------------------------------------------------------
# CPU-Only Synthetic Demo Bank (Zero GPU Load)
# ---------------------------------------------------------------------------


def create_demo_bank(out_dir: Path, *, dim: int = 64, seed: int = 42) -> SpanBank:
    """Create a synthetic multi-layer SpanBank simulating Qwen 3.8 L43..L49 vs L63 rank collapse.

    Runs strictly on CPU (~300 KB) so every UI instrument, geometry panel,
    `peer_ros` toggle, and X-ray readout can be explored while `heylook` / H3
    is using Metal.
    """
    rng = np.random.default_rng(seed)
    layers = (15, 27, 39, 43, 47, 49, 51, 55, 62, 63)
    layer_kinds = {l: ("FA" if l % 4 == 3 else "GDN") for l in layers}
    vocab = [
        " precisely", " equation", " theorem", " invariant", " latency", " kernel",
        " shimmering", " twilight", " lantern", " whisper", " velvet", " horizon",
        " directly", " false", " wrong", " stop", " exact", " verdict",
        " perhaps", " actually", " wait", " however", " delve", " tapestry",
        " landscape", " synergy", " vibrant", " multifaceted", " crucial", " foster",
        " why", " assume", " counterexample", " verify", " question", " test",
        " aperture", " shadow", " tungsten", " lens", " specular", " bokeh",
        " mood", " dreamy", " ethereal", " vibe", " cinematic", " aesthetic",
        " return", " def", " json", " assert", " int", " dict",
        " maybe", " arguably", " somewhat", " essentially", " basically", " generally",
        " concrete", " steel", " millimeter", " kelvin",
    ][:dim]

    # Synthetic lm_head (V, D) aligned with orthogonal semantic blocks
    lm_head = np.eye(len(vocab), dim, dtype=np.float32)

    def _basis_vec(indices: Sequence[int], sign: float = 1.0) -> np.ndarray:
        v = np.zeros((dim,), dtype=np.float64)
        for i in indices:
            if i < dim:
                v[i] = sign
        return v / max(float(np.linalg.norm(v)), 1e-8)

    # True semantic directions (nearly orthogonal at L43..L49, funneling at L63)
    shared_funnel = _basis_vec([18, 19, 20, 21, 22, 23, 24, 25])
    concept_axes = {
        "clinical_vs_poetic": (_basis_vec([6, 7, 8, 9, 10, 11]), _basis_vec([0, 1, 2, 3, 4, 5])),
        "blunt_vs_hedged": (_basis_vec([18, 19, 20, 21, 54, 55]), _basis_vec([12, 13, 14, 15, 16, 17])),
        "socratic_vs_declarative": (_basis_vec([12, 16, 17]), _basis_vec([30, 31, 32, 33, 34, 35])),
        "anti_slop": (_basis_vec([22, 23, 24, 25, 26, 27, 28, 29]), np.zeros((dim,), dtype=np.float64)),
        "vision_optical_vs_mood": (_basis_vec([42, 43, 44, 45, 46, 47]), _basis_vec([36, 37, 38, 39, 40, 41])),
    }
    hold_axis = _basis_vec([48, 49, 50, 51, 52, 53])

    concepts_meta: dict[str, dict[str, Any]] = {
        "clinical_vs_poetic": {
            "label": "Clinical Technical (vs. Poetic)",
            "operator": "regime_a",
            "phase": "content",
            "token_mask": "decode",
            "default_span": "answer",
            "layers": [39, 43, 47, 51],
            "scale_min": -2.0,
            "scale_max": 2.0,
            "scale_suggested": 1.0,
            "notes": "Demo Regime-A manifold replacement on the ~60-80% FA band (idx % 4 == 3).",
        },
        "blunt_vs_hedged": {
            "label": "Blunt Directness (vs. Hedged)",
            "operator": "regime_a",
            "phase": "content",
            "token_mask": "decode",
            "default_span": "answer",
            "layers": [39, 43, 47, 51],
            "scale_min": -2.0,
            "scale_max": 2.0,
            "scale_suggested": 1.0,
            "notes": "Demo Regime-A peer slider on the FA band demonstrating peer_ros.",
        },
        "socratic_vs_declarative": {
            "label": "Socratic Questioning (vs. Declarative)",
            "operator": "regime_a",
            "phase": "content",
            "token_mask": "decode",
            "default_span": "answer",
            "layers": [39, 43, 47, 51],
            "scale_min": -2.0,
            "scale_max": 2.0,
            "scale_suggested": 0.8,
            "notes": "Demo Regime-A third FA-band slider for multi-slider composition.",
        },
        "anti_slop": {
            "label": "Anti-AI-Slop Eraser (Regime B)",
            "operator": "regime_b",
            "phase": "content",
            "token_mask": "decode",
            "default_span": "answer",
            "layers": [63],
            "scale_min": 0.0,
            "scale_max": 1.0,
            "scale_suggested": 1.0,
            "m0": 0.08,
            "w": 0.22,
            "e_cap": 0.35,
            "notes": "Demo Regime-B pure suppressor at terminal L63 (ADDS=noise, ERASES=slop).",
        },
        "vision_optical_vs_mood": {
            "label": "Optical/Physical Vision (vs. Abstract Mood)",
            "operator": "regime_a",
            "phase": "all",
            "token_mask": "image_pad",
            "default_span": "image_pad",
            "layers": [27, 47],
            "scale_min": -2.0,
            "scale_max": 2.0,
            "scale_suggested": 1.0,
            "notes": "Demo VLM Dual-Lock concept captured on <|image_pad|> tokens.",
        },
        "cond_hedge_to_concrete": {
            "label": "IF Hedging -> THEN Concrete Nouns",
            "operator": "conditional",
            "phase": "content",
            "token_mask": "decode",
            "default_span": "answer",
            "layers": [47, 63],
            "scale_min": 0.0,
            "scale_max": 2.0,
            "scale_suggested": 1.0,
            "m0": 0.08,
            "w": 0.22,
            "notes": "Demo decoupled IF X -> THEN Y ReLU conditional neuron.",
        },
        "capability_holds": {
            "kind": "hold",
            "label": "Math / Code / JSON Capability Holds",
            "default_span": "answer",
            "layers": list(layers),
        },
    }

    tensors: dict[str, np.ndarray] = {}
    n_samples = 24
    for layer in layers:
        # Simulate the real L49..L51 -> L63 autoregressive funnel collapse!
        funnel_mix = 0.05 if layer <= 51 else (0.35 if layer == 55 else 0.72)
        for cid, (pole_a, pole_b) in concept_axes.items():
            sk = concepts_meta[cid]["default_span"]
            eff_a = pole_a + funnel_mix * shared_funnel
            eff_b = pole_b - 0.15 * funnel_mix * shared_funnel
            base = rng.normal(0.0, 0.25, size=(n_samples, dim))
            acts_a = (base + 2.4 * eff_a[None, :] + rng.normal(0.0, 0.18, size=(n_samples, dim))).astype(
                np.float32
            )
            acts_b = (base + 2.4 * eff_b[None, :] + rng.normal(0.0, 0.18, size=(n_samples, dim))).astype(
                np.float32
            )
            tensors[span_key(cid, "a", sk, layer)] = acts_a
            tensors[span_key(cid, "b", sk, layer)] = acts_b

        # Conditional neuron spans
        r_pos_dir = _basis_vec([54, 55, 56, 57, 58, 59])
        w_pos_dir = _basis_vec([60, 61, 62, 63])
        base = rng.normal(0.0, 0.2, size=(n_samples, dim))
        tensors[span_key("cond_hedge_to_concrete", "read_neg", "answer", layer)] = base.astype(np.float32)
        tensors[span_key("cond_hedge_to_concrete", "read_pos", "answer", layer)] = (
            base + 2.5 * r_pos_dir[None, :]
        ).astype(np.float32)
        tensors[span_key("cond_hedge_to_concrete", "write_neg", "answer", layer)] = base.astype(np.float32)
        tensors[span_key("cond_hedge_to_concrete", "write_pos", "answer", layer)] = (
            base + 2.5 * w_pos_dir[None, :]
        ).astype(np.float32)

        # Capability holds spans
        clean = (
            rng.normal(0.0, 0.12, size=(n_samples, dim))
            + rng.normal(0.0, 2.0, size=(n_samples, 1)) * hold_axis[None, :]
        ).astype(np.float32)
        tensors[span_key("capability_holds", "clean", "answer", layer)] = clean

    fp = ModelFingerprint(
        model_id="demo-qwen38-synth",
        hidden_size=dim,
        num_hidden_layers=64,
        config_sha256="demo0000synth064",
    )
    bank = SpanBank(
        fingerprint=fp,
        layers=layers,
        concepts=concepts_meta,
        tensors=tensors,
        layer_kinds=layer_kinds,
        xray_vocab=vocab,
        xray_lm_head=lm_head,
    )
    bank.save(out_dir)
    return bank
