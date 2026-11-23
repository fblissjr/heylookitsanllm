"""Local FastAPI + vanilla UI server for the MLX Steering Lab (default port 8111).

Loads model weights ONLY on explicit user request (`POST /api/model/load`) so
starting the workbench to explore cached span banks, geometry, `peer_ros`
composition, or `lm_head` X-rays exerts 0% Metal load on a running `heylook`
server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bank import (
    SpanBank,
    build_packs_from_bank,
    load_hold_pack,
    load_vector_spec,
)
from .chain_lens import (
    LinearDemoMTPAdapter,
    Qwen35MTPAdapter,
    build_probe_pair,
    fit_jlens_transport,
    run_chain_lens,
)
from .corpora import list_corpora_summary
from .operators import (
    LayerVector,
    VectorSpec,
    build_s3_random_layer,
    compose_regime_a,
)
from .runtime import (
    SteeringLabRuntime,
    SteeringPlan,
    classify_manifold_state,
    ngram_repetition_rate,
)

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


class ComposeRequest(BaseModel):
    layer: int = 47
    scales: dict[str, float] = Field(default_factory=dict)
    use_peer_ros: bool = True
    use_clamped_union: bool = True
    use_holds: bool = False
    use_s3_random: bool = False
    s3_seed: int = 1337
    knee: float = 0.0
    push_mode: str = "mu_d_norm"  # "mu_d_norm" | "norm_matched"
    norm_matched_frac: float = 0.15
    top_k: int = 10


class XrayRequest(BaseModel):
    concept_id: str
    layer: int = 47
    component: str = "mu_d"  # "mu_d" | "vr:0" | "w_read" | "w_write"
    top_k: int = 12


class ChainLensRequest(BaseModel):
    concept_id: Optional[str] = None
    layer: int = 47
    component: str = "mu_d"  # "mu_d" | "vr:0" | "w_read" | "w_write" | "blend"
    scales: dict[str, float] = Field(default_factory=dict)
    use_peer_ros: bool = True
    use_clamped_union: bool = True
    use_holds: bool = False
    use_jlens: bool = True
    use_exact_upper_layers: bool = False
    probe_scale: float = Field(default=1.0, ge=0.05, le=5.0)
    direction_sign: float = 1.0  # +1 for target pole / Adds, -1 for Pole A / Erases
    steps: int = Field(default=4, ge=1, le=6)
    beam_width: int = Field(default=6, ge=1, le=12)
    alpha: float = Field(default=0.75, ge=0.0, le=2.0)
    plausibility_beta: float = Field(default=0.1, ge=0.01, le=0.9)


class ArmPayload(BaseModel):
    label: str = ""
    scales: dict[str, float] = Field(default_factory=dict)
    use_peer_ros: bool = True
    use_clamped_union: bool = True
    use_holds: bool = False
    use_s3_random: bool = False
    s3_seed: int = 1337
    layer_filter: str = "all"
    phase_override: Optional[str] = None
    token_mask_override: Optional[str] = None
    center_mode: str = "prior_ema"
    rank_limit: Optional[int] = None
    m0_override: Optional[float] = None
    w_override: Optional[float] = None
    knee_override: Optional[float] = None
    push_mode: str = "mu_d_norm"
    norm_matched_frac: float = 0.15
    layer_scale_mode: str = "divide_k"  # "divide_k" | "divide_sqrt_k" | "raw"


class SweepRequest(BaseModel):
    prompt: str
    system: str = ""
    arms: list[ArmPayload] = Field(default_factory=list)
    max_tokens: int = Field(default=96, ge=4, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    enable_thinking: bool = True
    reasoning_effort: Optional[str] = None


class TokenDeltaRequest(BaseModel):
    prompt: str
    system: str = ""
    token_ids: list[int]
    target_index: int = Field(ge=0)
    arm: ArmPayload
    enable_thinking: bool = True
    top_k: int = Field(default=12, ge=1, le=50)


class LoadModelRequest(BaseModel):
    model_path: str
    model_id: Optional[str] = None


class LabState:
    def __init__(
        self,
        *,
        runtime: Optional[SteeringLabRuntime] = None,
        bank_dir: Optional[Path] = None,
        vectors_dir: Optional[Path] = None,
        default_model_path: Optional[str] = None,
    ) -> None:
        self.runtime = runtime or SteeringLabRuntime()
        self.bank_dir: Optional[Path] = bank_dir
        self.vectors_dir: Optional[Path] = vectors_dir
        self.default_model_path = default_model_path
        if bank_dir is not None and Path(bank_dir).is_dir():
            self.load_bank(Path(bank_dir))
        if vectors_dir is not None and Path(vectors_dir).is_dir():
            self.load_vectors(Path(vectors_dir))

    def load_bank(self, bank_dir: Path) -> None:
        self.bank_dir = Path(bank_dir)
        bank = SpanBank.load(self.bank_dir)
        self.runtime.bank = bank
        if self.runtime.fingerprint is None:
            self.runtime.fingerprint = bank.fingerprint
        # Auto-build in-memory vectors if none loaded yet so the UI works immediately.
        if not self.runtime.vectors:
            tmp_out = self.bank_dir / "_built_vectors"
            build_packs_from_bank(bank, tmp_out)
            self.load_vectors(tmp_out)

    def load_vectors(self, vectors_dir: Path) -> None:
        self.vectors_dir = Path(vectors_dir)
        loaded: dict[str, VectorSpec] = {}
        holds: dict[int, np.ndarray] = {}
        expected_fp = self.runtime.fingerprint
        for child in sorted(self.vectors_dir.iterdir()):
            if not child.is_dir():
                continue
            if (child / "holds.safetensors").is_file():
                holds = load_hold_pack(child, expected_fingerprint=expected_fp)
            elif (child / "vector.safetensors").is_file() and (child / "meta.json").is_file():
                spec, fp = load_vector_spec(child, expected_fingerprint=expected_fp)
                loaded[spec.id] = spec
                if expected_fp is None:
                    expected_fp = fp
                    self.runtime.fingerprint = fp
        self.runtime.vectors = loaded
        self.runtime.holds_by_layer = holds


def _plan_from_payload(arm: ArmPayload) -> SteeringPlan:
    return SteeringPlan(
        scales=dict(arm.scales),
        use_peer_ros=arm.use_peer_ros,
        use_clamped_union=arm.use_clamped_union,
        use_holds=arm.use_holds,
        use_s3_random=arm.use_s3_random,
        s3_seed=arm.s3_seed,
        layer_filter=arm.layer_filter,
        phase_override=arm.phase_override,
        token_mask_override=arm.token_mask_override,
        center_mode=arm.center_mode,
        rank_limit=arm.rank_limit,
        m0_override=arm.m0_override,
        w_override=arm.w_override,
        knee_override=arm.knee_override,
        push_mode=arm.push_mode,
        norm_matched_frac=arm.norm_matched_frac,
        layer_scale_mode=arm.layer_scale_mode,
        label=arm.label,
    )


def _demo_synthetic_sweep(state: LabState, req: SweepRequest) -> dict[str, Any]:
    """Deterministic CPU-only preview sweep when exploring UI with a demo bank and no 28 GB model."""
    arms_out: list[dict[str, Any]] = []
    vocab = (
        state.runtime.bank.xray_vocab
        if state.runtime.bank is not None and state.runtime.bank.xray_vocab
        else [" token", " mechanism", " invariant", " direct", " shadow", " perhaps"]
    )
    base_ent = 0.52
    base_mtp = 0.62

    for idx, arm in enumerate(req.arms):
        plan = _plan_from_payload(arm)
        active = plan.active_items()
        mag = float(sum(abs(s) for _, s in active))
        think_toks = ["First", " check", " the", " prompt", " structure", "."]
        if not active:
            content_toks = [
                "Unified", " memory", " shares", " one", " pool", " between", " CPU", " and", " GPU", "."
            ]
        elif plan.use_s3_random:
            content_toks = [
                "Random", " orthogonal", " perturbation", " leaves", " semantics", " mostly", " unshifted", "."
            ]
        else:
            top_id, top_s = max(active, key=lambda kv: abs(kv[1]))
            if top_id == "anti_slop":
                content_toks = [
                    "Skip", " throat-clearing.", " Direct", " memory", " bandwidth", " bounds", " decode", " speed", "."
                ]
            elif top_id == "clinical_vs_poetic" and top_s < 0:
                content_toks = [
                    "Like", " twilight", " lanterns", " along", " a", " quiet", " river,", " weights", " stream", " in", " unison", "."
                ]
            elif top_id == "socratic_vs_declarative" and top_s > 0:
                content_toks = [
                    "Which", " bottleneck", " hits", " first", " at", " B=1:", " compute", " FLOPs", " or", " memory", " bandwidth", "?"
                ]
            else:
                content_toks = [
                    "Conclusion:", " bandwidth", " dominates", " B=1", " decode;", " zero", " copy", " avoids", " PCIe", " stalls", "."
                ]

        tokens_rec: list[dict[str, Any]] = []
        t_i = 0
        for tok_str in think_toks:
            in_think_active = (plan.phase_override == "all") or (plan.phase_override == "think")
            g_t = min(1.0, 0.35 * mag) if in_think_active else 0.0
            tokens_rec.append(
                {
                    "index": t_i,
                    "token_id": t_i % len(vocab),
                    "text": tok_str,
                    "phase": "think",
                    "g_t": round(g_t, 4),
                    "ray_cos": round(0.08 + 0.22 * g_t, 4),
                    "rel_delta": round(0.05 * mag * g_t, 4),
                    "entropy": round(base_ent + 0.04 * (t_i % 3), 4),
                    "kl_vs_base": round(0.18 * mag * g_t, 4),
                    "mtp_accepted": True,
                }
            )
            t_i += 1
        for k, tok_str in enumerate(content_toks):
            g_t = min(1.0, (0.2 + 0.6 * ((k % 3) / 2.0)) * min(mag, 1.5)) if active else 0.0
            ray_c = round(0.04 + 0.31 * g_t, 4)
            rel_d = round(0.06 * mag * (0.4 + 0.6 * g_t), 4)
            ent_v = base_ent + (1.8 * max(0.0, mag - 2.6)) + 0.05 * (k % 3)
            kl_v = round(0.45 * mag * (0.3 + 0.7 * g_t), 4) if active else 0.0
            acc = not (mag > 0.7 and (k % 3 == 0))
            tokens_rec.append(
                {
                    "index": t_i,
                    "token_id": (t_i + 10) % len(vocab),
                    "text": tok_str,
                    "phase": "content",
                    "g_t": round(g_t, 4),
                    "ray_cos": ray_c,
                    "rel_delta": rel_d,
                    "entropy": round(ent_v, 4),
                    "kl_vs_base": kl_v,
                    "mtp_accepted": acc,
                }
            )
            t_i += 1

        mean_ent = float(np.mean([t["entropy"] for t in tokens_rec]))
        mean_kl = float(np.mean([t["kl_vs_base"] for t in tokens_rec]))
        mean_rel = float(np.mean([t["rel_delta"] for t in tokens_rec]))
        gate_fire = float(np.mean([1.0 if t["g_t"] > 1e-4 else 0.0 for t in tokens_rec]))
        mtp_share = float(np.mean([1.0 if t["mtp_accepted"] else 0.0 for t in tokens_rec]))
        rep4 = ngram_repetition_rate([t["token_id"] for t in tokens_rec], n=4)
        manifold = classify_manifold_state(
            mean_entropy=mean_ent,
            baseline_entropy=base_ent,
            mtp_share=mtp_share,
            baseline_mtp_share=base_mtp,
            mean_rel_delta=mean_rel,
            ngram_repeat_4=rep4,
            mean_kl_vs_base=mean_kl,
        )
        arms_out.append(
            {
                "arm_label": arm.label or ("Baseline (s=0)" if not active else f"Arm {idx}"),
                "plan": arm.model_dump(),
                "prompt": req.prompt,
                "thinking": "".join(think_toks),
                "content": "".join(content_toks),
                "tokens": tokens_rec,
                "summary": {
                    "think_tokens": len(think_toks),
                    "content_tokens": len(content_toks),
                    "total_tokens": len(tokens_rec),
                    "gate_fire_frac": round(gate_fire, 4),
                    "mean_rel_delta": round(mean_rel, 4),
                    "mean_entropy": round(mean_ent, 4),
                    "mean_kl_vs_base": round(mean_kl, 4),
                    "ngram_repeat_4": round(rep4, 4),
                    "mtp_share": round(mtp_share, 4),
                    "wall_s": 0.002,
                    "manifold": manifold,
                    "cache_signature": plan.cache_signature(
                        state.runtime.vectors,
                        state.runtime.fingerprint.num_hidden_layers
                        if state.runtime.fingerprint
                        else 64,
                    ),
                },
            }
        )
    return {"mode": "demo_synthetic", "arms": arms_out}


def _mount_heylook_access_gate(app: FastAPI) -> bool:
    """Attach heylook's HostCheckMiddleware, AccessGateMiddleware, and /login routes.

    Reuses ``~/.heylook/api_key`` / ``HEYLOOK_API_KEY``, browser session cookies,
    and always-on Origin/Sec-Fetch-Site/CSP guards so exposing Steering Lab behind
    a reverse proxy or subdomain shares identical security posture to ``heylook``.
    """
    try:
        from heylook_llm.auth import AccessGateMiddleware
        from heylook_llm.auth_api import auth_router
        from heylook_llm.frontend_static import FRONTEND_DIR
        from heylook_llm.host_check import HostCheckMiddleware
    except Exception:  # noqa: BLE001
        return False

    app.add_middleware(AccessGateMiddleware)
    app.add_middleware(HostCheckMiddleware)
    app.include_router(auth_router)

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Serve the exact four static assets listed in auth.EXEMPT_ROUTES for /login.
    login_assets = {
        "/css/app.css": FRONTEND_DIR / "css" / "app.css",
        "/js/login.js": FRONTEND_DIR / "js" / "login.js",
        "/icon.svg": FRONTEND_DIR / "icon.svg",
        "/favicon.ico": FRONTEND_DIR / "favicon.ico",
    }
    for route_path, asset_path in login_assets.items():
        if not asset_path.is_file():
            continue

        def _make_handler(p: Path) -> Any:
            def _serve() -> Any:
                return FileResponse(str(p))

            return _serve

        app.add_api_route(
            route_path,
            _make_handler(asset_path),
            methods=["GET", "HEAD"],
            include_in_schema=False,
        )
    return True


def create_app(state: Optional[LabState] = None, *, enable_access_gate: bool = True) -> FastAPI:
    lab = state or LabState()
    app = FastAPI(title="MLX Steering Lab", version="0.1.0")
    app.state.lab = lab
    gate_installed = _mount_heylook_access_gate(app) if enable_access_gate else False

    if UI_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        idx_file = UI_DIR / "index.html"
        if idx_file.is_file():
            return FileResponse(str(idx_file))
        return HTMLResponse("<h1>MLX Steering Lab</h1><p>UI directory missing.</p>")

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        rt = lab.runtime
        gate_on = False
        if gate_installed:
            try:
                from heylook_llm.auth import resolve_access_key

                gate_on = bool(resolve_access_key().enabled)
            except Exception:  # noqa: BLE001
                gate_on = False
        return {
            "model_loaded": rt.is_loaded,
            "default_model_path": lab.default_model_path,
            "fingerprint": rt.fingerprint.to_dict() if rt.fingerprint else None,
            "bank_loaded": rt.bank is not None,
            "bank_layers": list(rt.bank.layers) if rt.bank else [],
            "layer_kinds": {str(k): v for k, v in sorted(rt.layer_kinds().items())},
            "vector_count": len(rt.vectors),
            "vector_ids": sorted(rt.vectors.keys()),
            "hold_layers": sorted(rt.holds_by_layer.keys()),
            "corpora": list_corpora_summary(),
            "access_gate_enabled": gate_on,
        }

    @app.get("/api/vectors")
    def get_vectors() -> dict[str, Any]:
        items = []
        for cid, spec in sorted(lab.runtime.vectors.items()):
            meta = spec.public_meta()
            meta["layer_stats"] = {
                str(l): {
                    "rank": int(lv.vr.shape[0]),
                    "mu_d_norm": round(float(lv.mu_d_norm), 4),
                    "has_mu_clean": lv.mu_clean is not None,
                    "split_half_cos": lv.split_half_cos,
                    "flip_null_cos": lv.flip_null_cos,
                }
                for l, lv in sorted(spec.by_layer.items())
            }
            items.append(meta)
        return {"vectors": items, "hold_layers": sorted(lab.runtime.holds_by_layer.keys())}

    @app.get("/api/bank/geometry")
    def get_bank_geometry(span_kind: str = "answer") -> dict[str, Any]:
        if lab.runtime.bank is None:
            raise HTTPException(status_code=404, detail="No span bank loaded.")
        geom = lab.runtime.bank.geometry_by_layer(span_kind=span_kind)
        return {
            "fingerprint": lab.runtime.bank.fingerprint.to_dict(),
            "span_kind": span_kind,
            "layers": {str(k): v for k, v in sorted(geom.items())},
        }

    @app.post("/api/xray")
    def post_xray(req: XrayRequest) -> dict[str, Any]:
        spec = lab.runtime.vectors.get(req.concept_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown vector '{req.concept_id}'")
        lv = spec.by_layer.get(req.layer)
        if lv is None:
            raise HTTPException(
                status_code=400,
                detail=f"Vector '{req.concept_id}' has no layer {req.layer} (available: {list(spec.layers)})",
            )
        comp = req.component
        if comp == "mu_d":
            vec = lv.mu_d
        elif comp == "w_read" and lv.w_read is not None:
            vec = lv.w_read
        elif comp == "w_write" and lv.w_write is not None:
            vec = lv.w_write
        elif comp.startswith("vr:"):
            idx = int(comp.split(":", 1)[1])
            if idx < 0 or idx >= lv.vr.shape[0]:
                raise HTTPException(status_code=400, detail=f"vr index {idx} out of range")
            vec = lv.vr[idx]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported component '{comp}'")
        try:
            out = lab.runtime.xray_vector(vec, layer=req.layer, top_k=req.top_k, h_ref=lv.mu_clean)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        out["concept_id"] = req.concept_id
        out["component"] = comp
        return out

    @app.post("/api/chain-lens")
    def post_chain_lens(req: ChainLensRequest) -> dict[str, Any]:
        h_prior: Optional[np.ndarray] = None
        if req.component == "blend" or req.concept_id is None:
            active: list[tuple[LayerVector, float, float]] = []
            for cid, scale in sorted(req.scales.items()):
                if abs(float(scale)) <= 1e-8:
                    continue
                sp = lab.runtime.vectors.get(cid)
                if sp is None:
                    continue
                lv = sp.by_layer.get(req.layer)
                if lv is not None:
                    active.append((lv, float(scale), sp.erase_ratio))
                    if h_prior is None and lv.mu_clean is not None:
                        h_prior = lv.mu_clean
            hold = lab.runtime.holds_by_layer.get(req.layer) if req.use_holds else None
            composed = compose_regime_a(
                active,
                use_peer_ros=req.use_peer_ros,
                use_clamped_union=req.use_clamped_union,
                hold_basis=hold,
            )
            if composed is None:
                raise HTTPException(status_code=400, detail="No active sliders at this layer for blend.")
            target_vec = composed.mu_push * float(req.direction_sign)
            source_label = "multi-slider blend"
        else:
            spec = lab.runtime.vectors.get(req.concept_id)
            if spec is None:
                raise HTTPException(status_code=404, detail=f"Unknown vector '{req.concept_id}'")
            lv = spec.by_layer.get(req.layer)
            if lv is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vector '{req.concept_id}' has no layer {req.layer} (available: {list(spec.layers)})",
                )
            h_prior = lv.mu_clean
            if req.component == "mu_d":
                vec = lv.mu_d
            elif req.component == "w_read" and lv.w_read is not None:
                vec = lv.w_read
            elif req.component == "w_write" and lv.w_write is not None:
                vec = lv.w_write
            elif req.component.startswith("vr:"):
                idx = int(req.component.split(":", 1)[1])
                if idx < 0 or idx >= lv.vr.shape[0]:
                    raise HTTPException(status_code=400, detail=f"vr index {idx} out of range")
                vec = lv.vr[idx]
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported component '{req.component}'")
            target_vec = vec * float(req.direction_sign)
            source_label = f"{req.concept_id} ({req.component})"

        if h_prior is None and lab.runtime.bank is not None:
            h_prior = lab.runtime.bank.corpus_mean(req.layer)

        # Optional transport from mid-layer `req.layer` -> `terminal_layer` (`L63`):
        # either exact upper-layer forward (when live model loaded & requested) or Ridge J-Lens.
        num_layers = (
            lab.runtime.fingerprint.num_hidden_layers
            if lab.runtime.fingerprint is not None
            else (lab.runtime.bank.fingerprint.num_hidden_layers if lab.runtime.bank else 64)
        )
        terminal_layer = num_layers - 1
        jlens = None
        exact_upper = False
        target_frame_layer = req.layer

        if req.use_exact_upper_layers and lab.runtime.is_loaded and req.layer < terminal_layer:
            base_mid = np.zeros_like(target_vec) if h_prior is None else np.asarray(h_prior, dtype=np.float32)
            base_norm = float(np.linalg.norm(base_mid))
            v_norm = float(np.linalg.norm(target_vec))
            step_vec = (
                (float(req.probe_scale) * base_norm / v_norm) * target_vec
                if base_norm > 1e-6 and v_norm > 1e-8
                else float(req.probe_scale) * target_vec
            )
            h_base = lab.runtime.transport_upper_layers(base_mid, src_layer=req.layer, dst_layer=terminal_layer)
            h_target = lab.runtime.transport_upper_layers(
                base_mid + step_vec, src_layer=req.layer, dst_layer=terminal_layer
            )
            exact_upper = True
            target_frame_layer = terminal_layer
        else:
            if req.use_jlens and req.layer < terminal_layer and lab.runtime.bank is not None:
                paired = lab.runtime.bank.paired_layer_states(req.layer, terminal_layer)
                if paired is not None and paired[0].shape[0] >= 4:
                    jlens = fit_jlens_transport(
                        paired[0], paired[1], src_layer=req.layer, dst_layer=terminal_layer
                    )
                    target_frame_layer = terminal_layer
            h_target, h_base, _ = build_probe_pair(
                target_vec,
                h_prior=h_prior,
                probe_scale=req.probe_scale,
                jlens=jlens,
                norm_relative=True,
            )

        # Build peer bank directions & calibration states in `target_frame_layer` for
        # anisotropy-resistant round-trip rank + percentile checks.
        bank_dirs: dict[str, np.ndarray] = {}
        for cid, sp in lab.runtime.vectors.items():
            other_lv = sp.by_layer.get(req.layer)
            if other_lv is None:
                continue
            dvec = other_lv.mu_d
            if jlens is not None:
                dvec = jlens.transport_vector(dvec)
            bank_dirs[cid] = dvec
        cal_states = (
            lab.runtime.bank.corpus_states(target_frame_layer)
            if lab.runtime.bank is not None
            else None
        )

        draft_model = getattr(lab.runtime, "draft_model", None)
        embed_fn = None
        if lab.runtime.is_loaded and draft_model is not None:
            adapter = Qwen35MTPAdapter(lab.runtime.lm, draft_model)
            mode = "live_mtp"
            # Forward the 3..5 token phrase to `target_frame_layer` (NOT Layer-0 embed_tokens!)
            # so round-trip cosine compares states in the exact same residual coordinate frame.
            embed_fn = lambda tids: lab.runtime.encode_phrase_hidden(tids, layer=target_frame_layer)
        elif lab.runtime.bank is not None and lab.runtime.bank.xray_lm_head is not None:
            lm_head = lab.runtime.bank.xray_lm_head
            adapter = LinearDemoMTPAdapter(lm_head)
            mode = "demo_mtp"
            embed_fn = lambda tids: lm_head[np.asarray(list(tids), dtype=np.int64) % lm_head.shape[0]].mean(axis=0)
        else:
            raise HTTPException(
                status_code=409,
                detail="No MTP head or demo bank lm_head available for Chain Lens rollout.",
            )

        out = run_chain_lens(
            h_target,
            adapter=adapter,
            decode_token=lab.runtime._decode_token,
            h_baseline=h_base,
            steps=req.steps,
            beam_width=req.beam_width,
            alpha=req.alpha,
            plausibility_beta=req.plausibility_beta,
            embed_tokens_fn=embed_fn,
            bank_directions=bank_dirs,
            target_direction_key=req.concept_id,
            calibration_states=cal_states,
        )
        out["mode"] = mode
        out["layer"] = req.layer
        out["target_frame_layer"] = target_frame_layer
        out["layer_kind"] = lab.runtime.layer_kinds().get(req.layer, "FA" if req.layer % 4 == 3 else "GDN")
        out["source"] = source_label
        out["direction_sign"] = req.direction_sign
        out["jlens_applied"] = jlens is not None
        out["jlens_train_r2"] = None if jlens is None else jlens.train_r2
        out["jlens_val_r2"] = None if jlens is None else jlens.val_r2
        out["jlens_sample_count"] = None if jlens is None else jlens.sample_count
        out["exact_upper_applied"] = exact_upper
        out["used_corpus_prior"] = h_prior is not None
        return out

    @app.post("/api/compose")
    def post_compose(req: ComposeRequest) -> dict[str, Any]:
        active: list[tuple[LayerVector, float, float]] = []
        for cid, scale in sorted(req.scales.items()):
            if abs(float(scale)) <= 1e-8:
                continue
            spec = lab.runtime.vectors.get(cid)
            if spec is None or spec.operator != "regime_a":
                continue
            lv = spec.by_layer.get(req.layer)
            if lv is None:
                continue
            if req.use_s3_random:
                lv = build_s3_random_layer(lv, seed=req.s3_seed)
            active.append((lv, float(scale), spec.erase_ratio))

        hold = lab.runtime.holds_by_layer.get(req.layer) if req.use_holds else None
        composed = compose_regime_a(
            active,
            use_peer_ros=req.use_peer_ros,
            use_clamped_union=req.use_clamped_union,
            hold_basis=hold,
            knee=req.knee,
            push_mode=req.push_mode,
            norm_matched_frac=req.norm_matched_frac,
        )
        if composed is None:
            return {"active": False, "layer": req.layer, "xray": None}

        xray = None
        try:
            xray = lab.runtime.xray_vector(composed.mu_push, layer=req.layer, top_k=req.top_k)
        except RuntimeError:
            xray = None
        return {
            "active": True,
            "layer": req.layer,
            "union_rank": int(composed.u_erase.shape[0]),
            "max_raw_eig": round(float(composed.max_raw_eig), 4),
            "eigs": [round(float(e), 4) for e in composed.eig_erase.tolist()],
            "clamped_eigs": composed.clamped_eigs,
            "peer_ros_applied": composed.peer_ros_applied,
            "holds_applied": composed.holds_applied,
            "push_mode": composed.push_mode,
            "push_norm": round(float(np.linalg.norm(composed.mu_push)), 4),
            "xray": xray,
        }

    @app.post("/api/sweep")
    def post_sweep(req: SweepRequest) -> dict[str, Any]:
        if not req.arms:
            raise HTTPException(status_code=400, detail="At least one sweep arm is required.")
        if not lab.runtime.is_loaded:
            return _demo_synthetic_sweep(lab, req)
        plans = [_plan_from_payload(a) for a in req.arms]
        traces = lab.runtime.sweep(
            req.prompt,
            plans,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            seed=req.seed,
            enable_thinking=req.enable_thinking,
            reasoning_effort=req.reasoning_effort,
        )
        return {"mode": "live_mlx", "arms": [t.to_dict() for t in traces]}

    @app.post("/api/token-delta")
    def post_token_delta(req: TokenDeltaRequest) -> dict[str, Any]:
        if not lab.runtime.is_loaded:
            # Deterministic synthetic demo delta when exploring without a loaded 28 GB checkpoint.
            vocab = (
                lab.runtime.bank.xray_vocab
                if lab.runtime.bank is not None and lab.runtime.bank.xray_vocab
                else [" concrete", " direct", " mechanism", " perhaps", " tapestry", " delve"]
            )
            return {
                "mode": "demo_synthetic",
                "target_index": req.target_index,
                "causal": False,
                "note": "Demo readout (load an MLX model for live causal Δ logprob).",
                "adds": [
                    {"token_id": i, "token": vocab[i % len(vocab)], "delta_logprob": round(1.8 - 0.25 * i, 3)}
                    for i in range(min(6, len(vocab)))
                ],
                "erases": [
                    {
                        "token_id": (i + 18) % len(vocab),
                        "token": vocab[(i + 18) % len(vocab)],
                        "delta_logprob": round(-2.1 + 0.22 * i, 3),
                    }
                    for i in range(min(6, len(vocab)))
                ],
            }
        plan = _plan_from_payload(req.arm)
        out = lab.runtime.token_delta_logprob(
            req.prompt,
            req.token_ids,
            req.target_index,
            plan=plan,
            system=req.system,
            enable_thinking=req.enable_thinking,
            top_k=req.top_k,
        )
        out["mode"] = "live_mlx"
        return out

    @app.post("/api/model/load")
    def post_model_load(req: LoadModelRequest) -> dict[str, Any]:
        p = Path(req.model_path).expanduser()
        if not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Model directory not found: {p}")
        fp = lab.runtime.load_model_dir(p, model_id=req.model_id)
        return {"loaded": True, "fingerprint": fp.to_dict()}

    @app.post("/api/model/unload")
    def post_model_unload() -> dict[str, Any]:
        lab.runtime.unload_model()
        return {"loaded": False}

    return app
