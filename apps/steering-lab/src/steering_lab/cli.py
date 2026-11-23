"""CLI entrypoint for the standalone MLX Steering Lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .bank import SpanBank, build_packs_from_bank, create_demo_bank, load_vector_spec
from .chain_lens import (
    LinearDemoMTPAdapter,
    build_probe_pair,
    fit_jlens_transport,
    run_chain_lens,
)
from .runtime import DEFAULT_QWEN38_LAYERS, SteeringLabRuntime, SteeringPlan
from .server import LabState, create_app


def _resolve_model_path(model_arg: str) -> Path:
    p = Path(model_arg).expanduser()
    if p.is_dir():
        return p
    # Fall back to repo heylook.toml scan folders if available (never hardcode user paths).
    try:
        import tomllib

        repo_toml = Path(__file__).resolve().parents[4] / "heylook.toml"
        if repo_toml.is_file():
            cfg = tomllib.loads(repo_toml.read_text(encoding="utf-8"))
            for folder in (cfg.get("scan") or {}).get("folders") or []:
                root = Path(folder).expanduser()
                if not root.is_dir():
                    continue
                for cand in root.rglob("config.json"):
                    if cand.parent.name == model_arg:
                        return cand.parent
    except Exception:  # noqa: BLE001
        pass
    raise FileNotFoundError(
        f"Model '{model_arg}' not found as a directory or under heylook.toml [scan].folders."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="steering-lab", description="MLX Activation Steering Lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo-bank", help="Generate CPU-only synthetic demo SpanBank (zero GPU)")
    p_demo.add_argument("--out", required=True, help="Output directory for demo bank")
    p_demo.add_argument("--dim", type=int, default=64)
    p_demo.add_argument("--seed", type=int, default=42)

    p_cap = sub.add_parser("capture", help="Capture multi-layer raw spans from a live MLX model")
    p_cap.add_argument("--model", required=True, help="Model ID or local directory")
    p_cap.add_argument("--out", required=True, help="Output SpanBank directory")
    p_cap.add_argument("--corpus", default="all", help="Comma-separated corpus IDs or 'all'")
    p_cap.add_argument(
        "--layers",
        default=",".join(str(x) for x in DEFAULT_QWEN38_LAYERS),
        help="Comma-separated layer indices",
    )
    p_cap.add_argument("--max-tokens", type=int, default=896)
    p_cap.add_argument("--max-tokens-per-span", type=int, default=24)
    p_cap.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Reasoning effort passed to chat template (default: medium; use 'none' to omit)",
    )
    p_cap.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable <think> generation during span capture",
    )
    p_cap.add_argument("--image-dir", help="Optional folder of real photos/screenshots for VLM Dual-Lock capture")
    p_cap.add_argument("--seed", type=int, default=42)

    p_build = sub.add_parser("build", help="Build Regime A/B/Conditional packs from a cached SpanBank (CPU)")
    p_build.add_argument("--bank", required=True, help="Input SpanBank directory")
    p_build.add_argument("--out", required=True, help="Output vectors root directory")
    p_build.add_argument("--rank", type=int, default=8)
    p_build.add_argument("--hold-rank", type=int, default=16)
    p_build.add_argument("--unsupervised-ac", action="store_true", help="Use SVD(A_c) instead of paired vr_diff")

    p_insp = sub.add_parser("inspect-bank", help="Print cross-layer bank rank & cosine diagnostics (CPU)")
    p_insp.add_argument("--bank", required=True)
    p_insp.add_argument("--span", default="answer")

    p_xray = sub.add_parser("xray", help="Project a vector through lm_head (demo bank or loaded model)")
    p_xray.add_argument("--vector-dir", required=True)
    p_xray.add_argument("--layer", type=int, required=True)
    p_xray.add_argument("--bank", help="Demo SpanBank providing xray_lm_head (CPU)")
    p_xray.add_argument("--model", help="Live MLX model ID or directory")
    p_xray.add_argument("--top-k", type=int, default=12)

    p_cl = sub.add_parser("chain-lens", help="Multi-token contrastive MTP Chain Lens readout")
    p_cl.add_argument("--vector-dir", required=True)
    p_cl.add_argument("--layer", type=int, required=True)
    p_cl.add_argument("--bank", required=True, help="SpanBank providing demo lm_head/vocab (CPU)")
    p_cl.add_argument("--steps", type=int, default=4)
    p_cl.add_argument("--beam-width", type=int, default=6)
    p_cl.add_argument("--alpha", type=float, default=0.75)
    p_cl.add_argument("--sign", type=float, default=1.0)
    p_cl.add_argument("--no-jlens", action="store_true", help="Disable J-Lens ridge transport to terminal layer")

    p_serve = sub.add_parser("serve", help="Launch the local Steering Lab FastAPI + UI workbench")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8111)
    p_serve.add_argument("--bank", help="SpanBank directory to open at startup")
    p_serve.add_argument("--vectors", help="Built vector packs root directory")
    p_serve.add_argument("--model", help="Default model ID/path shown in Load control (loaded lazily)")

    args = parser.parse_args(argv)

    if args.cmd == "demo-bank":
        bank = create_demo_bank(Path(args.out), dim=args.dim, seed=args.seed)
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "fingerprint": bank.fingerprint.to_dict(),
                    "layers": list(bank.layers),
                    "concepts": sorted(bank.concepts.keys()),
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "build":
        bank = SpanBank.load(Path(args.bank))
        res = build_packs_from_bank(
            bank,
            Path(args.out),
            rank=args.rank,
            hold_rank=args.hold_rank,
            use_paired_diff=not args.unsupervised_ac,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "inspect-bank":
        bank = SpanBank.load(Path(args.bank))
        geom = bank.geometry_by_layer(span_kind=args.span)
        for layer, row in sorted(geom.items()):
            kind = row.get("layer_kind", bank.layer_kind(layer))
            print(
                f"L{layer:02d} [{kind:3s}] | concepts={row['count']} | eff_rank={row['effective_rank']:5.2f} | "
                f"top5={row['top5_pc_energy']:.3f} | mean|cos|={row['mean_abs_cos']:.3f} | "
                f"p90|cos|={row['p90_abs_cos']:.3f}"
            )
        return 0

    if args.cmd == "capture":
        model_dir = _resolve_model_path(args.model)
        rt = SteeringLabRuntime()
        rt.load_model_dir(model_dir, model_id=model_dir.name)
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        cids = None if args.corpus == "all" else [x.strip() for x in args.corpus.split(",") if x.strip()]
        effort = None if str(args.reasoning_effort).lower() in ("none", "") else str(args.reasoning_effort)
        bank = rt.capture_span_bank(
            Path(args.out),
            corpus_ids=cids,
            layers=layers,
            max_tokens=args.max_tokens,
            max_tokens_per_span=args.max_tokens_per_span,
            enable_thinking=not args.no_thinking,
            reasoning_effort=effort,
            image_dir=Path(args.image_dir) if args.image_dir else None,
            seed=args.seed,
        )
        print(f"Captured {len(bank.tensors)} span tensors to {args.out}")
        return 0

    if args.cmd == "xray":
        spec, _ = load_vector_spec(Path(args.vector_dir))
        lv = spec.by_layer[args.layer]
        rt = SteeringLabRuntime()
        if args.bank:
            rt.bank = SpanBank.load(Path(args.bank))
        if args.model:
            rt.load_model_dir(_resolve_model_path(args.model))
        out = rt.xray_vector(lv.mu_d, layer=args.layer, top_k=args.top_k, h_ref=lv.mu_clean)
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "chain-lens":
        import numpy as np

        spec, _ = load_vector_spec(Path(args.vector_dir))
        lv = spec.by_layer[args.layer]
        bank = SpanBank.load(Path(args.bank))
        if bank.xray_lm_head is None:
            raise RuntimeError("Demo bank has no xray_lm_head; use live server with MTP head.")
        rt = SteeringLabRuntime()
        rt.bank = bank
        terminal = bank.fingerprint.num_hidden_layers - 1
        jlens = None
        if not args.no_jlens and args.layer < terminal:
            paired = bank.paired_layer_states(args.layer, terminal)
            if paired is not None and paired[0].shape[0] >= 4:
                jlens = fit_jlens_transport(paired[0], paired[1], src_layer=args.layer, dst_layer=terminal)
        h_prior = lv.mu_clean if lv.mu_clean is not None else bank.corpus_mean(args.layer)
        h_target, h_base, _ = build_probe_pair(
            lv.mu_d * float(args.sign), h_prior=h_prior, jlens=jlens
        )
        lm_head = bank.xray_lm_head
        adapter = LinearDemoMTPAdapter(lm_head)
        out = run_chain_lens(
            h_target,
            adapter=adapter,
            decode_token=rt._decode_token,
            h_baseline=h_base,
            steps=args.steps,
            beam_width=args.beam_width,
            alpha=args.alpha,
            embed_tokens_fn=lambda tids: lm_head[np.asarray(list(tids), dtype=np.int64) % lm_head.shape[0]].mean(axis=0),
            calibration_states=bank.corpus_states(terminal if jlens is not None else args.layer),
        )
        out["jlens_applied"] = jlens is not None
        out["jlens_train_r2"] = None if jlens is None else jlens.train_r2
        out["jlens_val_r2"] = None if jlens is None else jlens.val_r2
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "serve":
        import uvicorn

        default_model = None
        if args.model:
            try:
                default_model = str(_resolve_model_path(args.model))
            except FileNotFoundError:
                default_model = args.model
        state = LabState(
            bank_dir=Path(args.bank) if args.bank else None,
            vectors_dir=Path(args.vectors) if args.vectors else None,
            default_model_path=default_model,
        )
        app = create_app(state)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
