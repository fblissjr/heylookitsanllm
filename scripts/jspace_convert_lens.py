#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "huggingface_hub",
#     "jlens @ git+https://github.com/anthropics/jacobian-lens",
# ]
# ///
"""Convert a Jacobian-lens ``.pt`` into an mx-safetensors lens the j-space
endpoint can load, and register it at ``adapters/jspace/<model_id>/``.

The output dir name is the SERVED model id -- that's the key
``/v1/jspace/analyze`` looks up (``adapters/jspace`` is the registry default;
override with ``HEYLOOK_JSPACE_DIR``). ``adapters/`` is git-tracked (`.gitkeep`)
but its contents are gitignored, like ``modelzoo/``.

Needs torch + jlens, which are NOT in the MLX server venv. This file carries
inline (PEP 723) deps, so run it with `uv run <script>` (NOT `uv run python
<script>`) and uv provisions them in an isolated env:

    uv run scripts/jspace_convert_lens.py \\
        --hf-repo solarkyle/jspace-lenses --hf-file gemma-4-26b-a4b-it/lens.pt \\
        --model-id gemma-4-26b-a4b-it-8bit-mlx --softcap 30

Or from a local .pt (or a directory containing exactly one *_jacobian_lens.pt /
lens.pt, e.g. a neuronpedia model dir):

    uv run scripts/jspace_convert_lens.py \\
        --lens-pt path/to/neuronpedia/qwen3-32b --model-id Qwen3.5-27B-abliterated-8bit-mlx

``--model-id`` is a bare name (the served model id), NOT a path. See
docs/jspace_guide.md.
"""
import argparse
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_lens_pt(ap, lens_pt: str) -> str:
    """Accept a .pt file or a directory containing exactly one lens .pt."""
    p = Path(lens_pt).expanduser()
    if p.is_file():
        return str(p)
    if p.is_dir():
        cands = [c for c in glob.glob(str(p / "**" / "*.pt"), recursive=True)
                 if "lens" in Path(c).name.lower()]
        if len(cands) == 1:
            print(f"found lens file in directory: {cands[0]}")
            return cands[0]
        ap.error(f"--lens-pt {lens_pt!r} is a directory with {len(cands)} lens .pt "
                 f"files (need exactly 1): {cands}")
    ap.error(f"--lens-pt {lens_pt!r} is not a file or directory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True,
                    help="served model id = output subdir under --out-dir (a bare name, not a path)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lens-pt", help="local jlens .pt, or a dir containing exactly one")
    src.add_argument("--hf-repo", help="HF repo id, e.g. solarkyle/jspace-lenses")
    ap.add_argument("--hf-file", help="path within --hf-repo (required with --hf-repo)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "adapters" / "jspace"),
                    help="registry base dir (default: <repo>/adapters/jspace)")
    ap.add_argument("--hf-name", default="",
                    help="HF model name the lens was fit on (metadata only)")
    ap.add_argument("--softcap", type=float, default=None,
                    help="final_logit_softcapping (metadata; the live model is authoritative)")
    args = ap.parse_args()

    if "/" in args.model_id or "\\" in args.model_id:
        ap.error(f"--model-id must be a bare name (the served model id), not a path: "
                 f"{args.model_id!r}. The lens is written to --out-dir/<model-id>/.")

    try:
        import torch  # noqa: F401
        import jlens
        from safetensors.torch import save_file
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"missing dependency ({e.name}). Run this with uv so its inline deps are "
            f"provisioned:\n    uv run scripts/jspace_convert_lens.py ...\n"
            f"(NOT `uv run python ...`, which uses the torch-less server venv).")

    if args.hf_repo:
        if not args.hf_file:
            ap.error("--hf-file is required with --hf-repo")
        from huggingface_hub import hf_hub_download
        lens_pt = hf_hub_download(args.hf_repo, args.hf_file)
    else:
        lens_pt = _resolve_lens_pt(ap, args.lens_pt)

    lens = jlens.JacobianLens.load(lens_pt)
    # <out-dir>/<model-id>/. Tolerate the common mistake of passing --out-dir
    # already ending in the model id (else it double-nests and the registry,
    # which looks one level up, never sees the lens).
    out = Path(args.out_dir)
    if out.name != args.model_id:
        out = out / args.model_id
    out.mkdir(parents=True, exist_ok=True)
    # Serialize the sidecar FIRST so a non-JSON value can't leave an orphan
    # lens.safetensors (the registry requires BOTH files anyway).
    sidecar = json.dumps({
        "model_id": args.model_id,
        "hf_model_name": args.hf_name,
        "source_layers": [int(l) for l in lens.source_layers],
        "d_model": int(lens.d_model),
        "n_prompts": int(lens.n_prompts),
        "final_logit_softcapping": args.softcap,
        "apply": "unembed(residual @ J[l].T)",
    }, indent=2)
    save_file({str(l): lens.jacobians[l].contiguous().float() for l in lens.source_layers},
              str(out / "lens.safetensors"))
    (out / "lens.sidecar.json").write_text(sidecar)
    print(f"wrote {out}/lens.safetensors  "
          f"(d_model={lens.d_model}, layers={len(lens.source_layers)}, n_prompts={lens.n_prompts})")
    print(f"-> served as model_id {args.model_id!r} via GET /v1/jspace/models")


if __name__ == "__main__":
    main()
