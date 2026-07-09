#!/usr/bin/env python
"""Convert a Jacobian-lens ``.pt`` into an mx-safetensors lens the j-space
endpoint can load, and register it at ``adapters/jspace/<model_id>/``.

The output dir name is the SERVED model id -- that's the key
``/v1/jspace/analyze`` looks up (``adapters/jspace`` is the registry default;
override with ``HEYLOOK_JSPACE_DIR``). ``adapters/`` is git-tracked (`.gitkeep`)
but its contents are gitignored, like ``modelzoo/``.

Needs torch + jlens, which are NOT in the MLX server venv -- run in a separate
throwaway env, e.g.:

    uv run --with torch --with safetensors --with huggingface_hub \\
        --with "jlens @ git+https://github.com/anthropics/jacobian-lens" \\
        python scripts/jspace_convert_lens.py \\
        --hf-repo solarkyle/jspace-lenses --hf-file gemma-4-26b-a4b-it/lens.pt \\
        --model-id gemma-4-26b-a4b-it-8bit-mlx --softcap 30

or from a local .pt:

    python scripts/jspace_convert_lens.py --lens-pt path/to/lens.pt \\
        --model-id my-model-id

See docs/jspace_integration_plan.md.
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True,
                    help="served model id = output subdir under --out-dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lens-pt", help="local path to the jlens .pt")
    src.add_argument("--hf-repo", help="HF repo id, e.g. solarkyle/jspace-lenses")
    ap.add_argument("--hf-file", help="path within --hf-repo (required with --hf-repo)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "adapters" / "jspace"),
                    help="registry base dir (default: <repo>/adapters/jspace)")
    ap.add_argument("--hf-name", default="",
                    help="HF model name the lens was fit on (metadata only)")
    ap.add_argument("--softcap", type=float, default=None,
                    help="final_logit_softcapping (metadata; the live model is authoritative)")
    args = ap.parse_args()

    import torch  # noqa: F401
    import jlens
    from safetensors.torch import save_file

    if args.hf_repo:
        if not args.hf_file:
            ap.error("--hf-file is required with --hf-repo")
        from huggingface_hub import hf_hub_download
        lens_pt = hf_hub_download(args.hf_repo, args.hf_file)
    else:
        lens_pt = args.lens_pt

    lens = jlens.JacobianLens.load(lens_pt)
    out = Path(args.out_dir) / args.model_id
    out.mkdir(parents=True, exist_ok=True)
    # Serialize the sidecar FIRST so a non-JSON value can't leave an orphan
    # lens.safetensors (the registry requires BOTH files, but this avoids the
    # confusing half-written state entirely).
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
