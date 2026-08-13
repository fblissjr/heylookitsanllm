#!/usr/bin/env python3
"""Convert a local HF safetensors checkpoint to GGUF (text + optional mmproj).

The "easy way" wrapper around the canonical llama.cpp checkout's
convert_hf_to_gguf.py (same tree scripts/build_llama.py maintains -- one
checkout, one source). Zero footprint on the repo venv: torch rides a
`uv run --with` overlay for the conversion only.

    uv run python scripts/convert_gguf.py <hf-checkpoint-dir> \
        --name Muse-Glimmer-30B --dest modelzoo/meta/Muse-Glimmer-30B-GGUF

Produces <dest>/<name>-<OUTTYPE>.gguf (default q8_0 -- the near-lossless
choice on a RAM-rich box; convert_hf_to_gguf quantizes q8_0 directly, no
llama-quantize needed) and, unless --no-mmproj, <dest>/mmproj-<name>-f16.gguf
for models with a vision/perception encoder.

Honesty notes:
- Converter support for an architecture lands in llama.cpp's PYTHON side
  separately from the C++ inference side; a brand-new arch may need the
  checkout at master (`uv run python scripts/build_llama.py --rev master`).
  Failures from the converter surface verbatim -- this wrapper adds nothing.
- After converting, verify with the probe before registering:
      uv run python scripts/gguf_probe.py <dest>
  then add a models.toml entry (or `heylookllm import --scan <dest>`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HOME_SUBDIR = (".heylook", "llama.cpp")  # keep in sync with build_llama.py


def checkout_dir() -> Path:
    env = os.environ.get("HEYLOOK_LLAMA_CPP_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*HOME_SUBDIR).resolve()


def run_convert(checkout: Path, src: Path, outfile: Path, outtype: str,
                mmproj: bool) -> None:
    convert = checkout / "convert_hf_to_gguf.py"
    if not convert.is_file():
        sys.exit(f"no converter at {convert} -- run scripts/build_llama.py first")
    # --no-project: the converter must NOT inherit the repo venv -- the
    # repo pins transformers (deliberately, contract-tested), and a new
    # checkpoint's tokenizer_class can postdate that pin (Muse-Glimmer's
    # "TokenizersBackend" did, 2026-08-13). The converter's world is
    # llama.cpp's, not heylook's: current transformers, its own gguf-py.
    cmd = [
        "uv", "run", "--no-project",
        "--with", "torch", "--with", "transformers",
        "--with", "sentencepiece", "--with", "safetensors",
        "python", str(convert), str(src),
        "--outfile", str(outfile), "--outtype", outtype,
    ]
    if mmproj:
        cmd.append("--mmproj")
    print(f"$ {' '.join(cmd)}")
    # cwd = the checkout so its own gguf-py is what the converter imports
    subprocess.run(cmd, cwd=checkout, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("src", help="local HF checkpoint dir (config.json + safetensors)")
    ap.add_argument("--name", help="output stem (default: src dir name)")
    ap.add_argument("--dest", help="output dir (default: modelzoo/<name>-GGUF)")
    ap.add_argument("--outtype", default="q8_0",
                    choices=["f32", "f16", "bf16", "q8_0", "auto"],
                    help="text-model quantization (default q8_0)")
    ap.add_argument("--no-mmproj", action="store_true",
                    help="skip the vision/perception encoder conversion")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not (src / "config.json").is_file():
        sys.exit(f"{src} has no config.json -- point at the HF checkpoint dir")
    name = args.name or src.name
    dest = Path(args.dest).expanduser().resolve() if args.dest \
        else Path("modelzoo") / f"{name}-GGUF"
    dest.mkdir(parents=True, exist_ok=True)
    checkout = checkout_dir()

    run_convert(checkout, src, dest / f"{name}-{args.outtype.upper()}.gguf",
                args.outtype, mmproj=False)
    if not args.no_mmproj:
        # vision towers stay f16: quantized projector quality is untested
        # territory and the file is small relative to the text model
        run_convert(checkout, src, dest / f"mmproj-{name}-f16.gguf",
                    "f16", mmproj=True)

    print(f"\ndone -> {dest}")
    print(f"verify:  uv run python scripts/gguf_probe.py {dest}")


if __name__ == "__main__":
    main()
