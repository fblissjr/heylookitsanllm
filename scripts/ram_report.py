#!/usr/bin/env python3
"""Memory pre-flight: what is holding RAM, and will this model fit?

Two jobs, because on this hardware they are the same question:

1. **Report** the ceilings and who is under them. "Total RAM" is the wrong
   number to reason with -- see the Metal note below.
2. **Size** a model the way it is actually loaded (whole shard SET, plus the
   mmproj and drafter sidecars that land in the same process) and check it
   against every ceiling that can refuse it.

The Metal ceiling is the one that surprises people. On a 192 GiB M2 Ultra the
GPU's ``max_recommended_working_set_size`` is ~161 GiB, not 192 -- a limit that
has nothing to do with free RAM, and that ``iogpu.wired_limit_mb`` (default 0)
can raise. What it MEANS depends on the engine, so this script reports it per
engine rather than as one verdict:

- **MLX** treats it as hard. server.py sets ``mx.set_wired_limit`` to exactly
  the recommendation, and MLX refuses any larger value outright.
- **llama.cpp** wires through the same ``MTLResidencySet`` (on by default;
  ``GGML_METAL_NO_RESIDENCY=1`` disables, ``GGML_METAL_RESIDENCY_KEEP_ALIVE_S``
  defaults to 180 s) but checks the recommendation only as a debug-build
  warning. Past the line the model still loads -- Metal just stops guaranteeing
  residency and you degrade into paging. A performance warning, not a refusal.

``max_buffer_length`` (~121 GiB) is a separate per-allocation cap; sharded
GGUFs stay under it naturally, one giant single-file model may not.

Sizing traps this script exists to not repeat:

- A GGUF ``model_path`` points at ONE shard. ``DeepSeek-V4-...-00001-of-00005``
  is a 5 MB index shard; the set behind it is ~127 GiB. Sizing the named file is
  wrong by four orders of magnitude.
- ``mmproj_path`` and ``draft_model_path`` load into the same llama-server
  process and must be counted.

Usage::

    uv run python scripts/ram_report.py                     # ceilings + holders
    uv run python scripts/ram_report.py --model <id>        # + fit check
    uv run python scripts/ram_report.py --path <dir|gguf>   # + fit, unimported
    uv run python scripts/ram_report.py --model <id> --quiet --headroom 8

``--quiet`` prints one line and exits 1 when the model does not fit, which is
the form ``dev_server.sh`` consumes.
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Optional

GB = 1024 ** 3

# llama.cpp shard naming: `<prefix>-00001-of-00005.gguf`. Mirrors
# model_importer._SHARD_RE -- kept local so this script stays runnable
# without importing the server package.
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------

def available_gb() -> float:
    """RAM the OS can hand out now: free + inactive + purgeable.

    Matches dev_server.sh's vm_stat arithmetic. macOS "used" includes file
    cache it will evict on demand, so `total - used` understates headroom.
    """
    import psutil

    vm = psutil.virtual_memory()
    return vm.available / GB


def sysctl_wired_limit_mb() -> Optional[int]:
    """`iogpu.wired_limit_mb`, the SYSTEM-wide GPU wired ceiling.

    0 means "OS default" (~84% of total on a 192 GiB M2 Ultra). This is the
    only lever that RAISES the Metal working set -- heylook's own
    ``mx.set_wired_limit`` at startup consumes that budget for MLX, it does
    not enlarge it, and it has no effect at all on a llama-server subprocess.
    Returns None if the OID is unreadable (non-Apple-Silicon, or sandboxed).
    """
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def metal_ceilings() -> Optional[dict]:
    """GPU working-set limits, or None off Metal (or if MLX is unavailable).

    Returned in GiB. These are the limits llama.cpp/MLX actually allocate
    against; neither is derivable from total RAM.
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
    except Exception:
        return None
    working_set = info.get("max_recommended_working_set_size")
    if not working_set:
        return None
    return {
        "device": info.get("device_name", "?"),
        "working_set_gb": working_set / GB,
        "max_buffer_gb": (info.get("max_buffer_length") or 0) / GB,
        "sysctl_wired_mb": sysctl_wired_limit_mb(),
    }


# ---------------------------------------------------------------------------
# Who is holding it
# ---------------------------------------------------------------------------

_FAMILY_SUFFIXES = (" Helper (Renderer)", " Helper (GPU)", " Helper (Plugin)", " Helper")


def _family(name: str) -> str:
    """Roll `Google Chrome Helper (Renderer)` up into `Google Chrome`."""
    for suffix in _FAMILY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def top_holders(limit: int = 12) -> list[tuple[str, float, int]]:
    """(family, summed RSS GiB, process count), biggest first.

    RSS double-counts shared pages, so these sum to more than the machine
    holds. It is a ranking of who to close, not an accounting identity.
    """
    import psutil

    totals: dict[str, list] = {}
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            rss = proc.info["memory_info"].rss
            fam = _family(proc.info["name"] or "?")
        except Exception:
            continue
        entry = totals.setdefault(fam, [0, 0])
        entry[0] += rss
        entry[1] += 1
    ranked = sorted(totals.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(fam, rss / GB, n) for fam, (rss, n) in ranked[:limit]]


# ---------------------------------------------------------------------------
# Model sizing
# ---------------------------------------------------------------------------

def _shard_set_bytes(f: Path) -> int:
    """Bytes of the whole split set ``f`` belongs to, else ``f``'s own size."""
    m = _SHARD_RE.search(f.name)
    if m is None:
        return f.stat().st_size
    prefix = f.name[: m.start()]
    return sum(
        s.stat().st_size
        for s in f.parent.glob(f"{_glob.escape(prefix)}-*-of-*.gguf")
    )


def is_mlx_config(config: dict) -> bool:
    """True when this entry loads in-process through MLX.

    Decides whether the Metal working set is a hard ceiling or advisory --
    a `.gguf` model_path means a llama-server subprocess, anything else
    (a weights DIR) means MLX. Cheap and layout-based on purpose: no
    models.toml `provider` field is needed, so `--path` works too.
    """
    return not str(config.get("model_path") or "").lower().endswith(".gguf")


def size_config_gb(config: dict) -> tuple[float, list[str]]:
    """Resident weight bytes for a models.toml ``config`` block, plus notes.

    Counts the primary (whole shard set, not the named shard) and every
    sidecar that loads into the same process.
    """
    notes: list[str] = []
    total = 0
    primary = Path(config.get("model_path") or "")
    if primary.is_dir():
        total += sum(
            f.stat().st_size
            for pattern in ("*.safetensors", "*.gguf")
            for f in primary.rglob(pattern)
        )
    elif primary.is_file():
        shard_bytes = _shard_set_bytes(primary)
        if _SHARD_RE.search(primary.name):
            n = len(list(primary.parent.glob("*-of-*.gguf")))
            notes.append(f"primary is a {n}-shard set ({shard_bytes / GB:.1f} GiB), not the named shard")
        total += shard_bytes
    else:
        return 0.0, ["model_path does not exist"]

    for key, label in (("mmproj_path", "mmproj"), ("draft_model_path", "drafter")):
        sidecar = Path(config.get(key) or "")
        if sidecar.is_file():
            size = _shard_set_bytes(sidecar)
            total += size
            notes.append(f"+{label} {sidecar.name} ({size / GB:.1f} GiB)")
    return total / GB, notes


def load_model_config(model_id: str, models_toml: Path) -> Optional[dict]:
    try:
        doc = tomllib.loads(models_toml.read_text())
    except OSError:
        return None
    for entry in doc.get("models", []):
        if entry.get("id") == model_id:
            return entry.get("config", {})
    return None


def config_from_path(path: Path) -> dict:
    """Synthesise a config for an un-imported dir, reusing the real pickers.

    Deliberately the IMPORTER's logic, so this script and an actual import
    agree on which shard is primary and which sidecar is the drafter.
    """
    if path.is_file():
        return {"model_path": str(path)}
    try:
        from heylook_llm.model_importer import ModelImporter

        importer = ModelImporter()
        primary = importer._pick_primary_gguf(path)
        if primary is None:
            return {"model_path": str(path)}  # MLX dir: size the whole tree
        config = {"model_path": str(primary)}
        if (mmproj := importer._pick_mmproj(path)) is not None:
            config["mmproj_path"] = str(mmproj)
        # `primary` matters: for a per-quant variant folder the drafter lives
        # at the repo root one level up, and the picker needs the primary to
        # know it is in one.
        if (draft := importer._pick_draft(path, primary)) is not None:
            config["draft_model_path"] = str(draft)
        return config
    except Exception:
        return {"model_path": str(path)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def check_fit(size_gb: float, headroom_gb: float, hard_working_set: bool = True) -> tuple[bool, list[str]]:
    """(fits, one line per ceiling).

    ``hard_working_set`` distinguishes the two engines, which treat the Metal
    recommendation differently and must not be reported the same way:

    - **MLX (True)**: server.py calls ``mx.set_wired_limit`` at exactly the
      recommendation, and MLX REFUSES anything above it
      ("Setting a wired limit larger than the maximum working set size is not
      allowed"). Over the line is a hard stop.
    - **llama.cpp (False)**: it wires through the same ``MTLResidencySet``,
      but ``recommendedMaxWorkingSetSize`` is only a debug-build warning --
      never a refusal. Over the line the model still loads; Metal just stops
      guaranteeing residency and you degrade into paging. That is a
      performance warning, not a refusal, and calling it FAIL would be wrong.
    """
    need = size_gb + headroom_gb
    avail = available_gb()
    lines = []
    ok = True

    ram_ok = need <= avail
    ok &= ram_ok
    lines.append(
        f"  {'PASS' if ram_ok else 'FAIL'}  available RAM      "
        f"need {need:6.1f} GiB have {avail:6.1f} GiB"
    )

    metal = metal_ceilings()
    if metal:
        ws_ok = need <= metal["working_set_gb"]
        ok &= ws_ok or not hard_working_set
        verdict = ("PASS" if ws_ok else "FAIL") if hard_working_set else ("PASS" if ws_ok else "WARN")
        engine = "hard limit, MLX" if hard_working_set else "advisory, llama.cpp pages past it"
        lines.append(
            f"  {verdict}  Metal working set  "
            f"need {need:6.1f} GiB have {metal['working_set_gb']:6.1f} GiB  ({engine})"
        )
        if not ws_ok and metal.get("sysctl_wired_mb") == 0:
            # Only actionable while the sysctl is at its default; if someone
            # already raised it, the ceiling is a deliberate choice and this
            # hint would just be noise.
            suggest = int((need + 8) * 1024)
            lines.append(
                f"        ^ this ceiling is the OS DEFAULT "
                f"(iogpu.wired_limit_mb=0). To raise it:\n"
                f"          sudo sysctl iogpu.wired_limit_mb={suggest}   "
                f"# resets on reboot; leave the OS real headroom"
            )
        # Per-allocation cap. Not a hard fail: llama.cpp/MLX split weights
        # across buffers, so exceeding it is a warning to check the layout
        # (one unsharded file over the cap is the case that actually bites).
        if size_gb > metal["max_buffer_gb"]:
            lines.append(
                f"  WARN  Metal max buffer   "
                f"weights {size_gb:6.1f} GiB exceed the {metal['max_buffer_gb']:.1f} GiB "
                f"per-allocation cap -- needs a sharded/split layout"
            )
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--model", help="models.toml entry id to size")
    ap.add_argument("--path", type=Path, help="model dir or .gguf to size (need not be imported)")
    ap.add_argument("--headroom", type=float, default=8.0,
                    help="GiB to leave for KV cache + compute buffers (default: 8)")
    ap.add_argument("--models-toml", type=Path, default=Path("models.toml"))
    ap.add_argument("--quiet", action="store_true",
                    help="one line + exit status only (for scripts)")
    ap.add_argument("--top", type=int, default=12, help="how many RAM holders to list")
    args = ap.parse_args()

    config = None
    label = None
    if args.model:
        config = load_model_config(args.model, args.models_toml)
        if config is None:
            print(f"model '{args.model}' not found in {args.models_toml}", file=sys.stderr)
            return 2
        label = args.model
    elif args.path:
        if not args.path.exists():
            print(f"path does not exist: {args.path}", file=sys.stderr)
            return 2
        config = config_from_path(args.path)
        label = args.path.name

    if args.quiet:
        if config is None:
            print(f"{available_gb():.0f} GiB available")
            return 0
        size_gb, _ = size_config_gb(config)
        fits, _ = check_fit(size_gb, args.headroom, is_mlx_config(config))
        verdict = "OK" if fits else "FAILED"
        print(
            f"RAM pre-flight {verdict}: {label} ~{size_gb:.0f} GiB "
            f"+ {args.headroom:.0f} GiB headroom, ~{available_gb():.0f} GiB available"
        )
        return 0 if fits else 1

    print("Ceilings")
    import psutil

    print(f"  total RAM            {psutil.virtual_memory().total / GB:6.1f} GiB")
    print(f"  available now        {available_gb():6.1f} GiB  (free + inactive; macOS evicts file cache on demand)")
    metal = metal_ceilings()
    if metal:
        wired_mb = metal.get("sysctl_wired_mb")
        origin = (
            "OS default, raisable" if wired_mb == 0
            else f"set by sysctl to {wired_mb / 1024:.0f} GiB" if wired_mb
            else "sysctl unreadable"
        )
        print(f"  Metal working set    {metal['working_set_gb']:6.1f} GiB  ({metal['device']}; {origin}) <- the real ceiling for GPU-resident weights")
        print(f"  Metal max buffer     {metal['max_buffer_gb']:6.1f} GiB   per single allocation")
    swap = psutil.swap_memory()
    print(f"  swap in use          {swap.used / GB:6.1f} GiB")

    print("\nHolding RAM (RSS, rolled up by app; shared pages double-count)")
    for family, rss_gb, count in top_holders(args.top):
        suffix = f"  x{count}" if count > 1 else ""
        print(f"  {rss_gb:6.2f} GiB {family}{suffix}")

    if config is not None:
        size_gb, notes = size_config_gb(config)
        print(f"\nFit check: {label}")
        print(f"  weights              {size_gb:6.1f} GiB")
        for note in notes:
            print(f"    {note}")
        fits, lines = check_fit(size_gb, args.headroom, is_mlx_config(config))
        print(f"  headroom requested   {args.headroom:6.1f} GiB")
        for line in lines:
            print(line)
        print(f"\n  => {'FITS' if fits else 'DOES NOT FIT'}")
        return 0 if fits else 1
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    sys.exit(main())
