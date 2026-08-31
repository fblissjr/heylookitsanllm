#!/usr/bin/env python3
"""Memory pre-flight: what is holding RAM, and will this model fit?

Two jobs, because on this hardware they are the same question:

1. **Report** the ceilings and who is under them. "Total RAM" is the wrong
   number to reason with -- see the Metal note below.
2. **Size** a model the way it is actually loaded (whole shard SET, plus the
   mmproj and drafter sidecars that land in the same process) and check it
   against every ceiling that can refuse it.

All computation lives in ``heylook_llm.ram_fit`` (shared with the admin fit
endpoint, so this script and the API can never disagree); this file is the
CLI renderer plus the who-is-holding-RAM report. The engine asymmetry, the
reclaimable-RAM rationale, and the sizing traps are documented there.

Usage::

    uv run python scripts/ram_report.py                     # ceilings + holders
    uv run python scripts/ram_report.py --model <id>        # + fit check
    uv run python scripts/ram_report.py --path <dir|gguf>   # + fit, unimported
    uv run python scripts/ram_report.py --model <id> --quiet --headroom 8

``--quiet`` prints one line and exits 1 when the model does not fit, which is
the form ``dev_server.sh`` consumes. It exits 2 with nothing on stdout when
the model could not be sized at all -- an unknown id, or files that no longer
read. That is a bad ``--model``, not a memory refusal, and the two must stay
distinguishable to the caller.

``--model`` resolves through the SERVER's registry merge, not models.toml
alone: models.toml is override-only, so most served models are never written
into it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Optional

# Make the package importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heylook_llm.model_registry import discover, merge_discovered  # noqa: E402
from heylook_llm.ram_fit import (  # noqa: E402
    GB,
    FitReport,
    available_gb,
    fit_for_config,
    metal_ceilings,
    reclaimable_gb,
    unsizeable_reason,
    usable_gb,
)


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
# Config resolution
# ---------------------------------------------------------------------------

def load_model_config(model_id: str, models_toml: Path) -> Optional[dict]:
    """Resolve an id the way the SERVER resolves it, not the way models.toml reads.

    models.toml is OVERRIDE-ONLY (``model_registry``): every model under
    ``[scan].folders`` is served with derived defaults and is never written
    down. A models.toml-only lookup is therefore blind to most of what is
    servable -- which is how sizing a discovered model made the dev_server
    pre-flight refuse to start with "not in models.toml" for a model the
    server happily serves.

    Same merge, same precedence (explicit entries win, discovery can only
    add) as ``ModelRouter._load_config``, reused rather than restated, so
    this script and the server cannot disagree about which config an id
    names. It costs a folder scan on every ``--model`` call; that is seconds
    against the multi-minute load this pre-flight gates.
    """
    try:
        doc = tomllib.loads(models_toml.read_text())
    except OSError:
        return None
    merged = merge_discovered(doc, discover(doc))
    for entry in merged.get("models", []):
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
# Rendering (the CLI face of ram_fit's structured report)
# ---------------------------------------------------------------------------

def render_fit(report: FitReport) -> list[str]:
    lines: list[str] = []
    for line in report.lines:
        verdict = line.verdict.upper()
        if line.ceiling == "reclaimable_ram":
            lines.append(
                f"  {verdict:4s}  reclaimable RAM    "
                f"need {line.need_gb:6.1f} GiB have {line.have_gb:6.1f} GiB"
            )
        elif line.ceiling == "metal_working_set":
            engine = ("hard limit, MLX" if report.hard_working_set
                      else "advisory, llama.cpp pages past it")
            lines.append(
                f"  {verdict:4s}  Metal working set  "
                f"need {line.need_gb:6.1f} GiB have {line.have_gb:6.1f} GiB  ({engine})"
            )
            if report.sysctl_suggest_mb:
                lines.append(
                    f"        ^ this ceiling is the OS DEFAULT "
                    f"(iogpu.wired_limit_mb=0). To raise it:\n"
                    f"          sudo sysctl iogpu.wired_limit_mb={report.sysctl_suggest_mb}   "
                    f"# resets on reboot; leave the OS real headroom"
                )
        elif line.ceiling == "metal_max_buffer":
            lines.append(
                f"  WARN  Metal max buffer   "
                f"weights {line.need_gb:6.1f} GiB exceed the {line.have_gb:.1f} GiB "
                f"per-allocation cap -- needs a sharded/split layout"
            )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--model", help="served model id to size (a models.toml "
                                    "entry, or one discovered under [scan].folders)")
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
            print(
                f"model '{args.model}' is not served: no {args.models_toml} "
                f"entry and nothing under [scan].folders derives that id",
                file=sys.stderr,
            )
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
            print(f"{usable_gb():.0f} GiB reclaimable")
            return 0
        report = fit_for_config(config, args.headroom)
        # Reason on stderr, stdout left EMPTY, exit 2 -- the same shape an
        # unknown id uses, because dev_server.sh reads empty stdout as "bad
        # --model" and anything else as a memory refusal. This is the former.
        if (why := unsizeable_reason(report)) is not None:
            print(f"could not size model '{label}': {why}", file=sys.stderr)
            return 2
        verdict = "OK" if report.fits else "FAILED"
        print(
            f"RAM pre-flight {verdict}: {label} ~{report.weights_gb:.0f} GiB "
            f"+ {args.headroom:.0f} GiB headroom, ~{report.reclaimable_gb:.0f} GiB reclaimable"
        )
        return 0 if report.fits else 1

    print("Ceilings")
    import psutil

    print(f"  total RAM            {psutil.virtual_memory().total / GB:6.1f} GiB")
    print(f"  free right now       {available_gb():6.1f} GiB  (free + inactive -- conservative, NOT the gate)")
    reclaim = reclaimable_gb()
    if reclaim is not None:
        print(f"  reclaimable          {reclaim:6.1f} GiB  (total - anonymous - wired) <- what an mmap-backed load can reach")
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
        report = fit_for_config(config, args.headroom)
        print(f"\nFit check: {label}")
        print(f"  weights              {report.weights_gb:6.1f} GiB")
        for note in report.sizing_notes:
            print(f"    {note}")
        print(f"  headroom requested   {args.headroom:6.1f} GiB")
        for line in render_fit(report):
            print(line)
        if (why := unsizeable_reason(report)) is not None:
            print(f"\n  => CANNOT SIZE ({why}) -- no verdict, not a pass")
            return 2
        print(f"\n  => {'FITS' if report.fits else 'DOES NOT FIT'}")
        return 0 if report.fits else 1
    return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    sys.exit(main())
