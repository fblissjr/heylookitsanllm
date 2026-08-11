# src/heylook_llm/ram_fit.py
"""Memory ceilings + model sizing + the fit verdict, as a library.

Extracted from ``scripts/ram_report.py`` (which now renders this module's
structured output for the CLI) so the admin fit endpoint and the script can
never disagree -- the design doc's rule is that fit is computed in exactly
one place, server-side, and the UI is a thin renderer.

The engine asymmetry this module exists to report correctly:

- **MLX** treats the Metal ``max_recommended_working_set_size`` as HARD:
  server.py sets ``mx.set_wired_limit`` to exactly the recommendation and
  MLX refuses any larger value outright -> over the line is FAIL.
- **llama.cpp** wires through the same ``MTLResidencySet`` but checks the
  recommendation only as a debug-build warning: past the line the model
  still loads, Metal just stops guaranteeing residency and you degrade
  into paging -> over the line is WARN, and calling it FAIL would be wrong.

``hard_working_set`` carries that distinction; the server derives it from
the entry's provider, the CLI from the config's layout (``is_mlx_config``).

Sizing counts the WHOLE llama.cpp shard set behind a ``model_path`` (the
named shard of ``...-00001-of-00005.gguf`` can be a 5 MB index; the set
behind it ~127 GiB) plus the mmproj/drafter sidecars that load into the
same process. The RAM gate is ``usable_gb()`` -- reclaimable (total minus
anonymous minus wired), NOT free+inactive, because both engines mmap their
weights and macOS will evict any clean file page on demand; the
conservative figure refuses loads that then run with zero swap traffic.
"""

from __future__ import annotations

import glob as _glob
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

GB = 1024 ** 3

# llama.cpp shard naming: `<prefix>-00001-of-00005.gguf`. Mirrors
# model_importer._SHARD_RE (string-level duplicate kept tiny on purpose --
# importing model_importer here would drag its scan machinery into every
# fit call).
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------

def available_gb() -> float:
    """RAM the OS can hand out WITHOUT reclaiming anything: free + inactive.

    The conservative figure. Reported for context, but the wrong number to
    gate a model load on -- see :func:`reclaimable_gb`.
    """
    import psutil

    return psutil.virtual_memory().available / GB


def _vm_stat_pages() -> dict[str, int]:
    """`vm_stat` counters in BYTES. Empty dict off macOS or if unreadable."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out.stdout)
    if m:
        page = int(m.group(1))
    stats: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        digits = value.strip().rstrip(".").replace(",", "")
        if digits.isdigit():
            stats[key.strip().lower()] = int(digits) * page
    return stats


def reclaimable_gb() -> Optional[float]:
    """RAM a model load can actually reach: total - anonymous - wired.

    Anonymous pages are the genuinely unavailable ones (no backing file --
    only compressible or swappable, never droppable); wired cannot even be
    that. Everything else is negotiable: mmap-backed weights are clean
    file pages and macOS evicts other clean file pages on demand, including
    the recently-touched ones parked in the ACTIVE queue where
    ``free + inactive`` cannot see them. Returns None off macOS.
    """
    stats = _vm_stat_pages()
    anonymous = stats.get("anonymous pages")
    wired = stats.get("pages wired down")
    if anonymous is None or wired is None:
        return None
    import psutil

    return (psutil.virtual_memory().total - anonymous - wired) / GB


def usable_gb() -> float:
    """The figure to gate on: reclaimable where measurable, else conservative."""
    reclaimable = reclaimable_gb()
    return reclaimable if reclaimable is not None else available_gb()


def sysctl_wired_limit_mb() -> Optional[int]:
    """`iogpu.wired_limit_mb`, the SYSTEM-wide GPU wired ceiling.

    0 means "OS default" (~84% of total on a 192 GiB M2 Ultra). The only
    lever that RAISES the Metal working set; heylook's own
    ``mx.set_wired_limit`` consumes that budget, it does not enlarge it,
    and it has no effect on a llama-server subprocess. None if unreadable.
    """
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


# The device's working-set/buffer caps are static per boot; the sysctl is
# not (a `sudo sysctl` changes it live), so only the mx.device_info() half
# is cached. The cache also keeps repeated fit calls (the UI debounces one
# per field edit) from re-entering MLX at all.
_device_info_cache: Optional[dict] = None


def metal_ceilings() -> Optional[dict]:
    """GPU working-set limits in GiB, or None off Metal / without MLX."""
    global _device_info_cache
    if _device_info_cache is None:
        try:
            import mlx.core as mx

            info = mx.device_info()
        except Exception:
            return None
        working_set = info.get("max_recommended_working_set_size")
        # The isinstance guard doubles as mock-safety: under a mocked MLX
        # (contract tests) device_info() yields MagicMocks, which are truthy
        # -- caching one would poison every later fit call in the process.
        if not isinstance(working_set, (int, float)) or not working_set:
            return None
        _device_info_cache = {
            "device": info.get("device_name", "?"),
            "working_set_gb": working_set / GB,
            "max_buffer_gb": (info.get("max_buffer_length") or 0) / GB,
        }
    return {**_device_info_cache, "sysctl_wired_mb": sysctl_wired_limit_mb()}


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

    Layout-based on purpose (a `.gguf` model_path means a llama-server
    subprocess, anything else means MLX) so the CLI's ``--path`` works on
    un-imported directories. The server derives the flag from the entry's
    ``provider`` instead and passes it explicitly.
    """
    return not str(config.get("model_path") or "").lower().endswith(".gguf")


def size_config_gb(config: dict) -> tuple[float, list[str]]:
    """Resident weight GiB for a models.toml ``config`` block, plus notes.

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
            notes.append(
                f"primary is a {n}-shard set ({shard_bytes / GB:.1f} GiB), not the named shard"
            )
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


# ---------------------------------------------------------------------------
# The fit verdict, structured
# ---------------------------------------------------------------------------

@dataclass
class FitLine:
    """One ceiling's row of the fit table."""
    ceiling: str          # "reclaimable_ram" | "metal_working_set" | "metal_max_buffer"
    verdict: str          # "pass" | "warn" | "fail"
    need_gb: float
    have_gb: float
    note: str = ""


@dataclass
class FitReport:
    weights_gb: float
    headroom_gb: float
    reclaimable_gb: float
    working_set_gb: Optional[float]      # None off Metal
    max_buffer_gb: Optional[float]
    sysctl_wired_mb: Optional[int]
    # The sysctl line worth suggesting, or None. Set ONLY while the sysctl is
    # at its OS default (0): once someone has raised it, the ceiling is a
    # deliberate choice and the hint would be noise (ram_report's rule).
    sysctl_suggest_mb: Optional[int]
    kv_headroom_gb: Optional[float]      # working set minus weights (the §5 readout line)
    hard_working_set: bool
    verdict: str                         # worst of lines: "pass" | "warn" | "fail"
    lines: list[FitLine] = field(default_factory=list)
    sizing_notes: list[str] = field(default_factory=list)
    # All numbers above are measured (file sizes, device properties, vm_stat).
    # Flips to True the day a component is approximated (e.g. expert-offload
    # deltas); the UI renders estimates in a different visual register.
    estimated: bool = False

    @property
    def fits(self) -> bool:
        return self.verdict != "fail"


def evaluate_fit(size_gb: float, headroom_gb: float, hard_working_set: bool,
                 sizing_notes: Optional[list[str]] = None) -> FitReport:
    """Check ``size_gb`` + ``headroom_gb`` against every ceiling that can
    refuse (or degrade) a load. Pure of I/O except the ceiling reads."""
    need = size_gb + headroom_gb
    avail = usable_gb()
    lines: list[FitLine] = []

    lines.append(FitLine(
        ceiling="reclaimable_ram",
        verdict="pass" if need <= avail else "fail",
        need_gb=need, have_gb=avail,
        note="total - anonymous - wired; what an mmap-backed load can reach",
    ))

    metal = metal_ceilings()
    working_set_gb = max_buffer_gb = None
    sysctl_wired_mb = sysctl_suggest_mb = None
    kv_headroom_gb = None
    if metal:
        working_set_gb = metal["working_set_gb"]
        max_buffer_gb = metal["max_buffer_gb"]
        sysctl_wired_mb = metal["sysctl_wired_mb"]
        kv_headroom_gb = working_set_gb - size_gb
        ws_ok = need <= working_set_gb
        lines.append(FitLine(
            ceiling="metal_working_set",
            verdict="pass" if ws_ok else ("fail" if hard_working_set else "warn"),
            need_gb=need, have_gb=working_set_gb,
            note="hard limit, MLX refuses above it" if hard_working_set
            else "advisory for llama.cpp -- it loads past it and degrades into paging",
        ))
        if not ws_ok and sysctl_wired_mb == 0:
            sysctl_suggest_mb = int((need + 8) * 1024)
        if size_gb > max_buffer_gb:
            lines.append(FitLine(
                ceiling="metal_max_buffer",
                verdict="warn",
                need_gb=size_gb, have_gb=max_buffer_gb,
                note="per-allocation cap -- needs a sharded/split layout, not more RAM",
            ))

    verdict = "pass"
    if any(l.verdict == "warn" for l in lines):
        verdict = "warn"
    if any(l.verdict == "fail" for l in lines):
        verdict = "fail"

    return FitReport(
        weights_gb=size_gb,
        headroom_gb=headroom_gb,
        reclaimable_gb=avail,
        working_set_gb=working_set_gb,
        max_buffer_gb=max_buffer_gb,
        sysctl_wired_mb=sysctl_wired_mb,
        sysctl_suggest_mb=sysctl_suggest_mb,
        kv_headroom_gb=kv_headroom_gb,
        hard_working_set=hard_working_set,
        verdict=verdict,
        lines=lines,
        sizing_notes=list(sizing_notes or []),
    )


def fit_for_config(config: dict, headroom_gb: float = 8.0,
                   hard_working_set: Optional[bool] = None) -> FitReport:
    """Size a models.toml ``config`` block and evaluate it.

    ``hard_working_set``: pass the provider-derived truth when you have it
    (the server does); None falls back to the layout heuristic for the
    CLI's un-imported ``--path`` case.
    """
    if hard_working_set is None:
        hard_working_set = is_mlx_config(config)
    size_gb, notes = size_config_gb(config)
    return evaluate_fit(size_gb, headroom_gb, hard_working_set, sizing_notes=notes)
