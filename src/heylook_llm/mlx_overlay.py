# src/heylook_llm/mlx_overlay.py
"""Zero-config canonical local runtime overlay discovery.

Mirrors ``LlamaServerProvider.DEFAULT_BUILD`` (``~/.heylook/llama.cpp/...``) for:
- the compiled MLX C++/Metal backend built out-of-repo by ``scripts/build_mlx.py``
  at ``~/.heylook/mlx/overlay`` (``$HEYLOOK_MLX_DIR/overlay``); and
- pure-Python upstream overlays (e.g. ``mlx-vlm``, ``transformers``) staged
  out-of-repo by ``scripts/sync_overlay.py`` at ``~/.heylook/overlays/<pkg>``
  (``$HEYLOOK_OVERLAYS_DIR/<pkg>``).

Precedence before importing a package:

1. Already imported in ``sys.modules`` -> no-op.
2. Global disable ``$HEYLOOK_OVERLAYS=0`` (for pure-Python overlays) or
   per-package ``$HEYLOOK_MLX_OVERLAY`` / ``$HEYLOOK_<PKG>_OVERLAY`` override or
   disable (``0``, ``off``, ``false``, ``no``, ``none``, ``pypi``, ``disabled``).
3. ``$PYTHONPATH`` entry containing the package directory -> already on ``sys.path``.
4. Canonical build under ``~/.heylook/``, automatically standing down on ABI
   mismatch (for compiled ``mlx``) or whenever ``.venv``'s PyPI version is
   strictly newer than the overlay's version.
5. ``.venv`` PyPI fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import sysconfig
from typing import Mapping, Optional

DEFAULT_MLX_DIR = Path.home() / ".heylook" / "mlx"
DEFAULT_OVERLAY = DEFAULT_MLX_DIR / "overlay"
DEFAULT_OVERLAYS_DIR = Path.home() / ".heylook" / "overlays"
ENV_OVERLAY = "HEYLOOK_MLX_OVERLAY"
ENV_MLX_DIR = "HEYLOOK_MLX_DIR"
ENV_OVERLAYS = "HEYLOOK_OVERLAYS"
ENV_OVERLAYS_DIR = "HEYLOOK_OVERLAYS_DIR"
DISABLE_VALUES = frozenset({"0", "off", "false", "no", "none", "pypi", "disabled"})

_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)(?:\.dev(\d+))?")
_ENV_PKG_OVERLAY_RE = re.compile(r"^HEYLOOK_([A-Z0-9_]+)_OVERLAY$")
_ACTIVE_DECISIONS: dict[str, "OverlayDecision"] = {}


@dataclass(frozen=True)
class OverlayDecision:
    """Outcome of resolving which runtime a process should load for a package."""

    action: str
    path: Optional[Path]
    source: str
    is_override: bool = False
    label: Optional[str] = None
    warning: Optional[str] = None
    package: str = "mlx"

    def describe(self) -> str:
        """Short description for the startup log line."""
        if self.action == "canonical":
            return f"canonical build {self.label}" if self.label else "canonical build"
        if self.action in ("override", "pythonpath"):
            base = f"{self.source} {self.label}" if self.label else self.source
            return f"{base} ({self.path})" if self.path else base
        if self.action == "disabled":
            return f".venv (disabled via {self.source})"
        if self.action == "abi_mismatch":
            return ".venv (overlay ABI mismatch)"
        if self.action == "superseded_by_pypi":
            return ".venv (newer than canonical overlay)"
        if self.action == "already_imported":
            return "already imported before overlay bootstrap"
        return self.source


def canonical_overlay_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve ``$HEYLOOK_MLX_DIR/overlay`` when set, else ``DEFAULT_OVERLAY``."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_MLX_DIR) or "").strip()
    if raw:
        return Path(raw).expanduser() / "overlay"
    return DEFAULT_OVERLAY


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path


def _read_manifest_for_overlay(overlay: Path) -> Optional[dict]:
    """Read ``<dir>/build/heylook-build.json`` if its ``overlay`` matches ``overlay``."""
    manifest_file = overlay.parent / "build" / "heylook-build.json"
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    recorded = data.get("overlay")
    if not recorded:
        return None
    if _safe_resolve(Path(str(recorded))) != _safe_resolve(overlay):
        return None
    return data


def read_overlay_version(overlay: Path) -> Optional[str]:
    """Read the built ``mlx`` version from dist-info METADATA or build manifest."""
    for meta in sorted(overlay.glob("mlx-*.dist-info/METADATA")):
        if meta.parent.name.startswith(("mlx_metal-", "mlx-metal-")):
            continue
        try:
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("version:"):
                    ver = line.split(":", 1)[1].strip()
                    if ver:
                        return ver
        except OSError:
            continue
    data = _read_manifest_for_overlay(overlay)
    if data:
        ver = str(data.get("version") or "").strip()
        if ver:
            return ver
    return None


def read_manifest_label(overlay: Path) -> Optional[str]:
    """Format ``sha[:8]`` and applied PR numbers from ``heylook-build.json``."""
    data = _read_manifest_for_overlay(overlay)
    if not data:
        return None
    sha = str(data.get("sha") or data.get("rev") or "").strip()[:8]
    pr_nums: list[str] = []
    for item in data.get("prs") or []:
        if isinstance(item, dict) and item.get("number") is not None:
            pr_nums.append(f"#{item['number']}")
        elif isinstance(item, (int, str)) and str(item).strip():
            pr_nums.append(f"#{str(item).strip().lstrip('#')}")
    if sha and pr_nums:
        return f"{sha} ({', '.join(pr_nums)})"
    if sha:
        return sha
    if pr_nums:
        return f"({', '.join(pr_nums)})"
    return None


def _parse_version_fallback(ver: str) -> Optional[tuple[tuple[int, ...], int, int]]:
    m = _VERSION_RE.match(ver.strip())
    if not m:
        return None
    release = tuple(int(part) for part in m.group(1).split("."))
    is_dev = m.group(2) is not None
    dev_num = int(m.group(2)) if is_dev else 0
    # Non-dev releases sort after .dev pre-releases of the same release tuple.
    return (release, 0 if is_dev else 1, dev_num)


def _compare_versions(venv_ver: str, overlay_ver: str) -> Optional[int]:
    """Return ``1`` if ``venv_ver > overlay_ver``, ``0`` if equal, ``-1`` if ``<``."""
    try:
        from packaging.version import InvalidVersion, Version

        try:
            v1 = Version(venv_ver)
            v2 = Version(overlay_ver)
            if v1 > v2:
                return 1
            if v1 < v2:
                return -1
            return 0
        except InvalidVersion:
            pass
    except ImportError:
        pass

    p1 = _parse_version_fallback(venv_ver)
    p2 = _parse_version_fallback(overlay_ver)
    if p1 is None or p2 is None:
        return None
    if p1 > p2:
        return 1
    if p1 < p2:
        return -1
    return 0


def _check_abi(
    overlay: Path, ext_suffix: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """Verify compiled ``mlx/core*.so`` matches this interpreter's ABI suffix."""
    so_files = sorted((overlay / "mlx").glob("core*.so"))
    if not so_files:
        return True, None
    expected = (
        sysconfig.get_config_var("EXT_SUFFIX") if ext_suffix is None else ext_suffix
    )
    for so in so_files:
        if so.name.endswith(".abi3.so"):
            return True, None
        if expected and so.name.endswith(expected):
            return True, None
    names = ", ".join(p.name for p in so_files)
    return False, f"found {names} (expected *{expected or '.so'} or *.abi3.so)"


def _installed_venv_mlx_version() -> Optional[str]:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_mlx_overlay(
    environ: Optional[Mapping[str, str]] = None,
    *,
    venv_version: Optional[str] = None,
    ext_suffix: Optional[str] = None,
    already_imported: Optional[bool] = None,
) -> OverlayDecision:
    """Determine which MLX runtime should be active without mutating ``sys.path``."""
    imported = (
        ("mlx" in sys.modules or "mlx.core" in sys.modules)
        if already_imported is None
        else already_imported
    )
    if imported:
        return OverlayDecision(
            action="already_imported",
            path=None,
            source="sys.modules",
        )

    env = os.environ if environ is None else environ
    canonical = canonical_overlay_path(env)
    canonical_exists = (canonical / "mlx").is_dir()

    raw = (env.get(ENV_OVERLAY) or "").strip()
    if not raw:
        global_raw = (env.get(ENV_OVERLAYS) or "").strip()
        if global_raw and global_raw.lower() in DISABLE_VALUES:
            warn = (
                f"${ENV_OVERLAYS}={global_raw} disables canonical MLX overlay at {canonical}; "
                "using .venv PyPI mlx"
                if canonical_exists
                else None
            )
            return OverlayDecision(
                action="disabled",
                path=None,
                source=f"${ENV_OVERLAYS}",
                is_override=True,
                warning=warn,
                package="mlx",
            )
    if raw:
        if raw.lower() in DISABLE_VALUES:
            warn = (
                f"${ENV_OVERLAY}={raw} disables canonical MLX overlay at {canonical}; "
                "using .venv PyPI mlx"
                if canonical_exists
                else None
            )
            return OverlayDecision(
                action="disabled",
                path=None,
                source=f"${ENV_OVERLAY}",
                is_override=True,
                warning=warn,
            )

        override = Path(raw).expanduser()
        if not (override / "mlx").is_dir():
            raise RuntimeError(
                f"${ENV_OVERLAY} points to '{override}', which has no 'mlx/' package. "
                f"Run `uv run scripts/build_mlx.py` or unset ${ENV_OVERLAY}."
            )
        ok, abi_detail = _check_abi(override, ext_suffix)
        if not ok:
            raise RuntimeError(
                f"${ENV_OVERLAY} overlay at '{override}' has incompatible Python ABI "
                f"({abi_detail}). Run `uv run scripts/build_mlx.py --rebuild` or unset "
                f"${ENV_OVERLAY}."
            )
        differs = canonical_exists and _safe_resolve(override) != _safe_resolve(canonical)
        warn = (
            f"${ENV_OVERLAY}={override} -- OVERRIDING the canonical overlay at {canonical}"
            if differs
            else None
        )
        return OverlayDecision(
            action="override",
            path=override,
            source=f"${ENV_OVERLAY}",
            is_override=True,
            label=read_manifest_label(override),
            warning=warn,
        )

    pythonpath = env.get("PYTHONPATH") or ""
    for entry in pythonpath.split(os.pathsep):
        if not entry.strip():
            continue
        candidate = Path(entry).expanduser()
        if (candidate / "mlx").is_dir():
            differs = (
                canonical_exists
                and _safe_resolve(candidate) != _safe_resolve(canonical)
            )
            warn = (
                f"$PYTHONPATH entry {candidate} -- OVERRIDING the canonical overlay at {canonical}"
                if differs
                else None
            )
            return OverlayDecision(
                action="pythonpath",
                path=candidate,
                source="$PYTHONPATH",
                is_override=differs,
                label=read_manifest_label(candidate),
                warning=warn,
            )

    if not canonical_exists:
        return OverlayDecision(
            action="pypi",
            path=None,
            source=".venv",
            is_override=False,
        )

    ok, abi_detail = _check_abi(canonical, ext_suffix)
    if not ok:
        return OverlayDecision(
            action="abi_mismatch",
            path=None,
            source=".venv",
            is_override=False,
            warning=(
                f"canonical MLX overlay at {canonical} has incompatible ABI "
                f"({abi_detail}); using .venv PyPI mlx. Run "
                "`uv run scripts/build_mlx.py --rebuild`."
            ),
        )

    v_ver = venv_version if venv_version is not None else _installed_venv_mlx_version()
    o_ver = read_overlay_version(canonical)
    if v_ver and o_ver and _compare_versions(v_ver, o_ver) == 1:
        return OverlayDecision(
            action="superseded_by_pypi",
            path=None,
            source=".venv",
            is_override=False,
            warning=(
                f".venv PyPI mlx {v_ver} is newer than canonical overlay {o_ver} at "
                f"{canonical}; standing down to .venv PyPI. Run "
                "`uv run scripts/build_mlx.py` to rebuild or set "
                f"{ENV_OVERLAY}={canonical} to force."
            ),
        )

    return OverlayDecision(
        action="canonical",
        path=canonical,
        source="canonical build",
        is_override=False,
        label=read_manifest_label(canonical),
    )


def activate_mlx_overlay(
    environ: Optional[Mapping[str, str]] = None,
    *,
    log: bool = True,
) -> OverlayDecision:
    """Resolve the MLX overlay and prepend it to ``sys.path`` before ``import mlx``."""
    decision = resolve_mlx_overlay(environ=environ)
    if decision.action in ("canonical", "override") and decision.path is not None:
        entry = str(decision.path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    _ACTIVE_DECISIONS["mlx"] = decision

    if log:
        if decision.warning:
            logging.warning("[MLX] %s", decision.warning)
        elif decision.action == "canonical" and decision.path is not None:
            suffix = f" ({decision.label})" if decision.label else ""
            logging.info("[MLX] using canonical overlay at %s%s", decision.path, suffix)
        elif decision.action == "override" and decision.path is not None:
            suffix = f" ({decision.label})" if decision.label else ""
            logging.info(
                "[MLX] using overlay from %s at %s%s",
                decision.source,
                decision.path,
                suffix,
            )

    return decision


def canonical_overlays_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve ``$HEYLOOK_OVERLAYS_DIR`` when set, else ``~/.heylook/overlays``."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_OVERLAYS_DIR) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_OVERLAYS_DIR


def package_overlay_env_var(dist_name: str) -> str:
    """Per-package override env var, e.g. ``mlx-vlm`` -> ``HEYLOOK_MLX_VLM_OVERLAY``."""
    norm = re.sub(r"[^A-Za-z0-9]+", "_", dist_name.strip()).strip("_").upper()
    return f"HEYLOOK_{norm}_OVERLAY"


def _format_manifest_label(data: Optional[dict]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    sha = str(data.get("sha") or data.get("rev") or "").strip()[:8]
    pr_nums: list[str] = []
    for item in data.get("prs") or []:
        if isinstance(item, dict) and item.get("number") is not None:
            pr_nums.append(f"#{item['number']}")
        elif isinstance(item, (int, str)) and str(item).strip():
            pr_nums.append(f"#{str(item).strip().lstrip('#')}")
    if sha and pr_nums:
        return f"{sha} ({', '.join(pr_nums)})"
    if sha:
        return sha
    if pr_nums:
        return f"({', '.join(pr_nums)})"
    return None


def read_package_manifest(overlay: Path) -> Optional[dict]:
    """Read ``<overlay>/heylook-build.json`` if present."""
    manifest_file = overlay / "heylook-build.json"
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_package_overlay_version(overlay: Path, dist_name: str) -> Optional[str]:
    """Read built package version from dist-info METADATA or ``heylook-build.json``."""
    norm_under = re.sub(r"[-_.]+", "_", dist_name.strip()).lower()
    norm_dash = norm_under.replace("_", "-")
    for meta in sorted(overlay.glob("*.dist-info/METADATA")):
        prefix = meta.parent.name.split("-", 1)[0].lower()
        if prefix not in (norm_under, norm_dash):
            continue
        try:
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("version:"):
                    ver = line.split(":", 1)[1].strip()
                    if ver:
                        return ver
        except OSError:
            continue
    data = read_package_manifest(overlay)
    if data:
        ver = str(data.get("version") or "").strip()
        if ver:
            return ver
    return None


def _installed_venv_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_package_target(directory: Path, import_name: str) -> bool:
    return (directory / import_name).is_dir() or (directory / f"{import_name}.py").is_file()


def resolve_python_overlay(
    dist_name: str,
    import_name: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    *,
    venv_version: Optional[str] = None,
    already_imported: Optional[bool] = None,
) -> OverlayDecision:
    """Determine which runtime a pure-Python package should load without mutating ``sys.path``."""
    pkg = dist_name.strip().lower()
    imp = (import_name or pkg.replace("-", "_")).strip()
    imported = (imp in sys.modules) if already_imported is None else already_imported
    if imported:
        return OverlayDecision(
            action="already_imported",
            path=None,
            source="sys.modules",
            package=pkg,
        )

    env = os.environ if environ is None else environ
    canonical = canonical_overlays_dir(env) / pkg
    canonical_exists = _has_package_target(canonical, imp)
    env_var = package_overlay_env_var(pkg)
    raw = (env.get(env_var) or "").strip()

    if not raw:
        global_raw = (env.get(ENV_OVERLAYS) or "").strip()
        if global_raw and global_raw.lower() in DISABLE_VALUES:
            warn = (
                f"${ENV_OVERLAYS}={global_raw} disables canonical {pkg} overlay at {canonical}; "
                f"using .venv PyPI {pkg}"
                if canonical_exists
                else None
            )
            return OverlayDecision(
                action="disabled",
                path=None,
                source=f"${ENV_OVERLAYS}",
                is_override=True,
                warning=warn,
                package=pkg,
            )

    if raw:
        if raw.lower() in DISABLE_VALUES:
            warn = (
                f"${env_var}={raw} disables canonical {pkg} overlay at {canonical}; "
                f"using .venv PyPI {pkg}"
                if canonical_exists
                else None
            )
            return OverlayDecision(
                action="disabled",
                path=None,
                source=f"${env_var}",
                is_override=True,
                warning=warn,
                package=pkg,
            )
        override = Path(raw).expanduser()
        if not _has_package_target(override, imp):
            raise RuntimeError(
                f"${env_var} points to '{override}', which has no '{imp}' package. "
                f"Run `uv run scripts/sync_overlay.py {pkg}` or unset ${env_var}."
            )
        differs = canonical_exists and _safe_resolve(override) != _safe_resolve(canonical)
        warn = (
            f"${env_var}={override} -- OVERRIDING the canonical {pkg} overlay at {canonical}"
            if differs
            else None
        )
        return OverlayDecision(
            action="override",
            path=override,
            source=f"${env_var}",
            is_override=True,
            label=_format_manifest_label(read_package_manifest(override)),
            warning=warn,
            package=pkg,
        )

    pythonpath = env.get("PYTHONPATH") or ""
    for entry in pythonpath.split(os.pathsep):
        if not entry.strip():
            continue
        candidate = Path(entry).expanduser()
        if _has_package_target(candidate, imp):
            differs = canonical_exists and _safe_resolve(candidate) != _safe_resolve(canonical)
            warn = (
                f"$PYTHONPATH entry {candidate} -- OVERRIDING the canonical {pkg} overlay at {canonical}"
                if differs
                else None
            )
            return OverlayDecision(
                action="pythonpath",
                path=candidate,
                source="$PYTHONPATH",
                is_override=differs,
                label=_format_manifest_label(read_package_manifest(candidate)),
                warning=warn,
                package=pkg,
            )

    if not canonical_exists:
        return OverlayDecision(
            action="pypi",
            path=None,
            source=".venv",
            is_override=False,
            package=pkg,
        )

    v_ver = venv_version if venv_version is not None else _installed_venv_version(pkg)
    o_ver = read_package_overlay_version(canonical, pkg)
    if v_ver and o_ver and _compare_versions(v_ver, o_ver) == 1:
        return OverlayDecision(
            action="superseded_by_pypi",
            path=None,
            source=".venv",
            is_override=False,
            warning=(
                f".venv PyPI {pkg} {v_ver} is newer than canonical overlay {o_ver} at "
                f"{canonical}; standing down to .venv PyPI. Run "
                f"`uv run scripts/sync_overlay.py {pkg}` to refresh or set "
                f"{env_var}={canonical} to force."
            ),
            package=pkg,
        )

    return OverlayDecision(
        action="canonical",
        path=canonical,
        source="canonical build",
        is_override=False,
        label=_format_manifest_label(read_package_manifest(canonical)),
        package=pkg,
    )


def discover_python_overlays(
    environ: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, str]]:
    """Find pure-Python overlays staged under ``~/.heylook/overlays`` or named via env."""
    env = os.environ if environ is None else environ
    found: dict[str, str] = {}

    root = canonical_overlays_dir(env)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = read_package_manifest(child)
            if manifest is None:
                continue
            pkg = str(manifest.get("package") or child.name).strip().lower()
            if not pkg or pkg == "mlx":
                continue
            imp = str(manifest.get("import_name") or pkg.replace("-", "_")).strip()
            found[pkg] = imp

    for key, val in sorted(env.items()):
        if key == ENV_OVERLAY or not (val or "").strip():
            continue
        m = _ENV_PKG_OVERLAY_RE.match(key)
        if not m:
            continue
        token = m.group(1).lower()
        if token in ("mlx",):
            continue
        pkg = token.replace("_", "-")
        found.setdefault(pkg, token)

    return [(pkg, imp) for pkg, imp in sorted(found.items())]


def activate_python_overlays(
    environ: Optional[Mapping[str, str]] = None,
    *,
    log: bool = True,
) -> list[OverlayDecision]:
    """Resolve and prepend every discovered pure-Python overlay to ``sys.path``."""
    decisions: list[OverlayDecision] = []
    for pkg, imp in discover_python_overlays(environ=environ):
        decision = resolve_python_overlay(pkg, imp, environ=environ)
        if decision.action in ("canonical", "override") and decision.path is not None:
            entry = str(decision.path)
            while entry in sys.path:
                sys.path.remove(entry)
            sys.path.insert(0, entry)
        _ACTIVE_DECISIONS[pkg] = decision
        decisions.append(decision)
        if log:
            tag = pkg.upper()
            if decision.warning:
                logging.warning("[OVERLAY:%s] %s", tag, decision.warning)
            elif decision.action == "canonical" and decision.path is not None:
                suffix = f" ({decision.label})" if decision.label else ""
                logging.info(
                    "[OVERLAY:%s] using canonical overlay at %s%s",
                    tag,
                    decision.path,
                    suffix,
                )
            elif decision.action == "override" and decision.path is not None:
                suffix = f" ({decision.label})" if decision.label else ""
                logging.info(
                    "[OVERLAY:%s] using overlay from %s at %s%s",
                    tag,
                    decision.source,
                    decision.path,
                    suffix,
                )
    return decisions


def activate_all_overlays(
    environ: Optional[Mapping[str, str]] = None,
    *,
    log: bool = True,
) -> dict[str, OverlayDecision]:
    """Activate compiled ``mlx`` overlay and all discovered pure-Python overlays."""
    out: dict[str, OverlayDecision] = {}
    out["mlx"] = activate_mlx_overlay(environ=environ, log=log)
    for dec in activate_python_overlays(environ=environ, log=log):
        out[dec.package] = dec
    return out


def active_overlay_decisions() -> dict[str, OverlayDecision]:
    """Snapshot of overlay decisions recorded by this process."""
    return dict(_ACTIVE_DECISIONS)


def log_mlx_runtime(
    mx_version: str,
    mx_file: Optional[str],
    decision: Optional[OverlayDecision],
) -> None:
    """Log the active ``mlx`` version and source in the server startup banner."""
    desc = decision.describe() if decision is not None else ".venv"
    logging.info("[MLX] runtime: mlx %s (%s)", mx_version, desc)
