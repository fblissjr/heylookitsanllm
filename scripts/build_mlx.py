#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Clone and build ml-explore/mlx from source, outside the repo.

Mirrors ``scripts/build_llama.py`` for the MLX C++/Metal backend. It never edits
``pyproject.toml`` or ``uv.lock`` -- source builds land in ``~/.heylook/mlx``
(relocatable via ``$HEYLOOK_MLX_DIR`` or ``--dir``) and write a provenance
manifest at ``<dir>/build/heylook-build.json``.

Two delivery modes:
  --overlay   Unpack the built wheel into ``<dir>/overlay`` (default).
              ``heylookllm`` picks up ``~/.heylook/mlx/overlay`` automatically
              with zero config, unless ``HEYLOOK_MLX_OVERLAY`` overrides or
              disables (``0``) it, ``PYTHONPATH`` already supplies ``mlx``, or
              ``.venv`` PyPI ``mlx`` is strictly newer than the overlay.
  --install   Install the built wheel directly into ``--python``'s environment
              and clean stale ``mlx_metal-*.dist-info`` metadata. Plain ``uv run``
              syncs ``.venv`` back to ``uv.lock`` unless ``--no-sync`` is passed.

Examples:
  uv run scripts/build_mlx.py --rev main --pr 4890 --pr 4826
  uv run scripts/build_mlx.py --status
  uv run scripts/build_mlx.py --rebuild --overlay
  HEYLOOK_MLX_OVERLAY=0 uv run --no-sync heylookllm
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import NoReturn
import zipfile

GIT_URL = "https://github.com/ml-explore/mlx"
HOME_SUBDIR = (".heylook", "mlx")

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s: str) -> str:
    return _c("1", s)


def green(s: str) -> str:
    return _c("32", s)


def yellow(s: str) -> str:
    return _c("33", s)


def red(s: str) -> str:
    return _c("31", s)


def dim(s: str) -> str:
    return _c("2", s)


def die(msg: str) -> NoReturn:
    print(red(f"error: {msg}"), file=sys.stderr)
    raise SystemExit(1)


def run(
    cmd: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    print(dim("$ " + " ".join(cmd)))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and proc.returncode != 0:
        if capture and proc.stderr:
            print(proc.stderr, file=sys.stderr)
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return (proc.stdout or "").strip()


def mlx_dir(cli_dir: str | None) -> Path:
    """--dir > $HEYLOOK_MLX_DIR > the fixed home location."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()
    env = os.environ.get("HEYLOOK_MLX_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*HOME_SUBDIR).resolve()


def default_python() -> Path:
    """Prefer the repo .venv interpreter without dereferencing its symlink."""
    repo_root = Path(__file__).resolve().parents[1]
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        return venv_py
    return Path(os.path.abspath(sys.executable))


def resolve_python(cli_python: str | None) -> Path:
    if not cli_python:
        return default_python()
    return Path(os.path.abspath(os.path.expanduser(cli_python)))


_VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def latest_release_tag() -> str:
    """Newest ml-explore/mlx vX.Y.Z release tag from upstream git remotes."""
    out = run(["git", "ls-remote", "--tags", "--refs", GIT_URL], capture=True)
    best: tuple[tuple[int, int, int], str] | None = None
    for line in out.splitlines():
        tag = line.rsplit("/", 1)[-1].strip()
        m = _VERSION_TAG_RE.match(tag)
        if m:
            ver = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if best is None or ver > best[0]:
                best = (ver, tag)
    if best is None:
        die(f"no vX.Y.Z release tags found at {GIT_URL}")
    return best[1]


def _remote_key(url: str) -> str:
    s = url.strip().removesuffix(".git")
    s = re.sub(r"^[a-z+]+://", "", s)
    s = s.replace(":", "/", 1) if s.startswith("git@") else s
    return s.removeprefix("git@").lower()


def ensure_checkout(path: Path) -> None:
    """Clone if absent; verify origin matches GIT_URL if present."""
    if (path / ".git").exists():
        origin = run(
            ["git", "-C", str(path), "remote", "get-url", "origin"], capture=True
        )
        if _remote_key(origin) != _remote_key(GIT_URL):
            die(
                f"{path} tracks {origin}, not {GIT_URL}. "
                f"Point --dir elsewhere or fix the remote."
            )
        return
    if path.exists() and any(path.iterdir()):
        die(f"{path} exists and is not an mlx checkout -- refusing to clone over it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", GIT_URL, str(path)])


def _ref_exists(path: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def resolve_fetched_rev(path: Path, rev: str) -> str:
    """Map a branch name onto origin/<branch>; pass tags, SHAs, and HEAD through."""
    if rev == "HEAD":
        return rev
    remote = f"origin/{rev}"
    if not _ref_exists(path, remote):
        return rev
    if _ref_exists(path, rev):
        local = run(["git", "-C", str(path), "rev-parse", "--short", rev], capture=True)
        tracking = run(
            ["git", "-C", str(path), "rev-parse", "--short", remote], capture=True
        )
        if local != tracking:
            print(
                f"  {rev!r} is a branch: building {remote} ({tracking}), not "
                f"the local branch ({local}), which fetch never advances"
            )
    return remote


def checkout_rev(path: Path, rev: str) -> str:
    """Detach onto ``rev``. Refuses over tracked local modifications."""
    dirty = run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture=True,
    )
    if dirty:
        die(
            f"{path} has uncommitted changes to TRACKED files -- refusing to "
            f"check out {rev} over them. Commit or stash first:\n{dirty}"
        )
    shallow = run(
        ["git", "-C", str(path), "rev-parse", "--is-shallow-repository"],
        capture=True,
    )
    if shallow == "true":
        print(yellow("  checkout is shallow -- repairing (fetch --unshallow)"))
        run(
            ["git", "-C", str(path), "fetch", "--unshallow", "--tags", "--force", "origin"]
        )
    else:
        run(["git", "-C", str(path), "fetch", "--tags", "--force", "origin"])
    run(["git", "-C", str(path), "checkout", "--detach", resolve_fetched_rev(path, rev)])
    return run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)


def apply_prs(path: Path, prs: list[int | dict]) -> list[dict[str, object]]:
    """Fetch and merge PR refs (or pinned SHAs during --rebuild) atop HEAD."""
    applied: list[dict[str, object]] = []
    for item in prs:
        if isinstance(item, dict):
            num = int(item["number"])
            pinned_sha = str(item["sha"])
            if not _ref_exists(path, pinned_sha):
                run(["git", "-C", str(path), "fetch", "origin", f"refs/pull/{num}/head"])
            target = pinned_sha
            pr_sha = pinned_sha
        else:
            num = int(item)
            run(["git", "-C", str(path), "fetch", "origin", f"refs/pull/{num}/head"])
            target = "FETCH_HEAD"
            pr_sha = run(["git", "-C", str(path), "rev-parse", "FETCH_HEAD"], capture=True)
        print(f"  merging PR #{num} ({pr_sha[:12]})")
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=heylook-build",
                "-c",
                "user.email=heylook-build@local",
                "merge",
                "--no-edit",
                "--no-gpg-sign",
                target,
            ],
            check=False,
        )
        if proc.returncode != 0:
            subprocess.run(["git", "-C", str(path), "merge", "--abort"], check=False)
            die(f"PR #{num} ({pr_sha[:12]}) failed to merge cleanly onto {path}")
        applied.append({"number": num, "sha": pr_sha})
    return applied


def _manifest_dir(path: Path) -> Path:
    return path / "build"


def manifest_path(path: Path) -> Path:
    return _manifest_dir(path) / "heylook-build.json"


def read_manifest(path: Path) -> dict:
    p = manifest_path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _drop_cmake_caches(build_root: Path) -> bool:
    removed = False
    if not build_root.exists():
        return False
    for cache_file in build_root.rglob("CMakeCache.txt"):
        cache_file.unlink(missing_ok=True)
        removed = True
    return removed


def build_wheel(
    path: Path,
    *,
    target_py: Path,
    jobs: int,
    clean: bool,
    cmake_args: list[str],
) -> Path:
    uv = shutil.which("uv")
    if not uv:
        die("uv not found on PATH")
    if not shutil.which("cmake"):
        die("cmake not found on PATH (brew install cmake)")
    if not target_py.exists():
        die(f"target python interpreter not found: {target_py}")

    build_dir = _manifest_dir(path)
    dist_dir = path / "dist"
    tmp_dist = dist_dir / ".current"
    if clean:
        for tree in (build_dir, path / "python" / "mlx" / "core"):
            if tree.exists():
                print(dim(f"removing {tree}"))
                shutil.rmtree(tree)
        for old_whl in dist_dir.glob("mlx-*.whl"):
            old_whl.unlink(missing_ok=True)
    if tmp_dist.exists():
        shutil.rmtree(tmp_dist)
    tmp_dist.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MLX_BUILD_FRONTEND_PACKAGE": "0",
            "MLX_BUILD_BACKEND_PACKAGE": "0",
            "PYPI_RELEASE": "0",
            "DEV_RELEASE": "0",
            "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs),
        }
    )
    if cmake_args:
        existing = env.get("CMAKE_ARGS", "").strip()
        env["CMAKE_ARGS"] = " ".join([existing, *cmake_args]).strip()

    cmd = [
        uv,
        "build",
        "--wheel",
        "--python",
        str(target_py),
        "--out-dir",
        str(tmp_dist),
        str(path),
    ]
    print(dim("$ " + " ".join(cmd)))
    had_build = build_dir.exists()
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0 and had_build and _drop_cmake_caches(build_dir):
        print(
            yellow(
                "  wheel build failed against an existing build tree; "
                "dropped CMakeCache.txt and retrying once"
            )
        )
        for leftover in tmp_dist.glob("*.whl"):
            leftover.unlink(missing_ok=True)
        proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        die(f"uv build --wheel failed ({proc.returncode})")

    wheels = sorted(tmp_dist.glob("mlx-*.whl"))
    if len(wheels) != 1:
        die(f"expected 1 wheel in {tmp_dist}, found {[w.name for w in wheels]}")

    built = dist_dir / wheels[0].name
    if built.exists():
        built.unlink()
    os.replace(wheels[0], built)
    shutil.rmtree(tmp_dist, ignore_errors=True)
    return built


def unpack_wheel_to_overlay(wheel: Path, overlay_dir: Path) -> Path:
    """Atomically unpack ``wheel`` into ``overlay_dir`` preserving exec bits."""
    overlay_dir = Path(os.path.abspath(os.path.expanduser(str(overlay_dir))))
    overlay_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = overlay_dir.parent / f".{overlay_dir.name}.staging"
    backup = overlay_dir.parent / f".{overlay_dir.name}.old"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)
    staging.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            rel = Path(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                die(f"unsafe wheel entry path: {info.filename}")
            dest = staging / rel
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(dest, mode)
            elif rel.suffix in (".so", ".dylib"):
                cur = dest.stat().st_mode
                os.chmod(dest, cur | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    moved_old = False
    try:
        if overlay_dir.exists():
            os.replace(overlay_dir, backup)
            moved_old = True
        os.replace(staging, overlay_dir)
    except Exception:
        if moved_old and not overlay_dir.exists() and backup.exists():
            os.replace(backup, overlay_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return overlay_dir


def verify_version_stamp(sha: str, version: str, *, short_sha: str | None = None) -> None:
    """Confirm imported ``mlx.__version__`` matches the checked-out SHA."""
    if not version:
        die("mlx.__version__ is empty -- cannot verify build provenance")
    needle = short_sha or sha[:7]
    if needle not in version and sha[:7] not in version:
        die(
            f"built {sha[:12]} ({needle}) but imported mlx reports {version!r} -- "
            "stale wheel or wrong PYTHONPATH. Re-run with --clean."
        )


def _probe_python_mlx(target_py: Path, overlay_dir: Path | None) -> dict:
    probe = (
        "import json, mlx.core as mx; "
        "from pathlib import Path; "
        "sdpa = mx.fast.scaled_dot_product_attention; "
        "q = mx.zeros((1, 1, 1, 8), dtype=mx.float32); "
        "ws_ok = False\n"
        "try:\n"
        "    mx.eval(sdpa(q, q, q, scale=1.0, window_size=-1, stream=mx.cpu)); ws_ok = True\n"
        "except (TypeError, ValueError):\n"
        "    pass\n"
        "print(json.dumps({"
        "'file': getattr(mx, '__file__', ''), "
        "'version': getattr(mx, '__version__', ''), "
        "'metal': bool(mx.metal.is_available()), "
        "'window_size': ws_ok, "
        "'metallib': (Path(mx.__file__).parent / 'lib' / 'mlx.metallib').is_file() "
        "if getattr(mx, '__file__', None) else False"
        "}))"
    )
    env = os.environ.copy()
    if overlay_dir is not None:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{overlay_dir}{os.pathsep}{prev}" if prev else str(overlay_dir)
        )
    else:
        env.pop("PYTHONPATH", None)
    out = run([str(target_py), "-c", probe], capture=True, env=env)
    try:
        return json.loads(out.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        die(f"failed to parse mlx verification output ({out!r}): {exc}")


def verify_overlay(
    target_py: Path, overlay_dir: Path, sha: str, *, short_sha: str | None = None
) -> dict:
    info = _probe_python_mlx(target_py, overlay_dir)
    imported_file = info.get("file", "")
    # Namespace package `mlx` may report __file__ == None; check `mlx.core` or overlay path.
    metallib = overlay_dir / "mlx" / "lib" / "mlx.metallib"
    libmlx = overlay_dir / "mlx" / "lib" / "libmlx.dylib"
    if not metallib.is_file() or not libmlx.is_file():
        die(f"overlay {overlay_dir} is missing Metal runtime files (mlx.metallib/libmlx.dylib)")
    if imported_file and not Path(os.path.abspath(imported_file)).is_relative_to(overlay_dir):
        die(f"overlay check imported mlx from {imported_file}, expected inside {overlay_dir}")
    if not info.get("metal"):
        die("mx.metal.is_available() is False in built overlay")
    verify_version_stamp(sha, info.get("version", ""), short_sha=short_sha)
    return info


def install_wheel(target_py: Path, wheel: Path, sha: str, *, short_sha: str | None = None) -> dict:
    uv = shutil.which("uv")
    if not uv:
        die("uv not found on PATH")
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(target_py),
            "--no-deps",
            "--reinstall",
            str(wheel),
        ]
    )
    # Single-package source wheel bundles Metal libs inside `mlx/lib/`, so
    # remove stale PyPI `mlx_metal-*.dist-info` metadata (never `mlx/lib/`).
    site_probe = run(
        [str(target_py), "-c", "import mlx, pathlib; print(pathlib.Path(mlx.__path__[0]).parent)"],
        capture=True,
    )
    site_dir = Path(site_probe.splitlines()[-1].strip())
    for dist_info in site_dir.glob("mlx_metal-*.dist-info"):
        print(dim(f"removing stale metadata {dist_info}"))
        shutil.rmtree(dist_info, ignore_errors=True)
    info = _probe_python_mlx(target_py, None)
    if not info.get("metal"):
        die("mx.metal.is_available() is False after installing wheel")
    verify_version_stamp(sha, info.get("version", ""), short_sha=short_sha)
    print(
        yellow(
            "  note: installed into Python env. Plain `uv run` syncs back to "
            "uv.lock; use `uv run --no-sync` or `.venv/bin/python`."
        )
    )
    return info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _venv_mlx_version(target_py: Path) -> str:
    if not target_py.exists():
        return "python missing"
    proc = subprocess.run(
        [str(target_py), "-c", "import mlx.core as mx; print(mx.__version__)"],
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return "unavailable"
    return (proc.stdout or "").strip()


def print_status(path: Path, target_py: Path) -> None:
    manifest = read_manifest(path)
    print(f"\n{bold('mlx')}  {dim(str(path))}")
    if not manifest:
        print(yellow("  not built -- run: uv run scripts/build_mlx.py --rev main --overlay"))
    else:
        print(f"  rev      {manifest.get('rev')}  {dim(manifest.get('version', ''))}")
        print(f"  sha      {manifest.get('sha', '')[:12]}")
        prs = manifest.get("prs") or []
        if prs:
            pr_desc = ", ".join(f"#{p['number']}@{str(p['sha'])[:8]}" for p in prs)
            print(f"  prs      {pr_desc}")
        overlay = manifest.get("overlay", "")
        if overlay:
            print(f"  overlay  {green(overlay) if Path(overlay).exists() else yellow(overlay + ' (missing)')}")
        wheel = manifest.get("wheel", "")
        if wheel:
            print(f"  wheel    {dim(wheel)}")
        if "has_window_size_sdpa" in manifest:
            print(f"  sdpa_ws  {manifest['has_window_size_sdpa']}")
    canonical_overlay = path / "overlay"
    if (canonical_overlay / "mlx").is_dir():
        print(f"  auto     {green(str(canonical_overlay))} (heylookllm canonical build)")
    else:
        print(f"  auto     {dim('inactive (no canonical overlay at ' + str(canonical_overlay) + ')')}")
    env = (os.environ.get("HEYLOOK_MLX_OVERLAY") or "").strip()
    if env:
        if env.lower() in {"0", "off", "false", "no", "none", "pypi", "disabled"}:
            print(yellow(f"  note: $HEYLOOK_MLX_OVERLAY={env} disables canonical overlay discovery."))
        elif not canonical_overlay.exists() or Path(env).expanduser().resolve() != canonical_overlay.resolve():
            print(yellow(
                f"  note: $HEYLOOK_MLX_OVERLAY points at {env} -- heylookllm "
                "will use THAT overlay instead of the canonical build."
            ))
    print(f"  venv     {target_py} -> {_venv_mlx_version(target_py)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rev",
        default=None,
        help="tag, branch, or SHA to build (default: newest vX.Y.Z tag; `main` = origin/main)",
    )
    ap.add_argument(
        "--pr",
        action="append",
        type=int,
        default=[],
        help="GitHub PR number to merge atop --rev before building (repeatable)",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the recorded base_sha + PRs (or sha) from the existing manifest",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="report current build manifest and .venv mlx version; no network, no writes",
    )
    ap.add_argument(
        "--dir",
        default=None,
        help="checkout location (default: ~/.heylook/mlx; $HEYLOOK_MLX_DIR also overrides)",
    )
    ap.add_argument(
        "--overlay-dir",
        default=None,
        help="unpacked overlay directory (default: <dir>/overlay)",
    )
    ap.add_argument(
        "--python",
        default=None,
        help="target python interpreter for wheel ABI and verification (default: repo .venv)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 8,
        help="parallel build jobs",
    )
    ap.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra cmake argument passed via CMAKE_ARGS (repeatable)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove build/ and stale wheels before building",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="unpack wheel to --overlay-dir for PYTHONPATH isolation (default behavior)",
    )
    ap.add_argument(
        "--install",
        action="store_true",
        help="install built wheel into --python environment (replaces PyPI mlx in .venv)",
    )
    args = ap.parse_args()

    path = mlx_dir(args.dir)
    target_py = resolve_python(args.python)

    if args.status:
        print_status(path, target_py)
        return

    overlay_dir = (
        Path(os.path.abspath(os.path.expanduser(args.overlay_dir)))
        if args.overlay_dir
        else path / "overlay"
    )

    if args.rebuild:
        prev = read_manifest(path)
        if not prev:
            die("--rebuild needs an existing build manifest")
        recorded_prs = prev.get("prs") or []
        rev = str(
            (prev.get("base_sha") if recorded_prs else None)
            or prev.get("sha")
            or prev.get("rev")
            or die("--rebuild manifest is missing sha/rev")
        )
        prs_to_apply: list[int | dict] = list(recorded_prs)
    else:
        rev = args.rev or latest_release_tag()
        prs_to_apply = list(args.pr)

    fresh = not (path / ".git").exists()
    print(
        f"\n{bold('mlx')} {yellow(rev)} -> {path}"
        f"{dim('  (fresh clone + build tree)') if fresh else ''}"
    )

    ensure_checkout(path)
    base_sha = checkout_rev(path, rev)
    applied_prs = apply_prs(path, prs_to_apply) if prs_to_apply else []
    head_sha = run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)
    short_sha = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture=True)

    wheel = build_wheel(
        path,
        target_py=target_py,
        jobs=args.jobs,
        clean=args.clean,
        cmake_args=args.cmake_arg,
    )
    unpack_wheel_to_overlay(wheel, overlay_dir)
    info = verify_overlay(target_py, overlay_dir, head_sha, short_sha=short_sha)
    if args.install:
        info = install_wheel(target_py, wheel, head_sha, short_sha=short_sha)

    build_dir = _manifest_dir(path)
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rev": rev,
        "base_sha": base_sha,
        "sha": head_sha,
        "prs": applied_prs,
        "cmake_args": list(args.cmake_arg),
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "overlay": str(overlay_dir),
        "installed": bool(args.install),
        "version": info.get("version", ""),
        "has_window_size_sdpa": bool(info.get("window_size", False)),
        "python": str(target_py),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  built   {green(str(wheel))}")
    print(f"  overlay {green(str(overlay_dir))}")
    print(
        f"  version {info.get('version')}  "
        f"(metal={info.get('metal')}, window_size={info.get('window_size')})"
    )
    canonical_overlay = path / "overlay"
    if overlay_dir.resolve() == canonical_overlay.resolve():
        current = (os.environ.get("HEYLOOK_MLX_OVERLAY") or "").strip()
        if current:
            print(yellow(f"  note    $HEYLOOK_MLX_OVERLAY={current} overrides auto-discovery."))
        else:
            print(dim(
                f"  auto    heylookllm uses {overlay_dir} automatically "
                "(HEYLOOK_MLX_OVERLAY=0 for .venv PyPI)"
            ))
    else:
        print(dim(f'  use     HEYLOOK_MLX_OVERLAY="{overlay_dir}" heylookllm ...'))


if __name__ == "__main__":
    main()
