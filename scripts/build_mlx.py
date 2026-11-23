#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Unified MLX stack updater: syncs pure-Python overlays + builds ml-explore/mlx.

Mirrors ``scripts/build_llama.py`` for the entire MLX stack, keeping
``pyproject.toml`` and ``uv.lock`` on clean PyPI releases while your machine
tracks upstream ``main`` (and recommended open PRs) under ``~/.heylook/``:
  - ``mlx-vlm`` (and optional ``--with`` / ``--only`` pure-Python packages) ->
    ``~/.heylook/overlays/<pkg>`` (``$HEYLOOK_OVERLAYS_DIR``)
  - ``mlx`` C++/Metal wheel unpacked -> ``~/.heylook/mlx/overlay``
    (``$HEYLOOK_MLX_DIR/overlay``)

``src/heylook_llm/mlx_overlay.py`` auto-discovers every staged overlay at
``heylookllm`` and worker startup with zero config, standing down automatically
on ABI mismatch or whenever ``.venv``'s PyPI release is strictly newer. Passing
``HEYLOOK_OVERLAYS=0`` (or per-package ``HEYLOOK_MLX_OVERLAY=0`` /
``HEYLOOK_MLX_VLM_OVERLAY=0``) returns to ``.venv`` PyPI for a single run.

PRs in ``RECOMMENDED_MLX_PRS`` that have already merged into ``origin/main`` are
detected via git ancestry/tree checks and skipped automatically (pure-Python
overlays like ``mlx-vlm`` track upstream git commits directly with no PRs), and
unchanged revisions skip staging and C++ compilation unless ``--rebuild`` or
``--clean`` is passed.

Examples:
  uv run scripts/build_mlx.py                         # update mlx-vlm (main) + mlx (main + recommended PRs)
  uv run scripts/build_mlx.py --status                # inspect mlx + pure-Python overlays + .venv versions
  uv run scripts/build_mlx.py --only mlx-vlm          # refresh mlx-vlm only (fast, no C++ compile)
  uv run scripts/build_mlx.py --only mlx --no-prs     # build clean mlx main with no PRs
  uv run scripts/build_mlx.py --only mlx --rev release --no-prs
  uv run scripts/build_mlx.py --with transformers     # also track transformers main alongside mlx-vlm + mlx
  uv run scripts/build_mlx.py --remove mlx-vlm        # remove one overlay (or `--remove` / `--remove all`)
  uv run scripts/build_mlx.py --upgrade-pypi          # opt-in `uv lock --upgrade && uv sync` before syncing
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
OVERLAYS_SUBDIR = (".heylook", "overlays")

DEFAULT_MLX_REV = "main"
# #4890 merged into upstream main on 2026-11-21; #4826 remains open.
# `apply_prs` checks `git merge-base --is-ancestor` and tree equivalence, so any
# recommended PR that later merges upstream is skipped cleanly without failing.
RECOMMENDED_MLX_PRS: tuple[int, ...] = (4826,)

PYTHON_PRESETS: dict[str, dict[str, str]] = {
    "mlx-vlm": {
        "repo": "https://github.com/Neywa-Labs/mlx-vlm",
        "import_name": "mlx_vlm",
        "default_rev": "main",
    },
    "transformers": {
        "repo": "https://github.com/huggingface/transformers",
        "import_name": "transformers",
        "default_rev": "main",
    },
}
DEFAULT_PYTHON_PACKAGES: tuple[str, ...] = ("mlx-vlm",)

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
    """--dir > $HEYLOOK_MLX_DIR > ~/.heylook/mlx."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()
    env = os.environ.get("HEYLOOK_MLX_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*HOME_SUBDIR).resolve()


def overlays_dir(cli_dir: str | None) -> Path:
    """--overlays-dir > $HEYLOOK_OVERLAYS_DIR > ~/.heylook/overlays."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()
    env = os.environ.get("HEYLOOK_OVERLAYS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*OVERLAYS_SUBDIR).resolve()


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


def normalize_pkg_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _parse_pr_arg(raw: str) -> int:
    s = str(raw).strip().lstrip("#")
    if not s.isdigit():
        die(f"invalid --pr value '{raw}' (expected integer PR number)")
    return int(s)


def resolve_targets(
    only: list[str] | tuple[str, ...] = (),
    with_pkgs: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Compute ordered update targets: defaults (``mlx-vlm``, ``mlx``) or ``--only``."""
    if only and with_pkgs:
        die("cannot combine --only and --with")
    if only:
        out: list[str] = []
        for item in only:
            norm = normalize_pkg_name(item)
            if norm not in out:
                out.append(norm)
        return out

    out = [normalize_pkg_name(p) for p in DEFAULT_PYTHON_PACKAGES]
    for item in with_pkgs:
        norm = normalize_pkg_name(item)
        if norm != "mlx" and norm not in out:
            out.append(norm)
    if "mlx" not in out:
        out.append("mlx")
    return out


def resolve_mlx_prs(
    cli_prs: list[int] | tuple[int, ...] = (),
    *,
    no_prs: bool = False,
) -> list[int]:
    """Resolve PR list for MLX: explicit ``--no-prs`` or ``--pr``, else recommended PRs."""
    if no_prs:
        return []
    if cli_prs:
        return list(dict.fromkeys(int(p) for p in cli_prs))
    return list(RECOMMENDED_MLX_PRS)


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


def _local_latest_release_tag(path: Path) -> str:
    """Newest local vX.Y.Z release tag when running ``--rev release --offline``."""
    out = run(["git", "-C", str(path), "tag", "-l", "v*"], capture=True)
    best: tuple[tuple[int, int, int], str] | None = None
    for tag in out.splitlines():
        tag = tag.strip()
        m = _VERSION_TAG_RE.match(tag)
        if m:
            ver = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if best is None or ver > best[0]:
                best = (ver, tag)
    if best is None:
        die(f"no local vX.Y.Z release tags found in {path}")
    return best[1]


def _remote_key(url: str) -> str:
    s = url.strip().removesuffix(".git")
    s = re.sub(r"^[a-z+]+://", "", s)
    s = s.replace(":", "/", 1) if s.startswith("git@") else s
    return s.removeprefix("git@").lower()


def ensure_checkout(path: Path, *, offline: bool = False) -> None:
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
    if offline:
        die(f"mlx checkout missing at {path} (cannot clone with --offline)")
    if path.exists() and any(path.iterdir()):
        die(f"{path} exists and is not an mlx checkout -- refusing to clone over it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", GIT_URL, str(path)])


def ensure_python_repo(repo_dir: Path, repo_url: str, *, offline: bool) -> None:
    """Ensure local git mirror exists for a pure-Python upstream repo and fetch updates."""
    if not (repo_dir / ".git").exists():
        if repo_dir.exists() and any(repo_dir.iterdir()):
            die(f"{repo_dir} exists and is not a git checkout -- refusing to overwrite.")
        if offline:
            die(f"repo mirror missing at {repo_dir} (cannot clone with --offline)")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", repo_url, str(repo_dir)])
        return

    origin = run(["git", "-C", str(repo_dir), "remote", "get-url", "origin"], capture=True)
    if _remote_key(origin) != _remote_key(repo_url):
        run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    if not offline:
        run(["git", "-C", str(repo_dir), "fetch", "--prune", "--tags", "--force", "origin"])


def _ref_exists(path: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _is_ancestor(path: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "merge-base", "--is-ancestor", ancestor, descendant],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
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
                f"  {rev!r} is a branch: checking out {remote} ({tracking}), not "
                f"the local branch ({local}), which fetch never advances"
            )
    return remote


def resolve_python_rev(repo_dir: Path, rev: str) -> str:
    """Resolve branch names to ``origin/<branch>`` commit SHA, falling back to tag/SHA."""
    for candidate in (f"origin/{rev}", rev):
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    die(f"could not resolve rev '{rev}' in {repo_dir}")


def checkout_rev(path: Path, rev: str, *, offline: bool = False) -> str:
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
    if not offline:
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
    target = resolve_fetched_rev(path, rev)
    run(["git", "-C", str(path), "checkout", "--detach", target])
    return run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)


def apply_prs(
    path: Path, prs: list[int | dict], *, offline: bool = False
) -> list[dict[str, object]]:
    """Fetch and merge unmerged PR refs (or pinned SHAs) atop HEAD, skipping merged PRs."""
    applied: list[dict[str, object]] = []
    for item in prs:
        if isinstance(item, dict):
            num = int(item["number"])
            pinned_sha = str(item["sha"])
            if not _ref_exists(path, pinned_sha):
                if offline:
                    die(f"PR #{num} pinned SHA {pinned_sha[:12]} missing locally with --offline")
                run(["git", "-C", str(path), "fetch", "origin", f"refs/pull/{num}/head"])
            target = pinned_sha
            pr_sha = pinned_sha
        else:
            num = int(item)
            if offline:
                die(f"cannot fetch PR #{num} with --offline")
            run(["git", "-C", str(path), "fetch", "origin", f"refs/pull/{num}/head"])
            target = "FETCH_HEAD"
            pr_sha = run(["git", "-C", str(path), "rev-parse", "FETCH_HEAD"], capture=True)

        if _is_ancestor(path, pr_sha, "HEAD"):
            print(dim(f"  PR #{num} ({pr_sha[:12]}) is already merged into HEAD -- skipping"))
            continue

        before_head = run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)
        before_tree = run(["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], capture=True)

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

        after_tree = run(["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], capture=True)
        if after_tree == before_tree:
            run(["git", "-C", str(path), "reset", "--hard", before_head], capture=True)
            print(dim(f"  PR #{num} ({pr_sha[:12]}) already applied in HEAD -- skipping"))
            continue

        applied.append({"number": num, "sha": pr_sha})
    return applied


def _manifest_dir(path: Path) -> Path:
    return path / "build"


def manifest_path(path: Path) -> Path:
    return _manifest_dir(path) / "heylook-build.json"


def _read_json(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_manifest(path: Path) -> dict:
    return _read_json(manifest_path(path))


def _drop_cmake_caches(build_root: Path) -> bool:
    removed = False
    if not build_root.exists():
        return False
    for cache_file in build_root.rglob("CMakeCache.txt"):
        cache_file.unlink(missing_ok=True)
        removed = True
    return removed


def mlx_is_up_to_date(
    path: Path,
    overlay_dir: Path,
    *,
    base_sha: str,
    applied_prs: list[dict[str, object]],
    cmake_args: list[str],
    target_py: Path,
) -> dict | None:
    """Return existing manifest if built MLX overlay matches resolved source + config."""
    prev = read_manifest(path)
    if not prev:
        return None
    if prev.get("base_sha") != base_sha:
        return None
    if (prev.get("prs") or []) != applied_prs:
        return None
    if (prev.get("cmake_args") or []) != list(cmake_args):
        return None
    if prev.get("python") != str(target_py):
        return None
    prev_overlay = prev.get("overlay")
    if not prev_overlay or Path(os.path.expanduser(str(prev_overlay))).resolve() != overlay_dir.resolve():
        return None
    if not (overlay_dir / "mlx" / "lib" / "mlx.metallib").is_file():
        return None
    if not (overlay_dir / "mlx" / "lib" / "libmlx.dylib").is_file():
        return None
    return prev


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


def _atomic_replace_dir(staging: Path, target_dir: Path) -> None:
    backup = target_dir.parent / f".{target_dir.name}.old"
    if backup.exists():
        shutil.rmtree(backup)
    moved_old = False
    try:
        if target_dir.exists():
            os.replace(target_dir, backup)
            moved_old = True
        os.replace(staging, target_dir)
    except Exception:
        if moved_old and not target_dir.exists() and backup.exists():
            os.replace(backup, target_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def unpack_wheel_to_overlay(wheel: Path, overlay_dir: Path) -> Path:
    """Atomically unpack ``wheel`` into ``overlay_dir`` preserving exec bits."""
    overlay_dir = Path(os.path.abspath(os.path.expanduser(str(overlay_dir))))
    overlay_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = overlay_dir.parent / f".{overlay_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
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

        _atomic_replace_dir(staging, overlay_dir)
    finally:
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


# ── Pure-Python overlay helpers (mlx-vlm, transformers, etc.) ─────────────────


def _read_dist_metadata(staging: Path, pkg: str) -> tuple[str | None, list[str]]:
    norm_under = re.sub(r"[-_.]+", "_", pkg).lower()
    norm_dash = norm_under.replace("_", "-")
    version: str | None = None
    requires: list[str] = []
    for meta in sorted(staging.glob("*.dist-info/METADATA")):
        prefix = meta.parent.name.split("-", 1)[0].lower()
        if prefix not in (norm_under, norm_dash):
            continue
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            if version is None and line.lower().startswith("version:"):
                version = line.split(":", 1)[1].strip()
            elif line.lower().startswith("requires-dist:"):
                requires.append(line.split(":", 1)[1].strip())
        break
    return version, requires


def _check_missing_deps(python_exe: Path, requires: list[str]) -> list[str]:
    """Compare upstream non-extra Requires-Dist against packages installed in python_exe."""
    if not requires:
        return []
    req_names: list[str] = []
    for raw in requires:
        if "; " in raw and "extra ==" in raw.split("; ", 1)[1]:
            continue
        if ";" in raw and "extra" in raw.split(";", 1)[1]:
            continue
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)", raw)
        if m:
            req_names.append(normalize_pkg_name(m.group(1)))
    if not req_names:
        return []

    code = (
        "import importlib.metadata as m, json, re\n"
        "have = {re.sub(r'[-_.]+', '-', d.metadata['Name']).lower() "
        "for d in m.distributions() if d.metadata and d.metadata.get('Name')}\n"
        "print(json.dumps(sorted(have)))\n"
    )
    proc = subprocess.run(
        [str(python_exe), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    try:
        installed = set(json.loads(proc.stdout))
    except ValueError:
        return []
    return sorted({r for r in req_names if r not in installed})


def python_overlay_is_up_to_date(
    target_overlay: Path,
    *,
    import_name: str,
    sha: str,
    target_py: Path,
) -> dict | None:
    """Return manifest if pure-Python overlay already matches ``sha`` + Python."""
    prev = _read_json(target_overlay / "heylook-build.json")
    if not prev:
        return None
    has_pkg = (
        (target_overlay / import_name).is_dir()
        or (target_overlay / f"{import_name}.py").is_file()
    )
    if not has_pkg:
        return None
    if prev.get("sha") != sha:
        return None
    if prev.get("python") != str(target_py):
        return None
    return prev


def stage_python_overlay(
    repo_dir: Path,
    target_overlay: Path,
    *,
    pkg: str,
    import_name: str,
    repo_url: str,
    rev: str,
    sha: str,
    python_exe: Path,
) -> dict:
    """Install pure-Python repo with ``--no-deps --target`` and swap into ``target_overlay``."""
    uv_bin = shutil.which("uv")
    if not uv_bin:
        die("`uv` not found on PATH; required to stage pure-Python overlays")

    target_overlay = Path(os.path.abspath(os.path.expanduser(str(target_overlay))))
    target_overlay.parent.mkdir(parents=True, exist_ok=True)
    staging = target_overlay.parent / f".{target_overlay.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        run(
            [
                uv_bin,
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(staging),
                "--python",
                str(python_exe),
                str(repo_dir),
            ]
        )

        if not ((staging / import_name).is_dir() or (staging / f"{import_name}.py").is_file()):
            die(
                f"staged overlay at {staging} has no '{import_name}' module/package; "
                "pass --import-name if the import name differs from the distribution name"
            )

        version, requires = _read_dist_metadata(staging, pkg)
        missing = _check_missing_deps(python_exe, requires)
        if missing:
            print(
                yellow(
                    f"warning: upstream {pkg} declares dependencies not in {python_exe}: "
                    + ", ".join(missing)
                )
            )

        manifest = {
            "package": pkg,
            "import_name": import_name,
            "repo": repo_url,
            "rev": rev,
            "sha": sha,
            "version": version or "",
            "overlay": str(target_overlay),
            "python": str(python_exe),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        (staging / "heylook-build.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        _atomic_replace_dir(staging, target_overlay)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return manifest


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


def _venv_dist_versions(target_py: Path, dist_names: list[str]) -> dict[str, str]:
    if not dist_names or not target_py.exists():
        return {}
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out = {}\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        out[name] = m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [str(target_py), "-c", code, *dist_names],
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return {}
    try:
        parsed = json.loads(proc.stdout)
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


def print_python_overlays_status(root: Path, python_exe: Path) -> None:
    """Print status of staged pure-Python overlays under ``root`` (no network/writes)."""
    print(f"\n{bold('pure-python overlays')}  {dim(str(root))}")
    if not root.is_dir():
        print(dim("  (none staged -- server uses .venv PyPI releases)"))
        return

    found = [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and (d / "heylook-build.json").is_file()
    ]
    if not found:
        print(dim("  (none staged -- server uses .venv PyPI releases)"))
        return

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from heylook_llm.mlx_overlay import resolve_python_overlay

    env = dict(os.environ)
    env["HEYLOOK_OVERLAYS_DIR"] = str(root)

    records: list[tuple[Path, dict, str, str]] = []
    for ov in found:
        data = _read_json(ov / "heylook-build.json")
        pkg = str(data.get("package") or ov.name)
        imp = str(data.get("import_name") or pkg.replace("-", "_"))
        records.append((ov, data, pkg, imp))

    venv_vers = _venv_dist_versions(python_exe, [pkg for _, _, pkg, _ in records])

    for ov, data, pkg, imp in records:
        sha = str(data.get("sha") or "")[:10]
        ver = str(data.get("version") or "?")
        prs = [
            f"#{p['number']}"
            for p in (data.get("prs") or [])
            if isinstance(p, dict) and p.get("number")
        ]
        pr_str = f" ({', '.join(prs)})" if prs else ""
        v_ver = venv_vers.get(pkg)
        dec = resolve_python_overlay(
            pkg,
            imp,
            environ=env,
            venv_version=v_ver,
            already_imported=False,
        )
        state = (
            green(dec.action)
            if dec.action in ("canonical", "override")
            else yellow(dec.action)
        )
        venv_note = f"  venv={v_ver}" if v_ver else ""
        print(f"  {bold(pkg):<18} {ver} @ {sha}{pr_str}  [{state}]{venv_note}  {dim(str(ov))}")
        if dec.warning:
            print(f"    {yellow(dec.warning)}")


def print_status(
    path: Path, target_py: Path, overlays_root: Path | None = None
) -> None:
    manifest = read_manifest(path)
    print(f"\n{bold('mlx')}  {dim(str(path))}")
    if not manifest:
        print(yellow("  not built -- run: uv run scripts/build_mlx.py"))
    else:
        print(f"  rev      {manifest.get('rev')}  {dim(manifest.get('version', ''))}")
        print(f"  sha      {manifest.get('sha', '')[:12]}")
        prs = manifest.get("prs") or []
        if prs:
            pr_desc = ", ".join(f"#{p['number']}@{str(p['sha'])[:8]}" for p in prs)
            print(f"  prs      {pr_desc}")
        overlay = manifest.get("overlay", "")
        if overlay:
            print(
                f"  overlay  {green(overlay) if Path(overlay).exists() else yellow(overlay + ' (missing)')}"
            )
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
            print(
                yellow(
                    f"  note: $HEYLOOK_MLX_OVERLAY points at {env} -- heylookllm "
                    "will use THAT overlay instead of the canonical build."
                )
            )
    print(f"  venv     {target_py} -> {_venv_mlx_version(target_py)}")

    if overlays_root is not None:
        print_python_overlays_status(overlays_root, target_py)


# ── Target runners ────────────────────────────────────────────────────────────


def sync_python_package(
    pkg: str,
    *,
    overlays_root: Path,
    target_py: Path,
    repo_url: str,
    import_name: str,
    rev: str,
    rebuild: bool,
    clean: bool,
    offline: bool,
) -> dict:
    target = overlays_root / pkg
    repo_dir = overlays_root / ".repos" / pkg
    fresh = not (repo_dir / ".git").exists()
    print(
        f"\n{bold(pkg)} {yellow(rev)} -> {target}"
        f"{dim('  (fresh clone)') if fresh else ''}"
    )

    ensure_python_repo(repo_dir, repo_url, offline=offline)
    sha = resolve_python_rev(repo_dir, rev)

    dirty = run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        capture=True,
    )
    if dirty:
        run(["git", "-C", str(repo_dir), "merge", "--abort"], check=False)
    run(["git", "-C", str(repo_dir), "checkout", "--detach", "--force", sha])
    run(["git", "-C", str(repo_dir), "clean", "-fd"])

    if not clean and not rebuild:
        existing = python_overlay_is_up_to_date(
            target,
            import_name=import_name,
            sha=sha,
            target_py=target_py,
        )
        if existing is not None:
            ver = existing.get("version") or "?"
            print(
                green(
                    f"  up to date  {pkg} {ver} @ {sha[:10]} "
                    f"(pass --rebuild to force)"
                )
            )
            return existing

    manifest = stage_python_overlay(
        repo_dir,
        target,
        pkg=pkg,
        import_name=import_name,
        repo_url=repo_url,
        rev=rev,
        sha=sha,
        python_exe=target_py,
    )
    print(
        green(
            f"  staged  {pkg} {manifest['version'] or '?'} @ {sha[:10]} -> {target}"
        )
    )
    return manifest


def sync_mlx(
    *,
    path: Path,
    overlay_dir: Path,
    target_py: Path,
    rev: str,
    prs_to_apply: list[int | dict],
    jobs: int,
    clean: bool,
    rebuild: bool,
    offline: bool,
    install: bool,
    cmake_args: list[str],
) -> dict:
    fresh = not (path / ".git").exists()
    print(
        f"\n{bold('mlx')} {yellow(rev)} -> {path}"
        f"{dim('  (fresh clone + build tree)') if fresh else ''}"
    )

    ensure_checkout(path, offline=offline)
    base_sha = checkout_rev(path, rev, offline=offline)
    applied_prs = apply_prs(path, prs_to_apply, offline=offline) if prs_to_apply else []
    head_sha = run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)
    short_sha = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture=True)

    if not clean and not rebuild and not install:
        existing = mlx_is_up_to_date(
            path,
            overlay_dir,
            base_sha=base_sha,
            applied_prs=applied_prs,
            cmake_args=cmake_args,
            target_py=target_py,
        )
        if existing is not None:
            pr_note = (
                f" (+ {', '.join('#' + str(p['number']) for p in applied_prs)})"
                if applied_prs
                else ""
            )
            print(
                green(
                    f"  up to date  {existing.get('version', '')} @ {head_sha[:12]}{pr_note} "
                    f"(pass --rebuild to force)"
                )
            )
            return existing

    wheel = build_wheel(
        path,
        target_py=target_py,
        jobs=jobs,
        clean=clean,
        cmake_args=cmake_args,
    )
    unpack_wheel_to_overlay(wheel, overlay_dir)
    info = verify_overlay(target_py, overlay_dir, head_sha, short_sha=short_sha)
    if install:
        info = install_wheel(target_py, wheel, head_sha, short_sha=short_sha)

    build_dir = _manifest_dir(path)
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rev": rev,
        "base_sha": base_sha,
        "sha": head_sha,
        "prs": applied_prs,
        "cmake_args": list(cmake_args),
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "overlay": str(overlay_dir),
        "installed": bool(install),
        "version": info.get("version", ""),
        "has_window_size_sdpa": bool(info.get("window_size", False)),
        "python": str(target_py),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
            print(
                dim(
                    f"  auto    heylookllm uses {overlay_dir} automatically "
                    "(HEYLOOK_MLX_OVERLAY=0 for .venv PyPI)"
                )
            )
    else:
        print(dim(f'  use     HEYLOOK_MLX_OVERLAY="{overlay_dir}" heylookllm ...'))
    return manifest


def remove_overlays(
    target_spec: str,
    *,
    only: list[str],
    mlx_overlay_dir: Path,
    overlays_root: Path,
) -> None:
    """Remove staged overlay directories so ``.venv`` PyPI releases take over cleanly."""
    if only and target_spec == "all":
        targets = [normalize_pkg_name(p) for p in only]
    elif target_spec == "all":
        removed_any = False
        if mlx_overlay_dir.exists():
            shutil.rmtree(mlx_overlay_dir)
            print(green(f"removed mlx overlay: {mlx_overlay_dir}"))
            removed_any = True
        if overlays_root.is_dir():
            for child in sorted(overlays_root.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    shutil.rmtree(child)
                    print(green(f"removed overlay: {child}"))
                    removed_any = True
        if not removed_any:
            print(dim("no overlays staged"))
        return
    else:
        targets = [normalize_pkg_name(target_spec)]

    for pkg in dict.fromkeys(targets):
        if pkg == "mlx":
            if mlx_overlay_dir.exists():
                shutil.rmtree(mlx_overlay_dir)
                print(green(f"removed mlx overlay: {mlx_overlay_dir}"))
            else:
                print(dim(f"no mlx overlay staged at {mlx_overlay_dir}"))
        else:
            ov = overlays_root / pkg
            if ov.exists():
                shutil.rmtree(ov)
                print(green(f"removed overlay: {ov}"))
            else:
                print(dim(f"no overlay staged at {ov}"))


def upgrade_pypi_env() -> None:
    """Opt-in refresh of ``uv.lock`` + ``.venv`` PyPI packages."""
    uv = shutil.which("uv")
    if not uv:
        die("uv not found on PATH")
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "pyproject.toml").is_file():
        die(f"pyproject.toml not found at {repo_root}")
    print(f"\n{bold('pypi')} upgrading uv.lock and syncing .venv in {repo_root}")
    run([uv, "lock", "--upgrade"], cwd=repo_root)
    run([uv, "sync"], cwd=repo_root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PKG",
        help="update only specified target(s) (`mlx`, `mlx-vlm`, `transformers`, or custom --repo); repeatable",
    )
    ap.add_argument(
        "--with",
        dest="with_pkgs",
        action="append",
        default=[],
        metavar="PKG",
        help="also stage additional pure-Python package(s) (e.g. `transformers`) alongside default mlx-vlm + mlx; repeatable",
    )
    ap.add_argument(
        "--rev",
        default=None,
        help="tag, branch, or SHA (`main` = origin/main, `release` = newest mlx vX.Y.Z tag). Applies to --only target or mlx in default mode.",
    )
    ap.add_argument(
        "--pr",
        action="append",
        type=_parse_pr_arg,
        default=[],
        help="GitHub PR number to merge atop mlx --rev (repeatable; mlx only, overrides recommended PRs)",
    )
    ap.add_argument(
        "--no-prs",
        action="store_true",
        help="build clean mlx --rev without merging recommended MLX PRs",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="force rebuild/re-stage even if already up to date (rebuilds manifest base_sha+PRs for mlx when no --rev/--pr given)",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="report current mlx + pure-Python overlays and .venv versions; no network, no writes",
    )
    ap.add_argument(
        "--remove",
        nargs="?",
        const="all",
        default=None,
        metavar="TARGET",
        help="remove staged overlay(s) (`mlx`, `mlx-vlm`, `<pkg>`, or `all`) so .venv PyPI takes over",
    )
    ap.add_argument(
        "--upgrade-pypi",
        action="store_true",
        help="run `uv lock --upgrade && uv sync` before updating overlays (does not touch overlays unless they are behind PyPI)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="do not clone or fetch from git remotes",
    )
    ap.add_argument(
        "--repo",
        default=None,
        help="git repository URL for a custom non-preset `--only <pkg>` target",
    )
    ap.add_argument(
        "--import-name",
        default=None,
        help="Python import name if different from package name for `--only <pkg>`",
    )
    ap.add_argument(
        "--dir",
        default=None,
        help="mlx checkout location (default: ~/.heylook/mlx; $HEYLOOK_MLX_DIR also overrides)",
    )
    ap.add_argument(
        "--overlay-dir",
        default=None,
        help="unpacked mlx overlay directory (default: <dir>/overlay)",
    )
    ap.add_argument(
        "--overlays-dir",
        default=None,
        help="pure-Python overlays directory (default: ~/.heylook/overlays; $HEYLOOK_OVERLAYS_DIR also overrides)",
    )
    ap.add_argument(
        "--python",
        default=None,
        help="target python interpreter for ABI, dep checks, and verification (default: repo .venv)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 8,
        help="parallel mlx build jobs",
    )
    ap.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra cmake argument passed via CMAKE_ARGS for mlx (repeatable)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove mlx build/ and stale wheels, and force re-stage pure-Python overlays",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="unpack wheel to --overlay-dir for PYTHONPATH isolation (default behavior)",
    )
    ap.add_argument(
        "--install",
        action="store_true",
        help="install built mlx wheel into --python environment (replaces PyPI mlx in .venv)",
    )
    args = ap.parse_args(argv)

    mlx_path = mlx_dir(args.dir)
    py_root = overlays_dir(args.overlays_dir)
    target_py = resolve_python(args.python)
    overlay_dir = (
        Path(os.path.abspath(os.path.expanduser(args.overlay_dir)))
        if args.overlay_dir
        else mlx_path / "overlay"
    )

    if args.status:
        print_status(mlx_path, target_py, overlays_root=py_root)
        return 0

    if args.remove is not None:
        remove_overlays(
            args.remove,
            only=args.only,
            mlx_overlay_dir=overlay_dir,
            overlays_root=py_root,
        )
        return 0

    targets = resolve_targets(args.only, args.with_pkgs)

    if (args.repo or args.import_name) and (
        len(args.only) != 1 or normalize_pkg_name(args.only[0]) == "mlx"
    ):
        die("--repo and --import-name require a single pure-Python --only <package> target")
    if (args.pr or args.no_prs) and "mlx" not in targets:
        die("--pr and --no-prs only apply when building mlx")
    if len(args.only) > 1 and args.rev is not None:
        die("--rev is ambiguous across multiple --only targets; run one target at a time")
    if args.install and "mlx" not in targets:
        die("--install only applies when building mlx")

    if args.upgrade_pypi:
        upgrade_pypi_env()

    single_python_only = len(args.only) == 1 and targets[0] != "mlx"

    for pkg in targets:
        if pkg == "mlx":
            if (
                args.rebuild
                and args.rev is None
                and not args.pr
                and not args.no_prs
            ):
                prev = read_manifest(mlx_path)
                if not prev:
                    die("--rebuild needs an existing build manifest (or pass --rev)")
                recorded_prs = prev.get("prs") or []
                rev = str(
                    (prev.get("base_sha") if recorded_prs else None)
                    or prev.get("sha")
                    or prev.get("rev")
                    or die("--rebuild manifest is missing sha/rev")
                )
                prs_to_apply: list[int | dict] = list(recorded_prs)
            else:
                raw_rev = args.rev or DEFAULT_MLX_REV
                if raw_rev == "release":
                    rev = (
                        _local_latest_release_tag(mlx_path)
                        if args.offline
                        else latest_release_tag()
                    )
                else:
                    rev = raw_rev
                prs_to_apply = list(resolve_mlx_prs(args.pr, no_prs=args.no_prs))

            sync_mlx(
                path=mlx_path,
                overlay_dir=overlay_dir,
                target_py=target_py,
                rev=rev,
                prs_to_apply=prs_to_apply,
                jobs=args.jobs,
                clean=args.clean,
                rebuild=args.rebuild,
                offline=args.offline,
                install=args.install,
                cmake_args=args.cmake_arg,
            )
        else:
            preset = PYTHON_PRESETS.get(pkg, {})
            repo_url = (args.repo if single_python_only else None) or preset.get("repo")
            if not repo_url:
                die(f"unknown package '{pkg}' -- pass `--only {pkg} --repo <git-url>`")
            import_name = (
                (args.import_name if single_python_only else None)
                or preset.get("import_name")
                or pkg.replace("-", "_")
            )
            rev = (
                (args.rev if single_python_only else None)
                or preset.get("default_rev")
                or "main"
            )
            sync_python_package(
                pkg,
                overlays_root=py_root,
                target_py=target_py,
                repo_url=repo_url,
                import_name=import_name,
                rev=rev,
                rebuild=args.rebuild,
                clean=args.clean,
                offline=args.offline,
            )

    print(
        dim(
            "\nheylookllm discovers staged ~/.heylook/ overlays automatically "
            "(pass HEYLOOK_OVERLAYS=0 to fall back to .venv PyPI)."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
