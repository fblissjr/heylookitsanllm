#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Sync pure-Python upstream repos into out-of-repo local overlays.

Mirrors ``scripts/build_llama.py`` (``~/.heylook/llama.cpp``) and
``scripts/build_mlx.py`` (``~/.heylook/mlx/overlay``) for fast-moving
pure-Python upstream packages such as ``mlx-vlm`` and ``transformers``.

It never touches ``pyproject.toml`` or ``uv.lock`` -- public commits stay on
clean PyPI releases while your local machine tracks upstream ``main`` (and
optional PRs) under ``~/.heylook/overlays/<pkg>`` (relocatable via
``$HEYLOOK_OVERLAYS_DIR`` or ``--dir``). ``heylookllm`` auto-discovers every
staged overlay at startup and stands down automatically whenever ``.venv``'s
PyPI release is strictly newer.

Examples:
  uv run scripts/sync_overlay.py mlx-vlm                  # latest origin/main
  uv run scripts/sync_overlay.py mlx-vlm --rev a22a364a   # exact commit
  uv run scripts/sync_overlay.py mlx-vlm --pr 2767        # main + PR #2767
  uv run scripts/sync_overlay.py --status                 # inspect local overlays
  uv run scripts/sync_overlay.py mlx-vlm --remove         # return to .venv PyPI
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import NoReturn

HOME_SUBDIR = (".heylook", "overlays")

PRESETS: dict[str, dict[str, str]] = {
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


def overlays_dir(cli_dir: str | None) -> Path:
    """--dir > $HEYLOOK_OVERLAYS_DIR > ~/.heylook/overlays."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()
    env = os.environ.get("HEYLOOK_OVERLAYS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*HOME_SUBDIR).resolve()


def default_python() -> Path:
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
    s = raw.strip().lstrip("#")
    if not s.isdigit():
        die(f"invalid --pr value '{raw}' (expected integer PR number)")
    return int(s)


def ensure_repo(repo_dir: Path, repo_url: str, *, offline: bool) -> None:
    if not repo_dir.exists():
        if offline:
            die(f"repo mirror missing at {repo_dir} (cannot clone with --offline)")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(repo_dir)])
        return

    origin = run(["git", "remote", "get-url", "origin"], cwd=repo_dir, capture=True)
    if origin.rstrip("/").removesuffix(".git") != repo_url.rstrip("/").removesuffix(".git"):
        run(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir)
    if not offline:
        run(["git", "fetch", "--prune", "--tags", "origin"], cwd=repo_dir)


def resolve_rev(repo_dir: Path, rev: str) -> str:
    """Resolve branch names to origin/<branch> so local refs never go stale."""
    for candidate in (f"origin/{rev}", rev):
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    die(f"could not resolve rev '{rev}' in {repo_dir}")


def checkout_with_prs(
    repo_dir: Path,
    base_sha: str,
    prs: list[int],
    *,
    offline: bool,
) -> tuple[str, list[dict]]:
    run(["git", "checkout", "--detach", "--force", base_sha], cwd=repo_dir)
    run(["git", "clean", "-fd"], cwd=repo_dir)

    applied: list[dict] = []
    for pr_num in prs:
        ref = f"refs/pull/{pr_num}/head"
        if not offline:
            run(["git", "fetch", "origin", ref], cwd=repo_dir)
            pr_sha = run(["git", "rev-parse", "FETCH_HEAD"], cwd=repo_dir, capture=True)
        else:
            die(f"cannot fetch PR #{pr_num} with --offline")
        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "heylook-overlay")
        env.setdefault("GIT_AUTHOR_EMAIL", "overlay@localhost")
        env.setdefault("GIT_COMMITTER_NAME", "heylook-overlay")
        env.setdefault("GIT_COMMITTER_EMAIL", "overlay@localhost")
        run(
            ["git", "merge", "--no-edit", "--no-ff", pr_sha],
            cwd=repo_dir,
            env=env,
        )
        applied.append({"number": pr_num, "sha": pr_sha})

    final_sha = run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture=True)
    return final_sha, applied


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


def stage_overlay(
    repo_dir: Path,
    target_overlay: Path,
    *,
    pkg: str,
    import_name: str,
    repo_url: str,
    rev: str,
    sha: str,
    prs: list[dict],
    python_exe: Path,
) -> dict:
    uv_bin = shutil.which("uv")
    if not uv_bin:
        die("`uv` not found on PATH; required to stage pure-Python overlays")

    staging = target_overlay.parent / f".{target_overlay.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

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
        shutil.rmtree(staging, ignore_errors=True)
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
        "prs": prs,
        "overlay": str(target_overlay),
        "python": str(python_exe),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (staging / "heylook-build.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    backup = target_overlay.parent / f".{target_overlay.name}.old"
    if backup.exists():
        shutil.rmtree(backup)
    if target_overlay.exists():
        target_overlay.rename(backup)
    try:
        staging.rename(target_overlay)
    except Exception:
        if backup.exists() and not target_overlay.exists():
            backup.rename(target_overlay)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

    return manifest


def print_status(root: Path, python_exe: Path) -> None:
    print(bold(f"Local pure-Python overlays ({root})"))
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

    for ov in found:
        try:
            data = json.loads((ov / "heylook-build.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        pkg = str(data.get("package") or ov.name)
        imp = str(data.get("import_name") or pkg.replace("-", "_"))
        sha = str(data.get("sha") or "")[:10]
        ver = str(data.get("version") or "?")
        prs = [f"#{p['number']}" for p in (data.get("prs") or []) if isinstance(p, dict) and p.get("number")]
        pr_str = f" ({', '.join(prs)})" if prs else ""
        dec = resolve_python_overlay(pkg, imp, environ=env, already_imported=False)
        state = green(dec.action) if dec.action in ("canonical", "override") else yellow(dec.action)
        print(f"  {bold(pkg):<18} {ver} @ {sha}{pr_str}  [{state}]  {dim(str(ov))}")
        if dec.warning:
            print(f"    {yellow(dec.warning)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync pure-Python upstream packages into ~/.heylook/overlays/<pkg>."
    )
    parser.add_argument(
        "package",
        nargs="?",
        help=f"Distribution name (presets: {', '.join(sorted(PRESETS))})",
    )
    parser.add_argument("--repo", help="Git repository URL (optional for presets)")
    parser.add_argument("--import-name", help="Python import name (default: package with - -> _)")
    parser.add_argument("--rev", help="Git branch, tag, or SHA (default: preset branch or main)")
    parser.add_argument(
        "--pr",
        action="append",
        default=[],
        help="GitHub PR number to merge on top of --rev (repeatable)",
    )
    parser.add_argument("--dir", help="Override overlay root directory (default: ~/.heylook/overlays)")
    parser.add_argument("--python", help="Target Python interpreter for dependency check")
    parser.add_argument("--offline", action="store_true", help="Do not fetch from network")
    parser.add_argument("--status", action="store_true", help="Show staged local overlays and exit")
    parser.add_argument("--remove", action="store_true", help="Remove the staged overlay for <package>")

    args = parser.parse_args(argv)
    root = overlays_dir(args.dir)
    py = resolve_python(args.python)

    if args.status:
        print_status(root, py)
        return 0

    if not args.package:
        parser.error("provide <package> (e.g. `mlx-vlm`) or pass --status")

    pkg = normalize_pkg_name(args.package)
    target = root / pkg

    if args.remove:
        if target.exists():
            shutil.rmtree(target)
            print(green(f"removed overlay: {target}"))
        else:
            print(dim(f"no overlay staged at {target}"))
        return 0

    preset = PRESETS.get(pkg, {})
    repo_url = args.repo or preset.get("repo")
    if not repo_url:
        die(f"unknown package '{pkg}' -- pass --repo <git-url>")
    import_name = args.import_name or preset.get("import_name") or pkg.replace("-", "_")
    rev = args.rev or preset.get("default_rev") or "main"
    prs = [_parse_pr_arg(p) for p in args.pr]

    repo_dir = root / ".repos" / pkg
    ensure_repo(repo_dir, repo_url, offline=args.offline)
    base_sha = resolve_rev(repo_dir, rev)
    final_sha, applied_prs = checkout_with_prs(repo_dir, base_sha, prs, offline=args.offline)

    manifest = stage_overlay(
        repo_dir,
        target,
        pkg=pkg,
        import_name=import_name,
        repo_url=repo_url,
        rev=rev,
        sha=final_sha,
        prs=applied_prs,
        python_exe=py,
    )
    pr_note = (
        f" (+ {', '.join('#' + str(p['number']) for p in applied_prs)})"
        if applied_prs
        else ""
    )
    print(
        green(
            f"staged {pkg} {manifest['version'] or '?'} @ {final_sha[:10]}{pr_note} -> {target}"
        )
    )
    print(dim("heylookllm picks this up automatically (HEYLOOK_OVERLAYS=0 to use .venv PyPI)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
