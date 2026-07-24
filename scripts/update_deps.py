#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit>=0.13"]
# ///
"""Update dependencies to their latest commit or release and keep them pinned.

Tracked dev helper. Run with `uv run` -- the PEP 723 header above provisions
tomlkit in an ephemeral env, so nothing is added to the project environment.

Two update modes per package:

  git      Resolve the latest commit of the package's git source and write that
           exact SHA back into [tool.uv.sources] as `rev = "<sha>"`, then relock.
           The point: track HEAD but ALWAYS leave a concrete rev pinned in
           pyproject.toml + uv.lock (reproducible, never a floating source).

  release  Ask uv for the latest PyPI release allowed by the constraint, relock,
           and (with --pin) raise the `>=` floor in [project.dependencies] to the
           version uv resolved.

Examples
--------
  # Default: bump mlx-lm + mlx-vlm to latest commit, pin the SHAs, relock.
  uv run scripts/update_deps.py

  # Same but explicit, plus track a non-default branch for one of them.
  uv run scripts/update_deps.py mlx-lm mlx-vlm --branch main

  # Latest released transformers, and bump its floor to whatever uv resolved.
  uv run scripts/update_deps.py transformers --release --pin

  # See what would change, touch nothing.
  uv run scripts/update_deps.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import tomlkit

# Packages updated when no package names are given on the command line.
DEFAULT_GIT_PACKAGES = ["mlx-lm", "mlx-vlm"]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"

# ANSI helpers (skip when not a tty).
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def bold(s: str) -> str: return _c("1", s)
def green(s: str) -> str: return _c("32", s)
def yellow(s: str) -> str: return _c("33", s)
def red(s: str) -> str: return _c("31", s)
def dim(s: str) -> str: return _c("2", s)


def die(msg: str) -> NoReturn:
    print(red(f"error: {msg}"), file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> str:
    """Run a subprocess, echoing it. Returns stdout when capture=True."""
    print(dim("$ " + " ".join(cmd)))
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
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


def latest_git_sha(url: str, branch: str | None) -> str:
    """Resolve the tip commit of `branch` (or the remote default) for `url`."""
    ref = f"refs/heads/{branch}" if branch else "HEAD"
    out = run(["git", "ls-remote", url, ref], capture=True)
    if not out:
        die(f"no ref {ref!r} found at {url}")
    sha = out.split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        die(f"unexpected ls-remote output for {url}: {out!r}")
    return sha


def load_doc() -> "tomlkit.TOMLDocument":
    if not PYPROJECT.exists():
        die(f"{PYPROJECT} not found")
    return tomlkit.parse(PYPROJECT.read_text())


def sources_table(doc) -> dict:
    return doc.get("tool", {}).get("uv", {}).get("sources", {})


def _canon(name: str) -> str:
    """PEP 503 name normalization (what uv.lock / pyproject matching needs)."""
    return name.lower().replace("_", "-")


def lock_version(pkg: str) -> str | None:
    """Best-effort read of the resolved version for `pkg` from uv.lock."""
    if not LOCKFILE.exists():
        return None
    try:
        data = tomlkit.parse(LOCKFILE.read_text())
    except Exception:
        return None
    canon = _canon(pkg)
    for entry in data.get("package", []):
        # uv.lock stores PEP 503-normalized (lowercase) names; pyproject may not.
        if _canon(str(entry.get("name", ""))) == canon:
            return entry.get("version")
    return None


# ---------------------------------------------------------------------------
# git-source updates
# ---------------------------------------------------------------------------
def update_git_package(doc, pkg: str, branch: str | None, dry_run: bool) -> bool:
    """Pin `pkg`'s git source to the latest commit SHA. Returns True if changed."""
    sources = sources_table(doc)
    src = sources.get(pkg)
    if src is None or "git" not in src:
        die(
            f"{pkg!r} has no git entry under [tool.uv.sources]. "
            f"Use --release for a PyPI package, or add a git source first."
        )
    url = str(src["git"])
    branch = branch or (str(src["branch"]) if "branch" in src else None)
    old = str(src.get("rev", "")) or None

    print(f"\n{bold(pkg)}  {dim(url)}" + (f"  {dim('@ ' + branch)}" if branch else ""))
    sha = latest_git_sha(url, branch)

    if old == sha:
        print(f"  already at latest {green(sha[:12])} -- no change")
        return False

    print(f"  {yellow(old[:12] if old else '(floating)')} -> {green(sha[:12])}")
    if dry_run:
        return False

    src["rev"] = sha
    if branch and "branch" not in src:
        src["branch"] = branch  # record the tracked branch for next time
    return True


# ---------------------------------------------------------------------------
# release (PyPI) updates
# ---------------------------------------------------------------------------
_DEP_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")

def bump_floor(doc, pkg: str, version: str) -> bool:
    """Raise the `>=` floor for `pkg` in [project.dependencies]. Returns changed."""
    deps = doc.get("project", {}).get("dependencies")
    if deps is None:
        return False
    changed = False
    canon = _canon(pkg)
    for i, item in enumerate(deps):
        s = str(item)
        m = _DEP_RE.match(s)
        if not m or _canon(m.group(1)) != canon:
            continue
        # Split off an environment marker (`; sys_platform == ...`) to preserve it.
        head, sep, marker = s.partition(";")
        new_head = re.sub(
            r"(>=)\s*[0-9][0-9A-Za-z.\-+*]*", rf"\g<1>{version}", head, count=1
        )
        if ">=" not in head:  # no floor to raise; leave bare/other specifiers alone
            continue
        new = new_head.rstrip() + (f"; {marker.strip()}" if sep else "")
        if new != s:
            deps[i] = new
            changed = True
    return changed


def update_release_package(doc, pkg: str, pin: bool, dry_run: bool) -> bool:
    """Relock `pkg` to its latest allowed release; optionally raise its floor."""
    print(f"\n{bold(pkg)}  {dim('(PyPI release)')}")
    before = lock_version(pkg)
    if dry_run:
        print(f"  currently locked: {yellow(before or '?')}  "
              f"{dim('(dry-run: skipping uv lock)')}")
        return False

    run(["uv", "lock", "--upgrade-package", pkg])
    after = lock_version(pkg)
    if before == after:
        print(f"  already at latest allowed release {green(after or '?')}")
    else:
        print(f"  {yellow(before or '?')} -> {green(after or '?')}")

    if pin and after and bump_floor(doc, pkg, after):
        print(f"  pinned floor >= {after} in [project.dependencies]")
        return True  # pyproject changed -> caller relocks
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "packages", nargs="*",
        help=f"packages to update (default: {' '.join(DEFAULT_GIT_PACKAGES)})",
    )
    ap.add_argument(
        "--release", action="store_true",
        help="update to latest PyPI RELEASE instead of latest git commit",
    )
    ap.add_argument(
        "--pin", action="store_true",
        help="(release mode) raise the >= floor in pyproject to the resolved version",
    )
    ap.add_argument(
        "--branch", default=None,
        help="git branch to track (default: the source's branch, else remote HEAD)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="show what would change without editing files or locking",
    )
    args = ap.parse_args()

    packages = args.packages or DEFAULT_GIT_PACKAGES
    doc = load_doc()

    pyproject_changed = False
    for pkg in packages:
        if args.release:
            pyproject_changed |= update_release_package(doc, pkg, args.pin, args.dry_run)
        else:
            pyproject_changed |= update_git_package(doc, pkg, args.branch, args.dry_run)

    if args.dry_run:
        print(yellow("\ndry-run: no files written."))
        return

    if pyproject_changed:
        PYPROJECT.write_text(tomlkit.dumps(doc))
        print(green(f"\nwrote {PYPROJECT.relative_to(ROOT)}"))
        # Relock so uv.lock captures the new revs / floors from pyproject.
        run(["uv", "lock"])
        print(green("\nrelocked. Sync your env with:"))
        print("  uv sync")
    else:
        print(dim("\nnothing changed; pyproject.toml and uv.lock left as-is."))


if __name__ == "__main__":
    main()
