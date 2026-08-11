#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit>=0.13"]
# ///
"""Move this project's fast-moving upstreams, and leave every one of them pinned.

Tracked dev helper. Run with `uv run` -- the PEP 723 header above provisions
tomlkit in an ephemeral env, so nothing is added to the project environment.

This is the ONE place that bumps an upstream. Three are in scope:

  mlx-lm, mlx-vlm   Python packages, installed by uv.
  llama.cpp         a C++ binary (llama-server) -- cloned and BUILT here.
                    `uv sync` cannot build C++, so a llama.cpp bump is always
                    an explicit run of this script, never a side effect of sync.

Updating a dependency is a deliberate act
-----------------------------------------
Pulling a new upstream rev means running code nobody here has read -- on the
`latest` channel, literally whatever was pushed to a branch since the last pin.
So this script never moves anything you did not name:

  * a bare run CHANGES NOTHING -- it reports what you are currently pinned to;
  * you name the packages (or --all) to move them;
  * it resolves the target, prints the plan with a compare link per package,
    and asks before writing anything or building;
  * -y/--yes skips the prompt, and is REQUIRED when stdin is not a terminal, so
    an automated caller cannot drift the pins by accident.

Channels
--------
`[tool.heylook.deps]` in pyproject.toml says where each package comes from:

  stable   Published releases. PyPI for the Python packages, whose hashes uv
           records in uv.lock; the newest `b<N>` release tag for llama.cpp.
  latest   Tip of the tracked branch, pinned to the exact commit it resolved
           to -- a `rev` in [tool.uv.sources] for the Python packages, `rev`
           under [tool.heylook.llama-cpp] for llama.cpp. This is the mode that
           pulls unreviewed code; the plan says so every time.

`channel` sets the project default, `overrides` sets it per package, and
`--channel` changes and PERSISTS that decision (leaving the file claiming
"stable" while a git SHA is pinned would be a lie). Either way the resolved pin
is written back, so pyproject.toml always states what you run.

Examples
--------
  # Report the current pins. Touches nothing, hits no network.
  uv run scripts/update_deps.py

  # Resolve what a bump would pull, print compare links, write nothing.
  uv run scripts/update_deps.py --all --dry-run

  # Move one package, with a look at the plan before it lands.
  uv run scripts/update_deps.py mlx-lm

  # Build llama-server at the newest release tag (or at HEAD, if its channel
  # is `latest`), pin what it built, and print the export line.
  uv run scripts/update_deps.py llama.cpp

  # Put llama.cpp on tip-of-master from here on, and build it now.
  uv run scripts/update_deps.py llama.cpp --channel latest

  # Rebuild the ALREADY-pinned llama.cpp rev (new Xcode, changed flags, --clean).
  uv run scripts/update_deps.py llama.cpp --rebuild

  # Put the whole project on published releases, and raise the floors to match.
  uv run scripts/update_deps.py --all --channel stable --pin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import tomlkit

# Every upstream this script knows how to move. There is no default subset --
# naming what you are updating is the point.
PYTHON_PACKAGES = ("mlx-lm", "mlx-vlm")
LLAMA = "llama.cpp"
LLAMA_ALIASES = {"llama.cpp", "llama-cpp", "llama_cpp", "llamacpp", "llama-server"}
ALL_PACKAGES = (*PYTHON_PACKAGES, LLAMA)
CHANNELS = ("stable", "latest")

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


def run(cmd: list[str], *, capture: bool = False, check: bool = True,
        cwd: Path | None = None) -> str:
    """Run a subprocess, echoing it. Returns stdout when capture=True."""
    print(dim("$ " + " ".join(cmd)))
    proc = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
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


def _canon(name: str) -> str:
    """PEP 503 name normalization (what uv.lock / pyproject matching needs)."""
    return name.lower().replace("_", "-")


def _short(rev: str | None) -> str:
    """SHAs abbreviate; tags like `b10354` are already short, keep them whole."""
    if not rev:
        return "(unpinned)"
    return rev[:12] if re.fullmatch(r"[0-9a-f]{40}", rev) else rev


# ---------------------------------------------------------------------------
# pyproject access
# ---------------------------------------------------------------------------
def load_doc() -> "tomlkit.TOMLDocument":
    if not PYPROJECT.exists():
        die(f"{PYPROJECT} not found")
    return tomlkit.parse(PYPROJECT.read_text())


def sources_table(doc, *, create: bool = False) -> dict:
    """[tool.uv.sources], live.

    `.get(..., {})` chaining returns a THROWAWAY dict when a level is missing,
    and writes into it vanish on dump -- the script would report a pin it never
    wrote. The stable channel deletes entries, so an empty (or tidied-away)
    table is reachable in normal use. Pass create=True on any write path."""
    node = doc
    for key in ("tool", "uv", "sources"):
        child = node.get(key) if hasattr(node, "get") else None
        if child is None:
            if not create:
                return {}
            # `tool` and `tool.uv` are super-tables; `sources` holds the entries.
            child = tomlkit.table(key != "sources")
            node[key] = child
        node = child
    return node


def heylook_table(doc, *keys: str) -> dict:
    node = doc.get("tool", {}).get("heylook", {})
    for key in keys:
        node = node.get(key, {}) if hasattr(node, "get") else {}
    return node


def resolve_channel(doc, pkg: str, cli_channel: str | None) -> str:
    """Channel for `pkg`: --channel > per-package override > project default."""
    if cli_channel:
        return cli_channel
    deps = heylook_table(doc, "deps")
    overrides = deps.get("overrides", {})
    channel = str(overrides.get(pkg, deps.get("channel", "stable")))
    if channel not in CHANNELS:
        die(f"unknown channel {channel!r} for {pkg} (expected one of {CHANNELS})")
    return channel


def persist_channel(doc, pkg: str, channel: str, *, global_default: bool) -> bool:
    """Write a channel decision back. Returns True if the document changed."""
    deps = heylook_table(doc, "deps")
    if not deps:
        die("[tool.heylook.deps] is missing from pyproject.toml")
    if global_default:
        if str(deps.get("channel", "")) == channel:
            return False
        deps["channel"] = channel
        return True
    overrides = deps.get("overrides")
    if overrides is None:
        overrides = tomlkit.inline_table()
        deps["overrides"] = overrides
    project_default = str(deps.get("channel", "stable"))
    if channel == project_default:
        # Redundant override -- drop it rather than restate the default.
        if pkg in overrides:
            del overrides[pkg]
            return True
        return False
    if str(overrides.get(pkg, "")) == channel:
        return False
    overrides[pkg] = channel
    return True


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
# git helpers
# ---------------------------------------------------------------------------
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


_BUILD_TAG_RE = re.compile(r"^b(\d+)$")

def latest_release_tag(url: str) -> str:
    """Newest llama.cpp release tag. Upstream tags builds as `b<N>`, monotonic,
    so the highest N is the newest release -- no semver parsing needed."""
    out = run(["git", "ls-remote", "--tags", "--refs", url], capture=True)
    best: tuple[int, str] | None = None
    for line in out.splitlines():
        tag = line.rsplit("/", 1)[-1].strip()
        m = _BUILD_TAG_RE.match(tag)
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, tag)
    if best is None:
        die(f"no b<N> release tags found at {url}")
    return best[1]


def _remote_key(url: str) -> str:
    """host/owner/repo, so https and ssh forms of one remote compare equal."""
    s = url.strip().removesuffix(".git")
    s = re.sub(r"^[a-z+]+://", "", s)
    s = s.replace(":", "/", 1) if s.startswith("git@") else s
    return s.removeprefix("git@").lower()


def review_url(url: str, old: str | None, new: str) -> str | None:
    """A link that shows what this bump actually pulls in. The whole point of
    printing a plan is that you can go read the diff before saying yes."""
    if "github.com" not in url:
        return None
    base = url.strip().removesuffix(".git")
    return f"{base}/compare/{old}...{new}" if old else f"{base}/commits/{new}"


# ---------------------------------------------------------------------------
# Python packages
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


def bump_override_floor(doc, pkg: str, version: str) -> bool:
    """Keep `[tool.uv] override-dependencies` in step with the floor.

    transformers carries BOTH a floor in [project.dependencies] and an override
    that replaces every constraint in the graph; pyproject's own comment says
    they are kept equal "so the two don't appear to disagree". Raising one and
    not the other is how they silently drift apart."""
    overrides = doc.get("tool", {}).get("uv", {}).get("override-dependencies")
    if overrides is None:
        return False
    changed = False
    canon = _canon(pkg)
    for i, item in enumerate(overrides):
        s = str(item)
        m = _DEP_RE.match(s)
        if not m or _canon(m.group(1)) != canon or ">=" not in s:
            continue
        new = re.sub(r"(>=)\s*[0-9][0-9A-Za-z.\-+*]*", rf"\g<1>{version}", s, count=1)
        if new != s:
            overrides[i] = new
            changed = True
    return changed


def git_origin(doc, pkg: str, cli_branch: str | None) -> tuple[str, str | None]:
    """(url, branch) for `pkg`'s `latest` channel, from [tool.heylook.deps.git]
    with the live [tool.uv.sources] entry as a fallback."""
    entry = heylook_table(doc, "deps", "git").get(pkg)
    src = sources_table(doc).get(pkg)
    url = None
    branch = cli_branch
    if entry is not None:
        url = str(entry["git"])
        branch = branch or (str(entry["branch"]) if "branch" in entry else None)
    elif src is not None and "git" in src:
        url = str(src["git"])
        branch = branch or (str(src["branch"]) if "branch" in src else None)
    if not url:
        die(
            f"{pkg!r} has no git origin. Add it under [tool.heylook.deps.git] "
            f"before putting it on the `latest` channel."
        )
    return url, branch


# ---------------------------------------------------------------------------
# Planning -- resolve everything, mutate nothing
# ---------------------------------------------------------------------------
def plan_python(doc, pkg: str, channel: str, cli_branch: str | None) -> dict:
    """What moving `pkg` onto `channel` would do. Read-only (one ls-remote)."""
    sources = sources_table(doc)
    src = sources.get(pkg)
    old_rev = str(src.get("rev", "")) if src is not None else None

    if channel == "latest":
        url, branch = git_origin(doc, pkg, cli_branch)
        new_rev = latest_git_sha(url, branch)
        return {
            "pkg": pkg, "kind": "python", "channel": channel, "url": url,
            "branch": branch, "old": old_rev, "new": new_rev,
            "changes": old_rev != new_rev,
            "review": review_url(url, old_rev, new_rev) if old_rev != new_rev else None,
            "note": "pins an unreviewed upstream commit",
        }

    # stable: uv decides the version, so the plan is "let uv resolve", and what
    # we can state up front is the source change and where we are now.
    return {
        "pkg": pkg, "kind": "python", "channel": channel, "url": None,
        "branch": None, "old": lock_version(pkg), "new": "latest release (uv resolves)",
        "drop_git_source": src is not None,
        "changes": True,  # only uv can say; a lock run is always warranted
        "review": None,
        "note": "PyPI release, hashes recorded in uv.lock",
    }


def plan_llama(doc, channel: str, cli_branch: str | None, rebuild: bool) -> dict:
    cfg = heylook_table(doc, "llama-cpp")
    if not cfg:
        die("[tool.heylook.llama-cpp] is missing from pyproject.toml")
    url = str(cfg["git"])
    branch = cli_branch or str(cfg.get("branch", "master"))
    pinned = str(cfg.get("rev", "")) or None

    # Is there already a binary built from the rev we are targeting? "The pin
    # did not move" does NOT mean "there is nothing to build": on a fresh clone
    # the pin is inherited from git and no binary exists at all.
    build = llama_dir(cfg) / "build"
    manifest = read_manifest(build)
    have_binary = (build / "bin" / "llama-server").exists()

    def _plan(new: str, note: str) -> dict:
        built_this_rev = have_binary and manifest.get("rev") == new
        pin_moves = pinned != new
        if rebuild:
            why = "rebuild of the pinned rev -- no new code pulled"
        elif not have_binary:
            why = f"{note}; no binary built yet"
        elif not built_this_rev:
            why = f"{note}; current binary is {manifest.get('rev') or 'unknown'}"
        else:
            why = note
        return {
            "pkg": LLAMA, "kind": "llama", "channel": channel, "url": url,
            "branch": branch, "old": pinned, "new": new,
            "dir": str(llama_dir(cfg)),
            "changes": pin_moves,
            # Build whenever the artifact does not already match the target.
            "build": rebuild or pin_moves or not built_this_rev,
            "have_binary": have_binary,
            "built_rev": manifest.get("rev"),
            "review": review_url(url, pinned, new) if pin_moves else None,
            "note": why,
        }

    if rebuild:
        if not pinned:
            die("--rebuild needs an already-pinned rev; run without it first.")
        return _plan(pinned, "rebuild")
    if channel == "latest":
        return _plan(latest_git_sha(url, branch),
                     f"pins an unreviewed commit from {branch}, then BUILDS it")
    return _plan(latest_release_tag(url), "newest release tag, then BUILDS it")


def plan_acts(p: dict) -> bool:
    """Will this plan item DO anything? A llama.cpp item with an unmoved pin
    still acts when no binary was built from that pin yet."""
    return bool(p["changes"] or p.get("build"))


def print_plan(plans: list[dict]) -> bool:
    """Show what is about to happen. Returns True if anything would act.

    Every line here is something you are about to consent to, so it has to
    match what the apply step actually does -- including the build."""
    print(f"\n{bold('Plan')}")
    any_action = False
    for p in plans:
        head = f"  {bold(p['pkg']):<24} {dim('channel=' + p['channel'])}"
        if not plan_acts(p):
            print(f"{head}  {green(_short(p['old']))} {dim('-- already current')}")
            continue
        any_action = True
        old, new = _short(p["old"]), p["new"]
        new = _short(new) if p["kind"] != "python" or p["channel"] == "latest" else new
        if p["changes"]:
            print(f"{head}  {yellow(old)} -> {green(new)}")
        else:
            # Pin stays put but we still act (missing or mismatched binary).
            print(f"{head}  {green(old)} {dim('(pin unchanged)')}")
        if p.get("drop_git_source"):
            print(f"    {yellow('drops the git source')} -- back to published releases")
        if p["kind"] == "llama" and p.get("build"):
            print(f"    {yellow('BUILDS llama-server')} -- minutes, not seconds")
            # Say WHERE before writing, not after. This is a git clone plus a
            # multi-GB build tree landing on someone's disk; it should never be
            # a surprise, and it is deliberately outside the repo.
            print(f"    writes to: {p['dir']}"
                  f"{dim('  (clone + build tree, several GB)') if not p.get('have_binary') else ''}")
        print(f"    {dim(p['note'])}")
        if p.get("review"):
            print(f"    review: {p['review']}")
    return any_action


def confirm(assume_yes: bool) -> None:
    """Gate every mutation. A non-tty must pass --yes: an automated caller
    should never be able to move a pin just because it ran the script."""
    if assume_yes:
        return
    if not sys.stdin.isatty():
        die("stdin is not a terminal -- pass --yes to accept the plan above "
            "non-interactively.")
    try:
        answer = input(f"\n{bold('Apply this plan?')} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in ("y", "yes"):
        print(yellow("aborted; nothing written."))
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------
def apply_python(doc, plan: dict, pin: bool) -> tuple[bool, bool, bool]:
    """Move one Python package.

    Returns (pyproject_changed, needs_relock, lock_touched). The third is
    separate because the stable path runs `uv lock` itself: without it the
    summary can tell you uv.lock was left alone after it has already moved."""
    pkg = plan["pkg"]
    print(f"\n{bold(pkg)}  {dim('channel=' + plan['channel'])}")
    sources = sources_table(doc)

    if plan["channel"] == "latest":
        if not plan["changes"]:
            print(f"  already at {green(_short(plan['old']))} -- no change")
            return False, False, False
        entry = tomlkit.inline_table()
        entry["git"] = plan["url"]
        if plan["branch"]:
            entry["branch"] = plan["branch"]
        entry["rev"] = plan["new"]
        sources_table(doc, create=True)[pkg] = entry  # create: the table may be gone
        print(f"  pinned {green(_short(plan['new']))}")
        return True, True, False

    # stable: the git source must go first, or uv relocks it right back.
    source_removed = False
    if pkg in sources:
        del sources[pkg]
        source_removed = True
        print(f"  {yellow('dropped the git source')}")
        # uv reads pyproject from disk, so land the removal before locking.
        PYPROJECT.write_text(tomlkit.dumps(doc))

    before = lock_version(pkg)
    run(["uv", "lock", "--upgrade-package", pkg])
    after = lock_version(pkg)
    if before == after:
        print(f"  already at latest allowed release {green(after or '?')}")
    else:
        print(f"  {yellow(before or '?')} -> {green(after or '?')}")

    floor_bumped = False
    if pin and after:
        if bump_floor(doc, pkg, after):
            print(f"  pinned floor >= {after} in [project.dependencies]")
            floor_bumped = True
        if bump_override_floor(doc, pkg, after):
            print(f"  matched override-dependencies to >= {after}")
            floor_bumped = True
    # The source removal was already captured by the lock run above; only a
    # floor bump lands after it and needs the lock re-run.
    return (source_removed or floor_bumped), floor_bumped, True


# ---------------------------------------------------------------------------
# llama.cpp -- clone, check out, build
# ---------------------------------------------------------------------------
def llama_cmake_args(cfg, *, lto: bool, openmp: bool, ui: bool) -> list[str]:
    """Build settings for a headless Apple-Silicon llama-server.

    The shape of the decision: on Metal the arithmetic runs in GPU shaders, so
    anything that only speeds up the CPU-side glue (sampling, tokenizer, graph
    build, the HTTP layer) buys noise and costs build time.

      GGML_METAL_EMBED_LIBRARY  metallib inside the binary -- it can be moved
                                without dragging a shader file along.
      BUILD_SHARED_LIBS=OFF     one self-contained static binary, which is what
                                a subprocess-spawning provider wants.
      GGML_NATIVE               -mcpu=native for the host it is built on.
      GGML_LTO=OFF              whole-program opt at link touches only that CPU
                                glue; costs a multi-minute link and defeats
                                ccache (which caches compiles, not the LTO link).
                                Upstream enables it on no platform.
      GGML_OPENMP=OFF           ggml's own threadpool is the #else branch of the
                                same graph-compute entry point -- same work
                                partition, same affinity and priority handling,
                                so this is a choice of thread runtime, not
                                threading vs none. macOS ships no libomp, so it
                                is already what every upstream mac binary runs.
      LLAMA_BUILD_UI=OFF        the WebUI re-provisions from a network bucket on
                                every build; heylook drives llama-server over
                                HTTP and never serves it.

    Deliberately NOT set: GGML_METAL_NDEBUG. It compiles out load-time logging
    only -- including the "allocated size is greater than the recommended max
    working set size" warning, which is the ceiling that actually refuses loads
    on a big-unified-memory Mac. No throughput to gain, real diagnostics to lose.
    """
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_METAL=ON",
        "-DGGML_METAL_EMBED_LIBRARY=ON",
        "-DGGML_NATIVE=ON",
        "-DGGML_ACCELERATE=ON",
        "-DGGML_BLAS=ON",
        "-DGGML_BLAS_VENDOR=Apple",
        "-DGGML_CCACHE=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",      # llama-server lives under tools/
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        f"-DGGML_LTO={'ON' if lto else 'OFF'}",
        f"-DGGML_OPENMP={'ON' if openmp else 'OFF'}",
        f"-DLLAMA_BUILD_UI={'ON' if ui else 'OFF'}",
    ]
    args += [str(a) for a in cfg.get("extra_cmake_args", [])]
    return args


_CACHE_LINE = re.compile(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$")

# Flags whose EFFECTIVE value we record. Requested args are a statement of
# intent; cmake can silently downgrade one (OpenMP), and a reused cache can
# carry an arg that is no longer requested. The manifest has to describe the
# binary, not the wish.
_EFFECTIVE_KEYS = (
    "CMAKE_GENERATOR", "CMAKE_BUILD_TYPE", "GGML_OPENMP_ENABLED", "GGML_LTO",
    "GGML_METAL", "GGML_METAL_NDEBUG", "GGML_METAL_EMBED_LIBRARY", "GGML_NATIVE",
    "GGML_BLAS", "GGML_ACCELERATE", "LLAMA_BUILD_UI", "BUILD_SHARED_LIBS",
)


def read_cmake_cache(build: Path) -> dict[str, str]:
    """Parse CMakeCache.txt into {KEY: value}. Empty when there is no cache."""
    cache = build / "CMakeCache.txt"
    if not cache.exists():
        return {}
    out = {}
    for line in cache.read_text(errors="replace").splitlines():
        m = _CACHE_LINE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def read_manifest(build: Path) -> dict:
    """The heylook-build.json we wrote next to a previous build, if any."""
    path = build / "heylook-build.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def libomp_flags() -> list[str]:
    """CMake args that actually make find_package(OpenMP) succeed on AppleClang.

    `-DGGML_OPENMP=ON` alone is a TRAP: ggml only warns when OpenMP is missing
    (ggml/src/CMakeLists.txt:225-232) and builds the threadpool path anyway, so
    you get a binary that is byte-for-byte the non-OpenMP one while the flag
    says ON. Verified on this machine: with libomp installed but no hints,
    find_package reports "Could NOT find OpenMP" and GGML_OPENMP_ENABLED=OFF.
    `-DOpenMP_ROOT=` is not enough either -- AppleClang needs the flags spelled
    out. An A/B run against a silently-downgraded build would "prove" OpenMP
    makes no difference, which is the whole reason to be strict here."""
    brew = shutil.which("brew")
    if not brew:
        return []
    prefix = subprocess.run([brew, "--prefix", "libomp"], text=True, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    root = (prefix.stdout or "").strip()
    if prefix.returncode != 0 or not root or not Path(root).exists():
        return []
    inc = f"-Xclang -fopenmp -I{root}/include"
    return [
        f"-DOpenMP_C_FLAGS={inc}", "-DOpenMP_C_LIB_NAMES=omp",
        f"-DOpenMP_CXX_FLAGS={inc}", "-DOpenMP_CXX_LIB_NAMES=omp",
        f"-DOpenMP_omp_LIBRARY={root}/lib/libomp.dylib",
    ]


# Where the llama.cpp checkout lives when nothing overrides it. Deliberately
# OUTSIDE the repo: this is upstream C++ source plus a multi-GB build tree, and
# heylook must never ship, package, or otherwise carry either. Same absolute
# location on every machine, so docs and error messages can name it.
LLAMA_HOME_SUBDIR = (".heylook", "llama.cpp")


def default_llama_dir() -> Path:
    return Path.home().joinpath(*LLAMA_HOME_SUBDIR)


def llama_dir(cfg) -> Path:
    """Checkout location, in precedence order:

    1. $HEYLOOK_LLAMA_CPP_DIR  -- per-machine override
    2. `dir` in [tool.heylook.llama-cpp], if set -- interpreted relative to the
       repo when relative, so a deliberate in-repo checkout is still possible
    3. the fixed home location -- the default, outside the repo
    """
    env = os.environ.get("HEYLOOK_LLAMA_CPP_DIR")
    if env:
        return Path(env).expanduser().resolve()
    configured = str(cfg.get("dir", "") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    return default_llama_dir().resolve()


def _show(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute when it is not."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_checkout(path: Path, url: str) -> None:
    """Clone if absent; verify the remote if present. Blobless so it stays small
    while still allowing a checkout of any rev."""
    if (path / ".git").exists():
        origin = run(["git", "-C", str(path), "remote", "get-url", "origin"],
                     capture=True)
        if _remote_key(origin) != _remote_key(url):
            die(f"{_show(path)} tracks {origin}, not {url}. "
                f"Point --dir elsewhere or fix the remote.")
        return
    if path.exists() and any(path.iterdir()):
        die(f"{_show(path)} exists and is not a llama.cpp checkout -- refusing "
            f"to clone over it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", url, str(path)])


def checkout_rev(path: Path, rev: str) -> str:
    """Detach onto `rev`. Refuses over TRACKED modifications. Returns the SHA.

    Untracked files (a stray .DS_Store, CMakeUserPresets.json, an editor swap
    file) do not block: git carries them across a checkout, `git stash` alone
    would not clear them, and telling someone their .DS_Store is "real work" is
    a lie that leaves them stuck."""
    dirty = run(["git", "-C", str(path), "status", "--porcelain",
                 "--untracked-files=no"], capture=True)
    if dirty:
        die(f"{_show(path)} has uncommitted changes to TRACKED files -- "
            f"refusing to check out {rev} over them. Commit or stash first "
            f"(llama.cpp ignores build/, so this is real work, not build "
            f"output):\n{dirty}")
    untracked = run(["git", "-C", str(path), "ls-files", "--others",
                     "--exclude-standard"], capture=True)
    if untracked:
        count = len(untracked.splitlines())
        print(dim(f"  ({count} untracked file(s) present; carried across the "
                  f"checkout untouched)"))
    run(["git", "-C", str(path), "fetch", "--tags", "--force", "origin"])
    run(["git", "-C", str(path), "checkout", "--detach", rev])
    return run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)


def build_llama(path: Path, args: list[str], targets: list[str], jobs: int,
                clean: bool, *, want_openmp: bool) -> Path:
    cmake = shutil.which("cmake")
    if not cmake:
        die("cmake not found on PATH (brew install cmake)")
    build = path / "build"
    if clean and build.exists():
        print(dim(f"removing {_show(build)}"))
        shutil.rmtree(build)

    # Only choose a generator on a FRESH configure. Passing -G Ninja at a dir
    # first configured with Unix Makefiles is a hard cmake error ("does not
    # match the generator used previously"), which would surface here as a bare
    # "command failed (1)" with no hint that --clean is the fix.
    cached = read_cmake_cache(build)

    # A CMake build tree hard-codes the absolute source path it was configured
    # for, so MOVING the checkout leaves a tree cmake refuses to reuse ("does
    # not match the source used to generate cache") -- a hard error with no
    # hint that the fix is to delete it. It is pure build output; discard it.
    cached_src = cached.get("CMAKE_HOME_DIRECTORY")
    if cached_src and Path(cached_src) != path:
        print(yellow(f"  build tree was configured for {cached_src}, which is "
                     f"not where the source lives now -- discarding it"))
        shutil.rmtree(build, ignore_errors=True)
        cached = {}

    if cached.get("CMAKE_GENERATOR"):
        generator = []
        print(dim(f"reusing cached generator: {cached['CMAKE_GENERATOR']}"))
    else:
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []

    run([cmake, "-S", str(path), "-B", str(build), *generator, *args])

    # ggml only WARNS when OpenMP is missing and links the threadpool build
    # anyway, so "-DGGML_OPENMP=ON succeeded" proves nothing. Read back what
    # cmake actually resolved and refuse to hand over a silently-downgraded
    # binary that would poison the very A/B this flag exists for.
    effective = read_cmake_cache(build)
    if want_openmp and effective.get("GGML_OPENMP_ENABLED", "OFF").upper() != "ON":
        die("--openmp was requested but cmake resolved GGML_OPENMP_ENABLED=OFF "
            "(ggml only warns, it does not fail). Install libomp "
            "(`brew install libomp`) and re-run; the binary you would have got "
            "is byte-identical to a non-OpenMP build, so any A/B against it "
            "would be measuring nothing.")

    build_cmd = [cmake, "--build", str(build), "--config", "Release", "-j", str(jobs)]
    for target in targets:
        build_cmd += ["--target", target]
    run(build_cmd)
    return build


def binary_version(binary: Path) -> str:
    """The banner llama-server prints for --version, on stderr:

        version: 10354 (d2f83055d)
        built with AppleClang 21.0.0.21000101 for Darwin arm64

    The first line is the identifying one -- take it, not the last."""
    proc = subprocess.run([str(binary), "--version"], text=True, check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return next((ln for ln in lines if ln.startswith("version:")), lines[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def apply_llama(doc, plan: dict, *, clean: bool, lto: bool, openmp: bool,
                ui: bool, jobs: int) -> bool:
    """Check out and build. Returns pyproject_changed."""
    cfg = heylook_table(doc, "llama-cpp")
    targets = [str(t) for t in cfg.get("targets", ["llama-server"])] or ["llama-server"]
    path = llama_dir(cfg)
    rev = plan["new"]

    print(f"\n{bold(LLAMA)}  {dim(plan['url'])}  {dim('channel=' + plan['channel'])}")
    ensure_checkout(path, plan["url"])
    sha = checkout_rev(path, rev)
    args = llama_cmake_args(cfg, lto=lto, openmp=openmp, ui=ui)
    if openmp:
        args += libomp_flags()  # -DGGML_OPENMP=ON alone does not find libomp
    build = build_llama(path, args, targets, jobs, clean, want_openmp=openmp)

    binary = build / "bin" / "llama-server"
    if not binary.exists():
        die(f"build finished but {_show(binary)} is missing")
    version = binary_version(binary)

    # Manifest lives with the build (gitignored) -- what produced this binary,
    # and a digest so "is the binary I am running the one I built" is answerable.
    cache = read_cmake_cache(build)
    manifest = {
        "rev": rev,
        "sha": sha,
        "channel": plan["channel"],
        "targets": targets,
        "cmake_args": args,
        # What cmake RESOLVED, which is not always what we asked for (OpenMP
        # downgrades silently; a reused cache can retain a dropped arg). The
        # manifest has to describe the binary, not the request.
        "effective": {k: cache[k] for k in _EFFECTIVE_KEYS if k in cache},
        "binary": str(binary),
        "sha256": sha256(binary),
        "version": version,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (build / "heylook-build.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  built {green(str(binary))}")
    if version:
        print(f"  {dim(version)}")
    current = os.environ.get("HEYLOOK_LLAMA_SERVER")
    if current and Path(current).resolve() != binary.resolve():
        print(yellow(f"  note: $HEYLOOK_LLAMA_SERVER points at {current} -- the "
                     f"server will keep using THAT binary, not this build."))
    print("  point the server at it with:")
    print(f"    export HEYLOOK_LLAMA_SERVER={binary}")

    if str(cfg.get("rev", "")) == rev:
        return False
    cfg["rev"] = rev
    return True


# ---------------------------------------------------------------------------
# status (the bare invocation)
# ---------------------------------------------------------------------------
def print_status(doc) -> None:
    """What we are pinned to right now. No network, no writes."""
    deps = heylook_table(doc, "deps")
    sources = sources_table(doc)
    cfg = heylook_table(doc, "llama-cpp")

    print(f"\n{bold('Current pins')}  "
          f"{dim('(project channel: ' + str(deps.get('channel', 'stable')) + ')')}")
    for pkg in PYTHON_PACKAGES:
        channel = resolve_channel(doc, pkg, None)
        src = sources.get(pkg)
        if src is not None and "rev" in src:
            where = f"git {_short(str(src['rev']))}"
        else:
            where = f"PyPI {lock_version(pkg) or '?'}"
        print(f"  {bold(pkg):<24} {dim('channel=' + channel):<28} {green(where)}")

    channel = resolve_channel(doc, LLAMA, None)
    rev = str(cfg.get("rev", "")) or None
    path = llama_dir(cfg)
    build = path / "build"
    binary = build / "bin" / "llama-server"
    manifest = read_manifest(build)
    print(f"  {bold(LLAMA):<24} {dim('channel=' + channel):<28} "
          f"{green(_short(rev))}")
    if not binary.exists():
        built = yellow("not built -- run: uv run scripts/update_deps.py llama.cpp")
    elif not manifest:
        built = f"{binary}  {yellow('(no build manifest -- provenance unknown)')}"
    elif manifest.get("rev") != rev:
        # The pin and the artifact disagree: a build that died after checkout,
        # a hand-run cmake, an interrupted --clean. Say so; this is exactly
        # what the manifest is for.
        built = (f"{binary}\n  {'':<24} {'':<28} "
                 f"{yellow('built from ' + str(manifest.get('rev')) + ', but pinned ' + str(rev))}"
                 f" {dim('-- rebuild with --rebuild')}")
    else:
        built = green(str(binary))
    print(f"  {'':<24} {dim('binary'):<28} {built}")
    if manifest.get("effective"):
        eff = manifest["effective"]
        interesting = {k: eff[k] for k in ("GGML_OPENMP_ENABLED", "GGML_LTO")
                       if k in eff}
        if interesting:
            summary = "  ".join(f"{k}={v}" for k, v in interesting.items())
            print(f"  {'':<24} {dim('built with'):<28} {dim(summary)}")
    env = os.environ.get("HEYLOOK_LLAMA_SERVER")
    if env:
        marker = "" if binary.exists() and Path(env).resolve() == binary.resolve() \
            else yellow("  <- not this build")
        print(f"  {'':<24} {dim('$HEYLOOK_LLAMA_SERVER'):<28} {env}{marker}")

    print(f"\n{dim('Nothing was changed. To move an upstream, name it:')}")
    print(dim(f"  uv run scripts/update_deps.py {' '.join(ALL_PACKAGES)}"))
    print(dim("  uv run scripts/update_deps.py --all --dry-run"))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "packages", nargs="*",
        help=f"upstreams to move ({', '.join(ALL_PACKAGES)}). There is NO "
             f"default set -- with no names this only reports the current pins",
    )
    ap.add_argument(
        "--all", action="store_true", help="every upstream (still asks first)",
    )
    ap.add_argument(
        "--channel", choices=CHANNELS, default=None,
        help="switch the named packages to this channel (no packages named: the "
             "project default). PERSISTS to pyproject.toml",
    )
    ap.add_argument(
        "-y", "--yes", action="store_true",
        help="accept the plan without prompting (required when stdin is not a tty)",
    )
    ap.add_argument(
        "--pin", action="store_true",
        help="(stable channel) raise the >= floor in pyproject to the resolved version",
    )
    ap.add_argument(
        "--branch", default=None,
        help="git branch to track (default: the configured branch, else remote HEAD)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="resolve and print the plan, then stop: no writes, no clone, no build",
    )
    llama_opts = ap.add_argument_group(f"{LLAMA} build")
    llama_opts.add_argument(
        "--rebuild", action="store_true",
        help="rebuild the already-pinned rev instead of resolving a new one",
    )
    llama_opts.add_argument(
        "--clean", action="store_true", help="delete the build dir and reconfigure",
    )
    llama_opts.add_argument(
        "--lto", action="store_true",
        help="-DGGML_LTO=ON (slow link, noise-level gain on a Metal-bound load)",
    )
    llama_opts.add_argument(
        "--openmp", action="store_true",
        help="-DGGML_OPENMP=ON (needs libomp; ggml's own threadpool is used otherwise)",
    )
    llama_opts.add_argument(
        "--ui", action="store_true",
        help="build llama-server's embedded WebUI (off by default: headless here)",
    )
    llama_opts.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 8, help="parallel build jobs",
    )
    args = ap.parse_args()

    if args.all and args.packages:
        die("--all takes no package names")

    doc = load_doc()

    # No names, no --all: report and stop. Updating a dependency is something
    # you ask for by name, never something a bare run does for you.
    if not args.packages and not args.all:
        if args.channel:
            die("--channel with no package names would change the project default "
                "for everything. Name the packages, or pass --all.")
        print_status(doc)
        return

    requested = list(ALL_PACKAGES) if args.all else [
        LLAMA if _canon(p) in LLAMA_ALIASES else p for p in args.packages
    ]
    # Anything outside the registry is treated as a plain PyPI dependency
    # (transformers is the live case: a hand-managed floor plus a matching
    # override that must not drift). There is no git origin for those, so
    # `latest` is not offered -- say so instead of failing deep in resolution.
    extras = [p for p in requested if p not in ALL_PACKAGES]
    if extras and args.channel == "latest":
        die(f"{', '.join(extras)}: no git origin is configured, so `latest` has "
            f"no meaning. Add it under [tool.heylook.deps.git] first, or drop "
            f"--channel to take the published release.")

    # --branch names ONE ref. Applying it to every package silently means
    # "mlx-lm@main and llama.cpp@main", and llama.cpp has no `main` -- the run
    # would die after the others had already been resolved.
    if args.branch and (args.all or len(requested) > 1):
        die("--branch applies to a single package; name exactly one.")

    # Resolve everything first: the plan is printed and agreed to as a whole,
    # so a multi-package run cannot half-apply before you have seen it.
    plans = []
    for pkg in requested:
        channel = "stable" if pkg in extras else resolve_channel(doc, pkg, args.channel)
        if pkg == LLAMA:
            plans.append(plan_llama(doc, channel, args.branch, args.rebuild))
        else:
            plans.append(plan_python(doc, pkg, channel, args.branch))

    channel_moves = [p for p in plans if args.channel and p["pkg"] not in extras
                     and args.channel != resolve_channel(doc, p["pkg"], None)]
    if channel_moves:
        scope = "project default" if args.all else "override"
        print(f"\n{yellow('channel change')}: "
              f"{', '.join(p['pkg'] for p in channel_moves)} -> {args.channel} "
              f"{dim('(persisted to pyproject.toml as the ' + scope + ')')}")

    any_action = print_plan(plans)
    if args.dry_run:
        print(yellow("\ndry-run: no files written."))
        return
    if not any_action and not channel_moves:
        print(dim("\nnothing to do."))
        return
    confirm(args.yes)

    pyproject_changed = False
    if args.channel:
        if args.all:
            # A project-wide declaration: move the default and clear the
            # per-package overrides, or the file would claim `stable` while
            # every package is individually overridden to something else.
            pyproject_changed |= persist_channel(
                doc, "", args.channel, global_default=True)
            for pkg in ALL_PACKAGES:
                pyproject_changed |= persist_channel(
                    doc, pkg, args.channel, global_default=False)
        else:
            for pkg in requested:
                if pkg not in extras:
                    pyproject_changed |= persist_channel(
                        doc, pkg, args.channel, global_default=False)

    # Python first, llama.cpp last: the build is the long, failure-prone step,
    # and a build that dies should not strand a half-applied pyproject.
    ordered = sorted(plans, key=lambda p: p["kind"] == "llama")

    needs_relock = False
    lock_touched = False
    for plan in ordered:
        if not plan_acts(plan):
            continue  # the plan said "already current"; do not quietly build
        if plan["kind"] == "llama":
            pyproject_changed |= apply_llama(
                doc, plan, clean=args.clean, lto=args.lto, openmp=args.openmp,
                ui=args.ui, jobs=args.jobs)
            continue
        changed, relock, locked = apply_python(doc, plan, args.pin)
        pyproject_changed |= changed
        needs_relock |= relock
        lock_touched |= locked

    if pyproject_changed:
        PYPROJECT.write_text(tomlkit.dumps(doc))
        print(green(f"\nwrote {PYPROJECT.relative_to(ROOT)}"))
    if needs_relock or (pyproject_changed and not lock_touched):
        # Relock so uv.lock captures the new revs / floors from pyproject.
        run(["uv", "lock"])
        lock_touched = True
    if lock_touched:
        print(green("\nuv.lock updated. Sync your env with:"))
        print("  uv sync")
    elif not pyproject_changed:
        print(dim("\nnothing changed; pyproject.toml and uv.lock left as-is."))


if __name__ == "__main__":
    main()
