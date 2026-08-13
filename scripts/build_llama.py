#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Clone and build llama-server (llama.cpp). Nothing else.

`uv sync` cannot build C++, so this is the one explicit step behind the gguf
provider. It never touches pyproject.toml or uv.lock -- it clones llama.cpp
OUTSIDE the repo, builds the `llama-server` binary, and writes a manifest
recording exactly what was built.

By default it builds the newest release. llama.cpp has no semver: upstream
tags every merge to master as `b<N>`, and that tag IS the release. `--rev`
takes any tag/branch/SHA instead; `--rebuild` rebuilds whatever the last
build used (new Xcode, changed flags).

The checkout + build tree land in a fixed directory under your home
(`.heylook/llama.cpp`) -- outside the repo, so multi-GB upstream source can
never be committed or packaged, and the same path on every machine so docs
and error messages can name it. $HEYLOOK_LLAMA_CPP_DIR or --dir relocate it.
The server finds the binary there with zero config (its last-resort fallback);
`server_binary` in models.toml or $HEYLOOK_LLAMA_SERVER override it.

Examples:
  uv run scripts/build_llama.py                # newest release, build it
  uv run scripts/build_llama.py --status       # what is built now; no network
  uv run scripts/build_llama.py --rev b10362   # a specific release
  uv run scripts/build_llama.py --rev master   # tip of master (unreviewed code)
  uv run scripts/build_llama.py --rebuild --clean
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

GIT_URL = "https://github.com/ggml-org/llama.cpp"
TARGETS = ["llama-server"]
HOME_SUBDIR = (".heylook", "llama.cpp")

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
    print(dim("$ " + " ".join(cmd)))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
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


def llama_dir(cli_dir: str | None) -> Path:
    """--dir > $HEYLOOK_LLAMA_CPP_DIR > the fixed home location."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()
    env = os.environ.get("HEYLOOK_LLAMA_CPP_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home().joinpath(*HOME_SUBDIR).resolve()


_BUILD_TAG_RE = re.compile(r"^b(\d+)$")

def latest_release_tag() -> str:
    """Newest llama.cpp release. Tags are `b<N>`, monotonic -- highest N wins."""
    out = run(["git", "ls-remote", "--tags", "--refs", GIT_URL], capture=True)
    best: tuple[int, str] | None = None
    for line in out.splitlines():
        tag = line.rsplit("/", 1)[-1].strip()
        m = _BUILD_TAG_RE.match(tag)
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, tag)
    if best is None:
        die(f"no b<N> release tags found at {GIT_URL}")
    return best[1]


def _remote_key(url: str) -> str:
    """host/owner/repo, so https and ssh forms of one remote compare equal."""
    s = url.strip().removesuffix(".git")
    s = re.sub(r"^[a-z+]+://", "", s)
    s = s.replace(":", "/", 1) if s.startswith("git@") else s
    return s.removeprefix("git@").lower()


def ensure_checkout(path: Path) -> None:
    """Clone if absent; verify the remote if present. Blobless so it stays
    small while still allowing a checkout of any rev."""
    if (path / ".git").exists():
        origin = run(["git", "-C", str(path), "remote", "get-url", "origin"],
                     capture=True)
        if _remote_key(origin) != _remote_key(GIT_URL):
            die(f"{path} tracks {origin}, not {GIT_URL}. "
                f"Point --dir elsewhere or fix the remote.")
        return
    if path.exists() and any(path.iterdir()):
        die(f"{path} exists and is not a llama.cpp checkout -- refusing to "
            f"clone over it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", GIT_URL, str(path)])


def checkout_rev(path: Path, rev: str) -> str:
    """Detach onto `rev`. Refuses over TRACKED modifications. Returns the SHA.

    Untracked files (a stray .DS_Store, an editor swap file) do not block:
    git carries them across a checkout untouched."""
    dirty = run(["git", "-C", str(path), "status", "--porcelain",
                 "--untracked-files=no"], capture=True)
    if dirty:
        die(f"{path} has uncommitted changes to TRACKED files -- refusing to "
            f"check out {rev} over them. Commit or stash first:\n{dirty}")
    run(["git", "-C", str(path), "fetch", "--tags", "--force", "origin"])
    run(["git", "-C", str(path), "checkout", "--detach", rev])
    return run(["git", "-C", str(path), "rev-parse", "HEAD"], capture=True)


def cmake_args(*, lto: bool, openmp: bool, ui: bool,
               extra: list[str]) -> list[str]:
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
    return args + extra


_CACHE_LINE = re.compile(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$")

# Flags whose EFFECTIVE value the manifest records. Requested args are a
# statement of intent; cmake can silently downgrade one (OpenMP), and a reused
# cache can carry an arg that is no longer requested. The manifest has to
# describe the binary, not the wish.
_EFFECTIVE_KEYS = (
    "CMAKE_GENERATOR", "CMAKE_BUILD_TYPE", "GGML_OPENMP_ENABLED", "GGML_LTO",
    "GGML_METAL", "GGML_METAL_NDEBUG", "GGML_METAL_EMBED_LIBRARY", "GGML_NATIVE",
    "GGML_BLAS", "GGML_ACCELERATE", "LLAMA_BUILD_UI", "BUILD_SHARED_LIBS",
)


def read_cmake_cache(build: Path) -> dict[str, str]:
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
    and builds the threadpool path anyway, so you get a binary byte-identical
    to the non-OpenMP one while the flag says ON. AppleClang needs the flags
    spelled out; `-DOpenMP_ROOT=` is not enough."""
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


def build(path: Path, args: list[str], jobs: int, clean: bool,
          *, want_openmp: bool) -> Path:
    cmake = shutil.which("cmake")
    if not cmake:
        die("cmake not found on PATH (brew install cmake)")
    build_dir = path / "build"
    if clean and build_dir.exists():
        print(dim(f"removing {build_dir}"))
        shutil.rmtree(build_dir)

    cached = read_cmake_cache(build_dir)

    # A CMake build tree hard-codes the absolute source path it was configured
    # for; a moved checkout leaves a tree cmake refuses to reuse. It is pure
    # build output -- discard it.
    cached_src = cached.get("CMAKE_HOME_DIRECTORY")
    if cached_src and Path(cached_src) != path:
        print(yellow(f"  build tree was configured for {cached_src}, which is "
                     f"not where the source lives now -- discarding it"))
        shutil.rmtree(build_dir, ignore_errors=True)
        cached = {}

    # Only choose a generator on a FRESH configure: -G at a dir configured with
    # a different generator is a hard cmake error whose fix (--clean) it does
    # not name.
    if cached.get("CMAKE_GENERATOR"):
        generator = []
        print(dim(f"reusing cached generator: {cached['CMAKE_GENERATOR']}"))
    else:
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []

    run([cmake, "-S", str(path), "-B", str(build_dir), *generator, *args])

    # ggml only WARNS when OpenMP is missing and links the threadpool build
    # anyway. Read back what cmake resolved and refuse to hand over a
    # silently-downgraded binary that would poison an A/B.
    effective = read_cmake_cache(build_dir)
    if want_openmp and effective.get("GGML_OPENMP_ENABLED", "OFF").upper() != "ON":
        die("--openmp was requested but cmake resolved GGML_OPENMP_ENABLED=OFF "
            "(ggml only warns, it does not fail). Install libomp "
            "(`brew install libomp`) and re-run; the binary you would have got "
            "is byte-identical to a non-OpenMP build.")

    cmd = [cmake, "--build", str(build_dir), "--config", "Release", "-j", str(jobs)]
    for target in TARGETS:
        cmd += ["--target", target]
    run(cmd)
    return build_dir


def binary_version(binary: Path) -> str:
    """First line of `llama-server --version` (on stderr): `version: N (sha)`."""
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


def print_status(path: Path) -> None:
    build_dir = path / "build"
    binary = build_dir / "bin" / "llama-server"
    manifest = read_manifest(build_dir)
    print(f"\n{bold('llama-server')}  {dim(str(path))}")
    if not binary.exists():
        print(yellow("  not built -- run: uv run scripts/build_llama.py"))
        return
    print(f"  binary   {green(str(binary))}")
    if manifest:
        print(f"  rev      {manifest.get('rev')}  {dim(manifest.get('version', ''))}")
        eff = manifest.get("effective", {})
        interesting = {k: eff[k] for k in ("GGML_OPENMP_ENABLED", "GGML_LTO")
                       if k in eff}
        if interesting:
            print(f"  built    {dim('  '.join(f'{k}={v}' for k, v in interesting.items()))}")
    else:
        print(yellow("  no build manifest -- provenance unknown"))
    env = os.environ.get("HEYLOOK_LLAMA_SERVER")
    if env and Path(env).resolve() != binary.resolve():
        print(yellow(f"  note: $HEYLOOK_LLAMA_SERVER points at {env} -- the "
                     f"server will use THAT binary, not this build."))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev", default=None,
                    help="tag, branch, or SHA to build (default: newest b<N> "
                         "release tag; `master` = unreviewed tip)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the rev of the existing build (per its manifest)")
    ap.add_argument("--status", action="store_true",
                    help="report what is built; no network, no writes")
    ap.add_argument("--dir", default=None,
                    help="checkout location (default: the fixed dir under your "
                         "home; $HEYLOOK_LLAMA_CPP_DIR also overrides)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the build dir and reconfigure")
    ap.add_argument("--lto", action="store_true",
                    help="-DGGML_LTO=ON (slow link, noise-level gain on a Metal-bound load)")
    ap.add_argument("--openmp", action="store_true",
                    help="-DGGML_OPENMP=ON (needs libomp; ggml's own threadpool otherwise)")
    ap.add_argument("--ui", action="store_true",
                    help="build llama-server's embedded WebUI (off by default: headless here)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8,
                    help="parallel build jobs")
    ap.add_argument("--cmake-arg", action="append", default=[], metavar="ARG",
                    help="extra cmake argument (repeatable)")
    args = ap.parse_args()

    path = llama_dir(args.dir)

    if args.status:
        print_status(path)
        return

    if args.rebuild:
        manifest = read_manifest(path / "build")
        rev = manifest.get("rev") or die("--rebuild needs an existing build manifest")
    else:
        rev = args.rev or latest_release_tag()

    fresh = not (path / ".git").exists()
    print(f"\n{bold('llama.cpp')} {yellow(rev)} -> {path}"
          f"{dim('  (fresh clone + build tree, several GB)') if fresh else ''}")
    if args.rev in ("master", "HEAD") or (args.rev and not re.fullmatch(r"b\d+", args.rev)):
        print(dim("  building a non-release rev: unreviewed upstream code."))

    ensure_checkout(path)
    sha = checkout_rev(path, rev)
    cargs = cmake_args(lto=args.lto, openmp=args.openmp, ui=args.ui,
                       extra=args.cmake_arg)
    if args.openmp:
        cargs += libomp_flags()  # -DGGML_OPENMP=ON alone does not find libomp
    build_dir = build(path, cargs, args.jobs, args.clean, want_openmp=args.openmp)

    binary = build_dir / "bin" / "llama-server"
    if not binary.exists():
        die(f"build finished but {binary} is missing")
    version = binary_version(binary)

    cache = read_cmake_cache(build_dir)
    manifest = {
        "rev": rev,
        "sha": sha,
        "targets": TARGETS,
        "cmake_args": cargs,
        "effective": {k: cache[k] for k in _EFFECTIVE_KEYS if k in cache},
        "binary": str(binary),
        "sha256": sha256(binary),
        "version": version,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (build_dir / "heylook-build.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n  built {green(str(binary))}")
    if version:
        print(f"  {dim(version)}")
    current = os.environ.get("HEYLOOK_LLAMA_SERVER")
    if current and Path(current).resolve() != binary.resolve():
        print(yellow(f"  note: $HEYLOOK_LLAMA_SERVER points at {current} -- the "
                     f"server will keep using THAT binary, not this build."))
    else:
        print(dim("  the server picks this up automatically (its default "
                  "build location)."))


if __name__ == "__main__":
    main()
