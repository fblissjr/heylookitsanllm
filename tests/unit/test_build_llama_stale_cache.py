# tests/unit/test_build_llama_stale_cache.py
"""A configure failure caused by a STALE build tree must self-heal once.

cmake caches every `find_package` result as a FILEPATH and never re-checks
that the file is still there, so a package manager upgrading a library out
from under a build tree leaves dangling paths that STILL READ AS FOUND. On
2026-09-06 homebrew replaced openssl@3 3.6.3 with 3.6.4: FindOpenSSL's
REQUIRED_VARS (non-empty strings, not files) passed, its imported targets
(guarded on EXISTS) were never created, and llama.cpp's vendored cpp-httplib
died on `OpenSSL::SSL ... target was not found` -- with nothing in the message
pointing at the cache. The source was fine; the tree was stale.

Real cmake on purpose. The bug IS cmake's cache semantics; a mocked cmake
would assert my model of them. `CACHED_LIB` stands in for the find_package
result: passed once, cached forever, absent from the second run's args.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_llama import TARGETS, build, read_cmake_cache  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None,
                                reason="cmake not installed")

CMAKELISTS = """cmake_minimum_required(VERSION 3.16)
project(stale C)
if(DEFINED CACHED_LIB AND NOT EXISTS "${CACHED_LIB}")
  message(FATAL_ERROR "cached path is gone: ${CACHED_LIB}")
endif()
""" + "".join(f"add_executable({t} main.c)\n" for t in TARGETS)
# One stand-in executable per REAL target, derived from the script's own
# list: the build step asks for every target by name, so a stand-in that
# defined only llama-server went red the day llama-bench joined the list.


@pytest.fixture
def src(tmp_path):
    """The smallest project that fails to configure on a dangling cached path."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "main.c").write_text("int main(void) { return 0; }\n")
    (d / "CMakeLists.txt").write_text(CMAKELISTS)
    return d


def run_build(src, args, *, want_openmp=False):
    return build(src, args, jobs=1, clean=False, want_openmp=want_openmp)


def test_reused_tree_that_went_stale_is_discarded_and_retried(src, tmp_path):
    lib = tmp_path / "libstand-in.dylib"
    lib.write_text("")
    run_build(src, [f"-DCACHED_LIB={lib}"])

    lib.unlink()  # the upgrade: the cached path now names nothing

    build_dir = run_build(src, [])  # the arg is gone; the CACHE still has it

    # Not "the error went away" -- the tree was actually thrown out. A reused
    # tree would still carry CACHED_LIB, so its absence is what proves it.
    assert "CACHED_LIB" not in read_cmake_cache(build_dir)
    assert list(build_dir.rglob("llama-server")), "target was never built"


def test_fresh_tree_that_fails_is_not_retried(src, tmp_path):
    """Retrying a FRESH failure just prints the same error twice."""
    with pytest.raises(SystemExit):
        run_build(src, [f"-DCACHED_LIB={tmp_path / 'never-existed'}"])


def test_a_heal_that_fails_still_leaves_the_last_binary(src, tmp_path):
    """The heal drops a cache file, never a working llama-server.

    The tree is reused, so the retry fires -- but the source is broken now, so
    it cannot succeed. What must survive is the binary the last good build
    produced: until the new one exists, that one is the one the server runs.
    """
    build_dir = run_build(src, [])
    binaries = list(build_dir.rglob("llama-server"))
    assert binaries

    (src / "CMakeLists.txt").write_text(
        CMAKELISTS + 'message(FATAL_ERROR "no cache can fix this")\n')

    with pytest.raises(SystemExit):
        run_build(src, [])

    assert [b for b in binaries if b.exists()] == binaries


def test_openmp_downgrade_is_still_refused_after_the_retry(src):
    """The guard the retry block sits next to, and once deleted along with it.

    ggml only WARNS when OpenMP is missing and links its own threadpool anyway,
    so `--openmp` has to be verified against the RESOLVED cache or it ships a
    binary byte-identical to a non-OpenMP one while every log says ON. That
    read-back lives directly below the configure call this file's other tests
    are about; an edit to one is an edit next to the other.
    """
    with pytest.raises(SystemExit):
        run_build(src, [], want_openmp=True)
    assert not list((src / "build").rglob("llama-server")), "built anyway"
