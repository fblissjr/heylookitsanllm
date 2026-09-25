# tests/unit/test_build_llama_rev.py
"""`--rev <branch>` must build the FETCHED branch, not the stale local one.

`git fetch` advances `origin/<branch>` and deliberately leaves the local
branch of the same name where it was. So `git checkout --detach master` after
a fetch builds whatever the clone happened to have at clone time -- in a
checkout nobody ever runs `git pull` in, that is forever.

This is a real escape, not a hypothetical: the 2026-08-14 llama-server build
was made with `--rev master`, compiled 4-day-old source, and wrote a manifest
recording `rev: "master"` with no hint of the gap.

Real git repos here on purpose. The bug IS git's fetch/branch semantics; a
mocked `run()` would assert my model of git rather than git.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_llama import resolve_fetched_rev  # noqa: E402


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sha(cwd, ref):
    return subprocess.run(["git", "-C", str(cwd), "rev-parse", ref],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def clone(tmp_path):
    """An upstream that has moved on, and a clone that has only fetched."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q", "-b", "master")
    git(upstream, "config", "user.email", "t@e.st")
    git(upstream, "config", "user.name", "test")
    (upstream / "f.txt").write_text("one")
    git(upstream, "add", "f.txt")
    git(upstream, "commit", "-qm", "one")
    git(upstream, "tag", "b1000")

    work = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Upstream moves; the clone only FETCHES (never pulls), exactly like the
    # build script does.
    (upstream / "f.txt").write_text("two")
    git(upstream, "commit", "-qam", "two")
    git(upstream, "tag", "b1001")
    git(work, "fetch", "--tags", "--force", "origin")
    return work


@pytest.mark.unit
class TestResolveFetchedRev:
    def test_a_branch_resolves_to_the_fetched_remote_ref(self, clone):
        """A branch name resolves to origin/<branch>, and that ref's commit is
        the fetched one, not the stale local branch's.

        Guard the guard first: if git ever advanced local branches on fetch
        this whole fix would be pointless, and the assertions below would pass
        for the wrong reason."""
        assert sha(clone, "master") != sha(clone, "origin/master")

        resolved = resolve_fetched_rev(clone, "master")
        assert resolved == "origin/master"
        assert sha(clone, resolved) == sha(clone, "origin/master")
        assert sha(clone, resolved) != sha(clone, "master")

    # Everything that is not a plain branch comes back unchanged. `premise` is
    # a ref that must exist for the row to mean anything.
    # - tag: origin/b1001 does not exist, so a tag must not be rewritten.
    # - qualified: origin/origin/master does not exist -> falls through.
    # - unknown: not this function's job to validate; `git checkout` gives a
    #   better error than anything invented here.
    # - HEAD: `git clone` always writes refs/remotes/origin/HEAD, so the
    #   generic branch mapping would resolve --rev HEAD to the remote default
    #   branch tip -- arbitrarily newer code, and the opposite of "build what
    #   is checked out".
    @pytest.mark.parametrize("rev, premise", [
        ("b1001", "b1001"),
        ("<sha of origin/master>", None),
        ("origin/master", None),
        ("no-such-thing", None),
        ("HEAD", "origin/HEAD"),
    ], ids=["tag", "sha", "already_qualified_remote_ref", "unknown_rev_for_git_to_reject",
            "HEAD_is_not_a_branch"])
    def test_non_branch_revs_pass_through_untouched(self, clone, rev, premise):
        if rev == "<sha of origin/master>":
            rev = sha(clone, "origin/master")
        if premise:
            assert sha(clone, premise)  # the ref really does exist
        assert resolve_fetched_rev(clone, rev) == rev


@pytest.mark.unit
class TestRebuildUsesTheRecordedSha:
    """--rebuild means "same source, new toolchain" -- not "resolve that name
    again". A manifest recording rev: "master" must not rebuild whatever
    upstream merged since."""
    def test_rebuild_checks_out_the_recorded_sha_not_the_rev(self, clone, monkeypatch):
        """main() --rebuild over a real clone whose manifest recorded
        rev: "master" at an older sha, with HEAD parked on the newer
        origin/master. The rebuild must land on the recorded sha. Stubbed:
        the remote check (the clone's origin is a local repo, not GIT_URL)
        and the cmake build, which stops the run right after the checkout."""
        import json

        import build_llama

        class Stopped(Exception):
            pass

        old, new = sha(clone, "master"), sha(clone, "origin/master")
        assert old != new
        git(clone, "checkout", "-q", "--detach", "origin/master")
        (clone / "build").mkdir()
        (clone / "build" / "heylook-build.json").write_text(
            json.dumps({"rev": "master", "sha": old}))

        def stop(*_a, **_k):
            raise Stopped
        monkeypatch.setattr(build_llama, "ensure_checkout", lambda path: None)
        monkeypatch.setattr(build_llama, "build", stop)
        monkeypatch.setattr(sys, "argv", ["build_llama.py", "--rebuild", "--dir", str(clone)])
        with pytest.raises(Stopped):
            build_llama.main()
        assert sha(clone, "HEAD") == old
