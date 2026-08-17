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
    def test_the_premise_holds_local_master_is_stale_after_fetch(self, clone):
        """Guard the guard. If git ever advanced local branches on fetch this
        whole fix would be pointless, and every test below would pass for the
        wrong reason."""
        assert sha(clone, "master") != sha(clone, "origin/master")

    def test_a_branch_resolves_to_the_remote_tracking_ref(self, clone):
        assert resolve_fetched_rev(clone, "master") == "origin/master"

    def test_the_resolved_ref_is_the_fetched_commit(self, clone):
        resolved = resolve_fetched_rev(clone, "master")
        assert sha(clone, resolved) == sha(clone, "origin/master")
        assert sha(clone, resolved) != sha(clone, "master")

    def test_a_tag_is_passed_through_untouched(self, clone):
        # origin/b1001 does not exist, so a tag must not be rewritten.
        assert resolve_fetched_rev(clone, "b1001") == "b1001"
        assert sha(clone, "b1001")  # still resolvable

    def test_a_sha_is_passed_through_untouched(self, clone):
        head = sha(clone, "origin/master")
        assert resolve_fetched_rev(clone, head) == head

    def test_an_already_qualified_remote_ref_is_passed_through(self, clone):
        # origin/origin/master does not exist -> falls through unchanged.
        assert resolve_fetched_rev(clone, "origin/master") == "origin/master"

    def test_an_unknown_rev_is_passed_through_for_git_to_reject(self, clone):
        """Not this function's job to validate: `git checkout` gives a better
        error than anything invented here."""
        assert resolve_fetched_rev(clone, "no-such-thing") == "no-such-thing"
