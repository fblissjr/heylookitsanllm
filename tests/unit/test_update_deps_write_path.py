"""update_deps.py write-path guards.

The script rewrites pyproject.toml (tomlkit, comment-preserving) and then
locks. Three things here had zero coverage while carrying real failure
modes (2026-08-11 audit): the latest-channel sources entry shape (uv rejects
`branch` alongside `rev` outright, and the file lands on disk before uv gets
to complain -- the whole channel shipped broken once, 030119f), the
comment-preserving round-trip the design leans on, and the concurrent-edit
guard (parallel sessions are normal here and the llama.cpp path holds the
in-memory doc across a minutes-long build).
"""

import importlib.util
from pathlib import Path

import pytest
import tomlkit

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "update_deps_under_test", ROOT / "scripts" / "update_deps.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod():
    return _load_module()


@pytest.mark.unit
def test_latest_channel_writes_git_and_rev_only(mod):
    """`branch` in [tool.uv.sources] next to `rev` is a hard uv error, and it
    reaches disk before uv sees it. The entry must be exactly {git, rev}."""
    doc = tomlkit.parse("")
    plan = {
        "pkg": "mlx-lm",
        "kind": "python",
        "channel": "latest",
        "changes": True,
        "old": "0" * 40,
        "new": "1" * 40,
        "url": "https://github.com/ml-explore/mlx-lm",
    }
    changed, needs_relock, lock_touched = mod.apply_python(doc, plan, False)
    assert changed and needs_relock and not lock_touched
    entry = doc["tool"]["uv"]["sources"]["mlx-lm"]
    assert set(entry.keys()) == {"git", "rev"}, dict(entry)
    assert entry["rev"] == "1" * 40


@pytest.mark.unit
def test_tomlkit_roundtrip_preserves_the_real_pyproject():
    """The overwrite design is only comment-safe if a no-edit round-trip is
    byte-identical on the actual file (this is what separates it from the
    models.toml/tomli_w comment-loss failure)."""
    text = (ROOT / "pyproject.toml").read_text()
    assert tomlkit.dumps(tomlkit.parse(text)) == text


@pytest.mark.unit
def test_write_guard_refuses_a_concurrent_edit(mod, tmp_path):
    """An edit landed by another session between load and write must abort
    the write (their edit wins, this run re-runs) -- never last-writer-wins."""
    scratch = tmp_path / "pyproject.toml"
    scratch.write_text("a = 1\n")
    mod.PYPROJECT = scratch

    doc = mod.load_doc()
    doc["a"] = 2
    scratch.write_text("a = 3  # another session\n")

    with pytest.raises(SystemExit):
        mod.write_pyproject(doc)
    assert scratch.read_text() == "a = 3  # another session\n"


@pytest.mark.unit
def test_write_guard_allows_sequential_writes(mod, tmp_path):
    """Our own writes update the snapshot: write, mutate, write again."""
    scratch = tmp_path / "pyproject.toml"
    scratch.write_text("a = 1\n")
    mod.PYPROJECT = scratch

    doc = mod.load_doc()
    doc["a"] = 2
    mod.write_pyproject(doc)
    doc["a"] = 3
    mod.write_pyproject(doc)
    assert "a = 3" in scratch.read_text()


@pytest.mark.unit
def test_rollback_restores_and_resets_snapshot(mod, tmp_path):
    scratch = tmp_path / "pyproject.toml"
    scratch.write_text("a = 1\n")
    mod.PYPROJECT = scratch

    doc = mod.load_doc()
    doc["a"] = 2
    mod.write_pyproject(doc)
    mod.restore_pyproject("a = 1\n")
    assert scratch.read_text() == "a = 1\n"
    # and the snapshot follows the restore, so a fresh write is not refused
    doc2 = mod.load_doc()
    doc2["a"] = 4
    mod.write_pyproject(doc2)
    assert "a = 4" in scratch.read_text()
