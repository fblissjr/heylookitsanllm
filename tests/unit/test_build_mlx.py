# tests/unit/test_build_mlx.py
"""Offline tests for ``scripts/build_mlx.py`` (no network, no CMake)."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_mlx  # noqa: E402


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _sha(cwd: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def mlx_clone(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _git(upstream, "config", "user.email", "t@e.st")
    _git(upstream, "config", "user.name", "test")
    (upstream / "f.txt").write_text("one")
    _git(upstream, "add", "f.txt")
    _git(upstream, "commit", "-qm", "one")
    _git(upstream, "tag", "v0.32.4")

    work = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(work)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    (upstream / "f.txt").write_text("two")
    _git(upstream, "commit", "-qam", "two")
    _git(upstream, "tag", "v0.32.5")
    _git(work, "fetch", "--tags", "--force", "origin")
    return work


@pytest.mark.unit
class TestBuildMlxRevAndStamp:
    def test_resolve_fetched_rev_maps_branch_only(self, mlx_clone: Path):
        assert _sha(mlx_clone, "main") != _sha(mlx_clone, "origin/main")
        assert build_mlx.resolve_fetched_rev(mlx_clone, "main") == "origin/main"

        head_sha = _sha(mlx_clone, "origin/main")
        for passthrough in ("v0.32.5", head_sha, "origin/main", "no-such-ref", "HEAD"):
            assert build_mlx.resolve_fetched_rev(mlx_clone, passthrough) == passthrough

    def test_verify_version_stamp_accepts_sha_and_rejects_stale(self):
        sha = "fc344c986abcdef1234567890"
        build_mlx.verify_version_stamp(sha, f"0.32.5.dev20261119+{sha[:7]}")
        with pytest.raises(SystemExit):
            build_mlx.verify_version_stamp(sha, "0.32.4+deadbee")
        with pytest.raises(SystemExit):
            build_mlx.verify_version_stamp(sha, "")


@pytest.mark.unit
class TestOverlayAndStatus:
    def test_unpack_wheel_to_overlay_replaces_stale_files_atomically(self, tmp_path: Path):
        overlay = tmp_path / "overlay"
        stale = overlay / "mlx" / "stale_artifact.py"
        stale.parent.mkdir(parents=True)
        stale.write_text("# stale")

        whl = tmp_path / "mlx-0.32.5-cp314-cp314-macosx_14_0_arm64.whl"
        with zipfile.ZipFile(whl, "w") as zf:
            zf.writestr("mlx/__init__.py", "__version__ = '0.32.5'\n")
            zi = zipfile.ZipInfo("mlx/core.cpython-314-darwin.so")
            zi.external_attr = 0o100755 << 16
            zf.writestr(zi, b"fake-so")
            zf.writestr("mlx/lib/libmlx.dylib", b"fake-dylib")

        out = build_mlx.unpack_wheel_to_overlay(whl, overlay)
        assert out == overlay
        assert not stale.exists()
        assert (overlay / "mlx" / "__init__.py").read_text().strip() == "__version__ = '0.32.5'"
        assert (overlay / "mlx" / "lib" / "libmlx.dylib").read_bytes() == b"fake-dylib"
        so_mode = (overlay / "mlx" / "core.cpython-314-darwin.so").stat().st_mode
        assert so_mode & 0o111

    def test_print_status_with_and_without_manifest(self, tmp_path: Path, capsys):
        checkout = tmp_path / "mlx_checkout"
        checkout.mkdir()
        target_py = tmp_path / "missing-python"

        build_mlx.print_status(checkout, target_py)
        first = capsys.readouterr().out
        assert "not built" in first

        overlay = checkout / "overlay"
        overlay.mkdir()
        build_dir = checkout / "build"
        build_dir.mkdir()
        build_mlx.manifest_path(checkout).write_text(
            json.dumps(
                {
                    "rev": "main",
                    "sha": "fc344c986abcdef",
                    "prs": [{"number": 4890, "sha": "1234567890ab"}],
                    "overlay": str(overlay),
                    "wheel": str(checkout / "dist" / "mlx.whl"),
                    "version": "0.32.5.dev20261119+fc344c9",
                    "has_window_size_sdpa": True,
                }
            )
        )
        build_mlx.print_status(checkout, target_py)
        second = capsys.readouterr().out
        assert "fc344c986abc" in second
        assert "#4890@12345678" in second
        assert "0.32.5.dev20261119+fc344c9" in second
        assert "sdpa_ws  True" in second
