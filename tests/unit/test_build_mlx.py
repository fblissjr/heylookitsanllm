# tests/unit/test_build_mlx.py
"""Offline tests for ``scripts/build_mlx.py`` (no network, no CMake)."""

from __future__ import annotations

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

    def test_apply_prs_skips_ancestor_and_squash_and_merges_open(self, tmp_path: Path):
        upstream = tmp_path / "upstream"
        upstream.mkdir()
        _git(upstream, "init", "-q", "-b", "main")
        _git(upstream, "config", "user.email", "t@e.st")
        _git(upstream, "config", "user.name", "test")
        (upstream / "base.txt").write_text("base\n")
        _git(upstream, "add", "base.txt")
        _git(upstream, "commit", "-qm", "base")

        # PR 101: ancestor-merged into main via --no-ff.
        _git(upstream, "checkout", "-qb", "pr101")
        (upstream / "merged.txt").write_text("merged\n")
        _git(upstream, "add", "merged.txt")
        _git(upstream, "commit", "-qm", "pr101 commit")
        pr101_sha = _sha(upstream, "HEAD")
        _git(upstream, "update-ref", "refs/pull/101/head", pr101_sha)
        _git(upstream, "checkout", "-q", "main")
        _git(upstream, "merge", "--no-ff", "-qm", "merge 101", "pr101")

        # PR 102: separate commit whose exact file change is already on main (squash-merged).
        _git(upstream, "checkout", "-qb", "pr102")
        (upstream / "squash.txt").write_text("same-tree\n")
        _git(upstream, "add", "squash.txt")
        _git(upstream, "commit", "-qm", "pr102 squash branch")
        pr102_sha = _sha(upstream, "HEAD")
        _git(upstream, "update-ref", "refs/pull/102/head", pr102_sha)
        _git(upstream, "checkout", "-q", "main")
        (upstream / "squash.txt").write_text("same-tree\n")
        _git(upstream, "add", "squash.txt")
        _git(upstream, "commit", "-qm", "squash land on main")

        # PR 103: unmerged open PR atop base commit.
        _git(upstream, "checkout", "-qb", "pr103", "HEAD~2")
        (upstream / "open.txt").write_text("open\n")
        _git(upstream, "add", "open.txt")
        _git(upstream, "commit", "-qm", "pr103 open")
        pr103_sha = _sha(upstream, "HEAD")
        _git(upstream, "update-ref", "refs/pull/103/head", pr103_sha)
        _git(upstream, "checkout", "-q", "main")

        work = tmp_path / "work"
        subprocess.run(
            ["git", "clone", "-q", str(upstream), str(work)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base_sha = build_mlx.checkout_rev(work, "main")
        assert base_sha == _sha(upstream, "main")

        applied = build_mlx.apply_prs(work, [101, 102, 103])
        assert applied == [{"number": 103, "sha": pr103_sha}]
        assert (work / "merged.txt").read_text() == "merged\n"
        assert (work / "squash.txt").read_text() == "same-tree\n"
        assert (work / "open.txt").read_text() == "open\n"


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

    def test_print_status_reports_auto_and_override_note(
        self, tmp_path: Path, capsys, monkeypatch
    ):
        checkout = tmp_path / "mlx_checkout"
        overlay = checkout / "overlay"
        (overlay / "mlx").mkdir(parents=True)
        target_py = tmp_path / "missing-python"

        monkeypatch.delenv("HEYLOOK_MLX_OVERLAY", raising=False)
        build_mlx.print_status(checkout, target_py)
        out_default = capsys.readouterr().out
        assert "heylookllm canonical build" in out_default

        monkeypatch.setenv("HEYLOOK_MLX_OVERLAY", "0")
        build_mlx.print_status(checkout, target_py)
        out_disabled = capsys.readouterr().out
        assert "disables canonical overlay discovery" in out_disabled

        custom = tmp_path / "custom_overlay"
        custom.mkdir()
        monkeypatch.setenv("HEYLOOK_MLX_OVERLAY", str(custom))
        build_mlx.print_status(checkout, target_py)
        out_override = capsys.readouterr().out
        assert "will use THAT overlay instead of the canonical build" in out_override


@pytest.mark.unit
class TestUnifiedBuildMlx:
    def test_normalize_parse_pr_and_metadata_reader(self, tmp_path: Path):
        assert build_mlx.normalize_pkg_name("MLX_VLM") == "mlx-vlm"
        assert build_mlx.normalize_pkg_name("transformers") == "transformers"
        assert build_mlx._parse_pr_arg("4826") == 4826
        assert build_mlx._parse_pr_arg("#4826") == 4826
        with pytest.raises(SystemExit):
            build_mlx._parse_pr_arg("not-a-pr")

        staging = tmp_path / "staging"
        dist_info = staging / "mlx_vlm-0.8.0.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            "Name: mlx-vlm\n"
            "Version: 0.8.0\n"
            "Requires-Dist: mlx>=0.32.3\n"
            "Requires-Dist: gradio>=5.0; extra == 'ui'\n",
            encoding="utf-8",
        )
        ver, reqs = build_mlx._read_dist_metadata(staging, "mlx-vlm")
        assert ver == "0.8.0"
        assert len(reqs) == 2

    def test_check_missing_deps_filters_extras(self):
        missing = build_mlx._check_missing_deps(
            Path(sys.executable),
            [
                "pytest>=1.0",
                "definitely-not-installed-xyz>=0.1",
                "optional-only-pkg>=1.0; extra == 'train'",
            ],
        )
        assert missing == ["definitely-not-installed-xyz"]

    def test_resolve_targets_and_mlx_prs(self):
        assert build_mlx.resolve_targets() == ["mlx-vlm", "mlx"]
        assert build_mlx.resolve_targets(with_pkgs=["transformers", "mlx"]) == [
            "mlx-vlm",
            "transformers",
            "mlx",
        ]
        assert build_mlx.resolve_targets(only=["MLX_VLM", "mlx-vlm"]) == ["mlx-vlm"]
        assert build_mlx.resolve_targets(only=["mlx"]) == ["mlx"]
        with pytest.raises(SystemExit):
            build_mlx.resolve_targets(only=["mlx"], with_pkgs=["transformers"])

        assert build_mlx.resolve_mlx_prs() == list(build_mlx.RECOMMENDED_MLX_PRS)
        assert build_mlx.resolve_mlx_prs([4900, 4900]) == [4900]
        assert build_mlx.resolve_mlx_prs([4900], no_prs=True) == []

    def test_only_python_rejects_pr_and_no_prs(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            build_mlx.main(
                [
                    "--dir",
                    str(tmp_path / "mlx"),
                    "--overlays-dir",
                    str(tmp_path / "overlays"),
                    "--only",
                    "mlx-vlm",
                    "--pr",
                    "123",
                ]
            )
        with pytest.raises(SystemExit):
            build_mlx.main(
                [
                    "--dir",
                    str(tmp_path / "mlx"),
                    "--overlays-dir",
                    str(tmp_path / "overlays"),
                    "--only",
                    "mlx-vlm",
                    "--no-prs",
                ]
            )

    def test_up_to_date_checks_for_mlx_and_python_overlay(self, tmp_path: Path):
        target_py = tmp_path / "venv" / "bin" / "python"
        target_py.parent.mkdir(parents=True)
        target_py.write_text("")

        # MLX overlay check
        checkout = tmp_path / "mlx"
        ov = checkout / "overlay"
        (ov / "mlx" / "lib").mkdir(parents=True)
        (ov / "mlx" / "lib" / "mlx.metallib").write_bytes(b"m")
        (ov / "mlx" / "lib" / "libmlx.dylib").write_bytes(b"d")
        (checkout / "build").mkdir()
        prs = [{"number": 4826, "sha": "abcd1234"}]
        build_mlx.manifest_path(checkout).write_text(
            json.dumps(
                {
                    "base_sha": "base111",
                    "prs": prs,
                    "cmake_args": [],
                    "python": str(target_py),
                    "overlay": str(ov),
                }
            ),
            encoding="utf-8",
        )
        assert (
            build_mlx.mlx_is_up_to_date(
                checkout,
                ov,
                base_sha="base111",
                applied_prs=prs,
                cmake_args=[],
                target_py=target_py,
            )
            is not None
        )
        assert (
            build_mlx.mlx_is_up_to_date(
                checkout,
                ov,
                base_sha="base222",
                applied_prs=prs,
                cmake_args=[],
                target_py=target_py,
            )
            is None
        )
        assert (
            build_mlx.mlx_is_up_to_date(
                checkout,
                ov,
                base_sha="base111",
                applied_prs=[],
                cmake_args=[],
                target_py=target_py,
            )
            is None
        )
        (ov / "mlx" / "lib" / "mlx.metallib").unlink()
        assert (
            build_mlx.mlx_is_up_to_date(
                checkout,
                ov,
                base_sha="base111",
                applied_prs=prs,
                cmake_args=[],
                target_py=target_py,
            )
            is None
        )

        # Pure-Python overlay check
        py_ov = tmp_path / "overlays" / "mlx-vlm"
        (py_ov / "mlx_vlm").mkdir(parents=True)
        (py_ov / "heylook-build.json").write_text(
            json.dumps({"sha": "vlm111", "python": str(target_py)}),
            encoding="utf-8",
        )
        assert (
            build_mlx.python_overlay_is_up_to_date(
                py_ov,
                import_name="mlx_vlm",
                sha="vlm111",
                target_py=target_py,
            )
            is not None
        )
        assert (
            build_mlx.python_overlay_is_up_to_date(
                py_ov,
                import_name="mlx_vlm",
                sha="vlm222",
                target_py=target_py,
            )
            is None
        )
        assert (
            build_mlx.python_overlay_is_up_to_date(
                py_ov,
                import_name="mlx_vlm",
                sha="vlm111",
                target_py=tmp_path / "other-py",
            )
            is None
        )

    def test_unified_status_and_remove_modes(self, tmp_path: Path, capsys):
        mlx_dir = tmp_path / "mlx"
        (mlx_dir / ".git").mkdir(parents=True)
        mlx_ov = mlx_dir / "overlay"
        (mlx_ov / "mlx").mkdir(parents=True)
        (mlx_dir / "build").mkdir()
        build_mlx.manifest_path(mlx_dir).write_text(
            json.dumps(
                {
                    "rev": "main",
                    "sha": "deadbeef12345678",
                    "prs": [{"number": 4826, "sha": "abcd1234ef56"}],
                    "overlay": str(mlx_ov),
                    "version": "0.32.5.dev20261122+deadbee",
                }
            ),
            encoding="utf-8",
        )

        py_root = tmp_path / "overlays"
        repo_keep = py_root / ".repos" / "mlx-vlm"
        repo_keep.mkdir(parents=True)
        (repo_keep / "keep.txt").write_text("repo")

        vlm_ov = py_root / "mlx-vlm"
        (vlm_ov / "mlx_vlm").mkdir(parents=True)
        (vlm_ov / "heylook-build.json").write_text(
            json.dumps(
                {
                    "package": "mlx-vlm",
                    "import_name": "mlx_vlm",
                    "sha": "a22a364af9d7ee5e7e063881923cf18a3f42533d",
                    "version": "0.8.0",
                    "overlay": str(vlm_ov),
                }
            ),
            encoding="utf-8",
        )

        rc = build_mlx.main(
            [
                "--dir",
                str(mlx_dir),
                "--overlays-dir",
                str(py_root),
                "--python",
                str(tmp_path / "missing-py"),
                "--status",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "deadbeef1234" in out
        assert "#4826@abcd1234" in out
        assert "mlx-vlm" in out
        assert "a22a364af9" in out

        # --remove mlx-vlm leaves mlx overlay and .repos untouched.
        assert (
            build_mlx.main(
                ["--dir", str(mlx_dir), "--overlays-dir", str(py_root), "--remove", "mlx-vlm"]
            )
            == 0
        )
        assert not vlm_ov.exists()
        assert mlx_ov.exists()
        assert (repo_keep / "keep.txt").exists()

        # --remove mlx removes mlx overlay only, leaving .git and py_root overlays.
        (vlm_ov / "mlx_vlm").mkdir(parents=True)
        assert (
            build_mlx.main(
                ["--dir", str(mlx_dir), "--overlays-dir", str(py_root), "--remove", "mlx"]
            )
            == 0
        )
        assert not mlx_ov.exists()
        assert (mlx_dir / ".git").exists()
        assert vlm_ov.exists()

        # --remove (const='all') cleans both mlx overlay and pure-Python overlays, keeping .git and .repos.
        (mlx_ov / "mlx").mkdir(parents=True)
        ((py_root / "transformers") / "transformers").mkdir(parents=True)
        assert (
            build_mlx.main(
                ["--dir", str(mlx_dir), "--overlays-dir", str(py_root), "--remove"]
            )
            == 0
        )
        assert not mlx_ov.exists()
        assert not vlm_ov.exists()
        assert not (py_root / "transformers").exists()
        assert (mlx_dir / ".git").exists()
        assert (repo_keep / "keep.txt").exists()
