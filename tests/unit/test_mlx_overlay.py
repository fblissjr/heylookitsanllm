# tests/unit/test_mlx_overlay.py
"""Offline unit tests for zero-config canonical MLX overlay discovery."""

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

from heylook_llm import mlx_overlay


def _make_overlay(
    root: Path,
    *,
    version: str = "0.32.5.dev20261119+280cc67d",
    ext_suffix: str | None = None,
    sha: str = "280cc67d1234567890",
    prs: list[int] | None = None,
    with_manifest: bool = True,
) -> Path:
    """Create a minimal unpacked MLX overlay tree with optional build manifest."""
    overlay = root / "overlay"
    pkg = overlay / "mlx"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    suffix = sysconfig.get_config_var("EXT_SUFFIX") if ext_suffix is None else ext_suffix
    if suffix:
        (pkg / f"core{suffix}").write_bytes(b"")
    dist_info = overlay / f"mlx-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: mlx\nVersion: {version}\n",
        encoding="utf-8",
    )
    if with_manifest:
        build_dir = root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "heylook-build.json").write_text(
            json.dumps(
                {
                    "rev": "main",
                    "sha": sha,
                    "prs": [{"number": n, "sha": f"{n:08x}"} for n in (prs or [])],
                    "overlay": str(overlay),
                    "version": version,
                }
            ),
            encoding="utf-8",
        )
    return overlay


@pytest.mark.unit
class TestResolveMlxOverlay:
    def test_canonical_selected_when_newer_or_equal_to_venv(self, tmp_path: Path, monkeypatch):
        overlay = _make_overlay(
            tmp_path / "mlx",
            version="0.32.5.dev20261119+280cc67d",
            prs=[4890, 4826],
        )
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay)

        d = mlx_overlay.resolve_mlx_overlay(
            {}, venv_version="0.32.4", already_imported=False
        )
        assert d.action == "canonical"
        assert d.path == overlay
        assert d.is_override is False
        assert d.warning is None
        assert d.label == "280cc67d (#4890, #4826)"
        assert d.describe() == "canonical build 280cc67d (#4890, #4826)"

        overlay_eq = _make_overlay(tmp_path / "mlx_eq", version="0.32.4", prs=[])
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay_eq)
        d_eq = mlx_overlay.resolve_mlx_overlay(
            {}, venv_version="0.32.4", already_imported=False
        )
        assert d_eq.action == "canonical"
        assert d_eq.path == overlay_eq
        assert d_eq.warning is None

    def test_superseded_by_pypi_stands_down_with_warning(self, tmp_path: Path, monkeypatch):
        overlay = _make_overlay(tmp_path / "mlx", version="0.32.5.dev20261119+280cc67d")
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay)

        d = mlx_overlay.resolve_mlx_overlay(
            {}, venv_version="0.32.5", already_imported=False
        )
        assert d.action == "superseded_by_pypi"
        assert d.path is None
        assert d.source == ".venv"
        assert d.warning is not None
        assert "0.32.5 is newer than canonical overlay" in d.warning
        assert "build_mlx.py" in d.warning

    def test_abi_mismatch_falls_back_to_pypi_and_abi3_accepted(
        self, tmp_path: Path, monkeypatch
    ):
        overlay_bad = _make_overlay(
            tmp_path / "mlx_bad",
            ext_suffix=".cpython-399-darwin.so",
        )
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay_bad)

        d_bad = mlx_overlay.resolve_mlx_overlay(
            {},
            venv_version="0.32.4",
            ext_suffix=".cpython-314-darwin.so",
            already_imported=False,
        )
        assert d_bad.action == "abi_mismatch"
        assert d_bad.path is None
        assert d_bad.warning is not None
        assert "incompatible ABI" in d_bad.warning

        overlay_abi3 = _make_overlay(tmp_path / "mlx_abi3", ext_suffix=".abi3.so")
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay_abi3)
        d_abi3 = mlx_overlay.resolve_mlx_overlay(
            {},
            venv_version="0.32.4",
            ext_suffix=".cpython-314-darwin.so",
            already_imported=False,
        )
        assert d_abi3.action == "canonical"
        assert d_abi3.path == overlay_abi3

    @pytest.mark.parametrize("val", ["0", "off", "false", "no", "none", "pypi", "disabled", "OFF"])
    def test_env_disable_values_skip_canonical(self, val: str, tmp_path: Path, monkeypatch):
        overlay = _make_overlay(tmp_path / "mlx")
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay)

        d = mlx_overlay.resolve_mlx_overlay(
            {mlx_overlay.ENV_OVERLAY: val},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d.action == "disabled"
        assert d.path is None
        assert d.is_override is True
        assert d.warning is not None
        assert "disables canonical MLX overlay" in d.warning

        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", tmp_path / "absent")
        d_no_canon = mlx_overlay.resolve_mlx_overlay(
            {mlx_overlay.ENV_OVERLAY: val},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d_no_canon.action == "disabled"
        assert d_no_canon.warning is None

    def test_env_override_valid_missing_and_bad_abi(self, tmp_path: Path, monkeypatch):
        canonical = _make_overlay(tmp_path / "canon", version="0.32.5.dev20261119+11111111")
        custom = _make_overlay(tmp_path / "custom", version="0.32.5.dev20261119+22222222", prs=[4893])
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", canonical)

        d = mlx_overlay.resolve_mlx_overlay(
            {mlx_overlay.ENV_OVERLAY: str(custom)},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d.action == "override"
        assert d.path == custom
        assert d.is_override is True
        assert d.label == "280cc67d (#4893)"
        assert d.warning is not None
        assert "OVERRIDING the canonical overlay" in d.warning

        with pytest.raises(RuntimeError, match="has no 'mlx/' package"):
            mlx_overlay.resolve_mlx_overlay(
                {mlx_overlay.ENV_OVERLAY: str(tmp_path / "missing")},
                already_imported=False,
            )

        bad_abi = _make_overlay(tmp_path / "bad_abi", ext_suffix=".cpython-399-darwin.so")
        with pytest.raises(RuntimeError, match="incompatible Python ABI"):
            mlx_overlay.resolve_mlx_overlay(
                {mlx_overlay.ENV_OVERLAY: str(bad_abi)},
                ext_suffix=".cpython-314-darwin.so",
                already_imported=False,
            )

    def test_pythonpath_takes_precedence_with_conditional_shadow_warning(
        self, tmp_path: Path, monkeypatch
    ):
        canonical = _make_overlay(tmp_path / "canon", prs=[4890])
        other = _make_overlay(tmp_path / "other", prs=[9999])
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", canonical)

        d_other = mlx_overlay.resolve_mlx_overlay(
            {"PYTHONPATH": os.pathsep.join([str(tmp_path / "unrelated"), str(other)])},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d_other.action == "pythonpath"
        assert d_other.path == other
        assert d_other.is_override is True
        assert d_other.warning is not None
        assert "OVERRIDING the canonical overlay" in d_other.warning

        d_same = mlx_overlay.resolve_mlx_overlay(
            {"PYTHONPATH": str(canonical)},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d_same.action == "pythonpath"
        assert d_same.path == canonical
        assert d_same.is_override is False
        assert d_same.warning is None

    def test_missing_canonical_selects_pypi_without_warning(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", tmp_path / "none" / "overlay")
        d = mlx_overlay.resolve_mlx_overlay({}, already_imported=False)
        assert d.action == "pypi"
        assert d.path is None
        assert d.source == ".venv"
        assert d.warning is None

    def test_already_imported_and_activate_sys_path(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        overlay = _make_overlay(tmp_path / "mlx", prs=[4890])
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay)
        monkeypatch.setattr(sys, "path", list(sys.path))

        d_imported = mlx_overlay.resolve_mlx_overlay({}, already_imported=True)
        assert d_imported.action == "already_imported"
        assert d_imported.path is None

        # Simulate already-imported state inside activate_mlx_overlay -> no sys.path change.
        monkeypatch.setattr(mlx_overlay, "resolve_mlx_overlay", lambda **kw: d_imported)
        before = list(sys.path)
        mlx_overlay.activate_mlx_overlay({}, log=False)
        assert sys.path == before

        # Real activate_mlx_overlay prepends canonical without duplicates.
        monkeypatch.undo()
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", overlay)
        monkeypatch.setattr(
            mlx_overlay,
            "resolve_mlx_overlay",
            lambda environ=None: mlx_overlay.OverlayDecision(
                action="canonical",
                path=overlay,
                source="canonical build",
                label="280cc67d (#4890)",
            ),
        )
        monkeypatch.setattr(sys, "path", ["/already", str(overlay), "/tail"])
        with caplog.at_level("INFO"):
            res = mlx_overlay.activate_mlx_overlay({}, log=True)
        assert res.action == "canonical"
        assert sys.path[0] == str(overlay)
        assert sys.path.count(str(overlay)) == 1
        assert "using canonical overlay at" in caplog.text

    def test_heylook_mlx_dir_overrides_canonical_path(self, tmp_path: Path, monkeypatch):
        custom_root = tmp_path / "custom_mlx_root"
        overlay = _make_overlay(custom_root, version="0.32.5.dev20261119+abcdef12")
        monkeypatch.setattr(mlx_overlay, "DEFAULT_OVERLAY", tmp_path / "unused" / "overlay")

        d = mlx_overlay.resolve_mlx_overlay(
            {mlx_overlay.ENV_MLX_DIR: str(custom_root)},
            venv_version="0.32.4",
            already_imported=False,
        )
        assert d.action == "canonical"
        assert d.path == overlay

    def test_version_comparator_handles_dev_vs_final_and_fallback(self):
        assert mlx_overlay._compare_versions("0.32.5", "0.32.5.dev20261119+280cc67d") == 1
        assert mlx_overlay._compare_versions("0.32.4", "0.32.5.dev20261119+280cc67d") == -1
        assert mlx_overlay._compare_versions("0.32.4", "0.32.4") == 0

        assert mlx_overlay._parse_version_fallback("0.32.5") > mlx_overlay._parse_version_fallback(
            "0.32.5.dev20261119+280cc67d"
        )
        assert mlx_overlay._parse_version_fallback("0.32.5.dev20261119+280cc67d") > mlx_overlay._parse_version_fallback(
            "0.32.4"
        )

    def test_importing_server_does_not_eagerly_import_mlx(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import heylook_llm.server, sys; assert 'mlx' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
