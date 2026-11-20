"""Unit tests for scripts/sync_overlay.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_overlay.py"


def _load_sync_overlay():
    spec = importlib.util.spec_from_file_location("sync_overlay", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestSyncOverlay:
    def test_normalize_and_metadata_reader(self, tmp_path: Path):
        mod = _load_sync_overlay()
        assert mod.normalize_pkg_name("MLX_VLM") == "mlx-vlm"
        assert mod.normalize_pkg_name("transformers") == "transformers"

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
        ver, reqs = mod._read_dist_metadata(staging, "mlx-vlm")
        assert ver == "0.8.0"
        assert len(reqs) == 2

    def test_status_and_remove(self, tmp_path: Path, capsys):
        mod = _load_sync_overlay()
        root = tmp_path / "overlays"
        ov = root / "mlx-vlm"
        (ov / "mlx_vlm").mkdir(parents=True)
        (ov / "heylook-build.json").write_text(
            json.dumps(
                {
                    "package": "mlx-vlm",
                    "import_name": "mlx_vlm",
                    "sha": "a22a364af9d7ee5e7e063881923cf18a3f42533d",
                    "version": "0.8.0",
                    "prs": [{"number": 2767}],
                    "overlay": str(ov),
                }
            ),
            encoding="utf-8",
        )

        assert mod.main(["--dir", str(root), "--status"]) == 0
        out = capsys.readouterr().out
        assert "mlx-vlm" in out
        assert "a22a364af9" in out
        assert "#2767" in out

        assert mod.main(["--dir", str(root), "mlx-vlm", "--remove"]) == 0
        assert not ov.exists()

    def test_check_missing_deps_filters_extras(self):
        mod = _load_sync_overlay()
        missing = mod._check_missing_deps(
            Path(sys.executable),
            [
                "pytest>=1.0",
                "definitely-not-installed-xyz>=0.1",
                "optional-only-pkg>=1.0; extra == 'train'",
            ],
        )
        assert missing == ["definitely-not-installed-xyz"]
