"""Unit tests for heylookllm companion app launcher (`--steering-lab`, `--apps`, `app <name>`)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from heylook_llm import app_launcher


def test_apps_list_and_unknown_app(capsys: pytest.CaptureFixture[str]) -> None:
    rc = app_launcher.maybe_dispatch_app_shortcut(["--apps"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--steering-lab" in out
    assert "--batch-labeler" in out

    rc_list = app_launcher.maybe_dispatch_app_shortcut(["app", "list"])
    assert rc_list == 0

    rc_bad = app_launcher.maybe_dispatch_app_shortcut(["app", "no-such-app"])
    assert rc_bad == 2
    assert "Unknown app" in capsys.readouterr().err

    # Regular server flags must fall through cleanly (`None`).
    assert app_launcher.maybe_dispatch_app_shortcut(["--port", "8000"]) is None
    assert app_launcher.maybe_dispatch_app_shortcut(["key", "status"]) is None
    assert app_launcher.maybe_dispatch_app_shortcut([]) is None


def test_steering_lab_shortcut_defaults_to_serve_and_auto_attaches_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build a fake repo layout under tmp_path with a bank + matching vectors dir.
    src_dir = tmp_path / "apps" / "steering-lab" / "src" / "steering_lab"
    src_dir.mkdir(parents=True)
    bank_dir = tmp_path / "apps" / "steering-lab" / "data" / "banks" / "swift_core"
    bank_dir.mkdir(parents=True)
    (bank_dir / "bank.json").write_text("{}", encoding="utf-8")
    vec_dir = tmp_path / "apps" / "steering-lab" / "data" / "vectors" / "swift_core"
    vec_dir.mkdir(parents=True)

    captured_argv: list[list[str]] = []

    class _FakeCli:
        @staticmethod
        def main(argv: list[str]) -> int:
            captured_argv.append(list(argv))
            return 0

    monkeypatch.setattr(app_launcher, "_repo_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "steering_lab.cli", _FakeCli)

    # Bare `--steering-lab --port 8123` defaults to `serve` + auto bank/vectors/model.
    rc = app_launcher.maybe_dispatch_app_shortcut(["--steering-lab", "--port", "8123"])
    assert rc == 0
    assert captured_argv[-1][0] == "serve"
    assert "--port" in captured_argv[-1] and "8123" in captured_argv[-1]
    assert "--bank" in captured_argv[-1] and str(bank_dir) in captured_argv[-1]
    assert "--vectors" in captured_argv[-1] and str(vec_dir) in captured_argv[-1]
    assert "--model" in captured_argv[-1]

    # Explicit subcommand (`capture --help`) passes through untouched without prepending `serve`.
    rc2 = app_launcher.maybe_dispatch_app_shortcut(
        ["app", "steering-lab", "capture", "--model", "foo", "--out", "bar"]
    )
    assert rc2 == 0
    assert captured_argv[-1] == ["capture", "--model", "foo", "--out", "bar"]


def test_server_apps_flag_does_not_import_mlx() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "heylook_llm.server", "--apps"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--steering-lab" in proc.stdout
