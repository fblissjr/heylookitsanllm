"""Tests for scripts/guard_stable_channel.sh commit guard behavior."""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "guard_stable_channel.sh"
HEX40 = "e267ccf740b8b9dfc4529665b5efd8b9e1eff187"


def _run_guard(
    tmp_path: Path,
    *,
    pyproject: str | None = None,
    uv_lock: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    to_stage: list[str] = []
    if pyproject is not None:
        (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        to_stage.append("pyproject.toml")
    if uv_lock is not None:
        (tmp_path / "uv.lock").write_text(uv_lock, encoding="utf-8")
        to_stage.append("uv.lock")
    if to_stage:
        subprocess.run(["git", "add", *to_stage], cwd=tmp_path, check=True)

    env = os.environ.copy()
    env.pop("HEYLOOK_ALLOW_CHANNEL_COMMIT", None)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", str(GUARD_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def test_allows_clean_pypi_and_self_editable_without_override(tmp_path: Path) -> None:
    pyproject = (
        '[project]\nname = "demo"\n'
        'dependencies = ["mlx-vlm>=0.8.0"]\n\n'
        "[tool.coverage.run]\nbranch = true\n"
    )
    uv_lock = (
        '[[package]]\nname = "demo"\nsource = { editable = "." }\n\n'
        '[[package]]\nname = "mlx-vlm"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    res = _run_guard(tmp_path, pyproject=pyproject, uv_lock=uv_lock)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    ("source_block", "expected_msg"),
    [
        (
            f'[tool.uv.sources]\nmlx-vlm = {{ git = "https://github.com/Neywa-Labs/mlx-vlm", rev = "{HEX40}" }}\n',
            "git source",
        ),
        (
            '[tool.uv.sources]\nmlx-vlm = { git = "https://github.com/example/mlx-vlm", branch = "main" }\n',
            "git source",
        ),
        (
            '[tool.uv.sources]\nmlx-vlm = { git = "https://github.com/example/mlx-vlm", tag = "v0.4.0" }\n',
            "git source",
        ),
        (
            '[tool.uv.sources]\nmlx-vlm = { git = "https://github.com/example/mlx-vlm" }\n',
            "git source",
        ),
        (
            '[tool.uv.sources.mlx-vlm]\ngit = "https://github.com/example/mlx-vlm"\n',
            "git source",
        ),
        (
            '[tool.uv.sources]\nmlx-vlm = { path = "../mlx-vlm", editable = true }\n',
            "local path source",
        ),
        (
            '[tool.uv]\noverrides = ["mlx==latest"]\n',
            "channel override",
        ),
    ],
)
def test_blocks_git_or_local_pyproject_sources(
    tmp_path: Path, source_block: str, expected_msg: str
) -> None:
    res = _run_guard(tmp_path, pyproject=source_block)
    assert res.returncode != 0
    assert expected_msg in (res.stdout + res.stderr)


@pytest.mark.parametrize(
    ("lock_line", "expected_msg"),
    [
        (
            f'source = {{ git = "https://github.com/Neywa-Labs/mlx-vlm?rev={HEX40}#{HEX40}" }}\n',
            "external git dependency",
        ),
        (
            'source = { git = "https://github.com/example/mlx-vlm?branch=main#main" }\n',
            "external git dependency",
        ),
        (
            'source = { editable = "../mlx-vlm" }\n',
            "non-self editable source",
        ),
        (
            'source = { path = "../mlx-vlm" }\n',
            "local path dependency",
        ),
    ],
)
def test_blocks_git_or_local_uv_lock_sources(
    tmp_path: Path, lock_line: str, expected_msg: str
) -> None:
    res = _run_guard(tmp_path, uv_lock=lock_line)
    assert res.returncode != 0
    assert expected_msg in (res.stdout + res.stderr)


def test_override_env_allows_blocked_source(tmp_path: Path) -> None:
    pyproject = f'[tool.uv.sources]\nmlx-vlm = {{ git = "https://github.com/Neywa-Labs/mlx-vlm", rev = "{HEX40}" }}\n'
    res = _run_guard(
        tmp_path,
        pyproject=pyproject,
        extra_env={"HEYLOOK_ALLOW_CHANNEL_COMMIT": "1"},
    )
    assert res.returncode == 0, res.stderr
