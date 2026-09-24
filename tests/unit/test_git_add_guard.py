"""The PreToolUse guard refuses broad git staging and lets named staging through.

Why: AGENTS.md "Repo rules" (parallel sessions share this checkout, so a
broad `git add` sweeps another session's uncommitted files into a commit).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "git_add_guard.py"
_spec = importlib.util.spec_from_file_location("git_add_guard", SCRIPT)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

BLOCKED = [
    "git add -A",
    "git add --all",
    "git add -u",
    "git add .",
    "git add :/",
    "cd repo && git add -A && git commit -m x",
    "git -c commit.gpgsign=false commit -am 'msg'",
    "git commit --all -m x",
    "FOO=1 git add -u src/",
]
ALLOWED = [
    "git add src/a.py tests/test_a.py",
    "git add -p src/a.py",
    "git -c commit.gpgsign=false commit -q -m 'add all the things'",
    "git commit --amend --no-edit",
    "git status --short",
    "echo 'git add -A'",
    "git commit -q -F - <<'EOF'\nnever run git add -A here\nEOF",
]


@pytest.mark.parametrize("command", BLOCKED)
def test_broad_staging_is_refused(command):
    assert guard.refusal(command) is not None


@pytest.mark.parametrize("command", ALLOWED)
def test_named_staging_and_other_commands_pass(command):
    assert guard.refusal(command) is None


def test_the_hook_exits_2_through_its_real_entry_point():
    payload = json.dumps({"tool_input": {"command": "git add -A"}})
    r = subprocess.run([sys.executable, str(SCRIPT)], input=payload,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "stage files by name" in r.stderr
