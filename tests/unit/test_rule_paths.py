"""Every path pattern in a .claude/rules/ file matches something in the tree.

Claude Code loads a path-scoped rule only when a file matching one of its
patterns is read. A pattern naming a renamed or deleted file matches nothing,
and its rules stop loading with no error. This makes that a failing suite.
Why: AGENTS.md "Rules by area".
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RULES = sorted((ROOT / ".claude" / "rules").glob("*.md"))


def _patterns(rule: Path) -> list[str]:
    front = rule.read_text().split("---")[1]
    return re.findall(r'^\s*-\s*"([^"]+)"', front, re.M)


def _expand(pattern: str) -> list[str]:
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end():]
    return [x for alt in m.group(1).split(",") for x in _expand(head + alt + tail)]


def _ignored(path: str) -> bool:
    # A gitignored local file (models.toml) is absent from a fresh clone by design.
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def test_rule_files_exist():
    assert RULES, "no .claude/rules/*.md found; AGENTS.md indexes them"


@pytest.mark.parametrize("rule", RULES, ids=lambda p: p.name)
def test_every_pattern_matches_a_file(rule):
    patterns = _patterns(rule)
    assert patterns, f"{rule.name}: no paths: frontmatter, so it loads every session"
    dead = [e for p in patterns for e in _expand(p)
            if not any(ROOT.glob(e)) and not _ignored(e)]
    assert not dead, f"{rule.name}: patterns that match nothing: {dead}"
