#!/usr/bin/env python3
"""Claude Code PreToolUse hook (Bash): refuse broad git staging.

Wired in `.claude/settings.json` (matcher Bash). Parallel sessions share this
checkout and often hold uncommitted work (AGENTS.md "Repo rules"), so a
command that stages everything sweeps another session's files into your
commit. This refuses `git add -A/--all/-u/--update/./:/` and `git commit -a/
--all` (which stages every tracked change), and exits 2 with the reason, which
Claude Code feeds back to the model. Stage files by name instead.

Heredoc bodies are skipped, so a commit message that mentions these commands
is not a false positive. Anything it cannot parse passes: this is a guard
against a habit, not a sandbox.
"""
from __future__ import annotations

import json
import re
import shlex
import sys

BROAD_ADD_ARGS = {"-A", "--all", "-u", "--update", ".", ":/", ":(top)", "*"}
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)
_SEGMENT = re.compile(r"&&|\|\||[;|\n]")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _short_flags(arg: str) -> str:
    """The letters of a short-option cluster (`-am` -> `am`), else empty."""
    return arg[1:] if arg.startswith("-") and not arg.startswith("--") else ""


def refusal(command: str) -> str | None:
    """The reason to refuse `command`, or None when it may run."""
    for segment in _SEGMENT.split(_HEREDOC.sub("", command)):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        while tokens and _ENV_ASSIGN.match(tokens[0]):
            tokens.pop(0)
        if not tokens or tokens[0] != "git":
            continue
        i = 1
        while i < len(tokens) and tokens[i].startswith("-"):
            i += 2 if tokens[i] in ("-c", "-C") else 1
        if i >= len(tokens):
            continue
        sub, args = tokens[i], tokens[i + 1:]
        if sub == "add":
            for arg in args:
                if arg in BROAD_ADD_ARGS or set(_short_flags(arg)) & {"A", "u"}:
                    return f"`git add {arg}` stages every change in the checkout"
        elif sub == "commit":
            for arg in args:
                if arg == "--all" or "a" in _short_flags(arg):
                    return f"`git commit {arg}` stages every tracked change"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    reason = refusal(command)
    if reason is None:
        return 0
    print(
        f"Blocked: {reason}. Other sessions may have uncommitted work here; "
        "stage files by name (`git add path/one path/two`) and check "
        "`git status` first.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
