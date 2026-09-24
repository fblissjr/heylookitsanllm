#!/usr/bin/env python3
"""Claude Code PostToolUse hook: validate heylook.toml right after an edit.

Wired in `.claude/settings.json` (matcher Edit|Write|MultiEdit). Reads the
hook payload on stdin and ignores any file that is not a `heylook.toml`. It
then runs the SAME validation the server runs at startup. On failure it exits
2 with the error on stderr, which Claude Code feeds back to the model. A typo'd
key is therefore caught at edit time, not at the next server start.
(`extra="forbid"` rejects unknown keys, including fields removed in past
releases.)

It replaced a hookify reminder that only told the model to run this command
itself. That reminder silently stopped firing when the plugin was disabled.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # not our payload shape; never block on our own confusion
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""
    if Path(file_path).name != "heylook.toml":
        return 0
    path = Path(file_path)
    if not path.is_file():
        return 0
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        print(f"heylook.toml no longer parses as TOML: {exc}", file=sys.stderr)
        return 2
    try:
        from heylook_llm.config import AppConfig
        cfg = AppConfig(**data)
    except ImportError as exc:
        # The venv is not synced. That is not a heylook.toml problem, so do not
        # block the edit over it.
        print(f"heylook.toml hook skipped: {exc}", file=sys.stderr)
        return 0
    except Exception as exc:  # pydantic ValidationError and friends
        print(
            "heylook.toml fails the validation the server runs at startup, so "
            "the server would refuse to start. Fix the entry before moving on.\n"
            f"{exc}",
            file=sys.stderr,
        )
        return 2
    # Success is silent on purpose: a hook that talks on every edit trains
    # its reader to skim it.
    _ = cfg
    return 0


if __name__ == "__main__":
    sys.exit(main())
