#!/usr/bin/env python3
"""Pre-commit guard: the newest CHANGELOG heading must equal `__version__`.

Reads the STAGED blobs (`git show :path`), not the working tree, so it judges
what is being committed. It is stdlib-only and run directly, never under
`uv run`, because a pre-commit hook must not depend on the venv being synced.

Why it exists: `__version__` feeds `/v1/capabilities.server_version` and the
package metadata. It once sat seven releases behind the changelog, found only
by accident. A hookify reminder was meant to catch that; it never ran while
the plugin was disabled. A commit check cannot be skipped by a disabled
plugin, and it catches every editor and every session, not just one.

Fails CLOSED: if either blob cannot be read, the commit is refused with the
reason, instead of passing a check that looked at nothing.
"""
from __future__ import annotations

import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"
INIT = "src/heylook_llm/__init__.py"
HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.M)
VERSION = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)


def staged(path: str) -> str:
    proc = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"version-sync: cannot read staged {path}: {proc.stderr.strip()}")
    return proc.stdout


def main() -> int:
    heading = HEADING.search(staged(CHANGELOG))
    version = VERSION.search(staged(INIT))
    if not heading or not version:
        print(f"version-sync: could not find a `## [x.y.z]` heading in {CHANGELOG} "
              f"or `__version__` in {INIT}", file=sys.stderr)
        return 1
    if heading.group(1) != version.group(1):
        print(f"version-sync: {CHANGELOG} is at {heading.group(1)} but "
              f"{INIT} says {version.group(1)}. Bump __version__ in the same "
              f"commit as the changelog entry (the heading is the source of "
              f"truth).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
