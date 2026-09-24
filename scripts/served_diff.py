#!/usr/bin/env python3
"""What would a models.toml edit do to the served set? Ask the server's merge.

A dry run over `heylook_llm.model_registry.served_diff`, which merges and
validates each side with the router's own code (`served`), so this cannot
disagree with what the server would serve. models.toml is gitignored: an edit
to it has no history to recover from, and every past surprise here came from
predicting the merge instead of running it (plan_registry_sidecars.md,
Phase 0).

Usage::

    uv run python scripts/served_diff.py --prune              # each entry, deleted alone
    uv run python scripts/served_diff.py --against cand.toml  # current vs a candidate
    uv run python scripts/served_diff.py --prune --json

``--prune`` is the inventory: for every ``[[models]]`` entry it reports what
deleting that entry alone would change. "none" means the entry is redundant
with discovery; anything else is what the entry is holding in place.

Read-only: it never writes models.toml. It scans once per distinct
``[scan]`` section. Exit 0 always; a scan that failed is printed as
UNRELIABLE, since a lost model under it may only be unread.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heylook_llm.model_registry import ServedDiff, scan, served_diff  # noqa: E402


def _load(path: str) -> dict:
    with open(Path(path).expanduser(), "rb") as f:
        return tomllib.load(f)


def _describe(d: ServedDiff) -> str:
    if d.empty:
        return "none"
    parts = []
    if d.lost:
        parts.append("LOSES " + ", ".join(d.lost))
    if d.gained:
        parts.append("gains " + ", ".join(d.gained))
    for old, new in d.renamed.items():
        parts.append(f"renames {old} -> {new}")
    for mid, fields in d.changed.items():
        parts.append(f"changes {mid}: " + "; ".join(
            f"{k} {a!r} -> {b!r}" for k, (a, b) in fields.items()))
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="What would a models.toml edit do to the served set? Read-only.")
    ap.add_argument("--config", default="models.toml", help="the current config (default models.toml)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prune", action="store_true", help="each entry, deleted alone")
    mode.add_argument("--against", metavar="TOML", help="a candidate config to compare with")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    current = _load(args.config)
    found = scan(current)

    if args.against:
        candidate = _load(args.against)
        same_scan = candidate.get("scan") == current.get("scan")
        d = served_diff((current, found), (candidate, found if same_scan else scan(candidate)))
        if args.json:
            print(json.dumps(dataclasses.asdict(d), indent=1))
        else:
            print(_describe(d))
            if d.unreliable:
                print("UNRELIABLE: these sources failed to scan:", ", ".join(d.unreliable))
        return 0

    rows = []
    entries = current.get("models") or []
    for k, e in enumerate(entries):
        without = dict(current, models=entries[:k] + entries[k + 1:])
        rows.append((str(e.get("id")), served_diff((current, found), (without, found))))
    if args.json:
        print(json.dumps({mid: dataclasses.asdict(d) for mid, d in rows}, indent=1))
        return 0
    width = max((len(mid) for mid, _ in rows), default=0)
    for mid, d in rows:
        print(f"{mid:<{width}}  {_describe(d)}")
    if found.failed:
        print("UNRELIABLE: these sources failed to scan:", ", ".join(found.failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
