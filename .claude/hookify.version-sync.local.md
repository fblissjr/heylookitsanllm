---
name: version-sync
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: CHANGELOG\.md$
---

You just edited CHANGELOG.md. Check version sync before moving on:

1. Read the newest `## [x.y.z]` heading at the top of CHANGELOG.md.
2. Compare it to `__version__` in `src/heylook_llm/__init__.py`.
3. If they differ, update `__init__.py` to match in the SAME change/commit.

This drifted silently before (`__version__` sat 7 releases behind the changelog and was only found by accident during a docs reconciliation). The changelog heading is the source of truth; `__init__.py` follows it. No other files carry the version.
