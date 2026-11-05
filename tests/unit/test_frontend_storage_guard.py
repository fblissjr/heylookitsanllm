"""Enforce the unified browser storage contract (`frontend/js/storage.js`).

Every `localStorage`, `sessionStorage`, and `indexedDB` reference in `frontend/js/`
(excluding `frontend/js/vendor/`) must live inside `frontend/js/storage.js`.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_JS = REPO_ROOT / "frontend" / "js"
STORAGE_MODULE = FRONTEND_JS / "storage.js"
VENDOR_DIR = FRONTEND_JS / "vendor"

FORBIDDEN_STORAGE_RE = re.compile(r"\b(?:localStorage|sessionStorage|indexedDB)\b")


def test_only_storage_module_touches_browser_storage():
    assert STORAGE_MODULE.is_file(), f"Missing unified storage module: {STORAGE_MODULE}"
    offenders: list[str] = []
    for path in sorted(FRONTEND_JS.rglob("*.js")):
        if path == STORAGE_MODULE or VENDOR_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = FORBIDDEN_STORAGE_RE.search(line)
            if match:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{line_no}: found {match.group(0)!r}")
    assert not offenders, (
        "Browser storage globals (localStorage, sessionStorage, indexedDB) are only "
        "allowed in frontend/js/storage.js:\n  " + "\n  ".join(offenders)
    )


def test_storage_module_declares_registry_and_retired_db():
    src = STORAGE_MODULE.read_text(encoding="utf-8")
    assert "export const STORAGE_REGISTRY =" in src
    assert "export const RETIRED_DATABASES =" in src
    assert "'embed-kit:media-finder'" in src
    assert "'heylook'" in src
