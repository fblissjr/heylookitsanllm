"""Pydantic `Field` defaults must use the keyword form, repo-wide.

House rule since the 2026-07-20 sweep. This pyright build recognises only
`Field(default=None, ...)`. A positional default (`Field(None, ...)`) makes
every constructor of that model report false "arguments missing" errors. The
pattern crept back once, copied from a pre-sweep sibling class, and was caught
only by a multi-agent review. This scan replaces the hookify reminder that was
meant to catch it and never ran while the plugin was disabled.

`Field(..., description=...)` is fine: the positional ellipsis is pydantic's
required-field marker, not a default.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "heylook_llm"


def _is_field_call(node: ast.Call) -> bool:
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "Field") or (
        isinstance(f, ast.Attribute) and f.attr == "Field")


def _positional_defaults() -> list[str]:
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_field_call(node) and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value is Ellipsis:
                    continue
                hits.append(f"{path.relative_to(SRC.parents[1])}:{node.lineno}")
    return hits


def test_no_positional_field_defaults():
    # The scan must actually find Field calls, or a pass is vacuous.
    count = 0
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count += sum(1 for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and _is_field_call(n))
    assert count > 0, f"no Field() calls found under {SRC}; the scan looked at nothing"

    hits = _positional_defaults()
    assert not hits, (
        "Field() with a POSITIONAL default; use Field(default=...) instead:\n  "
        + "\n  ".join(hits))
