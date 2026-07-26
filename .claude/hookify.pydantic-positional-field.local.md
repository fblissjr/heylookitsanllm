---
name: pydantic-positional-field
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: src/.*\.py$
  - field: new_text
    operator: regex_match
    pattern: 'Field\(\s*(None|True|False|[0-9]|"|''|\[)'
---

This edit contains a pydantic `Field(...)` with a POSITIONAL default
(`Field(None, ...)`, `Field(0, ...)`, etc.). House rule (CLAUDE.md,
repo-wide sweep 2026-07-20): defaults MUST use the keyword form --
`Field(default=None, ...)` -- because this pyright build only recognizes
the keyword form; positional defaults make every constructor of that model
flag false "arguments missing" errors.

NOT affected: `Field(..., description=...)` -- the positional ellipsis is
the standard required-field marker and is fine.

Fix the default to keyword form before moving on. (This exact anti-pattern
was reintroduced once already -- copied from a pre-sweep sibling class --
and caught only by a five-agent code review on 2026-07-26.)
