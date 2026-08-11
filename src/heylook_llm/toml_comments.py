# src/heylook_llm/toml_comments.py
"""Carry comments forward across ``tomli_w`` rewrites of models.toml.

``model_service._write_toml`` regenerates the whole file through ``tomli_w``,
which emits no comments -- so before this module, any admin write silently
dropped every comment in the file. The fix deliberately does NOT make the old
file's formatting authoritative:

- Values, key order, quoting, and layout come from the fresh ``tomli_w``
  render of the data. Only COMMENTS are copied over, as injected lines.
- A comment is carried only while its anchor is unchanged, so a note can
  never outlive what it describes:

  * an inline comment on a top-level key, or a full-line block above one,
    carries iff that key's rendered value is identical;
  * every comment inside a ``[[models]]`` entry carries iff that model's
    values render byte-identically through ``tomli_w`` (normalized, so the
    old file's hand-formatting doesn't matter);
  * a block at the very end of a model's section sits visually above the
    NEXT model's header, so it additionally requires that following model
    to be unchanged and still immediately next;
  * the block above the first ``[[models]]`` header requires the first
    model to be unchanged and still first.

- Merging is best-effort and can never block or corrupt a write: on any
  parse failure, missing anchor, or a merged text that no longer parses to
  exactly the same values as the fresh render, the fresh (comment-less)
  render is written instead.

Why line injection instead of grafting comments into a tomlkit document:
mutating ANY item of a tomlkit-parsed array-of-tables (even just its
comment trivia) makes ``tomlkit.dumps`` re-render the AoT as an inline
array -- malformed for nested tables. That is the failure mode that sank
the previous whole-table-splice attempt, so tomlkit is used strictly
read-only here, for comment EXTRACTION.
"""

from __future__ import annotations

import logging
import re
import tomllib

import tomli_w
import tomlkit
from tomlkit.items import AoT, Comment, Table, Whitespace

logger = logging.getLogger(__name__)

_MODEL_HEADER = "[[models]]"
_SUB_HEADER_RE = re.compile(r"^\[models\.([^\]]+)\]$")


def merge_comments(old_text: str, new_text: str) -> str:
    """Return ``new_text`` with the still-valid comments of ``old_text``.

    Never raises; falls back to ``new_text`` unchanged whenever the merge
    can't be done safely.
    """
    if "#" not in old_text:
        return new_text
    try:
        merged = _merge(old_text, new_text)
    except Exception as e:
        logger.warning(f"Comment carry-forward failed, writing without comments: {e}")
        return new_text
    if merged == new_text:
        return new_text
    try:
        if tomllib.loads(merged) != tomllib.loads(new_text):
            logger.warning(
                "Comment carry-forward changed parsed values; writing without comments"
            )
            return new_text
    except Exception as e:
        logger.warning(f"Comment carry-forward produced unparseable TOML ({e}); writing without comments")
        return new_text
    return merged


# --- extraction (tomlkit, read-only) ---


def _extract_table(body, path: tuple, records: list) -> tuple:
    """Collect comment records from one table body; returns the path of the
    deepest-last sub-table (where tomlkit attributes a model's trailing
    comments, including blocks that sit above the NEXT model's header)."""
    pending: list[str] = []
    last_path = path
    for key, item in body:
        if isinstance(item, Whitespace):
            continue
        if isinstance(item, Comment):
            pending.append(item.trivia.comment)
            continue
        k = key.key
        if pending:
            records.append(("before", path, k, pending))
            pending = []
        if isinstance(item, Table):
            if item.trivia.comment:
                records.append(
                    ("header_inline", path + (k,), item.trivia.comment, item.trivia.comment_ws)
                )
            last_path = _extract_table(item.value.body, path + (k,), records)
        else:
            trivia = getattr(item, "trivia", None)
            if trivia is not None and trivia.comment:
                records.append(("inline", path, k, trivia.comment, trivia.comment_ws))
    if pending:
        records.append(("trailing", path, pending))
    return last_path


def _extract(doc) -> tuple[list, list[str] | None, list[dict]]:
    """Walk the old document. Returns (root_records, block above the first
    [[models]] header or None, per-model infos in file order)."""
    root_records: list = []
    before_first: list[str] | None = None
    model_infos: list[dict] = []
    pending: list[str] = []
    for key, item in doc.body:
        if isinstance(item, Whitespace):
            continue
        if isinstance(item, Comment):
            pending.append(item.trivia.comment)
            continue
        k = key.key
        if isinstance(item, AoT) and k == "models":
            if pending:
                before_first = pending
                pending = []
            for t in item:
                records: list = []
                if t.trivia.comment:
                    records.append(("header_inline", (), t.trivia.comment, t.trivia.comment_ws))
                last_path = _extract_table(t.value.body, (), records)
                mid = t.get("id")
                model_infos.append(
                    {
                        "id": str(mid) if mid is not None else None,
                        "records": records,
                        "last_path": last_path,
                    }
                )
        else:
            if pending:
                root_records.append(("before", k, pending))
                pending = []
            trivia = getattr(item, "trivia", None)
            if trivia is not None and trivia.comment:
                root_records.append(("inline", k, trivia.comment, trivia.comment_ws))
            # Root-level plain tables other than [[models]] don't exist in
            # models.toml; their inner comments are not carried.
    return root_records, before_first, model_infos


# --- fresh-render indexing (plain text from tomli_w) ---


def _index_fresh(lines: list[str]) -> tuple[int, list[dict]]:
    """Map the tomli_w render: root span end + per-model header/section spans.

    tomli_w emits headers at column 0, one line per key (multiline arrays
    keep ``key = [`` on the key line), so line-level scanning is exact.
    """
    root_end = len(lines)
    models: list[dict] = []
    cur: dict | None = None
    cur_path: tuple | None = None

    def close_section(idx: int) -> None:
        if cur is not None and cur_path is not None:
            cur["sections"][cur_path][1] = idx

    for i, ln in enumerate(lines):
        if ln == _MODEL_HEADER:
            close_section(i)
            if cur is None:
                root_end = min(root_end, i)
            cur = {"header": i, "sections": {(): [i + 1, len(lines)]}}
            cur_path = ()
            models.append(cur)
        elif (m := _SUB_HEADER_RE.match(ln)) and cur is not None:
            close_section(i)
            cur_path = tuple(m.group(1).split("."))
            cur["sections"][cur_path] = [i + 1, len(lines)]
    return root_end, models


def _find_key_line(lines: list[str], start: int, end: int, key: str) -> int | None:
    for candidate in (f"{key} = ", f'"{key}" = '):
        for i in range(start, end):
            if lines[i].startswith(candidate):
                return i
    return None


def _comment_lines(block: list[str]) -> list[str]:
    return [c if c.startswith("#") else f"# {c}" for c in block]


def _inline_suffix(comment: str, ws: str) -> str:
    return (ws if ws.strip() == "" and ws else "  ") + comment


# --- merge ---


def _merge(old_text: str, new_text: str) -> str:
    old_data = tomllib.loads(old_text)
    new_data = tomllib.loads(new_text)
    root_records, before_first, model_infos = _extract(tomlkit.parse(old_text))

    old_models = {m.get("id"): m for m in old_data.get("models", [])}
    new_models_list = new_data.get("models", [])
    new_ids = [m.get("id") for m in new_models_list]

    def render_model(m: dict) -> str:
        return tomli_w.dumps({"models": [m]})

    unchanged = {
        mid
        for mid, m in zip(new_ids, new_models_list)
        if mid is not None
        and mid in old_models
        and render_model(old_models[mid]) == render_model(m)
    }

    lines = new_text.splitlines()
    root_end, fresh_models = _index_fresh(lines)
    if len(fresh_models) != len(new_ids):
        raise ValueError("fresh render section count does not match its model count")
    fresh_by_id = {mid: (j, fresh_models[j]) for j, mid in enumerate(new_ids) if mid is not None}

    inserts: dict[int, list[str]] = {}
    appends: dict[int, str] = {}

    def insert_at(idx: int, block: list[str]) -> None:
        inserts.setdefault(idx, []).extend(_comment_lines(block))

    def root_key_unchanged(k: str) -> bool:
        if k not in old_data or k not in new_data:
            return False
        try:
            return tomli_w.dumps({k: old_data[k]}) == tomli_w.dumps({k: new_data[k]})
        except Exception:
            return False

    for rec in root_records:
        if rec[0] == "inline":
            _, k, comment, ws = rec
            if root_key_unchanged(k) and (i := _find_key_line(lines, 0, root_end, k)) is not None:
                appends[i] = _inline_suffix(comment, ws)
        elif rec[0] == "before":
            _, k, block = rec
            if root_key_unchanged(k) and (i := _find_key_line(lines, 0, root_end, k)) is not None:
                insert_at(i, block)

    old_order = [info["id"] for info in model_infos]
    if before_first and old_order and new_ids:
        if old_order[0] == new_ids[0] and old_order[0] in unchanged and fresh_models:
            insert_at(fresh_models[0]["header"], before_first)

    for pos, info in enumerate(model_infos):
        mid = info["id"]
        if mid not in unchanged or mid not in fresh_by_id:
            continue
        j, span = fresh_by_id[mid]
        sections = span["sections"]
        for rec in info["records"]:
            if rec[0] == "header_inline":
                _, path, comment, ws = rec
                if path == ():
                    appends[span["header"]] = _inline_suffix(comment, ws)
                elif path in sections:
                    appends[sections[path][0] - 1] = _inline_suffix(comment, ws)
            elif rec[0] == "inline":
                _, path, k, comment, ws = rec
                if path in sections:
                    start, end = sections[path]
                    if (i := _find_key_line(lines, start, end, k)) is not None:
                        appends[i] = _inline_suffix(comment, ws)
            elif rec[0] == "before":
                _, path, k, block = rec
                if path + (k,) in sections:
                    insert_at(sections[path + (k,)][0] - 1, block)
                elif path in sections:
                    start, end = sections[path]
                    if (i := _find_key_line(lines, start, end, k)) is not None:
                        insert_at(i, block)
            elif rec[0] == "trailing":
                _, path, block = rec
                if path == info["last_path"]:
                    # Sits above the NEXT model's header: that model must be
                    # unchanged too, and still immediately next.
                    old_next = old_order[pos + 1] if pos + 1 < len(old_order) else None
                    fresh_next = new_ids[j + 1] if j + 1 < len(new_ids) else None
                    if old_next != fresh_next:
                        continue
                    if old_next is not None and old_next not in unchanged:
                        continue
                if path in sections:
                    insert_at(sections[path][1], block)

    if not inserts and not appends:
        return new_text
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i in inserts:
            out.extend(inserts[i])
        out.append(ln + appends.get(i, ""))
    if len(lines) in inserts:
        out.extend(inserts[len(lines)])
    return "\n".join(out) + "\n"
