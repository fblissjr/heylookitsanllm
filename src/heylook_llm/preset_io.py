# src/heylook_llm/preset_io.py
"""Portable preset import/export (JSON & Markdown) and before/after diffing.

Shared by the `heylook` CLI (`heylook_llm.cli`) and mirrored by
`frontend/js/preset-io.js` so files exported from either surface import cleanly
on the other.

Formats
-------
- Single JSON: ``{"name": "...", "system_prompt": "..." | null, "params": {...}}``
- Bulk JSON:   ``{"presets": [ <single>, ... ]}`` (also accepts a bare JSON list)
- Markdown:    One preset per ``.md`` file using YAML-style frontmatter:

      ---
      name: Pirate
      params:
        temperature: 1.2
        enable_thinking: true
      ---

      Talk like a pirate.

  Empty/omitted body means ``system_prompt = None`` (settings-only preset).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
from pathlib import Path
import re
from typing import Any

from pydantic import TypeAdapter, ValidationError

from heylook_llm.schema.sampler_params import SamplerParams

_SAMPLER_ADAPTER = TypeAdapter(SamplerParams)

# Characters that require double-quoting a YAML scalar so both our stdlib
# parser and standard YAML parsers read it back identically.
_YAML_SPECIAL_START = set("-?:,[]{}#&*!|>'\"%@`")
_YAML_BOOL_OR_NULL = frozenset({"true", "false", "null", "yes", "no", "on", "off", "~"})
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")
_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_DASH_RE = re.compile(r"-{2,}")


@dataclass(frozen=True)
class PresetRecord:
    """Normalized portable representation of a preset."""

    name: str
    system_prompt: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ParamDiff:
    key: str
    before: Any  # _MISSING when added
    after: Any   # _MISSING when removed
    kind: str    # "added" | "removed" | "changed"


@dataclass(frozen=True)
class ImportItem:
    incoming: PresetRecord
    existing: dict[str, Any] | None  # stored preset dict when name matches
    changed: bool
    param_diffs: tuple[ParamDiff, ...] = ()
    prompt_diff: tuple[str, ...] = ()
    prompt_changed: bool = False


@dataclass(frozen=True)
class ImportPlan:
    items: tuple[ImportItem, ...]

    @property
    def new_items(self) -> list[ImportItem]:
        return [it for it in self.items if it.existing is None]

    @property
    def existing_items(self) -> list[ImportItem]:
        return [it for it in self.items if it.existing is not None]

    @property
    def changed_existing_items(self) -> list[ImportItem]:
        return [it for it in self.items if it.existing is not None and it.changed]

    @property
    def unchanged_existing_items(self) -> list[ImportItem]:
        return [it for it in self.items if it.existing is not None and not it.changed]


class PresetFormatError(ValueError):
    """Raised when an imported preset file cannot be parsed or validated."""


def normalize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PresetFormatError("Preset 'system_prompt' must be a string or null")
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned if cleaned.strip() else None


def normalize_params(value: Any, *, validate: bool = True) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PresetFormatError("Preset 'params' must be an object")
    filtered = {k: v for k, v in value.items() if v is not None}
    if not validate:
        return filtered
    try:
        return _SAMPLER_ADAPTER.validate_python(filtered)
    except ValidationError as exc:
        msgs = "; ".join(e.get("msg", str(e)) for e in exc.errors())
        raise PresetFormatError(f"Invalid preset params: {msgs}") from None


def normalize_preset(raw: dict[str, Any], *, validate_params: bool = True) -> PresetRecord:
    if not isinstance(raw, dict):
        raise PresetFormatError("Preset entry must be a JSON object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PresetFormatError("Preset 'name' must be a non-empty string")
    prompt = normalize_prompt(raw.get("system_prompt"))
    params = normalize_params(raw.get("params"), validate=validate_params)
    return PresetRecord(name=name.strip(), system_prompt=prompt, params=params)


def slugify_preset_name(name: str) -> str:
    """Safe filename stem derived from a preset name."""
    slug = _SLUG_UNSAFE_RE.sub("-", name.strip())
    slug = _MULTI_DASH_RE.sub("-", slug).strip("-.")
    return slug or "preset"


def unique_markdown_filenames(presets: list[PresetRecord] | list[dict[str, Any]]) -> list[str]:
    """Assign collision-free `<slug>.md` filenames for bulk Markdown export."""
    used: set[str] = set()
    names: list[str] = []
    for preset in presets:
        raw_name = preset.name if isinstance(preset, PresetRecord) else str(preset["name"])
        base = slugify_preset_name(raw_name)
        candidate = f"{base}.md"
        key = candidate.lower()
        if key in used:
            n = 2
            while f"{base}-{n}.md".lower() in used:
                n += 1
            candidate = f"{base}-{n}.md"
            key = candidate.lower()
        used.add(key)
        names.append(candidate)
    return names


# ---------------------------------------------------------------------------
# JSON serialization / parsing
# ---------------------------------------------------------------------------


def serialize_preset_json(preset: PresetRecord | dict[str, Any]) -> str:
    rec = preset if isinstance(preset, PresetRecord) else normalize_preset(preset, validate_params=False)
    return json.dumps(rec.to_export_dict(), indent=2, ensure_ascii=False) + "\n"


def serialize_presets_json(presets: list[PresetRecord] | list[dict[str, Any]]) -> str:
    entries = [
        (p if isinstance(p, PresetRecord) else normalize_preset(p, validate_params=False)).to_export_dict()
        for p in presets
    ]
    return json.dumps({"presets": entries}, indent=2, ensure_ascii=False) + "\n"


def parse_presets_json(text: str, *, validate_params: bool = True) -> list[PresetRecord]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PresetFormatError(f"Invalid JSON: {exc}") from None

    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        if "presets" in data and isinstance(data["presets"], list):
            raw_list = data["presets"]
        elif "name" in data:
            raw_list = [data]
        else:
            raise PresetFormatError("JSON must be a preset object with 'name', a list, or {'presets': [...]}")
    else:
        raise PresetFormatError("JSON root must be an object or array")

    return [normalize_preset(item, validate_params=validate_params) for item in raw_list]


# ---------------------------------------------------------------------------
# Markdown serialization / parsing (YAML-style frontmatter + prompt body)
# ---------------------------------------------------------------------------


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quote = (
        not text
        or text != text.strip()
        or text[0] in _YAML_SPECIAL_START
        or ": " in text
        or text.endswith(":")
        or " #" in text
        or "\n" in text
        or "\r" in text
        or "\t" in text
        or '"' in text
        or "\\" in text
        or text.lower() in _YAML_BOOL_OR_NULL
        or bool(_INT_RE.match(text))
        or bool(_FLOAT_RE.match(text))
    )
    return json.dumps(text, ensure_ascii=False) if needs_quote else text


def _parse_yaml_scalar(raw: str, *, as_string: bool = False) -> Any:
    s = raw.strip()
    if not s:
        return "" if as_string else None
    if s.startswith('"'):
        try:
            val = json.loads(s)
        except json.JSONDecodeError as exc:
            raise PresetFormatError(f"Invalid quoted string in frontmatter: {s!r} ({exc})") from None
        return str(val) if as_string else val
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        val = s[1:-1].replace("''", "'")
        return val
    if as_string:
        return s
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return s


def serialize_preset_markdown(preset: PresetRecord | dict[str, Any]) -> str:
    rec = preset if isinstance(preset, PresetRecord) else normalize_preset(preset, validate_params=False)
    lines = ["---", f"name: {_format_yaml_scalar(rec.name)}"]
    if rec.params:
        lines.append("params:")
        for key in sorted(rec.params.keys()):
            val = rec.params[key]
            if val is not None:
                lines.append(f"  {key}: {_format_yaml_scalar(val)}")
    lines.append("---")
    out = "\n".join(lines) + "\n"
    if rec.system_prompt:
        body = rec.system_prompt
        out += "\n" + body
        if not body.endswith("\n"):
            out += "\n"
    return out


def parse_preset_markdown(
    text: str,
    *,
    fallback_name: str | None = None,
    validate_params: bool = True,
) -> PresetRecord:
    cleaned = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = cleaned.split("\n")

    name: str | None = None
    params: dict[str, Any] = {}
    body_text = cleaned

    if lines and lines[0].strip() == "---":
        end_idx: int | None = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            raise PresetFormatError("Unclosed frontmatter block (missing closing '---')")

        fm_lines = lines[1:end_idx]
        in_params = False
        for line in fm_lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 0:
                in_params = False
                if ":" not in line:
                    raise PresetFormatError(f"Malformed frontmatter line: {line!r}")
                key, rest = line.split(":", 1)
                key = key.strip()
                rest_s = rest.strip()
                if key == "name":
                    name = _parse_yaml_scalar(rest_s, as_string=True)
                elif key == "params":
                    if not rest_s:
                        in_params = True
                    elif rest_s.startswith("{"):
                        try:
                            parsed = json.loads(rest_s)
                        except json.JSONDecodeError as exc:
                            raise PresetFormatError(f"Invalid inline params JSON: {exc}") from None
                        if not isinstance(parsed, dict):
                            raise PresetFormatError("Frontmatter 'params' must be an object")
                        params.update(parsed)
                    else:
                        raise PresetFormatError(f"Unexpected value for 'params': {rest_s!r}")
                # Ignore optional metadata fields like id/created_at/updated_at
            elif in_params:
                if ":" not in line:
                    raise PresetFormatError(f"Malformed param line: {line!r}")
                p_key, p_rest = line.strip().split(":", 1)
                p_key = p_key.strip()
                val = _parse_yaml_scalar(p_rest.strip())
                if val is not None:
                    params[p_key] = val
            else:
                raise PresetFormatError(f"Unexpected indented frontmatter line: {line!r}")

        rest_lines = lines[end_idx + 1:]
        # Strip the single separator blank line written after '---' and the POSIX EOF newline.
        if rest_lines and rest_lines[0] == "":
            rest_lines = rest_lines[1:]
        if rest_lines and rest_lines[-1] == "":
            rest_lines = rest_lines[:-1]
        body_text = "\n".join(rest_lines)
    else:
        # Plain markdown prompt file with no frontmatter: derive name from filename.
        body_text = cleaned.rstrip("\n")

    resolved_name = (name or fallback_name or "").strip()
    if not resolved_name:
        raise PresetFormatError("Markdown preset is missing 'name' in frontmatter")

    return normalize_preset(
        {"name": resolved_name, "system_prompt": body_text or None, "params": params},
        validate_params=validate_params,
    )


def load_presets_from_paths(
    paths: list[Path],
    *,
    validate_params: bool = True,
) -> list[PresetRecord]:
    """Load one or more presets from JSON/Markdown files or directories."""
    files: list[Path] = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            raise PresetFormatError(f"Path not found: {p}")
        if p.is_dir():
            found = sorted(
                child for child in p.iterdir()
                if child.is_file() and child.suffix.lower() in (".md", ".markdown", ".json")
            )
            if not found:
                raise PresetFormatError(f"No .json or .md preset files found in directory: {p}")
            files.extend(found)
        else:
            files.append(p)

    records: list[PresetRecord] = []
    seen_names: dict[str, Path] = {}
    for f in files:
        suffix = f.suffix.lower()
        text = f.read_text(encoding="utf-8")
        try:
            if suffix == ".json":
                batch = parse_presets_json(text, validate_params=validate_params)
            elif suffix in (".md", ".markdown"):
                batch = [parse_preset_markdown(text, fallback_name=f.stem, validate_params=validate_params)]
            else:
                # Sniff content if extension is non-standard
                stripped = text.lstrip("\ufeff \t\r\n")
                if stripped.startswith(("{", "[")):
                    batch = parse_presets_json(text, validate_params=validate_params)
                else:
                    batch = [parse_preset_markdown(text, fallback_name=f.stem, validate_params=validate_params)]
        except PresetFormatError as exc:
            raise PresetFormatError(f"{f}: {exc}") from None

        for rec in batch:
            if rec.name in seen_names:
                raise PresetFormatError(
                    f"Duplicate preset name {rec.name!r} in import input "
                    f"({seen_names[rec.name].name} and {f.name})"
                )
            seen_names[rec.name] = f
            records.append(rec)

    if not records:
        raise PresetFormatError("No presets found in input")
    return records


# ---------------------------------------------------------------------------
# Diffing & Import Planning
# ---------------------------------------------------------------------------


def _param_values_equal(a: Any, b: Any) -> bool:
    # In Python, True == 1 and False == 0; treat bool as distinct from int/float.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def diff_preset(existing: dict[str, Any], incoming: PresetRecord) -> tuple[bool, tuple[ParamDiff, ...], bool, tuple[str, ...]]:
    """Compute param and system_prompt diffs between stored and incoming presets."""
    old_params = {k: v for k, v in (existing.get("params") or {}).items() if v is not None}
    new_params = {k: v for k, v in incoming.params.items() if v is not None}

    param_diffs: list[ParamDiff] = []
    for key in sorted(set(old_params) | set(new_params)):
        in_old = key in old_params
        in_new = key in new_params
        if not in_old and in_new:
            param_diffs.append(ParamDiff(key=key, before=None, after=new_params[key], kind="added"))
        elif in_old and not in_new:
            param_diffs.append(ParamDiff(key=key, before=old_params[key], after=None, kind="removed"))
        elif not _param_values_equal(old_params[key], new_params[key]):
            param_diffs.append(ParamDiff(key=key, before=old_params[key], after=new_params[key], kind="changed"))

    old_prompt = normalize_prompt(existing.get("system_prompt"))
    new_prompt = incoming.system_prompt
    prompt_changed = old_prompt != new_prompt
    prompt_diff: list[str] = []
    if prompt_changed:
        old_lines = (old_prompt or "").splitlines()
        new_lines = (new_prompt or "").splitlines()
        prompt_diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="before (stored)",
                tofile="after (incoming)",
                lineterm="",
                n=3,
            )
        )

    changed = bool(param_diffs or prompt_changed)
    return changed, tuple(param_diffs), prompt_changed, tuple(prompt_diff)


def plan_import(incoming: list[PresetRecord], existing_presets: list[dict[str, Any]]) -> ImportPlan:
    by_name = {p["name"]: p for p in existing_presets}
    items: list[ImportItem] = []
    for rec in incoming:
        cur = by_name.get(rec.name)
        if cur is None:
            items.append(ImportItem(incoming=rec, existing=None, changed=True))
        else:
            changed, p_diffs, prompt_changed, prompt_diff = diff_preset(cur, rec)
            items.append(
                ImportItem(
                    incoming=rec,
                    existing=cur,
                    changed=changed,
                    param_diffs=p_diffs,
                    prompt_diff=prompt_diff,
                    prompt_changed=prompt_changed,
                )
            )
    return ImportPlan(items=tuple(items))


def format_item_diff(item: ImportItem) -> str:
    """Human-readable before/after diff for one existing preset."""
    lines: list[str] = [f"=== {item.incoming.name} (existing -> incoming) ==="]
    if not item.changed:
        lines.append("  (no differences)")
        return "\n".join(lines)

    if item.param_diffs:
        lines.append("Params:")
        for d in item.param_diffs:
            if d.kind == "added":
                lines.append(f"  + {d.key}: {json.dumps(d.after)}")
            elif d.kind == "removed":
                lines.append(f"  - {d.key}: {json.dumps(d.before)}")
            else:
                lines.append(f"  - {d.key}: {json.dumps(d.before)}")
                lines.append(f"  + {d.key}: {json.dumps(d.after)}")

    if item.prompt_changed:
        lines.append("System prompt:")
        for dl in item.prompt_diff:
            lines.append(f"  {dl}")

    return "\n".join(lines)


def format_combined_diffs(plan: ImportPlan) -> str:
    """Combined diff view across all existing presets in a bulk import."""
    blocks = [format_item_diff(it) for it in plan.existing_items]
    return "\n\n".join(blocks)
