# src/heylook_llm/thinking_controls.py
"""What thinking controls a model's in-force chat template offers (plan W2).

Owner rule (2026-09-23): no hardcoded thinking levels. The controls are
whatever the template itself does, in its own spellings, found by RENDERING
it rather than by reading a list anyone wrote down:

- ``switch``: the variable that turns thinking on, or None. It is
  ``enable_thinking`` -- the kwarg both providers send -- when the template
  reads it (the same signal the thinking capability has always used).
- ``depth``: the variable that sets how hard it thinks, or None. Its values
  are the groups a render produces: every candidate string is rendered with
  thinking on, and candidates that render the same prompt are ONE value (the
  first spelling the template itself uses names it; the rest are aliases).

Candidates are the template's own string literals plus a small fixed probe
set (``_PROBES``). The probe set only helps FIND aliases the template
accepts ("minimal" for "low"); it can never add a value to a template that
pastes any string in verbatim, whose values are only the literals it ties to
the variable itself.

Which variable is depth: one whose NAME says it is about thinking
(``_DEPTH_NAME``) and whose value changes the prompt. Rendering alone cannot
tell it apart from another free-form variable (gpt-oss pastes
``model_identity`` exactly the way it pastes ``reasoning_effort``), so the
name is the tie-breaker. It is a pattern over variable names, not a list of
levels.

Renders run in chat_template_files' engine-mirroring jinja environment.
Whether Python jinja matches llama.cpp's own engine on every gguf template is
unverified (the audit's caveat); llama-server's /apply-template is the
cross-check. Cached by template body: the same body always answers the same.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

SWITCH = "enable_thinking"
# Names a depth variable has carried on the templates seen so far:
# reasoning_effort (Qwen3.8, DeepSeek, gpt-oss), reasoning_strength (Muse),
# thinking_mode (MiniMax). History variables (reasoning_content,
# preserve_thinking, keep_reasoning, thinking_text, ...) match too and are
# rejected by rendering: their string values do not change the prompt.
_DEPTH_NAME = re.compile(r"reason|think|effort", re.IGNORECASE)
# Alias finders only; see the module docstring.
_PROBES = ("low", "medium", "high", "xhigh", "max", "minimal", "none", "off",
           "on", "auto", "enabled", "disabled", "adaptive")
_GARBAGE = "zq-not-a-level"
_LITERAL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")
# The conversation depth is rendered into: a system prompt and a multi-turn
# history, so a template that raises without a system message, or places
# depth relative to the turns, is exercised the way a chat exercises it.
_CONVERSATION = (
    {"role": "system", "content": "SYSTEM-PROMPT"},
    {"role": "user", "content": "FIRST-USER-TURN"},
    {"role": "assistant", "content": "FIRST-REPLY"},
    {"role": "user", "content": "SECOND-USER-TURN"},
)


def _render(template, **kw) -> Optional[str]:
    try:
        return template.render(messages=[dict(m) for m in _CONVERSATION],
                               add_generation_prompt=True, bos_token="",
                               eos_token="", **kw)
    except Exception:
        return None


def _associated_literals(ast, variable: str) -> list[str]:
    """String constants the template ties to ``variable``: compared with it,
    tested for membership against it, or given as its default."""
    from jinja2 import nodes

    out: list[str] = []

    def consts(node):
        if isinstance(node, nodes.Const) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, (nodes.List, nodes.Tuple)):
            for item in node.items:
                yield from consts(item)

    def names_var(node) -> bool:
        return any(isinstance(n, nodes.Name) and n.name == variable
                   for n in node.find_all(nodes.Name)) or (
            isinstance(node, nodes.Name) and node.name == variable)

    for cmp in ast.find_all(nodes.Compare):
        if names_var(cmp.expr) or any(names_var(op.expr) for op in cmp.ops):
            for part in [cmp.expr, *(op.expr for op in cmp.ops)]:
                out.extend(consts(part))
    for flt in ast.find_all(nodes.Filter):
        if flt.name == "default" and flt.node is not None and names_var(flt.node):
            for arg in flt.args:
                out.extend(consts(arg))
    for assign in ast.find_all(nodes.Assign):
        if isinstance(assign.target, nodes.Name) and assign.target.name == variable:
            out.extend(consts(assign.node))
    for cond in ast.find_all(nodes.CondExpr):
        if names_var(cond.test):
            out.extend(consts(cond.expr1))
            if cond.expr2 is not None:
                out.extend(consts(cond.expr2))
    return [s for s in dict.fromkeys(out) if _LITERAL.match(s)]


def _normalized_spellings(ast, group_of: dict[str, str]) -> set[str]:
    """Words the template normalizes depth to: string constants it assigns to
    a name that receives words from at least two different depth groups
    (``set _initial_effort = 'low'`` ... ``= 'xhigh'``). Those are the
    template's own names for its levels; the words it merely accepts are
    aliases of them."""
    from jinja2 import nodes

    by_target: dict[str, list[str]] = {}
    for assign in ast.find_all(nodes.Assign):
        target = assign.target
        name = target.name if isinstance(target, nodes.Name) else (
            target.attr if isinstance(target, nodes.NSRef) else None)
        if name is None:
            continue
        for const in assign.node.find_all(nodes.Const) if not isinstance(
                assign.node, nodes.Const) else [assign.node]:
            if isinstance(const.value, str) and const.value in group_of:
                by_target.setdefault(name, []).append(const.value)
    out: set[str] = set()
    for words in by_target.values():
        if len({group_of[w] for w in words}) >= 2:
            out.update(words)
    return out


def _depth(template, ast, body: str, variable: str, switch_on: dict) -> Optional[dict]:
    # None when the template raises without the variable: it then has no
    # default, which is an answer, not a reason to give up.
    absent = _render(template, **switch_on)
    garbage = _render(template, **switch_on, **{variable: _GARBAGE})
    if garbage is None:
        unknown = "raises"
    elif garbage == absent:
        unknown = "ignored"
    elif _GARBAGE in garbage:
        unknown = "verbatim"
    else:
        unknown = "fallback"

    tied = _associated_literals(ast, variable)
    if unknown == "verbatim":
        # Any string is pasted in; only what the template itself names is a
        # value. The probe set would invent levels here.
        candidates = tied
    else:
        literals = [s for s in dict.fromkeys(
            m.group(1) for m in re.finditer(r"['\"]([A-Za-z][A-Za-z0-9_-]{0,23})['\"]", body))]
        candidates = list(dict.fromkeys([*tied, *literals, *_PROBES]))

    groups: dict[str, list[str]] = {}
    for value in candidates:
        out = _render(template, **switch_on, **{variable: value})
        if out is None:
            continue
        groups.setdefault(out, []).append(value)
    canonical = _normalized_spellings(
        ast, {v: out for out, members in groups.items() for v in members})
    if unknown in ("ignored", "fallback") and garbage in groups:
        # The group an unknown word falls into holds every stray literal and
        # probe, so it is no value -- unless the template names it itself by
        # normalizing to it (a `medium` it assigns as its own default). Then
        # it is a level, spelled only by what the template names.
        named = [v for v in groups.pop(garbage) if v in canonical or v in tied]
        if named and canonical & set(named):
            groups[garbage] = named
    if not groups and unknown != "verbatim":
        return None

    # Where a spelling first appears, counting from the variable's first
    # mention: a word the template also uses for something else earlier
    # (`enable_thinking != 'false'`) must not name a depth.
    start = max(body.find(variable), 0)

    def first_seen(value: str) -> int:
        hits = [i for q in ("'", '"') for i in (body.find(f"{q}{value}{q}", start),
                                                body.find(f"{q}{value}{q}")) if i >= 0]
        after = [i for i in hits if i >= start]
        return min(after) if after else (min(hits) + len(body) if hits else 2 * len(body) + 1)

    values, aliases = [], {}
    default = None
    for out, members in groups.items():
        # The template's own normalized word names a group ahead of any
        # alias it accepts for it (`minimal` and `low` both become `low`).
        spelling = min(members, key=lambda v: (v not in canonical, first_seen(v),
                                               members.index(v)))
        values.append(spelling)
        for m in members:
            if m != spelling:
                aliases[m] = spelling
        if out == absent:
            default = spelling
    values.sort(key=first_seen)

    # Where depth enters the prompt: before the first user turn means a
    # mid-conversation change re-processes the whole conversation.
    # Any value against the absent render; the earliest divergence decides.
    changes_prefix = None
    base = absent if absent is not None else next(iter(groups), None)
    if base is None:
        base = ""
    first_user = base.find(_CONVERSATION[1]["content"])
    others = [out for out in groups if out != base]
    if unknown == "verbatim":
        others = [garbage] if garbage is not None and garbage != base else others
    if others and first_user >= 0:
        def diverge(a: str, b: str) -> int:
            n = min(len(a), len(b))
            return next((i for i in range(n) if a[i] != b[i]), n)
        changes_prefix = min(diverge(base, o) for o in others) < first_user

    return {
        "variable": variable,
        "values": values,
        "aliases": aliases,
        "default": default,
        "unknown": unknown,
        "changes_prefix": changes_prefix,
    }


@lru_cache(maxsize=128)
def detect(body: Optional[str]) -> Optional[dict]:
    """``{"switch": str|None, "depth": dict|None}`` for a template body, or
    None when there is no body to judge (unknown, not "no controls")."""
    if not body:
        return None
    from .chat_template_files import _engine_environment
    from jinja2 import meta

    env = _engine_environment()
    if env is None:
        return None
    try:
        ast = env.parse(body)
        template = env.from_string(body)
    except Exception:
        return None
    reads = meta.find_undeclared_variables(ast)
    switch = SWITCH if SWITCH in reads else None
    switch_on = {SWITCH: True} if switch else {}

    depth = None
    for variable in sorted(reads, key=lambda v: body.find(v)):
        if variable == SWITCH or not _DEPTH_NAME.search(variable):
            continue
        found = _depth(template, ast, body, variable, switch_on)
        if found is not None and (found["values"] or found["unknown"] == "verbatim"):
            depth = found
            break
    return {"switch": switch, "depth": depth}


def check_depth(value: Any, controls: Optional[dict]) -> Optional[str]:
    """Why ``value`` is not a depth this model offers, or None when it is.

    Unknown controls (no template to judge) accept anything: refusing on a
    detection that could not run would be a false refusal. A template that
    pastes values in verbatim takes any string.
    """
    if value is None or controls is None:
        return None
    depth = controls.get("depth")
    if depth is None:
        return "this model's chat template has no thinking-depth control"
    if depth["unknown"] == "verbatim":
        return None
    if value in depth["values"] or value in depth["aliases"]:
        return None
    offered = ", ".join(depth["values"]) or "none"
    return (f"thinking depth {value!r} is not offered by this model's chat "
            f"template ({depth['variable']}: {offered})")


def depth_variable(controls: Optional[dict]) -> str:
    """The template variable a requested depth is sent as: the detected one,
    else ``reasoning_effort`` (the wire field's own name) when the template
    could not be judged."""
    depth = (controls or {}).get("depth")
    return depth["variable"] if depth else "reasoning_effort"
