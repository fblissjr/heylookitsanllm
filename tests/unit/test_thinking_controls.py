# tests/unit/test_thinking_controls.py
"""Thinking controls detected from real templates (plan W2).

The fixtures are the in-force templates of served models, copied verbatim
(tests/fixtures/chat_templates/). The expected rows are the audit's by-hand
reading of the same templates (docs/testing/gguf_runtime_audit_2026-09-23.md
§6), an independent source: detection renders, the audit read.
"""
from pathlib import Path

import pytest

from heylook_llm.thinking_controls import check_depth, detect

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_templates"

# name -> (switch, variable, values, default, unknown)
EXPECTED = {
    "qwen3_8_official": ("enable_thinking", "reasoning_effort", ["xhigh", "medium", "low"], "xhigh", "raises"),
    "qwen3_8_unsloth_embedded": ("enable_thinking", "reasoning_effort", ["xhigh", "medium", "low"], "xhigh", "raises"),
    "deepseek_v4_unsloth": ("enable_thinking", "reasoning_effort", ["high", "max"], None, "ignored"),
    "deepseek_v4_ggml_org": ("enable_thinking", "reasoning_effort", ["max"], None, "ignored"),
    "minimax_m3": (None, "thinking_mode", ["enabled", "disabled", "adaptive"], "adaptive", "fallback"),
    "muse_glimmer": (None, "reasoning_strength", ["high"], "high", "verbatim"),
    "gpt_oss": (None, "reasoning_effort", ["medium"], "medium", "verbatim"),
    "qwen3_5": ("enable_thinking", None, None, None, None),
    # Read by hand 2026-09-25, not in the audit. The group a template spells
    # by the word it normalizes to (`_initial_effort = 'low'`), and a word
    # used earlier for something else (`enable_thinking != 'false'`) names no
    # depth; a default the template assigns itself is a level even where an
    # unknown word also falls to it.
    "qwen3_8_uncensored_sidecar": ("enable_thinking", "reasoning_effort",
                                   ["medium", "none", "low", "xhigh"], "medium", "ignored"),
    "qwen3_8_unsloth_override": ("enable_thinking", "reasoning_effort",
                                 ["auto", "none", "xhigh", "high", "low"], "auto", "raises"),
}


def _controls(name):
    return detect((FIXTURES / f"{name}.jinja").read_text())


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_detection_matches_the_audit(name):
    switch, variable, values, default, unknown = EXPECTED[name]
    controls = _controls(name)
    assert controls["switch"] == switch
    depth = controls["depth"]
    if variable is None:
        assert depth is None
        return
    assert (depth["variable"], depth["values"], depth["default"], depth["unknown"]) == \
        (variable, values, default, unknown)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_only_a_models_own_values_pass(name):
    """Every detected value and alias passes; a value only ANOTHER model
    offers, and a made-up one, are refused -- except where the template
    pastes any word in verbatim, which takes anything. A model with no depth
    control refuses any value."""
    controls = _controls(name)
    depth = controls["depth"]
    others = {v for n in EXPECTED if n != name
              for v in (_controls(n)["depth"] or {}).get("values", [])}
    if depth is None:
        assert check_depth("low", controls)
        return
    for value in [*depth["values"], *depth["aliases"]]:
        assert check_depth(value, controls) is None
    foreign = sorted(others - set(depth["values"]) - set(depth["aliases"])) + ["zzz-made-up"]
    for value in foreign:
        refused = check_depth(value, controls)
        assert (refused is None) == (depth["unknown"] == "verbatim"), value


@pytest.mark.unit
def test_no_template_means_unknown_not_none():
    """No body to judge is unknown: nothing is refused on it."""
    assert detect(None) is None
    assert check_depth("anything", None) is None
