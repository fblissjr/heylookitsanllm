"""The sampler bag must survive the round trip: panel -> DuckDB -> panel.

A document's `params` is written by the frontend from its own `PARAM_META` and
read back into that same panel. Nothing enforces that the two vocabularies
agree, and BOTH directions lose data silently when they drift:

  a key the SERVER accepts but the panel lacks -- `mergeKnown` cannot represent
  it, so it is reported as unknown and the next params PUT (which writes the
  whole `snapshotSettings()` bag) ERASES it from the store. A value the user set
  through the API, or through an older client, disappears the first time they
  touch a knob.

  a key the PANEL has but the server does not accept -- it is written into
  `params`, stored, read back, and dropped at `_SAMPLER_KEYS` on every
  generation. The panel shows a number that never reaches the model.

This is the repo's named defect class (a second hand-written copy of a list) in
its most expensive form, because the copies are in different LANGUAGES so no
import can tie them together. Reading the frontend's source is what is left.
"""
import re
from pathlib import Path

from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS

_SETTINGS_JS = Path(__file__).resolve().parents[2] / "frontend" / "js" / "settings.js"


def _panel_keys() -> set[str]:
    src = _SETTINGS_JS.read_text()
    start = src.index("export const PARAM_META")
    end = src.index("function emptySettings", start)
    # Two-space indent = a top-level entry of the object literal; nested option
    # objects are indented further and must not count.
    return set(re.findall(r"^\s{2}(\w+):\s*\{", src[start:end], re.M))


class TestSamplerBagRoundTrips:
    def test_settings_js_is_readable(self):
        """Guard the guard: a moved file or renamed export must fail LOUDLY.

        Without this, every assertion below degrades to comparing against an
        empty set the moment the parse stops matching -- and an empty set makes
        the 'server-only' check fire and the 'panel-only' check pass, which
        reads as a real finding and a real clean bill in the same run.
        """
        assert _SETTINGS_JS.is_file(), f"{_SETTINGS_JS} is gone; this check cannot run"
        assert _panel_keys(), (
            "parsed no keys out of PARAM_META -- the literal's shape changed, so "
            "this test is comparing against nothing"
        )

    def test_every_stored_sampler_key_has_a_panel_control(self):
        missing = sorted(set(REQUEST_SAMPLER_FIELDS) - _panel_keys())
        assert not missing, (
            f"the server accepts {missing} but the panel has no control for them, "
            "so a stored value is reported unrepresentable and ERASED by the next "
            "params PUT. Add them to PARAM_META in frontend/js/settings.js, or "
            "drop them from REQUEST_SAMPLER_FIELDS."
        )

    def test_every_panel_control_is_a_key_the_server_stores(self):
        extra = sorted(_panel_keys() - set(REQUEST_SAMPLER_FIELDS))
        assert not extra, (
            f"the panel offers {extra} but the server drops them at _SAMPLER_KEYS, "
            "so the control shows a number that never reaches the model."
        )


def _panel_entries() -> dict[str, str]:
    """PARAM_META's top-level entries, key -> the entry's source text."""
    src = _SETTINGS_JS.read_text()
    start = src.index("export const PARAM_META")
    end = src.index("function emptySettings", start)
    body = src[start:end]
    heads = list(re.finditer(r"^\s{2}(\w+):\s*\{", body, re.M))
    return {m.group(1): body[m.start():(heads[i + 1].start() if i + 1 < len(heads) else len(body))]
            for i, m in enumerate(heads)}


class TestPanelTwins:
    """The panel's other hand copies of server facts, in a different language,
    so reading the source is the only tie."""

    def test_requires_cap_is_the_servers_cap_gate(self):
        """PARAM_META `requiresCap` hides a control; `_CAP_GATED` drops the
        same key at /generate. A key gated on one side only is either a
        control that silently does nothing or one hidden for no reason."""
        from heylook_llm.conversation_generate_api import _CAP_GATED

        panel = {key: m.group(1) for key, text in _panel_entries().items()
                 if (m := re.search(r"requiresCap:\s*'(\w+)'", text))}
        assert panel, "parsed no requiresCap out of PARAM_META; this check compares nothing"
        assert panel == _CAP_GATED

    def test_widget_hints_stay_inside_the_servers_bounds(self):
        """min/max are widget hints, not validation (settings.js `valid`), and
        may be narrower than the server's. They must never suggest a value
        ChatRequest rejects."""
        from annotated_types import Ge, Gt, Le, Lt

        from heylook_llm.config import ChatRequest

        checked = 0
        for key, text in _panel_entries().items():
            field = ChatRequest.model_fields.get(key)
            if field is None:
                continue
            lo = re.search(r"\bmin:\s*([-\d.]+)", text)
            hi = re.search(r"\bmax:\s*([-\d.]+)", text)
            for bound in field.metadata:
                if isinstance(bound, (Ge, Gt)) and lo:
                    floor = bound.ge if isinstance(bound, Ge) else bound.gt
                    ok = float(lo.group(1)) >= floor if isinstance(bound, Ge) else float(lo.group(1)) > floor
                    assert ok, f"{key}: panel min {lo.group(1)} is below the server's {floor}"
                    checked += 1
                if isinstance(bound, (Le, Lt)) and hi:
                    ceil = bound.le if isinstance(bound, Le) else bound.lt
                    ok = float(hi.group(1)) <= ceil if isinstance(bound, Le) else float(hi.group(1)) < ceil
                    assert ok, f"{key}: panel max {hi.group(1)} is above the server's {ceil}"
                    checked += 1
        assert checked >= 6, f"only {checked} bounds compared; the parse or the schema moved"

    def test_diffusion_fields_are_panel_keys(self):
        """engine.decoding.request_fields hides every panel control not named
        in it, so a misspelled name hides a control the model does read."""
        from heylook_llm.providers.mlx_provider import DIFFUSION_REQUEST_FIELDS

        assert set(DIFFUSION_REQUEST_FIELDS) <= _panel_keys()
