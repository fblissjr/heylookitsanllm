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
