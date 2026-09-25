# tests/unit/test_settings_resolver.py
"""Resolution + validation for operational settings.

Single source of truth: DB-stored value > built-in default. There is NO env-var
override for operational settings (an env silently overriding the admin UI is a
footgun) -- env is reserved for bootstrap paths that have no UI counterpart. The
Pydantic SettingsSchema supplies types + defaults and rejects bad values.
"""

import pytest
from pydantic import ValidationError

from heylook_llm.settings import SettingsSchema, resolve_settings, resolve_settings_safe


# {} yields the documented defaults (observability off = file logging is
# opt-in, owner rule 2026-08-13); a stored value overrides its default.
_RESOLVE_ROWS = [
    pytest.param(
        {},
        {"observability_level": "off", "observability_retention_days": 30},
        id="empty_yields_documented_defaults",
    ),
    pytest.param(
        {"observability_level": "debug"},
        {"observability_level": "debug"},
        id="db_value_overrides_default",
    ),
    pytest.param(
        {"observability_level": "off", "observability_retention_days": 7},
        {"observability_level": "off", "observability_retention_days": 7},
        id="multiple_values",
    ),
]


class TestResolution:
    @pytest.mark.parametrize("stored, expected", _RESOLVE_ROWS)
    def test_resolve_table(self, stored, expected):
        s = resolve_settings(stored)
        for key, value in expected.items():
            assert getattr(s, key) == value


class TestRobustness:
    def test_unknown_db_key_ignored(self):
        # a stale/removed setting in the DB must not break resolution
        s = resolve_settings({"gone_setting": "x", "observability_level": "standard"})
        assert s.observability_level == "standard"

    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param({"observability_level": "loud"}, id="invalid_db_value_raises"),
            pytest.param({"observability_retention_days": -5}, id="out_of_range_raises"),
        ],
    )
    def test_invalid_stored_value_raises(self, stored):
        with pytest.raises(ValidationError):
            resolve_settings(stored)


class TestSafeResolution:
    # resolve_settings_safe returns (settings, None) on a good value, and
    # (defaults, an error naming the key) on a bad one.
    @pytest.mark.parametrize(
        "stored, expected_level, error_names",
        [
            pytest.param({"observability_level": "debug"}, "debug", None,
                         id="safe_ok_returns_none_error"),
            pytest.param({"observability_level": "loud"}, "off", "observability_level",
                         id="safe_bad_value_falls_back"),
        ],
    )
    def test_safe_resolution(self, stored, expected_level, error_names):
        s, err = resolve_settings_safe(stored)
        assert s.observability_level == expected_level
        if error_names is None:
            assert err is None
        else:
            assert err is not None and error_names in err


class TestSchema:
    def test_schema_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            SettingsSchema(bogus=1)
