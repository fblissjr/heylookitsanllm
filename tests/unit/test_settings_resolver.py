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


class TestDefaults:
    def test_empty_yields_documented_defaults(self):
        s = resolve_settings({})
        assert s.observability_level == "minimal"
        assert s.observability_retention_days == 30


class TestResolution:
    def test_db_value_overrides_default(self):
        s = resolve_settings({"observability_level": "debug"})
        assert s.observability_level == "debug"

    def test_multiple_values(self):
        s = resolve_settings({"observability_level": "off", "observability_retention_days": 7})
        assert s.observability_level == "off"
        assert s.observability_retention_days == 7


class TestRobustness:
    def test_unknown_db_key_ignored(self):
        # a stale/removed setting in the DB must not break resolution
        s = resolve_settings({"gone_setting": "x", "observability_level": "standard"})
        assert s.observability_level == "standard"

    def test_invalid_db_value_raises(self):
        with pytest.raises(ValidationError):
            resolve_settings({"observability_level": "loud"})

    def test_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            resolve_settings({"observability_retention_days": -5})


class TestSafeResolution:
    def test_safe_ok_returns_none_error(self):
        s, err = resolve_settings_safe({"observability_level": "debug"})
        assert s.observability_level == "debug"
        assert err is None

    def test_safe_bad_value_falls_back(self):
        s, err = resolve_settings_safe({"observability_level": "loud"})
        assert s.observability_level == "minimal"  # default
        assert err is not None and "observability_level" in err


class TestSchema:
    def test_schema_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            SettingsSchema(bogus=1)
