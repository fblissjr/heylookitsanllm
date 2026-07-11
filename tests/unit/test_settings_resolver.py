# tests/unit/test_settings_resolver.py
"""Precedence + validation for operational settings resolution.

Effective value = env override > DB-stored value > built-in default. The DB
holds only what's been explicitly set; the Pydantic SettingsSchema supplies
types + defaults and rejects bad values. Env stays the always-wins escape hatch.
"""

import pytest
from pydantic import ValidationError

from heylook_llm.settings import SettingsSchema, resolve_settings, setting_env_key


class TestDefaults:
    def test_empty_yields_documented_defaults(self):
        s = resolve_settings({}, env={})
        assert s.observability_level == "minimal"
        assert s.observability_retention_days == 30


class TestPrecedence:
    def test_db_value_overrides_default(self):
        s = resolve_settings({"observability_level": "debug"}, env={})
        assert s.observability_level == "debug"

    def test_env_overrides_db(self):
        s = resolve_settings(
            {"observability_level": "debug"},
            env={"HEYLOOK_OBSERVABILITY_LEVEL": "off"},
        )
        assert s.observability_level == "off"

    def test_env_overrides_default_with_coercion(self):
        # env values arrive as strings; Pydantic coerces to the field type
        s = resolve_settings({}, env={"HEYLOOK_OBSERVABILITY_RETENTION_DAYS": "14"})
        assert s.observability_retention_days == 14
        assert isinstance(s.observability_retention_days, int)


class TestRobustness:
    def test_unknown_db_key_ignored(self):
        # a stale/removed setting in the DB must not break resolution
        s = resolve_settings({"gone_setting": "x", "observability_level": "standard"}, env={})
        assert s.observability_level == "standard"

    def test_invalid_db_value_raises(self):
        with pytest.raises(ValidationError):
            resolve_settings({"observability_level": "loud"}, env={})

    def test_invalid_env_value_raises(self):
        with pytest.raises(ValidationError):
            resolve_settings({}, env={"HEYLOOK_OBSERVABILITY_RETENTION_DAYS": "-5"})


class TestEnvKeyConvention:
    def test_env_key_naming(self):
        assert setting_env_key("observability_level") == "HEYLOOK_OBSERVABILITY_LEVEL"

    def test_schema_forbids_unknown_fields(self):
        with pytest.raises(ValidationError):
            SettingsSchema(bogus=1)
