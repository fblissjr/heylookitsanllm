# tests/unit/test_access_gate.py
"""Unit tests for access-gate key resolution, TOML rejection, startup logging, and failure logs."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
import pytest
from starlette.testclient import TestClient

from heylook_llm.auth import (
    AccessGateMiddleware,
    access_config_view,
    is_loopback_host,
    log_startup_access_state,
    resolve_access_key,
    verify_api_key,
)
from heylook_llm.config import AppConfig
from heylook_llm.settings import SettingsSchema


@pytest.mark.unit
class TestKeyResolutionAndPermissions:
    def test_unset_by_default_when_env_and_file_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("HEYLOOK_API_KEY", raising=False)
        monkeypatch.delenv("HEYLOOK_API_KEY_FILE", raising=False)
        monkeypatch.setattr("heylook_llm.auth.DEFAULT_API_KEY_PATH", tmp_path / "nonexistent_api_key")

        state = resolve_access_key()
        assert state.enabled is False
        assert state.key is None
        assert state.source == "none"
        assert state.error is None

    def test_mode_600_and_400_files_succeed_and_env_overrides_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        key_file = tmp_path / "api_key"
        key_file.write_text("  file-secret-value \n", encoding="utf-8")
        key_file.chmod(0o600)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(key_file))

        state = resolve_access_key()
        assert state.enabled is True
        assert state.key == "file-secret-value"
        assert state.source == "file"
        assert state.error is None

        key_file.chmod(0o400)
        state_ro = resolve_access_key()
        assert state_ro.enabled is True
        assert state_ro.key == "file-secret-value"
        assert state_ro.error is None
        key_file.chmod(0o600)

        monkeypatch.setenv("HEYLOOK_API_KEY", "env-wins-secret")
        state_env = resolve_access_key()
        assert state_env.source == "env"
        assert state_env.key == "env-wins-secret"

    @pytest.mark.parametrize("bad_mode", [0o644, 0o640, 0o604, 0o666])
    def test_group_or_world_accessible_key_file_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_mode: int
    ):
        key_file = tmp_path / "api_key"
        key_file.write_text("leaked-secret\n", encoding="utf-8")
        key_file.chmod(bad_mode)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(key_file))

        state = resolve_access_key()
        assert state.enabled is True
        assert state.key is None
        assert state.error is not None
        assert "chmod 600" in state.error

        with pytest.raises(RuntimeError, match="chmod 600"):
            log_startup_access_state("127.0.0.1", 8000)

        mini = FastAPI()
        mini.add_middleware(AccessGateMiddleware)

        @mini.get("/health")
        def _health():
            return {"status": "ok"}

        @mini.get("/v1/models")
        def _models():
            return {"data": []}

        with TestClient(mini, raise_server_exceptions=False) as c:
            assert c.get("/health").status_code == 200
            res = c.get("/v1/models", headers={"x-api-key": "leaked-secret"})
            assert res.status_code == 503
            assert res.json()["error"]["code"] == "auth_misconfigured"

    def test_empty_or_missing_explicit_key_file_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        empty_file = tmp_path / "empty_key"
        empty_file.write_text("   \n", encoding="utf-8")
        empty_file.chmod(0o600)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(empty_file))
        state_empty = resolve_access_key()
        assert state_empty.enabled is True
        assert state_empty.key is None
        assert state_empty.error is not None

        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(tmp_path / "missing_explicit"))
        state_missing = resolve_access_key()
        assert state_missing.enabled is True
        assert state_missing.key is None
        assert state_missing.error is not None

    def test_verify_api_key_handles_unicode_in_constant_time(self):
        assert verify_api_key("correct-key", "correct-key") is True
        assert verify_api_key("wrong-key", "correct-key") is False
        assert verify_api_key("🔑-unicode-attempt", "correct-key") is False
        assert verify_api_key("", "correct-key") is False
        assert verify_api_key("correct-key", "") is False

    def test_display_path_never_exposes_home_username(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake_home_key = Path.home() / ".heylook" / "api_key"
        monkeypatch.delenv("HEYLOOK_API_KEY", raising=False)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(fake_home_key))
        view = access_config_view()
        assert view["source_detail"] == "~/.heylook/api_key"


@pytest.mark.unit
class TestTomlRejectsApiKey:
    def test_app_config_forbids_api_key_in_heylook_toml(self):
        with pytest.raises(ValueError, match="api_key must not be stored in heylook.toml"):
            AppConfig(models=[], api_key="forbidden-in-toml")

    def test_settings_schema_forbids_api_key_in_settings_table(self):
        with pytest.raises(ValueError, match="api_key must not be stored in heylook.toml"):
            SettingsSchema(api_key="forbidden-in-settings")


@pytest.mark.unit
class TestStartupAndFailureVisibility:
    @pytest.mark.parametrize(
        "host, expected",
        [
            ("127.0.0.1", True),
            ("127.0.0.2", True),
            ("localhost", True),
            ("::1", True),
            ("[::1]", True),
            ("0.0.0.0", False),
            ("::", False),
            ("192.0.2.10", False),
        ],
    )
    def test_is_loopback_host(self, host: str, expected: bool):
        assert is_loopback_host(host) is expected

    def test_startup_logs_gate_state_and_warns_loudly_when_non_loopback_without_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.INFO, logger="heylook_llm.auth")

        log_startup_access_state("127.0.0.1", 8000)
        assert any("Access gate: OFF" in r.message for r in caplog.records)
        assert not any("SECURITY WARNING" in r.message for r in caplog.records)

        caplog.clear()
        log_startup_access_state("0.0.0.0", 42193)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "SECURITY WARNING" in warnings[0].message
        assert "0.0.0.0:42193" in warnings[0].message

        caplog.clear()
        monkeypatch.setenv("HEYLOOK_API_KEY", "never-log-this-secret-value")
        log_startup_access_state("0.0.0.0", 42193)
        assert any("Access gate: ON (key source: env -> HEYLOOK_API_KEY)" in r.message for r in caplog.records)
        assert not any("SECURITY WARNING" in r.message for r in caplog.records)
        assert all("never-log-this-secret-value" not in r.message for r in caplog.records)

    def test_failed_attempts_log_path_and_peer_without_leaking_submitted_secret(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setenv("HEYLOOK_API_KEY", "real-server-secret")
        mini = FastAPI()
        mini.add_middleware(AccessGateMiddleware)

        @mini.get("/v1/models")
        def _models():
            return {"data": []}

        caplog.set_level(logging.WARNING, logger="heylook_llm.auth")
        with TestClient(mini, client=("192.0.2.99", 54321), raise_server_exceptions=False) as c:
            res = c.get(
                "/v1/models",
                headers={
                    "Authorization": "Bearer wrong-submitted-secret",
                    "X-Forwarded-For": "198.51.100.77",
                },
            )
            assert res.status_code == 401

        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "GET /v1/models" in msg
        assert "192.0.2.99:54321" in msg
        assert "xff=198.51.100.77" in msg
        assert "invalid_bearer_token" in msg
        assert "wrong-submitted-secret" not in msg
        assert "real-server-secret" not in msg
