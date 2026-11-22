# tests/unit/test_access_gate.py
"""Unit tests for access-gate key resolution, TOML rejection, startup logging, and failure logs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types

from fastapi import FastAPI
import pytest
from starlette.testclient import TestClient

from heylook_llm.auth import (
    AccessGateMiddleware,
    access_config_view,
    clear_access_key,
    create_access_key,
    get_session_store,
    is_loopback_host,
    log_startup_access_state,
    manage_access_key,
    op_ref_sidecar_path,
    parse_op_reference,
    resolve_access_key,
    save_key_to_1password,
    validate_op_vault,
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


@pytest.mark.unit
class TestKeySubcommandLifecycle:
    def test_create_enforces_0600_under_permissive_umask_and_refuses_overwrite_without_rotate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        key_file = tmp_path / "nested" / "api_key"
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(key_file))

        old_umask = os.umask(0o022)
        try:
            key1, written = create_access_key(rotate=False)
        finally:
            os.umask(old_umask)

        assert written == key_file
        assert key1.startswith("hlk_")
        assert len(key1) >= 40
        assert (key_file.stat().st_mode & 0o777) == 0o600
        assert resolve_access_key().key == key1

        # Accidental re-run without --rotate refuses and leaves key1 intact
        with pytest.raises(FileExistsError, match="--rotate"):
            create_access_key(rotate=False)
        assert key_file.read_text(encoding="utf-8").strip() == key1

        rc_refused = manage_access_key(
            types.SimpleNamespace(action="create", rotate=False, copy=False, local_only=True)
        )
        assert rc_refused == 1
        assert key_file.read_text(encoding="utf-8").strip() == key1

        # Mint a browser session under key1, then rotate -> session is invalidated
        tok = get_session_store().create(key1)
        assert get_session_store().verify(tok, key1) is True

        key2, _ = create_access_key(rotate=True)
        assert key2 != key1
        assert key2.startswith("hlk_")
        assert (key_file.stat().st_mode & 0o777) == 0o600
        assert resolve_access_key().key == key2
        assert get_session_store().verify(tok, key2) is False

    def test_create_refuses_symlink_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        victim = tmp_path / "victim.txt"
        victim.write_text("keep-me\n", encoding="utf-8")
        link = tmp_path / "api_key_link"
        link.symlink_to(victim)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(link))

        with pytest.raises(RuntimeError, match="symlink"):
            create_access_key(rotate=True)
        assert victim.read_text(encoding="utf-8") == "keep-me\n"

    @pytest.mark.parametrize("disallowed", ["Private", "private", "Personal", "Employee", "Shared"])
    def test_rejects_vaults_inaccessible_to_1password_service_accounts(self, disallowed: str):
        with pytest.raises(ValueError, match="service accounts"):
            validate_op_vault(disallowed)
        with pytest.raises(ValueError, match="service accounts"):
            parse_op_reference(f"op://{disallowed}/heylook/password")

        assert validate_op_vault("Homelab") == "Homelab"
        assert parse_op_reference("op://Homelab/heylook/password") == ("Homelab", "heylook", "password")

    def test_1password_master_copy_create_status_rotate_and_sync(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        key_file = tmp_path / "api_key"
        monkeypatch.delenv("HEYLOOK_API_KEY", raising=False)
        monkeypatch.delenv("HEYLOOK_OP_REF", raising=False)
        monkeypatch.delenv("HEYLOOK_OP_VAULT", raising=False)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(key_file))
        monkeypatch.setattr("heylook_llm.auth.shutil.which", lambda name: "/opt/homebrew/bin/op")

        # In-memory fake 1Password vault store keyed by (vault, title)
        vault_items: dict[tuple[str, str], dict] = {}
        op_locked = False
        copied_payloads: list[bytes] = []

        def _fake_subprocess_run(cmd, *, input=None, capture_output=False, text=False, **kwargs):
            if cmd == ["pbcopy"]:
                copied_payloads.append(input)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            assert cmd[0] == "op"
            # Secret values must NEVER appear on op's command line (argv)
            for item_doc in vault_items.values():
                for f in item_doc.get("fields", []):
                    val = f.get("value")
                    if val and val.startswith("hlk_"):
                        assert all(val not in arg for arg in cmd)

            if op_locked:
                return subprocess.CompletedProcess(cmd, 1, "", "[ERROR] 1Password is locked")

            if cmd[:3] == ["op", "read", "--no-newline"]:
                vault, title, field = parse_op_reference(cmd[3])
                doc = vault_items.get((vault, title))
                if not doc:
                    return subprocess.CompletedProcess(cmd, 1, "", f'"{title}" isn\'t an item')
                for f in doc.get("fields", []):
                    if f.get("id") == field or f.get("label") == field:
                        return subprocess.CompletedProcess(cmd, 0, f["value"], "")
                return subprocess.CompletedProcess(cmd, 1, "", "field not found")

            if cmd[:3] == ["op", "item", "get"]:
                title = cmd[3]
                vault = cmd[cmd.index("--vault") + 1]
                doc = vault_items.get((vault, title))
                if not doc:
                    return subprocess.CompletedProcess(
                        cmd, 1, "", f'[ERROR] "{title}" isn\'t an item in the "{vault}" vault.'
                    )
                return subprocess.CompletedProcess(cmd, 0, json.dumps(doc), "")

            if cmd[:3] == ["op", "item", "create"]:
                assert "-" in cmd
                assert input is not None
                vault = cmd[cmd.index("--vault") + 1]
                payload = json.loads(input.decode("utf-8"))
                # Verify secret was NOT in cmd argv during create either
                for f in payload.get("fields", []):
                    val = f.get("value")
                    if val and val.startswith("hlk_"):
                        assert all(val not in arg for arg in cmd)
                doc = {
                    "id": "item-uuid-1",
                    "title": payload["title"],
                    "category": payload["category"],
                    "vault": {"name": vault},
                    "fields": payload["fields"],
                    "urls": payload.get("urls", []),
                }
                vault_items[(vault, payload["title"])] = doc
                return subprocess.CompletedProcess(cmd, 0, json.dumps(doc), "")

            if cmd[:3] == ["op", "item", "edit"]:
                assert input is not None
                vault = cmd[cmd.index("--vault") + 1]
                payload = json.loads(input.decode("utf-8"))
                for f in payload.get("fields", []):
                    val = f.get("value")
                    if val and val.startswith("hlk_"):
                        assert all(val not in arg for arg in cmd)
                vault_items[(vault, payload["title"])] = payload
                return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

            raise AssertionError(f"Unexpected command: {cmd}")

        monkeypatch.setattr("heylook_llm.auth.subprocess.run", _fake_subprocess_run)

        # 1. Refuses built-in Private vault on create
        rc_priv = manage_access_key(
            types.SimpleNamespace(action="create", vault="Private", rotate=False, copy=False, local_only=False)
        )
        assert rc_priv == 1
        assert "service accounts" in capsys.readouterr().err
        assert not key_file.exists()

        # 2. Create in dedicated 'Homelab' vault writes both 1Password item and mode-0600 file
        rc_create = manage_access_key(
            types.SimpleNamespace(
                action="create",
                vault="Homelab",
                item="heylook",
                url="https://heylook.lan.example",
                rotate=False,
                copy=True,
                local_only=False,
            )
        )
        assert rc_create == 0
        out_create = capsys.readouterr()
        key1 = key_file.read_text(encoding="utf-8").strip()
        assert key1.startswith("hlk_")
        assert (key_file.stat().st_mode & 0o777) == 0o600
        assert op_ref_sidecar_path(key_file).read_text(encoding="utf-8").strip() == "op://Homelab/heylook/password"
        assert (op_ref_sidecar_path(key_file).stat().st_mode & 0o777) == 0o600
        assert copied_payloads == [key1.encode("utf-8")]
        assert key1 not in out_create.out
        assert key1 not in out_create.err

        # 3. status (with no flags -- uses .op_ref sidecar) reports IN SYNC without printing either key
        assert manage_access_key(types.SimpleNamespace(action="status")) == 0
        status_in_sync = capsys.readouterr().out
        assert "Access gate: ON" in status_in_sync
        assert "1Password: IN SYNC (op://Homelab/heylook/password)" in status_in_sync
        assert key1 not in status_in_sync

        # 4. External update in 1Password -> status reports OUT OF SYNC; sync rewrites mode-0600 file
        external_key = "hlk_updated_in_1password_master_copy_999"
        for f in vault_items[("Homelab", "heylook")]["fields"]:
            if f.get("id") == "password":
                f["value"] = external_key

        assert manage_access_key(types.SimpleNamespace(action="status")) == 0
        status_drift = capsys.readouterr().out
        assert "1Password: OUT OF SYNC" in status_drift
        assert key1 not in status_drift
        assert external_key not in status_drift

        assert manage_access_key(types.SimpleNamespace(action="sync", seed=False)) == 0
        capsys.readouterr()
        assert key_file.read_text(encoding="utf-8").strip() == external_key
        assert (key_file.stat().st_mode & 0o777) == 0o600
        assert resolve_access_key().key == external_key

        # 5. Locked 1Password NEVER fails closed: server & status keep using existing file,
        #    and sync / create --rotate leave the existing file untouched!
        op_locked = True
        assert resolve_access_key().key == external_key
        assert manage_access_key(types.SimpleNamespace(action="status")) == 0
        locked_status = capsys.readouterr().out
        assert "Access gate: ON" in locked_status
        assert "1Password: unavailable" in locked_status
        assert "keeping existing" in locked_status

        assert manage_access_key(types.SimpleNamespace(action="sync", seed=False)) == 1
        assert "keeping existing" in capsys.readouterr().err
        assert key_file.read_text(encoding="utf-8").strip() == external_key

        assert manage_access_key(
            types.SimpleNamespace(action="create", rotate=True, copy=False, local_only=False)
        ) == 1
        assert "keeping existing" in capsys.readouterr().err
        assert key_file.read_text(encoding="utf-8").strip() == external_key

        # 6. Unlocking and running create --rotate updates both 1Password and mode-0600 file
        op_locked = False
        assert manage_access_key(
            types.SimpleNamespace(action="create", rotate=True, copy=False, local_only=False)
        ) == 0
        capsys.readouterr()
        rotated_key = key_file.read_text(encoding="utf-8").strip()
        assert rotated_key != external_key
        assert rotated_key.startswith("hlk_")
        assert manage_access_key(types.SimpleNamespace(action="status")) == 0
        assert "1Password: IN SYNC" in capsys.readouterr().out

        # 7. Refuses JSON template edit if an existing item contains a PASSKEY field
        vault_items[("Homelab", "heylook")]["fields"].append({"id": "pk1", "type": "PASSKEY"})
        with pytest.raises(RuntimeError, match="passkey"):
            save_key_to_1password("hlk_other", op_ref="op://Homelab/heylook/password", overwrite=True)

        # 8. clear removes local file & sidecar while leaving 1Password item intact
        existed, _ = clear_access_key()
        assert existed is True
        assert not key_file.exists()
        assert not op_ref_sidecar_path(key_file).exists()
        assert ("Homelab", "heylook") in vault_items

    def test_sync_seed_uploads_existing_local_key_without_rotating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        key_file = tmp_path / "api_key"
        existing_secret = "existing-handmade-openssl-key-0123456789"
        key_file.write_text(f"{existing_secret}\n", encoding="utf-8")
        key_file.chmod(0o600)
        monkeypatch.setenv("HEYLOOK_API_KEY_FILE", str(key_file))
        monkeypatch.setattr("heylook_llm.auth.shutil.which", lambda name: "/opt/homebrew/bin/op")

        stored: dict[str, str] = {}

        def _fake_op(cmd, *, input=None, **kwargs):
            if cmd[:3] == ["op", "item", "get"]:
                if "secret" not in stored:
                    return subprocess.CompletedProcess(cmd, 1, "", 'isn\'t an item')
                doc = {
                    "id": "id1",
                    "title": "heylook",
                    "vault": {"name": "Homelab"},
                    "fields": [{"id": "password", "purpose": "PASSWORD", "value": stored["secret"]}],
                }
                return subprocess.CompletedProcess(cmd, 0, json.dumps(doc), "")
            if cmd[:3] == ["op", "item", "create"]:
                payload = json.loads(input.decode("utf-8"))
                for f in payload["fields"]:
                    if f["id"] == "password":
                        stored["secret"] = f["value"]
                return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        monkeypatch.setattr("heylook_llm.auth.subprocess.run", _fake_op)
        rc = manage_access_key(types.SimpleNamespace(action="sync", seed=True, vault="Homelab"))
        assert rc == 0
        out = capsys.readouterr()
        assert stored["secret"] == existing_secret
        assert key_file.read_text(encoding="utf-8").strip() == existing_secret
        assert existing_secret not in out.out
        assert op_ref_sidecar_path(key_file).read_text(encoding="utf-8").strip() == "op://Homelab/heylook/password"

    def test_server_cli_key_status_runs_fast_without_starting_uvicorn(self, tmp_path: Path):
        env = {
            **os.environ,
            "HEYLOOK_API_KEY": "",
            "HEYLOOK_API_KEY_FILE": "",
            "HEYLOOK_OP_REF": "",
            "HEYLOOK_OP_VAULT": "",
        }
        proc = subprocess.run(
            [sys.executable, "-m", "heylook_llm.server", "key", "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert proc.returncode == 0
        assert "Access gate: OFF" in proc.stdout
