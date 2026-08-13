# tests/unit/test_import_reimport.py
"""Phase 1 item 8: import/re-import correctness.

Two audited defects in the scan/import flow:

1. ``already_configured`` matched by id ONLY -- a scanned model whose
   weights path is already configured under a different id showed as
   unconfigured, inviting a duplicate entry pointing at the same weights.
   Now the resolved model_path is compared too (symlink-safe).

2. Re-import was skip-not-update: importing an id that already exists in
   models.toml silently skipped it, so there was no way to refresh an
   entry from a rescan (new path, new profile) short of hand-editing the
   TOML. Re-import now has PUT semantics: the existing entry is replaced
   with the freshly built one.
"""

import textwrap

import pytest

from heylook_llm.model_service import ModelService


def _write_config(tmp_path, model_path: str):
    config = tmp_path / "models.toml"
    config.write_text(textwrap.dedent(f"""
        default_model = "existing"
        max_loaded_models = 1

        [[models]]
        id = "existing"
        provider = "mlx"
        enabled = true
        description = "hand-tuned entry"
        config = {{ model_path = "{model_path}" }}
    """).strip())
    return config


@pytest.fixture
def weights_dir(tmp_path):
    d = tmp_path / "weights" / "model-a"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def service(tmp_path, weights_dir):
    return ModelService(str(_write_config(tmp_path, str(weights_dir))))


def _raw(model_id: str, path: str) -> dict:
    """Raw importer dict as produced by ModelImporter.scan_directory."""
    return {
        "id": model_id,
        "provider": "mlx",
        "config": {"model_path": path, "vision": False},
        "tags": [],
        "description": "",
    }


class TestAlreadyConfiguredMatchesPath:
    def _scan_one(self, service, raw, monkeypatch):
        from heylook_llm import model_importer

        class FakeImporter:
            def scan_directory(self, path):
                return [raw]

            def scan_hf_cache(self):
                return []

        monkeypatch.setattr(model_importer, "ModelImporter", FakeImporter)
        results = service.scan_directory("/anywhere")
        assert len(results) == 1
        return results[0]

    def test_same_id_still_matches(self, service, weights_dir, monkeypatch):
        scanned = self._scan_one(service, _raw("existing", str(weights_dir)), monkeypatch)
        assert scanned.already_configured

    def test_same_path_different_id_matches(self, service, weights_dir, monkeypatch):
        # The weights at this path are already configured under id
        # "existing"; a rescan that derives a different id must not present
        # them as unconfigured.
        scanned = self._scan_one(
            service, _raw("model-a-fresh-scan", str(weights_dir)), monkeypatch
        )
        assert scanned.already_configured, (
            "already_configured matched by id only -- same weights path "
            "under a new id shows as unconfigured"
        )

    def test_symlinked_path_matches(self, service, weights_dir, tmp_path, monkeypatch):
        link = tmp_path / "weights-link"
        link.symlink_to(weights_dir)
        scanned = self._scan_one(
            service, _raw("model-a-via-link", str(link)), monkeypatch
        )
        assert scanned.already_configured, (
            "path comparison must resolve symlinks before matching"
        )

    def test_unrelated_path_and_id_does_not_match(self, service, tmp_path, monkeypatch):
        other = tmp_path / "weights" / "model-b"
        other.mkdir(parents=True)
        scanned = self._scan_one(service, _raw("model-b", str(other)), monkeypatch)
        assert not scanned.already_configured


class TestScanPathsIdentityComputedOnce:
    def test_scan_paths_computes_identity_once(self, service, monkeypatch):
        # scan_directory/scan_hf_cache each call _configured_identity(), which
        # re-reads and re-validates the whole models.toml. scan_paths fans
        # out to multiple sources (here: 2 dirs + hf cache), so without
        # sharing one precomputed identity this would run 3 times instead
        # of 1.
        from heylook_llm import model_importer

        class FakeImporter:
            def scan_directory(self, path):
                return [_raw(f"model-from-{path}", f"/nonexistent/{path}")]

            def scan_hf_cache(self):
                return [_raw("model-from-hf", "/nonexistent/hf")]

        monkeypatch.setattr(model_importer, "ModelImporter", FakeImporter)

        original_identity = service._configured_identity
        calls = 0

        def counting_identity():
            nonlocal calls
            calls += 1
            return original_identity()

        monkeypatch.setattr(service, "_configured_identity", counting_identity)

        results = service.scan_paths(paths=["/path-a", "/path-b"], scan_hf=True)

        assert calls == 1, "scan_paths must compute _configured_identity once, not per source"
        assert {r.id for r in results} == {
            "model-from-/path-a",
            "model-from-/path-b",
            "model-from-hf",
        }


class TestImportDoesNotPickADefaultModel:
    """Import must not decide which model the server routes to by default.

    Claim: import used to stamp `default_model = imported[0].id` whenever the
    field was unset -- so deliberately clearing it (the way to say "load
    nothing until asked") was silently undone by the next scan, and the
    *arbitrary first* model in an import batch became the routing target.
    Delete this and clearing default_model stops sticking.
    """

    def test_import_leaves_unset_default_model_unset(self, tmp_path):
        config = tmp_path / "models.toml"
        config.write_text("max_loaded_models = 1\nmodels = []\n")
        service = ModelService(str(config))
        weights = tmp_path / "weights" / "model-a"
        weights.mkdir(parents=True)

        service.import_models(
            [{"id": "model-a", "path": str(weights), "provider": "mlx"}]
        )

        assert 'default_model' not in config.read_text()

    def test_import_preserves_an_existing_default_model(self, service, tmp_path):
        new_path = tmp_path / "weights" / "model-c"
        new_path.mkdir(parents=True)

        service.import_models(
            [{"id": "model-c", "path": str(new_path), "provider": "mlx"}]
        )

        assert service._read_toml().get("default_model") == "existing"


class TestReimportUpdates:
    def test_reimport_existing_id_updates_entry(self, service, tmp_path):
        new_path = tmp_path / "weights" / "model-a-v2"
        new_path.mkdir(parents=True)

        imported = service.import_models(
            [{"id": "existing", "path": str(new_path), "provider": "mlx"}]
        )

        assert len(imported) == 1, "re-import of an existing id was skipped, not updated"
        updated = service.get_config("existing")
        assert updated is not None
        assert updated.config.model_path == str(new_path)

    def test_reimport_does_not_duplicate_entry(self, service, tmp_path):
        new_path = tmp_path / "weights" / "model-a-v2"
        new_path.mkdir(parents=True)

        service.import_models(
            [{"id": "existing", "path": str(new_path), "provider": "mlx"}]
        )

        ids = [c.id for c in service.list_configs()]
        assert ids.count("existing") == 1

    def test_new_model_import_still_appends(self, service, tmp_path):
        new_path = tmp_path / "weights" / "model-c"
        new_path.mkdir(parents=True)

        imported = service.import_models(
            [{"id": "model-c", "path": str(new_path), "provider": "mlx"}]
        )

        assert [c.id for c in imported] == ["model-c"]
        ids = {c.id for c in service.list_configs()}
        assert ids == {"existing", "model-c"}


# ---------------------------------------------------------------------------
# CLI bulk import (`heylookllm import`): merge-preserve semantics (v1.62.6).
#
# The admin flow above has PUT semantics on purpose (explicit per-model
# refresh). The CLI bulk path is the opposite contract: whatever the existing
# file says goes right back out -- hand-tuned entries (server_binary,
# samplers) and top-level settings survive, scans only APPEND new ids, and
# --fresh is the only way to get the old wholesale rewrite.
# ---------------------------------------------------------------------------

import tomllib
from types import SimpleNamespace

from heylook_llm.model_importer import ModelImporter, import_models


def _cli_args(output, fresh=False):
    return SimpleNamespace(
        folder="scan-me", hf_cache=False, output=str(output),
        sampler=None, override=None, chat_template=None, fresh=fresh,
    )


HAND_TUNED = """\
# top-of-file note
default_model = "existing"
max_loaded_models = 2
idle_unload_seconds = 300

[[models]]
id = "existing"
provider = "gguf"
enabled = true
# hand-tuned: custom binary
[models.config]
model_path = "weights/existing.gguf"
server_binary = "custom/llama-server"
"""


def _scan_stub(entries):
    # Bypasses the scanners' own existing_ids bookkeeping ON PURPOSE: the
    # post-scan filter in import_models must hold even if a scanner forgets
    # to check.
    def scan(self, path):
        return [dict(e) for e in entries]
    return scan


NEW_MODEL = {"id": "brand-new", "provider": "mlx", "enabled": True,
             "config": {"model_path": "weights/new"}}


class TestCliImportMergePreserve:
    def test_reimport_puts_existing_values_right_back(self, tmp_path, monkeypatch):
        cfg = tmp_path / "models.toml"
        cfg.write_text(HAND_TUNED)
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([NEW_MODEL]))

        import_models(_cli_args(cfg))

        text = cfg.read_text()
        data = tomllib.loads(text)
        by_id = {m["id"]: m for m in data["models"]}
        assert by_id["existing"]["config"]["server_binary"] == "custom/llama-server"
        assert data["idle_unload_seconds"] == 300
        assert data["max_loaded_models"] == 2
        assert data["default_model"] == "existing"
        assert "brand-new" in by_id
        # comments ride the same toml_comments machinery as admin writes
        assert "# hand-tuned: custom binary" in text

    def test_a_configured_id_cannot_reenter_through_a_scan(self, tmp_path, monkeypatch):
        cfg = tmp_path / "models.toml"
        cfg.write_text(HAND_TUNED)
        rescan = {"id": "existing", "provider": "gguf", "enabled": True,
                  "config": {"model_path": "weights/rescanned.gguf"}}
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([rescan]))

        import_models(_cli_args(cfg))

        data = tomllib.loads(cfg.read_text())
        assert len(data["models"]) == 1
        entry = data["models"][0]
        assert entry["config"]["model_path"] == "weights/existing.gguf"
        assert entry["config"]["server_binary"] == "custom/llama-server"

    def test_no_new_models_leaves_file_byte_identical(self, tmp_path, monkeypatch):
        cfg = tmp_path / "models.toml"
        cfg.write_text(HAND_TUNED)
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([]))

        import_models(_cli_args(cfg))

        assert cfg.read_text() == HAND_TUNED

    def test_fresh_regenerates_from_scratch(self, tmp_path, monkeypatch):
        cfg = tmp_path / "models.toml"
        cfg.write_text(HAND_TUNED)
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([NEW_MODEL]))

        import_models(_cli_args(cfg, fresh=True))

        data = tomllib.loads(cfg.read_text())
        assert [m["id"] for m in data["models"]] == ["brand-new"]

    def test_unparseable_existing_file_refuses_instead_of_clobbering(self, tmp_path, monkeypatch):
        cfg = tmp_path / "models.toml"
        cfg.write_text("default_model = [broken\n")
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([NEW_MODEL]))

        with pytest.raises(SystemExit):
            import_models(_cli_args(cfg))
        assert cfg.read_text() == "default_model = [broken\n"

    def test_default_model_derivation_skips_idless_entries(self, tmp_path, monkeypatch):
        # An id-less entry is already invalid to the server, but deriving a
        # default_model from it must not crash the import with a KeyError.
        cfg = tmp_path / "models.toml"
        cfg.write_text(
            "[[models]]\n"
            'provider = "mlx"\n'
            "enabled = false\n"
        )
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([NEW_MODEL]))

        import_models(_cli_args(cfg))

        data = tomllib.loads(cfg.read_text())
        assert data["default_model"] == "brand-new"
        assert len(data["models"]) == 2

    def test_exotic_top_level_keys_round_trip_in_shape(self, tmp_path, monkeypatch):
        # Keys outside the models.toml schema still round-trip: values always,
        # and an array-of-tables keeps its [[section]] form instead of being
        # mangled into an inline array.
        cfg = tmp_path / "models.toml"
        cfg.write_text(
            'default_model = "existing"\n'
            "extra_scalar = 7\n"
            "\n"
            "[[notify]]\n"
            'url = "http://a"\n'
            "\n"
            "[[notify]]\n"
            'url = "http://b"\n'
            "[notify.headers]\n"
            'x = "y"\n'
            "\n"
            "[custom_table]\n"
            "a = 1\n"
            "\n"
            "[[models]]\n"
            'id = "existing"\n'
            'provider = "mlx"\n'
            "enabled = true\n"
        )
        monkeypatch.setattr(ModelImporter, "scan_directory", _scan_stub([NEW_MODEL]))

        import_models(_cli_args(cfg))

        text = cfg.read_text()
        data = tomllib.loads(text)
        assert data["extra_scalar"] == 7
        assert data["custom_table"] == {"a": 1}
        assert data["notify"] == [
            {"url": "http://a"},
            {"url": "http://b", "headers": {"x": "y"}},
        ]
        assert text.count("[[notify]]") == 2
        assert "[notify.headers]" in text
