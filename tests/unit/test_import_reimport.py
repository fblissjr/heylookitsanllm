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
