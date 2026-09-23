# tests/unit/test_import_reimport.py
"""Scan identity: ``already_configured`` matches on resolved path, not id.

A scanned model whose weights path is already configured under a different
id must still read as configured (symlink-safe). The import half of this
file (re-import PUT semantics, the CLI merge-preserve writer) went with
`heylookllm import` in v2.0.72; the scanner it shared stays, behind the
discovery cache.
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
