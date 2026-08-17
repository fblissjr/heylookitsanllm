# tests/unit/test_model_registry.py
"""Discovery-as-registry: models.toml overrides, the scan folders decide what exists.

Claims (what breaks if a test is deleted):

- merge tests: a new download stops being servable without an import, or --
  worse -- discovery starts overriding hand-written entries, which is the
  clobbering the whole design exists to prevent.
- materialization tests: editing a discovered model 404s, or a bulk edit
  reports success and changes nothing.
"""
import textwrap

import pytest

from heylook_llm.model_registry import merge_discovered, discover
from heylook_llm.model_service import ModelService


def entry(model_id, path, **config):
    return {"id": model_id, "provider": "gguf", "enabled": True,
            "config": {"model_path": str(path), **config}}


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    return d


@pytest.mark.unit
class TestMergeDiscovered:
    def test_unrepresented_model_is_appended(self, store):
        cfg = {"models": [entry("configured", store / "a.gguf")]}
        merged = merge_discovered(cfg, [entry("found", store / "b.gguf")])
        assert [m["id"] for m in merged["models"]] == ["configured", "found"]

    def test_explicit_entry_wins_and_is_not_duplicated(self, store):
        """The hand-written entry must survive verbatim, keeping its id."""
        blob = store / "a.gguf"
        cfg = {"models": [entry("my-nice-name", blob, supports_thinking=True)]}
        # Same file, derived id, and a WRONG value -- exactly what a rescan
        # produces for a renamed entry.
        merged = merge_discovered(
            cfg, [entry("a", blob, supports_thinking=False)])
        assert [m["id"] for m in merged["models"]] == ["my-nice-name"]
        assert merged["models"][0]["config"]["supports_thinking"] is True

    def test_symlinked_spelling_counts_as_the_same_file(self, tmp_path):
        """modelzoo/<vendor> symlinks mean two paths, one file.

        Without resolution these compare unequal and the model is served
        twice under two ids.
        """
        real = tmp_path / "store"
        real.mkdir()
        blob = real / "a.gguf"
        blob.write_text("x")
        link = tmp_path / "modelzoo"
        link.symlink_to(real, target_is_directory=True)

        cfg = {"models": [entry("via-link", link / "a.gguf")]}
        merged = merge_discovered(cfg, [entry("a", blob)])
        assert [m["id"] for m in merged["models"]] == ["via-link"]

    def test_derived_id_colliding_with_a_different_file_is_refused(self, store, caplog):
        """Serving both would make the id ambiguous; say so rather than guess."""
        cfg = {"models": [entry("dupe", store / "one.gguf")]}
        merged = merge_discovered(cfg, [entry("dupe", store / "two.gguf")])
        assert len(merged["models"]) == 1
        assert "already used by a different" in caplog.text

    def test_no_discoveries_preserves_the_written_entries(self, store):
        """Equal contents, not identity: the merge always returns a fresh dict
        carrying a `models` key (see test_scan_only_config_still_yields_a_models_key)."""
        cfg = {"models": [entry("configured", store / "a.gguf")]}
        merged = merge_discovered(cfg, [])
        assert merged["models"] == cfg["models"]

    def test_inputs_are_not_mutated(self, store):
        cfg = {"models": [entry("configured", store / "a.gguf")]}
        merge_discovered(cfg, [entry("found", store / "b.gguf")])
        assert len(cfg["models"]) == 1, "merge mutated the caller's config"

    def test_scan_only_config_still_yields_a_models_key(self):
        """AppConfig.models is REQUIRED, so an absent key is a dead server.

        The config shape that hits this is the one this design promotes: a
        models.toml carrying only [scan]. Empty folder, unmounted volume, or
        a failed scan all arrive here with discovered=[].
        """
        from heylook_llm.config import AppConfig

        cfg = {"scan": {"folders": ["/nope"]}, "default_model": "none"}
        merged = merge_discovered(cfg, [])
        assert "models" in merged
        AppConfig(**merged)  # must not raise

    def test_two_discoveries_of_one_file_are_served_once(self, store):
        """Dedup must apply within the batch, not only against models.toml.

        scan_hf_cache rewrites the id to `org/name` after the directory-name
        id is registered, and scan_directory follows symlinks -- so one file
        legitimately arrives twice under different ids.
        """
        blob = store / "same.gguf"
        merged = merge_discovered({"models": []}, [entry("a", blob), entry("b", blob)])
        assert [m["id"] for m in merged["models"]] == ["a"]


@pytest.mark.unit
class TestDiscoverIsBestEffort:
    def test_no_scan_section_discovers_nothing(self):
        assert discover({"models": []}) == []

    def test_scan_failure_degrades_to_models_toml(self, monkeypatch, caplog):
        """A broken scan must not stop the server coming up."""
        import heylook_llm.model_importer as mi

        def boom(self, path):
            raise OSError("disk gone")

        monkeypatch.setattr(mi.ModelImporter, "scan_directory", boom)
        assert discover({"scan": {"folders": ["/nope"]}}) == []
        assert "failed" in caplog.text

    def test_one_bad_folder_does_not_discard_the_healthy_ones(
            self, tmp_path, monkeypatch):
        """Per-source isolation: an unmounted volume must cost only itself."""
        import heylook_llm.model_importer as mi
        good = entry("survivor", tmp_path / "ok.gguf")

        def selective(self, path):
            if "broken" in str(path):
                raise OSError("unmounted")
            return [dict(good)]

        monkeypatch.setattr(mi.ModelImporter, "scan_directory", selective)
        found = discover({"scan": {"folders": ["/broken", "/healthy"]}})
        assert [e["id"] for e in found] == ["survivor"]

    def test_interval_zero_disables_discovery(self, monkeypatch):
        """The documented off switch must actually switch discovery off."""
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(
            mi.ModelImporter, "scan_directory",
            lambda self, path: pytest.fail("scanned despite interval 0"))
        assert discover({"scan": {"folders": ["/x"], "scan_interval_seconds": 0}}) == []


@pytest.mark.unit
class TestMaterializeOnWrite:
    """A discovered model has no entry; an edit has to create one."""

    def _service(self, tmp_path, store):
        cfg = tmp_path / "models.toml"
        cfg.write_text(textwrap.dedent(f"""
            default_model = "none"

            [scan]
            folders = ["{store}"]
        """).strip())
        return ModelService(str(cfg)), cfg

    def _stub_scan(self, monkeypatch, entries):
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(mi.ModelImporter, "scan_directory",
                            lambda self, path: [dict(e) for e in entries])

    def test_update_config_materializes_then_applies(
            self, tmp_path, store, monkeypatch):
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, cfg = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])

        updated, _ = svc.update_config(
            "found", {"config": {"chat_template_path": str(blob)}})

        assert updated.id == "found"
        import tomllib
        data = tomllib.loads(cfg.read_text())
        written = {m["id"]: m for m in data["models"]}
        assert "found" in written, "edit did not write an entry"
        assert written["found"]["config"]["chat_template_path"] == str(blob)

    def test_reading_does_not_materialize(self, tmp_path, store, monkeypatch):
        """Browsing the models page must not grow models.toml."""
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, cfg = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])
        before = cfg.read_text()

        svc.scan_paths(paths=[str(store)], scan_hf=False)

        assert cfg.read_text() == before

    def test_toggle_enabled_materializes_so_the_off_switch_sticks(
            self, tmp_path, store, monkeypatch):
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, cfg = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])

        result = svc.toggle_enabled("found")

        assert result.enabled is False
        import tomllib
        data = tomllib.loads(cfg.read_text())
        assert data["models"][0]["enabled"] is False

    def test_unknown_id_still_raises(self, tmp_path, store, monkeypatch):
        svc, _ = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [])
        with pytest.raises(ValueError, match="not found"):
            svc.toggle_enabled("ghost")

    def test_materialized_entry_is_thin(self, tmp_path, store, monkeypatch):
        """Only identity + the edited field. Derived facts re-derive at load.

        Freezing modalities/mmproj_path here is worse than elsewhere: the
        merge makes an explicit entry AUTHORITATIVE, so a stale copy silently
        overrides fresh detection after an in-place re-quantize.
        """
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, cfg = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [dict(
            entry("found", blob), config={
                "model_path": str(blob), "modalities": ["text", "vision"],
                "supports_thinking": True, "mmproj_path": str(store / "mm.gguf")})])

        svc.toggle_enabled("found")

        import tomllib
        written = tomllib.loads(cfg.read_text())["models"][0]
        assert set(written["config"]) == {"model_path"}, written["config"]
        assert written["enabled"] is False

    def test_deleting_a_disabled_override_is_refused(
            self, tmp_path, store, monkeypatch):
        """Otherwise the delete silently RE-ENABLES the model."""
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, _ = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])
        svc.toggle_enabled("found")  # materializes enabled = false

        with pytest.raises(ValueError, match="re-enable"):
            svc.remove_config("found")


@pytest.mark.unit
class TestAdminSurfaceSeesDiscovered:
    """/v1/admin/models and /v1/models must agree about what exists.

    list_configs backs the v3 models page. If it read models.toml alone, a
    discovered model would be servable and listed by /v1/models while being
    invisible in the page that manages models.
    """

    def _service(self, tmp_path, store):
        cfg = tmp_path / "models.toml"
        cfg.write_text(textwrap.dedent(f"""
            default_model = "none"

            [scan]
            folders = ["{store}"]
        """).strip())
        return ModelService(str(cfg))

    def _stub_scan(self, monkeypatch, entries):
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(mi.ModelImporter, "scan_directory",
                            lambda self, path: [dict(e) for e in entries])

    def test_list_configs_is_written_down_only(self, tmp_path, store, monkeypatch):
        """It feeds _configured_identity; folding discovery in here marks every
        scanned model already_configured and permanently empties
        GET /v1/admin/models/discovered."""
        blob = store / "found.gguf"
        blob.write_text("x")
        svc = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])

        assert [c.id for c in svc.list_configs()] == []
        assert svc.get_config("found") is None

    def test_discovered_endpoint_still_sees_unconfigured_models(
            self, tmp_path, store, monkeypatch):
        """The C3 cache drops anything already_configured -- so a discovered
        model must NOT be reported as configured."""
        blob = store / "found.gguf"
        blob.write_text("x")
        svc = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])

        scanned = svc.scan_paths(paths=[str(store)], scan_hf=False)
        assert [s.id for s in scanned] == ["found"]
        assert scanned[0].already_configured is False

    def test_listing_does_not_write(self, tmp_path, store, monkeypatch):
        blob = store / "found.gguf"
        blob.write_text("x")
        cfg = tmp_path / "models.toml"
        svc = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])
        before = cfg.read_text()

        svc.list_configs()
        svc.get_config("found")

        assert cfg.read_text() == before
