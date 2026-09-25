# tests/unit/test_model_registry.py
"""Discovery-as-registry: models.toml overrides, the scan folders decide what exists.

Claims (what breaks if a test is deleted):

- merge tests: a new download stops being servable without an import, or --
  worse -- discovery starts overriding hand-written entries, which is the
  clobbering the whole design exists to prevent.
- model-file tests: editing a discovered model 404s, freezes its derived
  config into models.toml, or loses what its model.heylook.toml already held.
"""
import textwrap

import pytest

from heylook_llm.model_registry import Discovery, discover, merge_discovered, scan, served_diff
from heylook_llm.model_service import ModelService


def entry(model_id, path, **config):
    return {"id": model_id, "provider": "gguf", "enabled": True,
            "config": {"model_path": str(path), **config}}


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    return d


def _dedup_written_entry(root):
    # Same file, derived id, and a WRONG value -- exactly what a rescan
    # produces for a renamed entry. The hand-written entry survives verbatim.
    blob = root / "a.gguf"
    written = entry("my-nice-name", blob, supports_thinking=True)
    return {"models": [written]}, [entry("a", blob, supports_thinking=False)], [written]


def _dedup_symlink(root):
    # Vendor symlinks in a model folder mean two paths, one file; without
    # resolution these compare unequal and the model is served twice.
    real = root / "real"
    real.mkdir()
    blob = real / "a.gguf"
    blob.write_text("x")
    link = root / "vendor-alias"
    link.symlink_to(real, target_is_directory=True)
    written = entry("via-link", link / "a.gguf")
    return {"models": [written]}, [entry("a", blob)], [written]


def _dedup_two_discoveries(root):
    # Dedup applies within the batch, not only against models.toml:
    # scan_directory follows symlinks, so one file arrives twice.
    blob = root / "same.gguf"
    first = entry("a", blob)
    return {"models": []}, [first, entry("b", blob)], [first]


@pytest.mark.unit
class TestMergeDiscovered:
    def test_unrepresented_model_is_appended(self, store):
        cfg = {"models": [entry("configured", store / "a.gguf")]}
        merged = merge_discovered(cfg, [entry("found", store / "b.gguf")])
        assert [m["id"] for m in merged["models"]] == ["configured", "found"]

    @pytest.mark.parametrize("setup", [
        pytest.param(_dedup_written_entry, id="explicit_entry_wins_and_is_not_duplicated"),
        pytest.param(_dedup_symlink, id="symlinked_spelling_counts_as_the_same_file"),
        pytest.param(_dedup_two_discoveries, id="two_discoveries_of_one_file_are_served_once"),
    ])
    def test_one_file_is_served_once_under_the_first_spelling(self, tmp_path, setup):
        """The same file reached by the written entry, a symlink, or two
        discoveries is served once, under the written (or first) id and values."""
        cfg, discovered, expected = setup(tmp_path)
        assert merge_discovered(cfg, discovered)["models"] == expected

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

        cfg = {"scan": {"folders": ["/nope"]}}
        merged = merge_discovered(cfg, [])
        assert "models" in merged
        AppConfig(**merged)  # must not raise

@pytest.mark.unit
class TestDiscoverIsBestEffort:
    def test_no_scan_section_discovers_nothing(self):
        assert discover({"models": []}) == []

    def test_one_bad_folder_costs_only_itself_and_is_named(
            self, tmp_path, monkeypatch):
        """Per-source isolation: a folder that raises and a folder that is not
        there each cost only themselves, and each is named as failed -- a
        missing folder must not read as "no models here"."""
        import heylook_llm.model_importer as mi
        good = entry("survivor", tmp_path / "ok.gguf")
        for d in ("broken", "healthy"):
            (tmp_path / d).mkdir()

        def selective(self, path):
            if "broken" in str(path):
                raise OSError("disk gone")
            return [dict(good)]

        monkeypatch.setattr(mi.ModelImporter, "scan_directory", selective)
        folders = [str(tmp_path / "broken"), str(tmp_path / "healthy"), str(tmp_path / "unmounted")]
        found = scan({"scan": {"folders": folders}})
        assert [e["id"] for e in found.entries] == ["survivor"]
        assert found.failed == [folders[0], folders[2]]

    def test_one_invalid_model_is_rejected_alone(self, tmp_path, monkeypatch):
        """One malformed model directory must not unserve its whole folder,
        which with a single scan folder is every discovered model."""
        import heylook_llm.model_importer as mi
        good = entry("good", tmp_path / "ok.gguf")
        bad = entry("bad", tmp_path / "bad.gguf", not_a_field=1)
        monkeypatch.setattr(mi.ModelImporter, "scan_directory",
                            lambda self, path: self._validate([dict(good), dict(bad)]))
        found = scan({"scan": {"folders": [str(tmp_path)]}})
        assert [e["id"] for e in found.entries] == ["good"]
        assert found.failed == [str(tmp_path / "bad.gguf")]

    def test_interval_zero_disables_discovery(self, monkeypatch):
        """The documented off switch must actually switch discovery off."""
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(
            mi.ModelImporter, "scan_directory",
            lambda self, path: pytest.fail("scanned despite interval 0"))
        assert discover({"scan": {"folders": ["/x"], "scan_interval_seconds": 0}}) == []


@pytest.mark.unit
class TestAnEditWritesTheModelsOwnFile:
    """Plan_registry_sidecars Phase 3: an admin edit to a discovered model
    writes its model.heylook.toml. It used to materialize a models.toml entry
    holding the whole derived config, which froze every derived value."""

    def _service(self, tmp_path, store):
        cfg = tmp_path / "models.toml"
        cfg.write_text(f'[scan]\nfolders = ["{store}"]\n')
        return ModelService(str(cfg)), cfg

    def _stub_scan(self, monkeypatch, entries):
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(mi.ModelImporter, "scan_directory",
                            lambda self, path: [dict(e) for e in entries])

    def test_set_then_reset_writes_only_what_was_set_and_reverts_by_deleting(
            self, tmp_path, store, monkeypatch):
        import tomllib
        blob = store / "found.gguf"
        blob.write_text("x")
        mm = store / "mm.gguf"
        mm.write_text("x")
        svc, cfg = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [dict(entry("found", blob), config={
            "model_path": str(blob), "mmproj_path": str(mm)})])
        before = cfg.read_text()

        updated, reload = svc.update_config("found", {"config": {"ctx_size": 8192}})
        assert updated.config.ctx_size == 8192 and "ctx_size" in reload
        assert updated.config.mmproj_path == str(mm)            # derived value kept
        assert tomllib.loads((store / "model.heylook.toml").read_text()) == {"ctx_size": 8192}
        assert cfg.read_text() == before                         # models.toml untouched

        svc.update_config("found", {"config": {"ctx_size": None}})
        assert not (store / "model.heylook.toml").exists()

    # Browsing the models page must not write anything. The two rows were
    # copies in two classes whose models.toml differed only by a trailing
    # newline; both spellings stay.
    @pytest.mark.parametrize("trailing", [
        pytest.param("\n", id="reading_does_not_write"),
        pytest.param("", id="listing_does_not_write"),
    ])
    def test_reading_writes_nothing(self, tmp_path, store, monkeypatch, trailing):
        blob = store / "found.gguf"
        blob.write_text("x")
        cfg = tmp_path / "models.toml"
        cfg.write_text(f'[scan]\nfolders = ["{store}"]{trailing}')
        svc = ModelService(str(cfg))
        self._stub_scan(monkeypatch, [entry("found", blob)])
        before = cfg.read_text()
        svc.list_configs()
        svc.get_config("found")
        assert cfg.read_text() == before and not (store / "model.heylook.toml").exists()

    def test_unknown_id_and_non_config_keys_are_refused(self, tmp_path, store, monkeypatch):
        blob = store / "found.gguf"
        blob.write_text("x")
        svc, _ = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])
        with pytest.raises(ValueError, match="not found"):
            svc.update_config("ghost", {"config": {"ctx_size": 1}})
        with pytest.raises(ValueError, match="config fields only"):
            svc.update_config("found", {"enabled": False})


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

            [scan]
            folders = ["{store}"]
        """).strip())
        return ModelService(str(cfg))

    def _stub_scan(self, monkeypatch, entries):
        import heylook_llm.model_importer as mi
        monkeypatch.setattr(mi.ModelImporter, "scan_directory",
                            lambda self, path: [dict(e) for e in entries])

    def test_list_configs_is_written_down_only(self, tmp_path, store, monkeypatch):
        """Folding discovery in here would list models the router's snapshot
        cannot load."""
        blob = store / "found.gguf"
        blob.write_text("x")
        svc = self._service(tmp_path, store)
        self._stub_scan(monkeypatch, [entry("found", blob)])

        assert [c.id for c in svc.list_configs()] == []
        assert svc.get_config("found") is None


@pytest.mark.unit
class TestScanConfigAccessors:
    """[scan] is server config, not a UI preference -- it decides what is served."""

    def _service(self, tmp_path, body=""):
        cfg = tmp_path / "models.toml"
        cfg.write_text(textwrap.dedent(f"""
            {body}
        """).strip())
        return ModelService(str(cfg)), cfg

    def test_defaults_when_no_scan_table(self, tmp_path):
        svc, _ = self._service(tmp_path)
        assert svc.get_scan_config() == {
            "folders": [], "scan_interval_seconds": 900}

    def test_partial_update_leaves_other_keys_alone(self, tmp_path):
        svc, _ = self._service(tmp_path, '\n[scan]\nfolders = ["a"]\nscan_interval_seconds = 0\n')
        out = svc.set_scan_config(folders=["b", "c"])
        assert out["folders"] == ["b", "c"]
        assert out["scan_interval_seconds"] == 0, "an absent field must not be reset"

    def test_duplicate_folders_are_dropped_in_order(self, tmp_path):
        svc, _ = self._service(tmp_path)
        # Two spellings of one folder just make discovery walk it twice.
        assert svc.set_scan_config(folders=["a", "b", "a", " ", "b"])["folders"] == ["a", "b"]

    def test_negative_interval_is_refused(self, tmp_path):
        svc, _ = self._service(tmp_path)
        with pytest.raises(ValueError, match="0 disables"):
            svc.set_scan_config(scan_interval_seconds=-1)

    # A comment survives a [scan] edit only while its anchor is untouched:
    # test_toml_comment_preservation.py TestThroughModelService (the scan rows).


@pytest.mark.unit
class TestServedDiff:
    """Phase 0 of plan_registry_sidecars: what an edit does to the served set,
    answered by the router's own merge rather than predicted from the rule."""

    def test_deleting_an_entry_that_reads_redundant_can_lose_the_model(self, store):
        """The twin: two entries claim one file. The plain one matches what
        discovery derives, so it reads as safe to delete -- and deleting it
        unserves that id, because the twin still claims the path and
        discovery therefore adds nothing back."""
        blob = store / "a.gguf"
        found = Discovery([entry("a", blob)], [])
        before = {"models": [entry("a", blob), entry("a-twin", blob)]}
        after = {"models": [entry("a-twin", blob)]}
        d = served_diff((before, found), (after, found))
        assert d.lost == ["a"] and d.gained == [] and d.changed == {}

    def test_an_edit_is_named_field_by_field(self, store):
        blob = store / "a.gguf"
        found = Discovery([entry("a", blob)], [])
        d = served_diff(({"models": []}, found),
                        ({"models": [entry("a", blob, ctx_size=8192)]}, found))
        assert d.changed == {"a": {"ctx_size": [None, 8192]}}
        assert not d.lost and not d.gained

    def test_a_degraded_scan_is_named_not_trusted(self, store):
        """A scan that failed and a mass deletion look identical by count, so
        the diff reports the failed source instead of guessing by magnitude."""
        found = Discovery([entry("a", store / "a.gguf"), entry("b", store / "b.gguf")], [])
        degraded = Discovery([], [str(store)])
        d = served_diff(({"models": []}, found), ({"models": []}, degraded))
        assert d.lost == ["a", "b"] and d.unreliable == [str(store)]

    def test_a_hand_named_entry_deleted_is_a_rename_with_its_fields(self, store):
        """Deleting a hand-named entry hands its file back to discovery under
        the derived id: that is a rename, and what the entry held shows as a
        field change rather than hiding behind lost-and-gained."""
        blob = store / "a.gguf"
        found = Discovery([entry("a", blob)], [])
        d = served_diff(({"models": [entry("my-name", blob, ctx_size=4096)]}, found),
                        ({"models": []}, found))
        assert d.renamed == {"my-name": "a"} and not d.lost and not d.gained
        assert d.changed == {"a": {"ctx_size": [4096, None]}}


@pytest.mark.unit
class TestModelHeylookToml:
    """Plan_registry_sidecars Phase 2: a model's own settings in its own
    folder, layered over what discovery derives."""

    def _model(self, root, name="m", sidecar=None):
        from helpers.gguf import STR, write_gguf
        d = root / name
        d.mkdir(parents=True)
        write_gguf(d / f"{name}.gguf", [("general.architecture", STR, "llama")])
        (d / "mtp-drafter.gguf").write_bytes(b"x")
        if sidecar is not None:
            (d / "model.heylook.toml").write_text(sidecar)
        return d

    def test_values_layer_over_derivation_and_unset_drops_one(self, tmp_path):
        d = self._model(tmp_path, sidecar='ctx_size = 8192\nchat_template_path = "t.jinja"\n'
                                           'unset = ["draft_model_path"]\n')
        found = scan({"scan": {"folders": [str(tmp_path)]}})
        (e,) = found.entries
        assert e["config"]["ctx_size"] == 8192
        assert e["config"]["chat_template_path"] == str(d / "t.jinja")   # relative to the folder
        assert "draft_model_path" not in e["config"]                       # spec decode off
        assert e["derived"]["draft_model_path"] == str(d / "mtp-drafter.gguf")
        assert found.failed == []

    @pytest.mark.parametrize("body", ['model_path = "/elsewhere"\n', 'unset = ["not_a_field"]\n',
                                      'ctx_size = \n', 'not_a_field = 1\n'])
    def test_a_bad_file_rejects_that_model_alone(self, tmp_path, body):
        self._model(tmp_path, "bad", sidecar=body)
        self._model(tmp_path, "good")
        found = scan({"scan": {"folders": [str(tmp_path)]}})
        assert [e["id"] for e in found.entries] == ["good"]
        assert len(found.failed) == 1 and "bad" in found.failed[0]

    def test_the_contract_names_the_file(self, tmp_path):
        from heylook_llm.providers import contract
        from heylook_llm.router import ModelRouter
        self._model(tmp_path, sidecar="ctx_size = 8192\n")
        router = object.__new__(ModelRouter)
        app = router._with_discovered({"models": [], "scan": {"folders": [str(tmp_path)]}})
        setting = contract.describe(app.models[0], router).settings["ctx_size"]
        assert setting.provenance == "configured" and setting.value == 8192
        assert "model.heylook.toml" in setting.reason

    def test_an_admin_edit_keeps_the_files_other_settings(self, tmp_path):
        """An edit rewrites the file with the edited field; what the file
        already held (including `unset`) survives, and models.toml gets no
        entry."""
        import tomllib
        store = tmp_path / "store"
        d = self._model(store, sidecar='ctx_size = 8192\nunset = ["draft_model_path"]\n')
        cfg = tmp_path / "models.toml"
        cfg.write_text(f'[scan]\nfolders = ["{store}"]\n')
        ModelService(str(cfg)).update_config("m", {"config": {"n_ubatch": 512}})
        assert tomllib.loads((d / "model.heylook.toml").read_text()) == {
            "ctx_size": 8192, "n_ubatch": 512, "unset": ["draft_model_path"]}
        assert "[[models]]" not in cfg.read_text()


@pytest.mark.unit
def test_a_read_only_instance_does_not_write_models_toml(tmp_path, monkeypatch):
    from heylook_llm.model_registry import READONLY_ENV, ModelConfigReadOnly
    cfg = tmp_path / "models.toml"
    cfg.write_text('[scan]\nfolders = ["a"]\n')
    monkeypatch.setenv(READONLY_ENV, "1")
    with pytest.raises(ModelConfigReadOnly):
        ModelService(str(cfg)).set_scan_config(folders=["a", "b"])
    assert cfg.read_text() == '[scan]\nfolders = ["a"]\n'


@pytest.mark.unit
class TestSpeculativeReport:
    """engine.speculative says what a gguf model drafts with, where discovery
    found it, and whether the running process drafts -- the "available, not in
    use" case the fit and load fallbacks create has to be visible."""

    def test_static_half_names_the_drafter_and_where_it_came_from(self, tmp_path):
        from heylook_llm.providers.gguf_describe import _speculative
        model = tmp_path / "vendor-a_Model" / "m.gguf"
        drafter = tmp_path / "vendor-b_Model" / "dspark-m.gguf"
        cfg = {"model_path": str(model), "draft_model_path": str(drafter)}
        slot = _speculative(cfg, {}, False)
        assert slot["drafter"].value == "dspark-m.gguf"
        assert "neighbouring folder vendor-b_Model/" in slot["drafter"].source
        assert str(tmp_path) not in slot["drafter"].source            # never a path

        builtin = _speculative({"model_path": str(model), "spec_type": "draft-mtp"}, {}, False)
        assert builtin["drafter"].value == "built-in MTP head"

        off = _speculative({"model_path": str(model)}, dict(cfg), "model.heylook.toml")
        assert off["drafter"].value is None and "turned off" in off["drafter"].source
        assert off["drafter"].provenance == "configured"

    def test_observed_half_says_why_a_found_drafter_is_not_in_force(self):
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider
        p = LlamaServerProvider("m", {"model_path": "/fake/m.gguf"}, False)
        assert p._spec_in_force().provenance == "unknown"             # not loaded
        p._proc = object()
        p.drafter_skipped = "short by 7.4 GiB"
        fact = p._spec_in_force()
        assert fact.value is False and fact.source == "short by 7.4 GiB"
        p.config["spec_type"] = "draft-mtp"
        assert p._spec_in_force().value is True
        p._proc = None
