# tests/unit/test_gguf_import.py
"""GGUF model-import support (plan Phase 7e).

The scan previously only knew MLX/safetensors layouts. It must now:
- recognize a GGUF model directory (a primary <name>.gguf, optional mmproj/
  mtp sidecars, optional imatrix_*.gguf_file calibration data that must
  never be mistaken for a model or sidecar)
- recognize + SKIP an HF-format ASSISTANT/drafter source checkpoint
  (config.json "architectures" containing "...Assistant...") -- these are
  not servable models and previously would have been misdetected as MLX
  (config.json + model.safetensors is the same shape).
- keep reporting accurate directory sizes for GGUF dirs (previously 0 GB,
  since size was only ever summed over *.safetensors).
- keep the CLI TOML generator producing entries that validate through
  ModelConfig/AppConfig for the new provider.

All fixtures use tmp_path with tiny fake files -- never modelzoo/ (real
GGUF files there are multi-GB and gitignored).
"""
import json

import pytest
import tomllib

from heylook_llm.config import AppConfig, ModelConfig
from heylook_llm.model_importer import ModelImporter
from heylook_llm.model_service import ModelService


def _write_bytes(path, n: int):
    path.write_bytes(b"\x00" * n)


@pytest.fixture
def importer():
    return ModelImporter()


# ---------------------------------------------------------------------------
# Detection: GGUF directories
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsGGUFModel:
    def test_primary_gguf_detected(self, importer, tmp_path):
        # If this regresses, a plain GGUF model dir (just the primary file)
        # stops being recognized as importable at all.
        _write_bytes(tmp_path / "model-a.gguf", 1000)
        assert importer._is_gguf_model(tmp_path)

    def test_mmproj_only_is_not_primary(self, importer, tmp_path):
        # A dir with ONLY an mmproj sidecar (no primary weight) must not be
        # treated as a servable GGUF model -- mmproj is a companion file.
        _write_bytes(tmp_path / "mmproj-F16.gguf", 1000)
        assert not importer._is_gguf_model(tmp_path)

    def test_mtp_only_is_not_primary(self, importer, tmp_path):
        # Same reasoning for a bare drafter sidecar with no primary model.
        _write_bytes(tmp_path / "mtp-model-a.gguf", 1000)
        assert not importer._is_gguf_model(tmp_path)

    def test_imatrix_gguf_file_extension_is_not_gguf(self, importer, tmp_path):
        # imatrix calibration data uses ".gguf_file", NOT ".gguf" -- if this
        # ever matched, calibration blobs would be misdetected as models.
        _write_bytes(tmp_path / "imatrix_unsloth.gguf_file", 1000)
        assert not importer._is_gguf_model(tmp_path)

    def test_non_gguf_dir_not_detected(self, importer, tmp_path):
        (tmp_path / "readme.txt").write_text("hi")
        assert not importer._is_gguf_model(tmp_path)


# ---------------------------------------------------------------------------
# Detection: HF-format assistant/drafter checkpoints must be skipped
# ---------------------------------------------------------------------------


def _assistant_dir(tmp_path, architecture: str):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": "gemma4"})
    )
    _write_bytes(tmp_path / "model.safetensors", 1000)
    return tmp_path


@pytest.mark.unit
class TestAssistantCheckpointSkipped:
    @pytest.mark.parametrize(
        "architecture",
        ["Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM"],
    )
    def test_assistant_checkpoint_not_imported(self, importer, tmp_path, architecture):
        # If this regresses, drafter SOURCE checkpoints (config.json +
        # safetensors, same shape as a real MLX model) get imported as
        # bogus, un-servable "mlx" entries.
        d = _assistant_dir(tmp_path, architecture)
        found = importer.scan_directory(str(tmp_path.parent))
        ids = [m["id"] for m in found]
        assert d.name not in ids

    def test_regular_mlx_dir_with_safetensors_still_detected(self, importer, tmp_path):
        # Regression guard: the assistant-checkpoint refusal must not
        # collaterally swallow legitimate MLX checkpoints.
        d = tmp_path / "real-mlx-model"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
        )
        _write_bytes(d / "model.safetensors", 1000)
        found = importer.scan_directory(str(tmp_path))
        ids = [m["id"] for m in found]
        assert "real-mlx-model" in ids
        assert next(m for m in found if m["id"] == "real-mlx-model")["provider"] == "mlx"


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def _make_gguf_dir(
    tmp_path, name="my-gguf-model", *, mmproj=None, mtp=None, mtp_subdir=None,
    imatrix=False, primary_size=10_000,
):
    d = tmp_path / name
    d.mkdir()
    _write_bytes(d / f"{name}.gguf", primary_size)  # primary -- largest file
    if mmproj:
        for fname in mmproj:
            _write_bytes(d / fname, 1_000)
    if mtp:
        _write_bytes(d / mtp, 2_000)
    if mtp_subdir:
        sub = d / "MTP"
        sub.mkdir()
        for fname in mtp_subdir:
            _write_bytes(sub / fname, 500)
    if imatrix:
        _write_bytes(d / "imatrix_unsloth.gguf_file", 3_000)
    return d


@pytest.mark.unit
class TestCreateGGUFEntry:
    def test_basic_entry_fields(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path)
        entry = importer._create_gguf_entry(d)
        assert entry is not None
        assert entry["id"] == "my-gguf-model"
        assert entry["provider"] == "gguf"
        assert entry["config"]["model_path"] == str(d / "my-gguf-model.gguf")
        # Derive-at-load (6a): no auto description/tags materialized.
        assert "tags" not in entry
        assert "description" not in entry

    def test_spec_type_deliberately_unset(self, importer, tmp_path):
        # spec_type turns speculative decoding ON at request time; whether it
        # actually helps is measured per-model, so import only pairs the
        # drafter PATH -- it must never auto-enable spec decode.
        d = _make_gguf_dir(tmp_path, mtp="mtp-my-gguf-model.gguf")
        entry = importer._create_gguf_entry(d)
        assert "spec_type" not in entry["config"]

    def test_mtp_root_level_paired_as_draft(self, importer, tmp_path):
        d = _make_gguf_dir(
            tmp_path,
            mtp="mtp-my-gguf-model.gguf",
            mtp_subdir=["mtp-my-gguf-model-F16.gguf", "mtp-my-gguf-model-BF16.gguf"],
        )
        entry = importer._create_gguf_entry(d)
        # Only the ROOT-level mtp file is used, never an MTP/ subdir variant.
        assert entry["config"]["draft_model_path"] == str(d / "mtp-my-gguf-model.gguf")

    def test_no_draft_when_no_mtp_sidecar(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path)
        entry = importer._create_gguf_entry(d)
        assert "draft_model_path" not in entry["config"]

    @pytest.mark.parametrize("prefix", ["mtp-", "dspark-", "dflash-", "eagle3-"])
    def test_every_drafter_family_is_paired(self, importer, tmp_path, prefix):
        # llama.cpp resolves drafter siblings by one prefix per speculative
        # family (common/download.cpp). Knowing only "mtp-" left DeepSeek-V4's
        # `dspark-*.gguf` unpaired -- and, being neither mmproj nor mtp, it
        # also polluted the PRIMARY candidate set.
        d = _make_gguf_dir(tmp_path, name=f"m-{prefix.strip('-')}")
        drafter = d / f"{prefix}drafter.gguf"
        _write_bytes(drafter, 2_000)
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["draft_model_path"] == str(drafter)
        assert entry["config"]["model_path"] == str(d / f"{d.name}.gguf")

    def test_sharded_model_loads_at_first_shard(self, importer, tmp_path):
        # llama_model_loader hard-errors ("model must be loaded with the first
        # split") on any shard but 00001, because it derives its siblings from
        # the given file's own split index. Picking the LARGEST shard -- which
        # a plain size-max does, since shard 1 is a tiny index shard -- makes
        # every multi-shard GGUF unloadable.
        d = tmp_path / "sharded"
        d.mkdir()
        _write_bytes(d / "sharded-00001-of-00003.gguf", 100)
        _write_bytes(d / "sharded-00002-of-00003.gguf", 90_000)
        _write_bytes(d / "sharded-00003-of-00003.gguf", 80_000)
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["model_path"] == str(d / "sharded-00001-of-00003.gguf")

    def test_sharded_set_outweighs_a_standalone_sibling(self, importer, tmp_path):
        # The first shard is a few MB; the SET is the servable weight. Sizing
        # the candidate by its own bytes would hand the entry to any standalone
        # .gguf sitting beside a much larger sharded model.
        d = tmp_path / "mixed"
        d.mkdir()
        _write_bytes(d / "big-00001-of-00002.gguf", 100)
        _write_bytes(d / "big-00002-of-00002.gguf", 90_000)
        _write_bytes(d / "small-standalone.gguf", 50_000)
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["model_path"] == str(d / "big-00001-of-00002.gguf")

    def test_mmproj_preference_f16_over_bf16_and_f32(self, importer, tmp_path):
        d = _make_gguf_dir(
            tmp_path, mmproj=["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"]
        )
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["mmproj_path"] == str(d / "mmproj-F16.gguf")
        assert "vision" in entry["config"]["modalities"]

    def test_mmproj_preference_bf16_over_f32_when_no_f16(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-BF16.gguf", "mmproj-F32.gguf"])
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["mmproj_path"] == str(d / "mmproj-BF16.gguf")

    def test_mmproj_arbitrary_name_used_when_no_known_precision(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-custom.gguf"])
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["mmproj_path"] == str(d / "mmproj-custom.gguf")

    def test_no_mmproj_means_text_only_modality(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path)
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["modalities"] == ["text"]
        assert "mmproj_path" not in entry["config"]

    def test_imatrix_file_never_treated_as_model_or_sidecar(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"], imatrix=True)
        entry = importer._create_gguf_entry(d)
        assert entry["config"]["model_path"] == str(d / "my-gguf-model.gguf")
        assert entry["config"]["mmproj_path"] == str(d / "mmproj-F16.gguf")

    def test_scan_directory_finds_gguf_dir(self, importer, tmp_path):
        _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"])
        found = importer.scan_directory(str(tmp_path))
        gguf_entries = [m for m in found if m["provider"] == "gguf"]
        assert len(gguf_entries) == 1
        assert gguf_entries[0]["id"] == "my-gguf-model"


# ---------------------------------------------------------------------------
# Regression: plain MLX dirs must still be detected as mlx (not swallowed by
# the new gguf/assistant-checkpoint branches)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMLXDetectionRegression:
    def test_mlx_dir_still_detected(self, importer, tmp_path):
        d = tmp_path / "plain-mlx-model"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
        _write_bytes(d / "model.safetensors", 5_000)
        found = importer.scan_directory(str(tmp_path))
        assert len(found) == 1
        assert found[0]["provider"] == "mlx"
        assert found[0]["id"] == "plain-mlx-model"


# ---------------------------------------------------------------------------
# Size computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGGUFSizeComputation:
    def test_weights_size_sums_gguf_bytes(self, tmp_path):
        # The one shared byte-summer (cache_defaults.weights_size_gb; the
        # importer's private copy died in the /simplify dedup) must count
        # *.gguf, not just *.safetensors -- a GGUF dir must never report 0 GB.
        from heylook_llm.cache_defaults import weights_size_gb

        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"])
        assert weights_size_gb(str(d)) > 0

    def test_scanned_model_size_gb_nonzero_for_gguf(self, tmp_path):
        config_path = tmp_path / "models.toml"
        config_path.write_text('default_model = "none"\nmax_loaded_models = 1\n')
        service = ModelService(str(config_path))

        gguf_dir = tmp_path / "scan-root"
        gguf_dir.mkdir()
        # Big enough that ScannedModel.size_gb (rounded to 2 decimals) is
        # distinguishably nonzero -- a KB-scale fixture rounds to 0.0 and
        # would pass even with the pre-fix always-0 behavior.
        _make_gguf_dir(gguf_dir, mmproj=["mmproj-F16.gguf"], primary_size=10_000_000)

        results = service.scan_directory(str(gguf_dir))
        gguf_results = [r for r in results if r.provider == "gguf"]
        assert len(gguf_results) == 1
        assert gguf_results[0].size_gb > 0


# ---------------------------------------------------------------------------
# TOML round-trip validity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGGUFTomlRoundTrip:
    def test_generated_gguf_entry_validates_through_model_config(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"], mtp="mtp-my-gguf-model.gguf")
        entry = importer._create_gguf_entry(d)
        # Round-trips through ModelConfig directly (no TOML dump needed to
        # catch a schema mismatch, e.g. an "extra field forbidden" error).
        validated = ModelConfig(**entry)
        assert validated.provider == "gguf"
        assert validated.config.mmproj_path == str(d / "mmproj-F16.gguf")

    def test_generated_toml_round_trips_through_app_config(self, importer, tmp_path):
        # If generate_toml ever emits a field GGUFModelConfig forbids (its
        # model_config is extra="forbid"), this fails loudly instead of
        # producing a models.toml that crashes on server startup.
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"])
        entry = importer._create_gguf_entry(d)
        toml_text = importer.generate_toml([entry])
        parsed = tomllib.loads(toml_text)
        app_config = AppConfig(**parsed)
        assert app_config.models[0].provider == "gguf"

    def test_gguf_section_header_present_when_generator_groups_by_section(self, importer, tmp_path):
        d = _make_gguf_dir(tmp_path, mmproj=["mmproj-F16.gguf"])
        entry = importer._create_gguf_entry(d)
        toml_text = importer.generate_toml([entry])
        assert "GGUF" in toml_text


# ---------------------------------------------------------------------------
# model_service.py provider gating must not crash on gguf entries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelServiceGGUFNoCrash:
    def _service(self, tmp_path):
        config_path = tmp_path / "models.toml"
        config_path.write_text('default_model = "none"\nmax_loaded_models = 1\n')
        return ModelService(str(config_path))

    def test_validate_config_accepts_gguf_provider(self, tmp_path):
        # If "gguf" regresses out of this allowlist, validate_config reports
        # every gguf entry as an "Unknown provider" error.
        service = self._service(tmp_path)
        gguf_file = tmp_path / "model.gguf"
        _write_bytes(gguf_file, 1000)
        result = service.validate_config(
            {
                "id": "gguf-model",
                "provider": "gguf",
                "config": {"model_path": str(gguf_file)},
            }
        )
        assert "Unknown provider: gguf" not in result.errors

    def test_import_models_does_not_crash_on_gguf(self, tmp_path):
        # import_models() used to build every non-embedding provider's
        # config with mlx-only fields ("vision", cache_type/kv_bits smart
        # defaults) -- GGUFModelConfig forbids extra fields, so a gguf
        # import silently failed validation and was dropped.
        service = self._service(tmp_path)
        gguf_file = tmp_path / "model.gguf"
        _write_bytes(gguf_file, 1000)

        imported = service.import_models(
            [
                {
                    "id": "gguf-model",
                    "provider": "gguf",
                    "path": str(gguf_file),
                    "config": {"model_path": str(gguf_file), "mmproj_path": None},
                }
            ]
        )
        assert len(imported) == 1, "gguf import was silently dropped (validation failure)"
        assert imported[0].provider == "gguf"

    def test_bulk_set_default_sampler_skips_gguf_gracefully(self, tmp_path):
        service = self._service(tmp_path)
        gguf_file = tmp_path / "model.gguf"
        _write_bytes(gguf_file, 1000)
        service.import_models(
            [{"id": "gguf-model", "provider": "gguf", "config": {"model_path": str(gguf_file)}}]
        )
        # Must not raise even though this provider isn't "mlx".
        updated = service.bulk_set_default_sampler(["gguf-model"], "balanced")
        assert len(updated) == 1


def test_mmproj_suffix_naming_detected(tmp_path):
    # Deleting this re-breaks google-style projector names (<model>-mmproj.gguf,
    # suffix not prefix) -- the entry silently imports text-only and the real
    # modelzoo google E4B dir loses vision (caught by live scan 2026-07-26).
    d = tmp_path / "google_style"
    d.mkdir()
    (d / "gemma-x-q4_0.gguf").write_bytes(b"0" * 1000)
    (d / "gemma-x-it-mmproj.gguf").write_bytes(b"0" * 100)
    from heylook_llm.model_importer import ModelImporter

    imp = ModelImporter()
    entry = imp._create_gguf_entry(d)
    assert entry is not None
    assert entry["config"]["model_path"].endswith("gemma-x-q4_0.gguf")
    assert entry["config"]["mmproj_path"].endswith("gemma-x-it-mmproj.gguf")
    assert "vision" in entry["config"]["modalities"]
