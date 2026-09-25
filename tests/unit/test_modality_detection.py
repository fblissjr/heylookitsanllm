# tests/unit/test_modality_detection.py
"""Unit tests for modality detection (Phase 6 refinement 2026-07-11).

``detect_modalities`` reads a model dir's declared capabilities from the
config's OWN blocks (``vision_config``/``audio_config`` + ``*_token_id`` keys --
the author's ground truth), falling back to weight/processor filenames for
sparse checkpoints. It is pure description (no library dependency): loader
routing (mlx-vlm vs mlx-lm) is resolved separately in the provider.

Download-free: fake config.json dicts on tmp_path.
"""
import json

import pytest

from heylook_llm.model_importer import ModelImporter


def _write(tmp_path, config: dict | None, *, files: list[str] | None = None):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    for name in files or []:
        (tmp_path / name).write_text("x")
    return tmp_path


@pytest.fixture
def importer():
    return ModelImporter()


@pytest.mark.unit
class TestDetectModalities:
    @pytest.mark.parametrize("config, files, expected", [
        pytest.param({"model_type": "llama"}, [], ["text"], id="text_only"),
        pytest.param({"model_type": "qwen3_5", "vision_config": {"depth": 32}}, [],
                     ["text", "vision"], id="vision_config_block"),
        # No vision_config, but the model routes image tokens -> still vision.
        pytest.param({"model_type": "x", "image_token_id": 12345}, [],
                     ["text", "vision"], id="image_token_id_signal"),
        # LLaVA/Mistral/Pixtral spell it image_token_INDEX; a stripped/converted
        # checkpoint may carry it without a vision_config block (found on
        # soundTeam/MS3.2-24b-Angel in the local model audit).
        pytest.param({"model_type": "llava", "image_token_index": 32000}, [],
                     ["text", "vision"], id="image_token_index_signal"),
        pytest.param({"model_type": "x", "audio_config": {"n_mels": 128}}, [],
                     ["text", "audio"], id="audio_config_block"),
        # gemma-4 shape: declares text + vision + audio.
        pytest.param({"model_type": "gemma4", "vision_config": {}, "audio_config": {},
                      "image_token_id": 1, "audio_token_id": 2}, [],
                     ["text", "vision", "audio"], id="vision_and_audio"),
        pytest.param({"model_type": "x", "vision_config": {}, "video_token_id": 9}, [],
                     ["text", "vision", "video"], id="video_signal"),
        # A draft/MTP head or a sparse dir with no config.json must never
        # crash -- default to text.
        pytest.param(None, [], ["text"], id="missing_config_is_text_only"),
        # Sparse checkpoint (GGUF/split) with no vision_config but an mmproj file.
        pytest.param({"model_type": "x"}, ["mmproj-model-f16.gguf"],
                     ["text", "vision"], id="vision_weight_file_fallback"),
    ])
    def test_detect_modalities(self, importer, tmp_path, config, files, expected):
        _write(tmp_path, config, files=files)
        assert importer.detect_modalities(tmp_path) == expected
