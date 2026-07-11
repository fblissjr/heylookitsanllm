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
    def test_text_only(self, importer, tmp_path):
        _write(tmp_path, {"model_type": "llama"})
        assert importer.detect_modalities(tmp_path) == ["text"]

    def test_vision_config_block(self, importer, tmp_path):
        _write(tmp_path, {"model_type": "qwen3_5", "vision_config": {"depth": 32}})
        assert importer.detect_modalities(tmp_path) == ["text", "vision"]

    def test_image_token_id_signal(self, importer, tmp_path):
        # No vision_config, but the model routes image tokens -> still vision.
        _write(tmp_path, {"model_type": "x", "image_token_id": 12345})
        assert importer.detect_modalities(tmp_path) == ["text", "vision"]

    def test_audio_config_block(self, importer, tmp_path):
        _write(tmp_path, {"model_type": "x", "audio_config": {"n_mels": 128}})
        assert importer.detect_modalities(tmp_path) == ["text", "audio"]

    def test_vision_and_audio(self, importer, tmp_path):
        # gemma-4 shape: declares text + vision + audio.
        _write(tmp_path, {"model_type": "gemma4", "vision_config": {},
                          "audio_config": {}, "image_token_id": 1, "audio_token_id": 2})
        assert importer.detect_modalities(tmp_path) == ["text", "vision", "audio"]

    def test_video_signal(self, importer, tmp_path):
        _write(tmp_path, {"model_type": "x", "vision_config": {}, "video_token_id": 9})
        mods = importer.detect_modalities(tmp_path)
        assert "video" in mods and mods[0] == "text"

    def test_missing_config_is_text_only(self, importer, tmp_path):
        # Robustness: a draft/MTP head or a sparse dir with no config.json must
        # never crash -- default to text.
        assert importer.detect_modalities(tmp_path) == ["text"]

    def test_vision_weight_file_fallback(self, importer, tmp_path):
        # Sparse checkpoint (GGUF/split) with no vision_config but an mmproj file.
        _write(tmp_path, {"model_type": "x"}, files=["mmproj-model-f16.gguf"])
        assert importer.detect_modalities(tmp_path) == ["text", "vision"]

    def test_is_vision_model_delegates(self, importer, tmp_path):
        _write(tmp_path, {"model_type": "x", "audio_config": {}})
        # audio-only multimodal must NOT read as a vision model.
        assert importer._is_vision_model(tmp_path) is False
        _write(tmp_path, {"model_type": "x", "vision_config": {}})
        assert importer._is_vision_model(tmp_path) is True
