# tests/unit/test_cache_defaults.py
"""weights_size_gb: the byte-summing size probe model_service uses. (The
RAM-relative MLX KV-cache defaults that shared this module were retired with
the mlx-vlm engine, plan W10 stage 3.)"""

import pytest

from heylook_llm.cache_defaults import weights_size_gb


@pytest.mark.unit
class TestWeightsSize:
    def test_sums_safetensors_and_gguf(self, tmp_path):
        (tmp_path / "model-00001.safetensors").write_bytes(b"\x00" * 2048)
        (tmp_path / "model-00002.safetensors").write_bytes(b"\x00" * 2048)
        assert weights_size_gb(str(tmp_path)) == pytest.approx(4096 / 1024**3)

    def test_missing_dir_is_zero(self):
        assert weights_size_gb("/nonexistent/nowhere") == 0.0
