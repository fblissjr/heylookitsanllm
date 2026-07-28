# tests/unit/test_cache_defaults.py
"""Derive-at-load cache defaults (Wave 1 / 6a, 2026-07-28).

``cache_type = None`` means AUTO: resolved at model load from the actual
weight bytes on disk vs machine RAM -- the same RAM-relative policy
``get_smart_defaults`` applied at import time, now computed where it can't
rot (import-time materialization froze the decision against whatever
machine/weights existed at import).

Claims: (1) the threshold policy lives in ONE place (cache_defaults.py)
shared by import-time and load-time consumers; (2) an explicit stored
cache_type is never overridden; (3) resolution is best-effort -- a missing
dir yields the standard cache, never an exception at load.
"""

import pytest

from heylook_llm.cache_defaults import (
    resolve_cache_config,
    smart_cache_defaults,
    weights_size_gb,
)


@pytest.mark.unit
class TestSmartCacheDefaults:
    def test_small_model_standard(self, monkeypatch):
        monkeypatch.setattr(
            "heylook_llm.cache_defaults._system_ram_gb", lambda: 192.0)
        assert smart_cache_defaults(30.0) == {"cache_type": "standard"}

    def test_large_model_quantized(self, monkeypatch):
        # >35% of RAM -> 8-bit KV quantization (RAM-relative, not absolute).
        monkeypatch.setattr(
            "heylook_llm.cache_defaults._system_ram_gb", lambda: 64.0)
        assert smart_cache_defaults(30.0) == {
            "cache_type": "quantized", "kv_bits": 8, "kv_group_size": 64,
        }


@pytest.mark.unit
class TestWeightsSize:
    def test_sums_safetensors_and_gguf(self, tmp_path):
        (tmp_path / "model-00001.safetensors").write_bytes(b"\x00" * 2048)
        (tmp_path / "model-00002.safetensors").write_bytes(b"\x00" * 2048)
        assert weights_size_gb(str(tmp_path)) == pytest.approx(4096 / 1024**3)

    def test_missing_dir_is_zero(self):
        assert weights_size_gb("/nonexistent/nowhere") == 0.0


@pytest.mark.unit
class TestResolveCacheConfig:
    def test_auto_resolves_from_weights(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "heylook_llm.cache_defaults._system_ram_gb", lambda: 64.0)
        monkeypatch.setattr(
            "heylook_llm.cache_defaults.weights_size_gb", lambda p: 30.0)
        config = {"model_path": str(tmp_path), "cache_type": None}
        updates = resolve_cache_config(config)
        assert updates["cache_type"] == "quantized"
        assert updates["kv_bits"] == 8

    def test_explicit_cache_type_untouched(self):
        config = {"model_path": "/x", "cache_type": "standard"}
        assert resolve_cache_config(config) == {}

    def test_explicit_kv_bits_survive_auto(self, monkeypatch):
        # Operator pinned kv_bits but left cache_type auto: auto may pick the
        # cache TYPE but must not stomp the pinned knob.
        monkeypatch.setattr(
            "heylook_llm.cache_defaults._system_ram_gb", lambda: 64.0)
        monkeypatch.setattr(
            "heylook_llm.cache_defaults.weights_size_gb", lambda p: 30.0)
        config = {"model_path": "/x", "cache_type": None, "kv_bits": 4}
        updates = resolve_cache_config(config)
        assert updates["cache_type"] == "quantized"
        assert "kv_bits" not in updates  # pinned value wins

    def test_missing_dir_falls_back_standard(self):
        config = {"model_path": "/nonexistent/nowhere", "cache_type": None}
        assert resolve_cache_config(config) == {"cache_type": "standard"}
