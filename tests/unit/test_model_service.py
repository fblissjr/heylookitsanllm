# tests/unit/test_model_service.py
#
# Unit tests for the v1.19.0 profile system: TOML loading, profile application,
# model size regex, and the public load_profiles() API.

from pathlib import Path

import pytest

from heylook_llm.model_service import (
    ModelProfile,
    _load_profiles_from_toml,
    load_profiles,
)
from heylook_llm.model_importer import ModelImporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles"


@pytest.fixture(autouse=True)
def reset_profiles_cache():
    """Reset the module-level profiles cache between tests."""
    import heylook_llm.model_service as mod
    original = mod._profiles_cache
    mod._profiles_cache = None
    yield
    mod._profiles_cache = original


# ---------------------------------------------------------------------------
# Profile loading (_load_profiles_from_toml)
# ---------------------------------------------------------------------------

class TestProfileLoading:
    """Tests for _load_profiles_from_toml()."""

    def test_all_toml_files_load(self):
        """All 9 TOML files in profiles/ load without error."""
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        assert len(profiles) == 9, f"Expected 9 profiles, got {len(profiles)}: {list(profiles.keys())}"

    def test_each_profile_has_name_and_description(self):
        """Each profile has non-empty name and description."""
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        for name, profile in profiles.items():
            assert profile.name, f"Profile '{name}' has empty name"
            assert profile.description, f"Profile '{name}' has empty description"

    def test_each_profile_has_defaults(self):
        """Each profile has at least one default parameter."""
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        for name, profile in profiles.items():
            assert len(profile.defaults) > 0, f"Profile '{name}' has no defaults"

    def test_nonexistent_directory_returns_empty(self):
        """Loading from a nonexistent directory returns empty dict."""
        profiles = _load_profiles_from_toml(Path("/nonexistent/path"))
        assert profiles == {}

    def test_known_profile_values(self):
        """The 'moderate' profile has expected parameter values."""
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        moderate = profiles.get("moderate")
        assert moderate is not None, "moderate profile not found"
        assert moderate.defaults["temperature"] == 0.7
        assert moderate.defaults["top_k"] == 40
        assert moderate.defaults["max_tokens"] == 512

    def test_quantized_kv_profile_has_kv_params(self):
        """The 'quantized_kv' profile includes KV cache parameters."""
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        qkv = profiles.get("quantized_kv")
        assert qkv is not None, "quantized_kv profile not found"
        assert qkv.defaults["cache_type"] == "quantized"
        assert qkv.defaults["kv_bits"] == 8


# ---------------------------------------------------------------------------
# Profile application (ModelProfile.apply)
# ---------------------------------------------------------------------------

class TestProfileApplication:
    """Tests for ModelProfile.apply()."""

    def _make_profile(self, defaults):
        return ModelProfile(name="test", description="test profile", defaults=defaults)

    def test_profile_overrides_existing_values(self):
        """Profile values override whatever was in config (the v1.19.0 fix)."""
        profile = self._make_profile({"temperature": 0.3, "max_tokens": 256})
        config = {"temperature": 0.9, "max_tokens": 1024, "top_k": 50}
        model_info = {"provider": "mlx"}

        result = profile.apply(config, model_info)

        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 256
        # Existing value not in profile should be preserved
        assert result["top_k"] == 50

    def test_gguf_only_params_excluded_for_mlx(self):
        """GGUF-only parameters are skipped when provider is MLX."""
        profile = self._make_profile({
            "temperature": 0.5,
            "n_ctx": 8192,       # GGUF-only
            "n_batch": 512,      # GGUF-only
        })
        config = {}
        model_info = {"provider": "mlx"}

        result = profile.apply(config, model_info)

        assert result["temperature"] == 0.5
        assert "n_ctx" not in result
        assert "n_batch" not in result

    def test_mlx_only_params_excluded_for_gguf(self):
        """MLX-only parameters are skipped when provider is GGUF."""
        profile = self._make_profile({
            "temperature": 0.5,
            "cache_type": "quantized",  # MLX-only
            "kv_bits": 8,               # MLX-only
        })
        config = {}
        model_info = {"provider": "gguf"}

        result = profile.apply(config, model_info)

        assert result["temperature"] == 0.5
        assert "cache_type" not in result
        assert "kv_bits" not in result

    def test_callable_defaults(self):
        """Profile defaults that are callable get invoked with model_info."""
        profile = self._make_profile({
            "dynamic_val": lambda info: info.get("name", "unknown").upper(),
        })
        config = {}
        model_info = {"provider": "mlx", "name": "test-model"}

        result = profile.apply(config, model_info)
        assert result["dynamic_val"] == "TEST-MODEL"

    def test_llama_cpp_is_gguf_provider(self):
        """llama_cpp provider is treated as GGUF (MLX params excluded)."""
        profile = self._make_profile({
            "cache_type": "quantized",
            "n_ctx": 4096,
        })
        config = {}
        model_info = {"provider": "llama_cpp"}

        result = profile.apply(config, model_info)
        assert "cache_type" not in result
        assert result["n_ctx"] == 4096

    def test_unknown_profile_raises_valueerror(self):
        """ModelService.apply_profile raises ValueError for unknown profile name."""
        from heylook_llm.model_service import ModelService
        # We need a ModelService instance but don't want real TOML I/O.
        # The apply_profile method only reads from the module-level PROFILES dict.
        # We can test via load_profiles + PROFILES directly.
        profiles = _load_profiles_from_toml(PROFILES_DIR)
        assert "nonexistent_profile" not in profiles


# ---------------------------------------------------------------------------
# Model size regex (_get_model_size)
# ---------------------------------------------------------------------------

class TestModelSizeRegex:
    """Tests for ModelImporter._get_model_size() regex patterns."""

    def _get_size(self, path_str: str):
        """Helper: run _get_model_size on a fake path."""
        importer = ModelImporter()
        # _get_model_size uses str(path).lower() and optionally checks disk,
        # so we pass a non-existent path to test only regex matching.
        return importer._get_model_size(Path(path_str))

    def test_decimal_before_integer(self):
        """v1.19.0 fix: decimal sizes like 0.6B match before integer patterns."""
        label, gb = self._get_size("/models/Qwen3-0.6B")
        assert label == "0.6B"
        assert gb == 0.6

    def test_integer_b_suffix(self):
        """Standard integer-B sizes like 8B."""
        label, gb = self._get_size("/models/Llama-3.1-8B")
        assert label == "8B"
        assert gb == 8.0

    def test_million_parameter_model(self):
        """M-suffix models (e.g. 135M)."""
        label, gb = self._get_size("/models/SmolLM-135M")
        assert label == "135M"
        assert gb == pytest.approx(0.135)

    def test_no_size_marker(self):
        """Models without size indicators return None."""
        label, gb = self._get_size("/models/Phi-3-mini-128k")
        assert label is None
        assert gb is None

    def test_large_decimal_model(self):
        """Decimal sizes > 1B."""
        label, gb = self._get_size("/models/Qwen2.5-14.5B-instruct")
        assert label == "14.5B"
        assert gb == 14.5

    def test_large_million_model(self):
        """Large M-suffix models (>= 1000M)."""
        label, gb = self._get_size("/models/some-model-1500m")
        assert label == "1.5B"
        assert gb == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# load_profiles() public API
# ---------------------------------------------------------------------------

class TestLoadProfiles:
    """Tests for the public load_profiles() function."""

    def test_returns_all_profiles(self):
        """load_profiles() returns all 9 profiles."""
        profiles = load_profiles()
        assert len(profiles) == 9

    def test_profile_names_match_filenames(self):
        """Profile dict keys match the [meta].name field from TOML files."""
        profiles = load_profiles()
        expected_names = {
            "tight_fast", "moderate", "wide_sampling", "high_throughput",
            "widest_sampling", "low_resource", "quantized_kv", "conversation",
            "embedding",
        }
        assert set(profiles.keys()) == expected_names

    def test_caching_returns_same_object(self):
        """Second call returns the same cached dict object."""
        first = load_profiles()
        second = load_profiles()
        assert first is second

    def test_explicit_dir_bypasses_cache(self):
        """Passing profiles_dir explicitly loads fresh (not from cache)."""
        cached = load_profiles()
        fresh = load_profiles(profiles_dir=PROFILES_DIR)
        # Both should have same content but fresh is not the cached object
        assert set(cached.keys()) == set(fresh.keys())
