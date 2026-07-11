# tests/unit/test_config.py
"""Unit tests for Pydantic config models."""
import pytest
from pydantic import ValidationError

from heylook_llm.config import (
    AppConfig,
    ChatMessage,
    ChatRequest,
    ImageContentPart,
    ImageUrl,
    MLX_RUNTIME_DEFAULT_FIELDS,
    MLXModelConfig,
    ModelConfig,
    TextContentPart,
)


@pytest.mark.unit
class TestChatMessage:
    def test_simple_text_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.thinking is None

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="hi there")
        assert msg.role == "assistant"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_multimodal_content(self):
        parts = [
            TextContentPart(type="text", text="What is this?"),
            ImageContentPart(type="image_url", image_url=ImageUrl(url="https://example.com/img.png")),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_thinking_roundtrip(self):
        msg = ChatMessage(role="assistant", content="answer", thinking="reasoning")
        assert msg.thinking == "reasoning"
        d = msg.model_dump()
        assert d["thinking"] == "reasoning"

    def test_thinking_excluded_when_none(self):
        msg = ChatMessage(role="assistant", content="hi")
        d = msg.model_dump(exclude_none=True)
        assert "thinking" not in d

    def test_optional_fields_default_none(self):
        msg = ChatMessage(role="user", content="test")
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None


@pytest.mark.unit
class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert req.model is None
        assert req.stream is False
        assert len(req.messages) == 1

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            ChatRequest(messages=[])

    def test_all_sampler_params(self):
        req = ChatRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            min_p=0.1,
            repetition_penalty=1.1,
            max_tokens=256,
            seed=42,
        )
        assert req.temperature == 0.5
        assert req.top_k == 40
        assert req.seed == 42

    def test_stream_options(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert req.stream is True
        assert req.stream_options["include_usage"] is True

    def test_logprobs_params(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            logprobs=True,
            top_logprobs=5,
        )
        assert req.logprobs is True
        assert req.top_logprobs == 5

    def test_enable_thinking(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            enable_thinking=True,
        )
        assert req.enable_thinking is True


@pytest.mark.unit
class TestModelConfig:
    def test_mlx_model_config(self):
        mc = ModelConfig(
            id="test-mlx",
            provider="mlx",
            config={"model_path": "/fake/path"},
        )
        assert mc.id == "test-mlx"
        assert mc.provider == "mlx"
        assert isinstance(mc.config, MLXModelConfig)
        assert mc.config.model_path == "/fake/path"

    def test_invalid_config_for_provider_rejected(self):
        """Config missing required fields raises validation error."""
        with pytest.raises(ValueError):
            ModelConfig(
                id="bad",
                provider="mlx",
                config={"bad_field_only": True},
            )

    def test_mlx_config_defaults(self):
        mc = MLXModelConfig(model_path="/fake")
        assert mc.vision is False
        assert mc.enable_thinking is False
        assert mc.cache_type == "standard"
        assert mc.default_hidden_layer == -2
        assert mc.default_max_length == 512

    def test_capabilities_list(self):
        mc = ModelConfig(
            id="test",
            provider="mlx",
            config={"model_path": "/fake"},
            capabilities=["chat", "thinking", "vision"],
        )
        assert "thinking" in mc.capabilities

    def test_enabled_default_true(self):
        mc = ModelConfig(
            id="test",
            provider="mlx",
            config={"model_path": "/fake"},
        )
        assert mc.enabled is True


@pytest.mark.unit
class TestAppConfig:
    def test_get_model_config(self):
        cfg = AppConfig(
            models=[
                ModelConfig(id="m1", provider="mlx", config={"model_path": "/a"}, enabled=True),
                ModelConfig(id="m2", provider="mlx", config={"model_path": "/b"}, enabled=False),
            ]
        )
        assert cfg.get_model_config("m1") is not None
        assert cfg.get_model_config("m2") is None  # disabled
        assert cfg.get_model_config("m3") is None  # missing

    def test_get_enabled_models(self):
        cfg = AppConfig(
            models=[
                ModelConfig(id="m1", provider="mlx", config={"model_path": "/a"}, enabled=True),
                ModelConfig(id="m2", provider="mlx", config={"model_path": "/b"}, enabled=False),
            ]
        )
        enabled = cfg.get_enabled_models()
        assert len(enabled) == 1
        assert enabled[0].id == "m1"

    def test_max_loaded_models_default(self):
        cfg = AppConfig(models=[])
        assert cfg.max_loaded_models == 1


@pytest.mark.unit
class TestMLXRuntimeDefaultFields:
    """Guardrail for the metadata-driven cache/speculative-decoding field set.

    MLX_RUNTIME_DEFAULT_FIELDS is derived from MLXModelConfig via
    ``json_schema_extra={"is_runtime_default": True}``. If someone adds a new
    cache or speculative-decoding field and forgets to annotate it, the
    hardcoded expectation below fails loudly -- which is the point. Update
    this list in the same commit as the field addition.
    """

    # quantized_kv_start was removed 2026-07-06: stored and forwarded but
    # never consumed by _build_cache_config/make_cache (dead config).
    EXPECTED_RUNTIME_DEFAULTS = frozenset({
        "cache_type",
        "kv_bits",
        "kv_group_size",
        "max_kv_size",
        "num_draft_tokens",
        "prefill_step_size",
    })

    def test_derived_set_matches_expected(self):
        assert MLX_RUNTIME_DEFAULT_FIELDS == self.EXPECTED_RUNTIME_DEFAULTS

    def test_every_runtime_default_is_optional(self):
        """Safety: runtime defaults must be omittable so models.toml entries
        that don't set them fall through to mlx-lm's own defaults."""
        for name in MLX_RUNTIME_DEFAULT_FIELDS:
            field = MLXModelConfig.model_fields[name]
            # Either the default is explicit OR the field allows None.
            assert not field.is_required(), (
                f"MLXModelConfig.{name} is marked is_runtime_default but is required; "
                f"that forces every models.toml entry to set it. Add a default."
            )


class TestMLXModelConfigValidation:
    """Config typos and impossible values must fail at load time, not at
    first generation (audit 2026-07-06)."""

    BASE = {"model_path": "/fake/model"}

    def test_unknown_key_rejected(self):
        # extra="forbid": a typo like `temperatue` must not silently vanish.
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, temperatue=0.9)

    def test_kv_bits_must_be_2_4_or_8(self):
        # MLX QuantizedKVCache supports only 2/4/8-bit.
        for bad in (1, 3, 5, 6, 7):
            with pytest.raises(ValidationError):
                MLXModelConfig(**self.BASE, kv_bits=bad)
        for good in (2, 4, 8):
            assert MLXModelConfig(**self.BASE, kv_bits=good).kv_bits == good

    def test_kv_group_size_constrained(self):
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, kv_group_size=48)
        for good in (32, 64, 128):
            assert MLXModelConfig(**self.BASE, kv_group_size=good).kv_group_size == good

    def test_rotating_cache_requires_max_kv_size(self):
        # Previously validated fine and raised at first generation
        # (cache_helpers.make_cache).
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, cache_type="rotating")
        cfg = MLXModelConfig(**self.BASE, cache_type="rotating", max_kv_size=4096)
        assert cfg.max_kv_size == 4096

    def test_max_queue_depth_is_a_real_field(self):
        # The provider reads config["max_queue_depth"]; without a field the
        # value was silently dropped by pydantic and unreachable.
        assert MLXModelConfig(**self.BASE).max_queue_depth == 8
        assert MLXModelConfig(**self.BASE, max_queue_depth=2).max_queue_depth == 2

    def test_quantized_kv_start_removed(self):
        # Dead config: stored and forwarded but never consumed by
        # _build_cache_config/make_cache. Removed outright.
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, quantized_kv_start=1024)


@pytest.mark.unit
class TestModalitiesAndLoader:
    """modalities/loader split (Phase 6 refinement 2026-07-11).

    ``vision: bool`` used to do two jobs -- DESCRIBE the model (has a vision
    tower) and SELECT the loader (mlx-vlm vs mlx-lm). ``modalities`` (list) is
    the description; ``loader`` (routing) is separate. ``vision`` is retained but
    demoted to a derived mirror of ``"vision" in modalities`` for back-compat.
    """

    BASE = {"model_path": "/fake/model"}

    def test_defaults_are_text_only(self):
        cfg = MLXModelConfig(**self.BASE)
        assert cfg.modalities == ["text"]
        assert cfg.vision is False
        assert cfg.loader == "auto"

    def test_legacy_vision_true_derives_modalities(self):
        # Old entries carry only ``vision = true``; modalities derives from it.
        cfg = MLXModelConfig(**self.BASE, vision=True)
        assert cfg.modalities == ["text", "vision"]
        assert cfg.vision is True

    def test_explicit_modalities_syncs_vision_true(self):
        cfg = MLXModelConfig(**self.BASE, modalities=["text", "vision", "audio"])
        assert cfg.vision is True                     # derived from modalities
        assert "audio" in cfg.modalities

    def test_explicit_modalities_without_vision_sets_vision_false(self):
        # A non-vision multimodal model (e.g. text+audio) must NOT read as vision.
        cfg = MLXModelConfig(**self.BASE, modalities=["text", "audio"])
        assert cfg.vision is False
        assert cfg.modalities == ["text", "audio"]

    def test_modalities_are_authoritative_over_vision(self):
        # Contradiction (vision=True but modalities lacks it): modalities wins,
        # since it is the richer, author-declared description.
        cfg = MLXModelConfig(**self.BASE, vision=True, modalities=["text"])
        assert cfg.vision is False
        assert cfg.modalities == ["text"]

    def test_text_always_present(self):
        # Every language model does text; normalize it in even if omitted.
        cfg = MLXModelConfig(**self.BASE, modalities=["vision"])
        assert cfg.modalities[0] == "text"
        assert "vision" in cfg.modalities

    def test_modalities_deduped(self):
        cfg = MLXModelConfig(**self.BASE, modalities=["text", "vision", "vision"])
        assert cfg.modalities == ["text", "vision"]

    def test_loader_default_auto(self):
        assert MLXModelConfig(**self.BASE).loader == "auto"

    def test_loader_accepts_explicit_engines(self):
        for good in ("auto", "mlx-vlm", "mlx-lm"):
            assert MLXModelConfig(**self.BASE, loader=good).loader == good

    def test_loader_rejects_unknown(self):
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, loader="mlx_embedding")
        with pytest.raises(ValidationError):
            MLXModelConfig(**self.BASE, loader="vllm")

    def test_modalities_and_loader_not_runtime_defaults(self):
        # They are model-level metadata, not per-request sampler defaults --
        # they must not leak into MLX_RUNTIME_DEFAULT_FIELDS.
        assert "modalities" not in MLX_RUNTIME_DEFAULT_FIELDS
        assert "loader" not in MLX_RUNTIME_DEFAULT_FIELDS
