# tests/unit/test_config.py
"""Unit tests for Pydantic config models."""
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from heylook_llm.config import (
    AppConfig,
    ChatMessage,
    ChatRequest,
    GGUFModelConfig,
    ImageContentPart,
    ImageUrl,
    MLX_RUNTIME_DEFAULT_FIELDS,
    MLXModelConfig,
    ModelConfig,
    TextContentPart,
)


# Pydantic construction round-trip: ChatMessage / ChatRequest / ModelConfig /
# AppConfig construct from every role and content shape, round-trip thinking,
# and omit None on dump. Rows: (build the object, project what the row checks,
# expected projection). Built inside the test so a constructor failure is a
# failed row, not a collection error.
_CONSTRUCTION_ROWS = [
    pytest.param(
        lambda: ChatMessage(role="user", content="hello"),
        lambda m: (m.role, m.content, m.thinking),
        ("user", "hello", None),
        id="simple_text_message",
    ),
    pytest.param(
        lambda: ChatMessage(role="assistant", content="hi there"),
        lambda m: m.role, "assistant", id="assistant_message",
    ),
    pytest.param(
        lambda: ChatMessage(role="system", content="You are helpful."),
        lambda m: m.role, "system", id="system_message",
    ),
    pytest.param(
        lambda: ChatMessage(role="user", content=[
            TextContentPart(type="text", text="What is this?"),
            ImageContentPart(type="image_url",
                             image_url=ImageUrl(url="https://example.com/img.png")),
        ]),
        lambda m: (isinstance(m.content, list), len(m.content)),
        (True, 2),
        id="multimodal_content",
    ),
    pytest.param(
        lambda: ChatMessage(role="assistant", content="answer", thinking="reasoning"),
        lambda m: (m.thinking, m.model_dump()["thinking"]),
        ("reasoning", "reasoning"),
        id="thinking_roundtrip",
    ),
    pytest.param(
        lambda: ChatMessage(role="assistant", content="hi"),
        lambda m: "thinking" in m.model_dump(exclude_none=True),
        False,
        id="thinking_excluded_when_none",
    ),
    pytest.param(
        lambda: ChatMessage(role="user", content="test"),
        lambda m: (m.name, m.tool_call_id, m.tool_calls),
        (None, None, None),
        id="optional_fields_default_none",
    ),
    pytest.param(
        lambda: ChatRequest(messages=[ChatMessage(role="user", content="hi")]),
        lambda r: (r.model, r.stream, len(r.messages)),
        (None, False, 1),
        id="minimal_request",
    ),
    pytest.param(
        lambda: ChatRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5, top_p=0.9, top_k=40, min_p=0.1,
            repetition_penalty=1.1, max_tokens=256, seed=42,
        ),
        lambda r: (r.temperature, r.top_k, r.seed),
        (0.5, 40, 42),
        id="all_sampler_params",
    ),
    pytest.param(
        lambda: ChatRequest(messages=[ChatMessage(role="user", content="hi")],
                            enable_thinking=True),
        lambda r: r.enable_thinking, True, id="enable_thinking",
    ),
    pytest.param(
        lambda: ModelConfig(id="test-mlx", provider="mlx",
                            config={"model_path": "/fake/path"}),
        lambda mc: (mc.id, mc.provider, isinstance(mc.config, MLXModelConfig),
                    mc.config.model_path),
        ("test-mlx", "mlx", True, "/fake/path"),
        id="mlx_model_config",
    ),
    # The provider's config class comes from PROVIDER_CONFIG_CLASSES, from the
    # dict form a models.toml entry arrives in.
    pytest.param(
        lambda: ModelConfig.model_validate(
            {"id": "m", "provider": "mlx", "config": {"model_path": "/tmp/fake"}}),
        lambda mc: isinstance(mc.config, MLXModelConfig), True,
        id="validator_uses_registry",
    ),
    pytest.param(
        lambda: ModelConfig.model_validate({
            "id": "m", "provider": "gguf",
            "config": {"model_path": "/x/model.gguf", "mmproj_path": "/x/mmproj.gguf"},
        }),
        lambda mc: (isinstance(mc.config, GGUFModelConfig), mc.config.mmproj_path),
        (True, "/x/mmproj.gguf"),
        id="gguf_model_config_builds",
    ),
    pytest.param(
        lambda: ModelConfig(id="test", provider="mlx", config={"model_path": "/fake"},
                            capabilities=["chat", "thinking", "vision"]),
        lambda mc: "thinking" in mc.capabilities, True, id="capabilities_list",
    ),
    pytest.param(
        lambda: AppConfig(models=[ModelConfig(id="m1", provider="mlx",
                                              config={"model_path": "/a"})]),
        # m3 is missing
        lambda c: (c.get_model_config("m1") is not None, c.get_model_config("m3")),
        (True, None),
        id="get_model_config",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("build, project, expected", _CONSTRUCTION_ROWS)
def test_pydantic_construction_round_trip(build, project, expected):
    assert project(build()) == expected


@pytest.mark.unit
class TestChatRequest:
    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            ChatRequest(messages=[])


@pytest.mark.unit
class TestModelConfig:
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
        assert mc.enable_thinking is None  # v1.79.62: unset = follow the thinking capability


@pytest.mark.unit
class TestAppConfig:
    def test_max_loaded_models_default(self):
        # The schema default flipped from 2 to 1 with idle unloading (C2): a
        # models.toml that doesn't set max_loaded_models loads one at a time.
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
    # The KV cache knobs and num_draft_tokens were retired with the mlx-vlm
    # engine (plan W10 stage 3).
    EXPECTED_RUNTIME_DEFAULTS = frozenset({
        "prefill_step_size",
    })

    def test_derived_set_matches_expected_and_each_is_optional(self):
        assert MLX_RUNTIME_DEFAULT_FIELDS == self.EXPECTED_RUNTIME_DEFAULTS
        # Safety: runtime defaults must be omittable so models.toml entries
        # that don't set them fall through to mlx-lm's own defaults.
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

    # dict[str, Any], not the inferred dict[str, str]: this is splatted into
    # config constructors with int/float/bool fields, and pyright flags every
    # call site otherwise.
    BASE: ClassVar[dict[str, Any]] = {"model_path": "/fake/model"}

    # extra="forbid", on both provider config classes: an unknown or retired
    # key fails validation. Each row is a list of kwargs sets, each refused on
    # its own.
    # - unknown_key: a typo like `temperatue` must not silently vanish.
    # - quantized_kv_start: dead config (stored and forwarded but never
    #   consumed by _build_cache_config/make_cache), removed outright.
    # - retired_mlx_fields: retired with the mlx-vlm engine (plan W10 stage 3);
    #   an entry still carrying one fails at load rather than doing nothing.
    # - gguf: the same rule on GGUFModelConfig.
    @pytest.mark.parametrize("config_cls, base, bad_kwargs", [
        pytest.param(MLXModelConfig, BASE, [{"temperatue": 0.9}], id="unknown_key_rejected"),
        pytest.param(MLXModelConfig, BASE, [{"quantized_kv_start": 1024}],
                     id="quantized_kv_start_removed"),
        pytest.param(MLXModelConfig, BASE,
                     [{"loader": "mlx-lm"}, {"cache_type": "quantized"},
                      {"kv_bits": 8}, {"draft_model_path": "/d"},
                      {"num_draft_tokens": 3}],
                     id="retired_mlx_fields_are_refused"),
        pytest.param(GGUFModelConfig, {"model_path": "/x.gguf"}, [{"surprise": True}],
                     id="gguf_extra_fields_forbidden"),
    ])
    def test_extra_keys_are_forbidden(self, config_cls, base, bad_kwargs):
        config_cls(**base)  # the base alone validates: the key is what fails
        for kwargs in bad_kwargs:
            with pytest.raises(ValidationError):
                config_cls(**base, **kwargs)

    def test_max_queue_depth_is_a_real_field(self):
        # The provider reads config["max_queue_depth"]; without a field the
        # value was silently dropped by pydantic and unreachable.
        assert MLXModelConfig(**self.BASE).max_queue_depth == 8
        assert MLXModelConfig(**self.BASE, max_queue_depth=2).max_queue_depth == 2



@pytest.mark.unit
class TestModalitiesAndLoader:
    """modalities/loader split (Phase 6 refinement 2026-07-11).

    ``vision: bool`` used to do two jobs -- DESCRIBE the model (has a vision
    tower) and SELECT the loader (mlx-vlm vs mlx-lm). ``modalities`` (list) is
    the description; ``loader`` (routing) is separate. ``vision`` is retained but
    demoted to a derived mirror of ``"vision" in modalities`` for back-compat.
    """

    # dict[str, Any], not the inferred dict[str, str]: this is splatted into
    # config constructors with int/float/bool fields, and pyright flags every
    # call site otherwise.
    BASE: ClassVar[dict[str, Any]] = {"model_path": "/fake/model"}

    # Normalization rows: (kwargs, expected modalities). Text first, deduped,
    # `vision` mirrors "vision" in modalities, modalities beat the legacy bool.
    # - legacy_vision_true: old entries carry only `vision = true`.
    # - without_vision: a non-vision multimodal model (text+audio) must NOT
    #   read as vision.
    # - authoritative: vision=True but modalities lacks it -> modalities wins,
    #   being the richer, author-declared description.
    # - text_always_present: every language model does text; normalized in.
    @pytest.mark.parametrize("kwargs, expected", [
        pytest.param({}, ["text"], id="defaults_are_text_only"),
        pytest.param({"vision": True}, ["text", "vision"],
                     id="legacy_vision_true_derives_modalities"),
        pytest.param({"modalities": ["text", "vision", "audio"]},
                     ["text", "vision", "audio"],
                     id="explicit_modalities_syncs_vision_true"),
        pytest.param({"modalities": ["text", "audio"]}, ["text", "audio"],
                     id="explicit_modalities_without_vision_sets_vision_false"),
        pytest.param({"vision": True, "modalities": ["text"]}, ["text"],
                     id="modalities_are_authoritative_over_vision"),
        pytest.param({"modalities": ["vision"]}, ["text", "vision"],
                     id="text_always_present"),
        pytest.param({"modalities": ["text", "vision", "vision"]}, ["text", "vision"],
                     id="modalities_deduped"),
    ])
    def test_modalities_normalization(self, kwargs, expected):
        cfg = MLXModelConfig(**self.BASE, **kwargs)
        assert cfg.modalities == expected
        assert cfg.vision is ("vision" in expected)  # derived mirror


@pytest.mark.unit
class TestModalitiesDeriveAtLoad:
    """Derive-at-load (Wave 1 / 6a, 2026-07-28): when an entry does not
    materialize ``modalities``, the config derives it at validation time from
    the model dir's own config.json -- the same ground truth the importer
    reads. Stored ``modalities`` is an explicit OVERRIDE and always wins.

    Claim: without this, thin entries silently regress to text-only (the
    legacy vision-bool fallback) and every import must keep materializing
    derived metadata that rots when the dir changes in place.
    """

    # Rows: (model dir config.json, or None for a fake path with no dir;
    # stored kwargs; expected modalities). `vision` mirrors the result.
    # - stored_override: the operator says text-only, the dir says vision;
    #   stored intent wins.
    # - no_config_json: fake paths (tests, HF repo ids) keep the pre-6a
    #   derivation from the legacy vision bool.
    @pytest.mark.parametrize("config_json, kwargs, expected", [
        pytest.param({"model_type": "gemma4", "vision_config": {}, "audio_config": {}},
                     {}, ["text", "vision", "audio"],
                     id="unset_modalities_detects_from_model_dir"),
        pytest.param({"model_type": "llama"}, {}, ["text"],
                     id="unset_modalities_text_only_dir"),
        pytest.param({"model_type": "x", "vision_config": {}}, {"modalities": ["text"]},
                     ["text"], id="stored_modalities_override_detection"),
        pytest.param(None, {"vision": True}, ["text", "vision"],
                     id="no_config_json_falls_back_to_legacy_vision_bool"),
    ])
    def test_modalities_derive_at_load(self, tmp_path, config_json, kwargs, expected):
        if config_json is None:
            path = "/fake/model"
        else:
            import json as _json
            (tmp_path / "config.json").write_text(_json.dumps(config_json))
            path = str(tmp_path)
        cfg = MLXModelConfig(model_path=path, **kwargs)
        assert cfg.modalities == expected
        assert cfg.vision is ("vision" in expected)  # mirror syncs to detection


@pytest.mark.unit
class TestModelsExampleToml:
    """models.example.toml is the tracked format reference (README points at
    it). Claim: every entry in it must round-trip the REAL Pydantic
    validators -- without this anchor, a field rename in config.py rots the
    example silently and only a future user's copy-paste fails.
    """

    def test_example_file_validates(self):
        import tomllib
        from pathlib import Path

        from heylook_llm.config import AppConfig

        example = Path(__file__).parents[2] / "models.example.toml"
        with open(example, "rb") as f:
            data = tomllib.load(f)
        cfg = AppConfig(**data)
        assert len(cfg.models) >= 3  # minimal MLX, override MLX, gguf
        providers = {m.provider for m in cfg.models}
        assert {"mlx", "gguf"} <= providers
