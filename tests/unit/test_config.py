# tests/unit/test_config.py
"""Unit tests for Pydantic config models."""
import re
import tomllib
from pathlib import Path
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


# Pydantic construction: a models.toml-shaped entry picks its provider's config
# class, AppConfig finds entries by id, and an unset thinking field stays off
# the wire. Rows: (build the object, project what the row checks, expected
# projection). Built inside the test so a constructor failure is a failed row,
# not a collection error. Rows that only echoed a declared field back were
# dropped (second prune pass, 2026-09-25).
_CONSTRUCTION_ROWS = [
    pytest.param(
        lambda: ChatMessage(role="assistant", content="hi"),
        lambda m: "thinking" in m.model_dump(exclude_none=True),
        False,
        id="thinking_excluded_when_none",
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

    def test_an_entry_that_leaves_thinking_unset_follows_the_capability(self):
        """v1.79.62: an MLX entry that never mentions enable_thinking thinks
        exactly when the model can. Built the way the router builds a
        provider's config (validate the entry, dump its config) and answered
        by the real cascade, so a field default that stops meaning "unset"
        turns a thinking model off by default and fails here."""
        from heylook_llm.samplers import thinking_default

        entry = ModelConfig.model_validate(
            {"id": "m", "provider": "mlx", "config": {"model_path": "/fake"}})
        provider_config = entry.config.model_dump()
        assert thinking_default(provider_config, thinking_capable=True) is True
        assert thinking_default(provider_config, thinking_capable=False) is False


@pytest.mark.unit
class TestAppConfig:
    def test_max_loaded_models_default(self):
        # The schema default flipped from 2 to 1 with idle unloading (C2): a
        # models.toml that doesn't set max_loaded_models loads one at a time.
        cfg = AppConfig(models=[])
        assert cfg.max_loaded_models == 1


@pytest.mark.unit
class TestMLXRuntimeDefaultFields:
    """MLX_RUNTIME_DEFAULT_FIELDS is derived from MLXModelConfig via
    ``json_schema_extra={"is_runtime_default": True}``. A runtime default must
    be omittable, so an entry that does not set it falls through to the
    engine's own default instead of forcing every entry to name it."""

    def test_every_runtime_default_is_optional(self):
        for name in MLX_RUNTIME_DEFAULT_FIELDS:
            field = MLXModelConfig.model_fields[name]
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

    @pytest.mark.parametrize("depth, busy", [
        pytest.param(None, False, id="default_depth_admits_a_second_waiter"),
        pytest.param(1, True, id="depth_1_turns_the_second_waiter_away"),
    ])
    def test_max_queue_depth_reaches_the_busy_answer(self, mock_mlx, depth, busy):  # noqa: ARG002
        """An entry's max_queue_depth decides when a request is turned away.
        Before the field existed, pydantic dropped the key and the provider
        ran at the default whatever the entry said. Entry -> config the router
        hands the provider -> provider -> the gate every generation queues
        in: one generation running and one queued, then a third request asks
        for capacity."""
        import threading

        from heylook_llm.providers.common.generation_gate import (
            ModelBusyError, get_process_gate,
        )
        from heylook_llm.providers.mlx_provider import MLXProvider

        cfg: dict[str, Any] = dict(self.BASE)
        if depth is not None:
            cfg["max_queue_depth"] = depth
        entry = ModelConfig.model_validate({"id": "q", "provider": "mlx", "config": cfg})
        provider = MLXProvider(model_id="q", config=entry.config.model_dump(), verbose=False)

        gate = get_process_gate(99)  # the gate the provider made; 99 is ignored
        gate.acquire()  # a generation in flight
        waiter = threading.Thread(target=lambda: (gate.acquire(), gate.release()))
        waiter.start()
        try:
            for _ in range(200):
                if gate.waiting == 1:
                    break
                threading.Event().wait(0.005)
            assert gate.waiting == 1
            if busy:
                with pytest.raises(ModelBusyError):
                    provider.check_capacity()
            else:
                provider.check_capacity()
        finally:
            gate.release()
            waiter.join(timeout=5)



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
class TestExampleConfigs:
    """heylook.example.toml and model.heylook.example.toml are the tracked
    format references (README points at them). Claim: what they show, the
    commented-out blocks included (those are what gets copied), passes the
    REAL validators. Without this anchor a field rename in config.py rots the
    examples silently and only a future user's copy-paste fails.
    """

    ROOT = Path(__file__).parents[2]

    @staticmethod
    def _uncommented(text: str) -> str:
        """The file with every commented-out TOML line (a key, [table] or
        [[table]]) switched on."""
        return re.sub(r"^# ?(\s*(\[\[?[a-z.]+\]\]?$|[a-z_]+ = .*$))", r"\1", text, flags=re.M)

    def test_server_config_example_validates(self):
        from heylook_llm.model_registry import served
        from heylook_llm.settings import SettingsSchema

        text = (self.ROOT / "heylook.example.toml").read_text()
        data = tomllib.loads(text)
        app = served(data, [])
        assert app.allowed_hosts  # a top-level key, not swallowed by [scan]
        assert app.scan is not None and app.scan.folders
        SettingsSchema(**data["settings"])
        full = tomllib.loads(self._uncommented(text))
        assert {m.provider for m in served(full, []).models} == {"mlx", "gguf"}
        SettingsSchema(**full["settings"])

    def test_model_file_example_validates(self):
        text = (self.ROOT / "model.heylook.example.toml").read_text()
        mlx_part, marker, rest = text.partition("# --- gguf model")
        gguf_part = marker + rest
        mlx = tomllib.loads(self._uncommented(mlx_part))
        gguf = tomllib.loads(self._uncommented(gguf_part))
        for data in (mlx, gguf):
            assert not any(isinstance(v, dict) for v in data.values())
            assert not {"model_path", "id"} & data.keys()
        MLXModelConfig(model_path="/m", **mlx)
        unset = gguf.pop("unset")
        assert set(unset) <= GGUFModelConfig.model_fields.keys()
        GGUFModelConfig(model_path="/m/w.gguf", **gguf)
