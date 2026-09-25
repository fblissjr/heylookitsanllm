# tests/unit/test_context_length.py
"""The context window: ONE resolver for every surface (v1.79.65).

Claims:
- `model_context_length` reads the number where each provider keeps it (a
  gguf header, an MLX config.json -- top-level or the nested text block a VLM
  wrapper puts the language head in) and answers None, never a guess, when
  the files do not say or the provider has no chat context.
- The MLX over-length guard in vlm_engine refuses a prompt longer than
  that number as the CLIENT's error (InvalidGenerationRequest -> 400), before
  any engine work, and stays silent when the number is unknown.
"""

import json

import pytest

from heylook_llm.capabilities import model_context_length
from heylook_llm.providers.mlx_describe import _config_json_context_length as _mlx_context_length


def _checkpoint(tmp_path, config: dict):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


_MISSING = object()  # a model dir that does not exist
_BAD_JSON = object()  # a config.json that does not parse


def _files(tmp_path, spec):
    """A model path from a row's spec: a dict is config.json, `_MISSING` a
    directory that is not there, `_BAD_JSON` an unparseable config.json."""
    if spec is _MISSING:
        return str(tmp_path / "missing")
    if spec is _BAD_JSON:
        (tmp_path / "config.json").write_text("{not json")
        return str(tmp_path)
    return _checkpoint(tmp_path, spec)


# MLX config.json context: top-level beats the nested text_config a VLM
# wrapper puts the language head in, alias keys count, and None (never a
# guess) when the files do not say, are missing or unparseable, or the value
# is not positive.
@pytest.mark.parametrize("spec, expected", [
    ({"max_position_embeddings": 40960}, 40960),
    ({"model_type": "gemma4", "text_config": {"max_position_embeddings": 131072}}, 131072),
    ({"max_position_embeddings": 8192, "text_config": {"max_position_embeddings": 4096}}, 8192),
    ({"max_seq_len": 2048}, 2048),
    ({"model_type": "x"}, None),
    (_MISSING, None),
    (_BAD_JSON, None),
    ({"max_position_embeddings": 0}, None),
], ids=["top_level", "nested_text_config_when_top_level_silent", "top_level_wins_over_nested",
        "alias_key", "files_do_not_say", "missing_dir", "bad_json", "non_positive_is_not_a_context"])
def test_mlx_context_length_from_config_json(tmp_path, spec, expected):
    assert _mlx_context_length(_files(tmp_path, spec)) == expected


# model_context_length routes by provider: mlx reads config.json, an
# unreadable gguf header is None, no path is None.
@pytest.mark.parametrize("provider, path, expected", [
    ("mlx", {"max_position_embeddings": 32768}, 32768),
    ("gguf", "nope.gguf", None),
    ("mlx", None, None),
    ("mlx", "", None),
], ids=["mlx_routes_to_config_json", "gguf_unreadable_header", "no_path_none", "no_path_empty"])
def test_model_context_length_routing(tmp_path, provider, path, expected):
    if isinstance(path, dict):
        path = _checkpoint(tmp_path, path)
    elif path:
        path = str(tmp_path / path)
    assert model_context_length(provider, path) == expected


class TestContextLengthOverride:
    """The entry's own `context_length` (MLXModelConfig) wins over the files:
    a YaRN-scaled checkpoint ships the ORIGINAL max_position_embeddings with
    the factor in rope_scaling, so the file alone would refuse a prompt the
    model takes. Absent = the file value; non-positive is not a window."""

    @pytest.mark.parametrize("spec, override, expected", [
        ({"max_position_embeddings": 32768}, 131072, 131072),
        (_MISSING, 8192, 8192),
        ({"max_position_embeddings": 32768}, None, 32768),
        ({"max_position_embeddings": 32768}, 0, 32768),
        ({"max_position_embeddings": 32768}, True, 32768),
    ], ids=["override_wins_over_config_json", "override_answers_when_files_do_not",
            "absent_override_is_file_value", "zero_override_ignored", "bool_override_ignored"])
    def test_override_against_the_file(self, tmp_path, spec, override, expected):
        assert model_context_length("mlx", _files(tmp_path, spec), override=override) == expected

    def test_config_field_rejects_a_non_positive_window(self):
        from pydantic import ValidationError
        from heylook_llm.config import MLXModelConfig
        assert MLXModelConfig(model_path="/fake", context_length=4096).context_length == 4096
        with pytest.raises(ValidationError):
            MLXModelConfig(model_path="/fake", context_length=0)

    def test_derived_facts_carry_the_override(self, tmp_path):
        from heylook_llm.capabilities import derived_model_facts
        from heylook_llm.config import ModelConfig
        path = _checkpoint(tmp_path, {"max_position_embeddings": 32768})
        with_override = ModelConfig(id="m", provider="mlx",
                                    config={"model_path": path, "context_length": 131072})
        without = ModelConfig(id="m", provider="mlx", config={"model_path": path})
        length = derived_model_facts(with_override).engine.context.length
        assert (length.value, length.provenance) == (131072, "configured")
        length = derived_model_facts(without).engine.context.length
        assert (length.value, length.provenance) == (32768, "derived")


class TestOverLengthGuard:
    def _gen(self, prompt_len: int, context_length):
        import mlx.core as mx

        from heylook_llm.providers.common import vlm_engine
        return vlm_engine.generate(
            model=None, processor=None, apc_manager=None,
            input_ids=mx.array([[1] * prompt_len]), raw_inputs={},
            sampler=None, processors=[], stop_tokens=(), max_tokens=4,
            model_id="m", context_length=context_length)

    def test_a_prompt_past_the_window_is_the_clients_error(self):
        from heylook_llm.providers.base import InvalidGenerationRequest
        with pytest.raises(InvalidGenerationRequest) as exc:
            next(self._gen(11, 10))
        assert "11 tokens" in str(exc.value) and "10 tokens" in str(exc.value)

    def test_an_unknown_window_does_not_guard(self):
        # Reaches the engine setup and fails THERE (no model), which is the
        # point: the guard did not fire.
        from heylook_llm.providers.base import InvalidGenerationRequest
        with pytest.raises(Exception) as exc:
            next(self._gen(11, None))
        assert not isinstance(exc.value, InvalidGenerationRequest)
