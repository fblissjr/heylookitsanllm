# tests/unit/test_context_length.py
"""The context window: ONE resolver for every surface (v1.79.65).

Claims:
- `model_context_length` reads the number where each provider keeps it (a
  gguf header, an MLX config.json -- top-level or the nested text block a VLM
  wrapper puts the language head in) and answers None, never a guess, when
  the files do not say or the provider has no chat context.
- The MLX over-length guard in run_generation refuses a prompt longer than
  that number as the CLIENT's error (InvalidGenerationRequest -> 400), before
  any engine work, and stays silent when the number is unknown.
"""

import json

import pytest

from heylook_llm.capabilities import _mlx_context_length, model_context_length


def _checkpoint(tmp_path, config: dict):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


class TestMlxContextLength:
    def test_top_level_max_position_embeddings(self, tmp_path):
        assert _mlx_context_length(_checkpoint(tmp_path, {"max_position_embeddings": 40960})) == 40960

    def test_nested_text_config_when_the_top_level_is_silent(self, tmp_path):
        cfg = {"model_type": "gemma4", "text_config": {"max_position_embeddings": 131072}}
        assert _mlx_context_length(_checkpoint(tmp_path, cfg)) == 131072

    def test_top_level_wins_over_nested(self, tmp_path):
        cfg = {"max_position_embeddings": 8192, "text_config": {"max_position_embeddings": 4096}}
        assert _mlx_context_length(_checkpoint(tmp_path, cfg)) == 8192

    def test_alias_keys(self, tmp_path):
        assert _mlx_context_length(_checkpoint(tmp_path, {"max_seq_len": 2048})) == 2048

    def test_none_when_the_files_do_not_say(self, tmp_path):
        assert _mlx_context_length(_checkpoint(tmp_path, {"model_type": "x"})) is None
        assert _mlx_context_length(str(tmp_path / "missing")) is None
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "config.json").write_text("{not json")
        assert _mlx_context_length(str(tmp_path / "bad")) is None

    def test_non_positive_values_are_not_a_context(self, tmp_path):
        assert _mlx_context_length(_checkpoint(tmp_path, {"max_position_embeddings": 0})) is None


class TestModelContextLength:
    def test_mlx_routes_to_config_json(self, tmp_path):
        assert model_context_length("mlx", _checkpoint(tmp_path, {"max_position_embeddings": 32768})) == 32768

    def test_gguf_unreadable_header_is_none(self, tmp_path):
        assert model_context_length("gguf", str(tmp_path / "nope.gguf")) is None

    def test_providers_without_a_chat_context_answer_none(self, tmp_path):
        assert model_context_length("mlx_embedding", _checkpoint(tmp_path, {"max_position_embeddings": 512})) is None

    def test_no_path_is_none(self):
        assert model_context_length("mlx", None) is None
        assert model_context_length("mlx", "") is None


class TestOverLengthGuard:
    def _gen(self, prompt_len: int, context_length):
        from heylook_llm.providers.common.generation_core import run_generation
        return run_generation(
            model=None, tokenizer=None, prompt_tokens=[1] * prompt_len,
            effective_request={}, sampler=None, processors=None,
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
