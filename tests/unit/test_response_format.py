# tests/unit/test_response_format.py
"""Structured output (`response_format`): the wire normalizes it, gguf hands
llama-server the schema, and MLX decides where the constraint starts.

The constraint itself is the engines' (llama-server's grammar, mlx-vlm's
llguidance processor); whether a real reply parses against a schema is the
live probe's job, not a unit test's.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heylook_llm.config import ChatRequest
from heylook_llm.providers.base import InvalidGenerationRequest
from heylook_llm.schema.converters import to_chat_request
from heylook_llm.schema.messages import MessageCreateRequest

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def _wire(**extra):
    return MessageCreateRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})


@pytest.mark.unit
@pytest.mark.parametrize("fmt, schema", [
    ({"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}, SCHEMA),
    ({"type": "json_object"}, {}),
    ({"type": "text"}, None),
])
def test_the_wire_normalizes_to_one_schema(fmt, schema):
    assert to_chat_request(_wire(response_format=fmt)).response_schema == schema


@pytest.mark.unit
def test_gguf_sends_response_format_not_top_level_json_schema():
    # A top-level `json_schema` also becomes a plain grammar in llama-server,
    # which broke every structured request on a PEG-parsed template
    # (2026-09-27, Muse-Glimmer); response_format takes only the
    # template's reasoning-aware path.
    from heylook_llm.providers.llama_server_provider import LlamaServerProvider

    p = LlamaServerProvider("g", {"model_path": "/fake/model.gguf"}, False)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    payload = p._build_payload(ChatRequest.model_validate({**body, "response_schema": SCHEMA}))
    assert payload["response_format"] == {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
    assert "json_schema" not in payload
    assert "response_format" not in p._build_payload(ChatRequest.model_validate(body))


@pytest.mark.unit
def test_mlx_constrains_the_reply_not_the_thinking():
    from heylook_llm.providers.mlx_provider import _structured_processors

    think = SimpleNamespace(has_harmony_structure=False, has_gemma_channel_structure=False,
                            has_thinking_markers=True)
    harmony = SimpleNamespace(has_harmony_structure=True, has_gemma_channel_structure=False,
                              has_thinking_markers=False)
    request = ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "response_schema": SCHEMA})
    built = object()

    class Wrapper:
        def __init__(self, processor, tokenizer, **kw):
            self.processor, self.kw = processor, kw

    with patch("mlx_vlm.structured.build_json_schema_logits_processor", return_value=built), \
            patch("mlx_vlm.structured.ThinkingAwareLogitsProcessor", Wrapper):
        # thinking off: the reply is the first token, constrain from it
        assert _structured_processors(request, {"enable_thinking": False}, think, None) == [built]
        # thinking on: held back until the format's closer
        (wrapped,) = _structured_processors(request, {"enable_thinking": True}, think, None)
        assert wrapped.processor is built and wrapped.kw["thinking_end_token"] == "</think>"
        # no schema, no processor
        free = ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
        assert _structured_processors(free, {"enable_thinking": True}, think, None) == []
        # the reply's start cannot be found: refused, not left unconstrained
        with pytest.raises(InvalidGenerationRequest, match="harmony"):
            _structured_processors(request, {"enable_thinking": False}, harmony, None)
