# tests/unit/test_tool_calling.py
"""Unit tests for engine-neutral Anthropic tool calling and `heylook presets assist`."""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import patch

import pytest

from heylook_llm import cli
from heylook_llm.config import ChatMessage, ChatRequest
from heylook_llm.messages_api import StreamingEventTranslator, _non_stream_messages, _stream_messages
from heylook_llm.perf_collector import PerfCollector
from heylook_llm.providers.base import GenerationChunk, InvalidGenerationRequest
from heylook_llm.providers.llama_server_provider import LlamaServerProvider
from heylook_llm.schema.content_blocks import TextBlock, ToolResultBlock, ToolUseBlock
from heylook_llm.schema.converters import from_openai_response_dict, to_chat_request, to_stop_reason
from heylook_llm.schema.messages import Message, MessageCreateRequest, ToolChoice, ToolDefinition
from heylook_llm.thinking_parser import HybridThinkingParser
from heylook_llm.tool_calling import (
    ToolCallStreamFilter,
    infer_tool_parser_name,
    normalize_tool_calls_for_jinja,
    normalize_tool_calls_for_openai_wire,
    prepare_tool_choice_policy,
)


def _sample_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_preset",
        description="Read a preset",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    )


def _fake_qwen_extractor(span: str):
    body = span.removeprefix("<tool_call>").removesuffix("</tool_call>").strip()
    data = json.loads(body)
    return (
        [
            {
                "id": "call_test_1",
                "type": "function",
                "function": {"name": data["name"], "arguments": json.dumps(data.get("arguments", {}))},
            }
        ],
        "",
    )


@pytest.mark.unit
class TestToolSchemaAndConverters:
    def test_to_chat_request_converts_tools_tool_choice_and_history_blocks(self):
        req = MessageCreateRequest(
            model="m",
            system="You are helpful.",
            tools=[_sample_tool()],
            tool_choice=ToolChoice(type="tool", name="get_preset"),
            messages=[
                Message(role="user", content="Read Pirate"),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Checking."),
                        ToolUseBlock(id="toolu_1", name="get_preset", input={"name": "Pirate"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="toolu_1", content='{"name":"Pirate"}'),
                    ],
                ),
            ],
        )
        chat = to_chat_request(req)
        assert chat.tools == [
            {
                "type": "function",
                "function": {
                    "name": "get_preset",
                    "description": "Read a preset",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                },
            }
        ]
        assert chat.tool_choice == {"type": "function", "function": {"name": "get_preset"}}
        assert [m.role for m in chat.messages] == ["system", "user", "assistant", "tool"]
        assert chat.messages[2].tool_calls[0]["function"]["name"] == "get_preset"
        assert chat.messages[3].tool_call_id == "toolu_1"
        assert chat.messages[3].content == '{"name":"Pirate"}'

    def test_tool_choice_requires_name_for_tool_type(self):
        with pytest.raises(ValueError, match="requires `name`"):
            ToolChoice(type="tool")

    def test_from_openai_response_dict_emits_tool_use_and_stop_reason(self):
        resp = from_openai_response_dict({
            "model": "m",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "thinking": "Need to check preset.",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {"name": "get_preset", "arguments": '{"name":"Pirate"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"input_tokens": 10, "completion_tokens": 5},
        })
        assert resp.stop_reason == "tool_use"
        assert [b.type for b in resp.content] == ["thinking", "tool_use"]
        assert resp.content[1].id == "call_9"
        assert resp.content[1].name == "get_preset"
        assert resp.content[1].input == {"name": "Pirate"}
        assert to_stop_reason("tool_calls") == "tool_use"

    def test_infer_tool_parser_name_detects_major_families(self):
        assert infer_tool_parser_name("{% if tools %}<tool_call>{% endif %}") == "json_tools"
        assert infer_tool_parser_name("<tool_call>\n<function=foo>") == "qwen3_coder"
        assert infer_tool_parser_name("...<|tool_call>call:foo{}<tool_call|>...") == "gemma4"
        assert infer_tool_parser_name("plain chat template with no tool markers") is None

    def test_normalize_tool_call_arguments_between_jinja_and_openai(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]
        for_jinja = normalize_tool_calls_for_jinja(calls)
        assert for_jinja[0]["function"]["arguments"] == {"a": 1}
        back_wire = normalize_tool_calls_for_openai_wire(for_jinja)
        assert json.loads(back_wire[0]["function"]["arguments"]) == {"a": 1}

    def test_prepare_tool_choice_policy_enforces_modes(self):
        tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
        msgs = [{"role": "user", "content": "hi"}]
        m_none, t_none = prepare_tool_choice_policy(msgs, tools, "none")
        assert t_none is None and m_none == msgs

        m_req, t_req = prepare_tool_choice_policy(msgs, tools, "required")
        assert len(t_req) == 2 and m_req[0]["role"] == "system"

        m_one, t_one = prepare_tool_choice_policy(msgs, tools, {"type": "function", "function": {"name": "b"}})
        assert [t["function"]["name"] for t in t_one] == ["b"]
        assert "call the 'b' function" in m_one[0]["content"]

        with pytest.raises(InvalidGenerationRequest, match="unknown tool"):
            prepare_tool_choice_policy(msgs, tools, {"type": "function", "function": {"name": "missing"}})


@pytest.mark.unit
class TestToolStreamingAndExtraction:
    def test_stream_filter_isolates_thinking_and_streams_tool_use_sse(self):
        filt = ToolCallStreamFilter("<tool_call>", "</tool_call>", _fake_qwen_extractor)
        translator = StreamingEventTranslator(
            "msg_1", "m", thinking_parser=HybridThinkingParser(), tool_filter=filt
        )

        # Notice <tool_call> inside <think> must stay in thinking, not trigger tool extraction!
        chunks = [
            "<think>Maybe use <tool_call> later</think>",
            "Checking now. <tool_",
            'call>{"name":"get_preset","arguments":{"name":"Pirate"}}</tool_call>',
        ]
        raw_events = []
        for c in chunks:
            raw_events.extend(translator.process_chunk(c))
        raw_events.extend(translator.flush())
        raw_events.append(translator.message_delta_event())

        parsed = [json.loads(ev.split("data: ", 1)[1]) for ev in raw_events]
        starts = [e["content_block"] for e in parsed if e["type"] == "content_block_start"]
        assert [b["type"] for b in starts] == ["thinking", "text", "tool_use"]
        assert starts[2]["name"] == "get_preset"

        json_deltas = [
            e["delta"]["partial_json"]
            for e in parsed
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
        ]
        assert json.loads("".join(json_deltas)) == {"name": "Pirate"}

        msg_delta = [e for e in parsed if e["type"] == "message_delta"][-1]
        assert msg_delta["delta"]["stop_reason"] == "tool_use"

    def test_non_streaming_extracts_inline_tool_call_after_thinking(self):
        filt = ToolCallStreamFilter("<tool_call>", "</tool_call>", _fake_qwen_extractor)
        req = MessageCreateRequest(
            model="m",
            messages=[Message(role="user", content="check")],
            tools=[_sample_tool()],
        )
        gen = (
            c for c in [
                GenerationChunk(text="<think>plan</think>"),
                GenerationChunk(text='<tool_call>{"name":"get_preset","arguments":{"name":"Pirate"}}</tool_call>', finish_reason="stop"),
            ]
        )
        with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
            resp = asyncio.run(
                _non_stream_messages(gen, req, "rid", 0.0, make_parser=HybridThinkingParser, tool_filter=filt)
            )
        assert resp.stop_reason == "tool_use"
        assert [b.type for b in resp.content] == ["thinking", "tool_use"]
        assert resp.content[1].input == {"name": "Pirate"}


@pytest.mark.unit
class TestGGUFToolCalling:
    def test_build_payload_forwards_tools_and_refuses_when_caps_false(self):
        provider = LlamaServerProvider("g", {"model_path": "/fake.gguf"}, verbose=False)
        provider.chat_template_caps = {"supports_tools": True}
        req = ChatRequest(
            model="g",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "list_presets", "parameters": {"type": "object"}}}],
            tool_choice="auto",
        )
        payload = provider._build_payload(req)
        assert payload["tools"] == req.tools
        assert payload["tool_choice"] == "auto"

        provider.chat_template_caps = {"supports_tools": False}
        with pytest.raises(InvalidGenerationRequest, match="does not support tool calling"):
            provider._build_payload(req)

    def test_stream_chunks_accumulates_fragmented_delta_tool_calls(self):
        provider = LlamaServerProvider("g", {"model_path": "/fake.gguf"}, verbose=False)
        frames = [
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_preset", "arguments": '{"na'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'me":"Pirate"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}},
        ]
        sse_lines = [f"data: {json.dumps(f)}\n".encode() for f in frames] + [b"data: [DONE]\n"]
        chunks = list(provider._stream_chunks(sse_lines, abort_event=None))
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].finish_reason == "tool_calls"
        assert tool_chunks[0].tool_calls == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_preset", "arguments": '{"name":"Pirate"}'},
            }
        ]


class _FakePresetStore:
    def __init__(self, initial: list[dict]):
        self._items = [
            {
                "id": f"p-{i + 1}",
                "name": row["name"],
                "system_prompt": row.get("system_prompt"),
                "params": dict(row.get("params") or {}),
            }
            for i, row in enumerate(initial)
        ]

    def list_presets(self) -> list[dict]:
        return sorted(self._items, key=lambda p: p["name"])

    def create_preset(self, rec) -> dict:
        row = {"id": f"p-{len(self._items) + 1}", "name": rec.name, "system_prompt": rec.system_prompt, "params": dict(rec.params)}
        self._items.append(row)
        return row

    def update_preset(self, preset_id: str, rec) -> dict:
        for row in self._items:
            if row["id"] == preset_id:
                row["system_prompt"] = rec.system_prompt
                row["params"] = dict(rec.params)
                return row
        raise RuntimeError(f"Unknown preset id: {preset_id}")

    def by_name(self) -> dict[str, dict]:
        return {p["name"]: p for p in self._items}


@pytest.mark.unit
class TestHeylookCliPresetAssist:
    def test_assist_runs_tool_loop_shows_combined_diff_and_applies_on_confirm(self):
        store = _FakePresetStore([
            {"name": "Coder", "system_prompt": "Write clean Python.", "params": {"temperature": 0.2}},
            {"name": "Reviewer", "system_prompt": "Review code carefully.", "params": {"temperature": 0.3}},
        ])

        seen_payloads: list[dict] = []
        turn_responses = iter([
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "list_presets", "input": {}},
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "get_preset", "input": {"name": "Coder"}},
                    {"type": "tool_use", "id": "tu_3", "name": "get_preset", "input": {"name": "Reviewer"}},
                ],
            },
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_4",
                        "name": "propose_preset_update",
                        "input": {
                            "name": "Coder",
                            "system_prompt": "Write clean Python. Avoid em-dashes.",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_5",
                        "name": "propose_preset_update",
                        "input": {
                            "name": "Reviewer",
                            "system_prompt": "Review code carefully. Avoid em-dashes.",
                        },
                    },
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Staged updates for Coder and Reviewer."}],
            },
        ])

        def fake_message(payload: dict) -> dict:
            seen_payloads.append(payload)
            return next(turn_responses)

        out = io.StringIO()
        err = io.StringIO()
        inputs = iter(["d", "a"])
        rc = cli.main(
            ["presets", "assist", "--model", "qwen3", "Weave in: avoid em-dashes"],
            store=store,
            message_fn=fake_message,
            out=out,
            err=err,
            input_fn=lambda _p: next(inputs),
        )
        assert rc == 0, err.getvalue()
        stdout = out.getvalue()
        assert "-> list_presets()" in stdout
        assert "-> propose_preset_update('Coder')" in stdout
        assert "Already exist (2):" in stdout
        assert "=== Coder (existing -> incoming) ===" in stdout
        assert "=== Reviewer (existing -> incoming) ===" in stdout
        assert "Import complete: 2 updated." in stdout

        # Verify params were preserved when model omitted `params`
        after = store.by_name()
        assert after["Coder"]["system_prompt"] == "Write clean Python. Avoid em-dashes."
        assert after["Coder"]["params"] == {"temperature": 0.2}
        assert after["Reviewer"]["system_prompt"] == "Review code carefully. Avoid em-dashes."
        assert after["Reviewer"]["params"] == {"temperature": 0.3}
        assert len(seen_payloads) == 4


@pytest.mark.unit
class TestGemma4AndDiffusionGuards:
    def test_strip_specials_preserves_gemma4_tool_delimiters_when_tools_active(self):
        from types import SimpleNamespace
        from heylook_llm.reasoning_parser import parse_reasoning, parser_factory_for

        info = SimpleNamespace(
            chat_template="...<|tool_call>...<tool_call|>...",
            has_gemma_channel_structure=True,
            has_harmony_structure=False,
            has_recipient_channel_structure=False,
            has_thinking_markers=False,
            prefills_thinking=False,
            special_tokens=frozenset({"<|tool_call>", "<tool_call|>", '<|"|>', "<turn|>"}),
        )

        class FakeProv:
            def template_info(self):
                return info

            def effective_thinking(self, req):
                return False

        raw = '<|tool_call>call:get_preset{name:<|"|>Coder<|"|>}<tool_call|><turn|>'
        req_with_tools = ChatRequest(
            model="gemma4",
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "get_preset", "parameters": {}}}],
        )
        content_with, _ = parse_reasoning(raw, parser_factory_for(FakeProv(), req_with_tools)())
        assert content_with == '<|tool_call>call:get_preset{name:<|"|>Coder<|"|>}<tool_call|>'

        req_no_tools = ChatRequest(
            model="gemma4",
            messages=[ChatMessage(role="user", content="hi")],
        )
        content_without, _ = parse_reasoning(raw, parser_factory_for(FakeProv(), req_no_tools)())
        assert "<|tool_call>" not in content_without
        assert "<turn|>" not in content_without

    def test_diffusion_checkpoint_does_not_advertise_tools(self, tmp_path):
        from heylook_llm.capabilities import template_supports_tools

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "diffusion_gemma", "canvas_length": 256}),
            encoding="utf-8",
        )
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{% if tools %}<|tool_call>{% endif %}"}),
            encoding="utf-8",
        )
        assert template_supports_tools(str(tmp_path)) is False

    def test_gemma4_post_tool_response_starts_inside_thinking(self):
        from types import SimpleNamespace
        from heylook_llm.reasoning_parser import parse_reasoning, parser_factory_for

        info = SimpleNamespace(
            chat_template="{% elif ns.prev_message_type == 'tool_response' and enable_thinking %}<|channel>thought\n{% endif %}<|tool_call>",
            has_gemma_channel_structure=True,
            has_harmony_structure=False,
            has_recipient_channel_structure=False,
            has_thinking_markers=False,
            prefills_thinking=False,
            special_tokens=frozenset({"<|tool_call>", "<tool_call|>", '<|"|>', "<turn|>"}),
        )

        class FakeProv:
            def template_info(self):
                return info

            def effective_thinking(self, req):
                return True

        req = ChatRequest(
            model="gemma4",
            messages=[
                ChatMessage(role="user", content="Update Coder"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[{"id": "c1", "type": "function", "function": {"name": "list_presets", "arguments": "{}"}}],
                ),
                ChatMessage(role="tool", tool_call_id="c1", content='{"presets": []}'),
            ],
            tools=[{"type": "function", "function": {"name": "list_presets", "parameters": {}}}],
        )
        raw = "All presets have been staged.<channel|>Done! I staged updates for all 3 presets.<turn|>"
        content, thinking = parse_reasoning(raw, parser_factory_for(FakeProv(), req)())
        assert thinking == "All presets have been staged."
        assert content == "Done! I staged updates for all 3 presets."
