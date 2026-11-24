# src/heylook_llm/tool_calling.py
"""Engine-neutral tool-calling helpers shared by schema, capabilities, and providers.

Kept strictly zero-MLX (pure Python stdlib) so capability probes, wire
converters, CLI, and unit tests run without importing MLX or Metal. At
generation time, MLXProvider delegates actual format parsing to
``mlx_vlm.tools`` (which owns the 16 model-family parsers), while GGUF
delegates to ``llama-server``'s native ``--jinja`` autoparser.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Optional, Sequence

from heylook_llm.providers.base import InvalidGenerationRequest

# Mirrors mlx_vlm.tools.registry.SPECS marker signatures so unloaded capability
# probes derive `"tools"` from the model's chat template without importing MLX.
_TEMPLATE_TOOL_SPECS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("atem", (("<atem:function_calls>", "<atem:invoke"),)),
    ("gemma4", (("<|tool_call>",),)),
    ("cohere2_moe", (("<|START_ACTION|>",),)),
    ("deepseek_v4", (("<｜DSML｜tool_calls>",),)),
    ("minimax_m3", (("]<]minimax[>[<tool_call>",), ("<mm:think>",))),
    ("minimax_m2", (("<minimax:tool_call>",),)),
    ("minicpm5", (("<function name=", "<param name="),)),
    ("longcat", (("<longcat_tool_call>",),)),
    ("glm47", (("<arg_key>",),)),
    ("pythonic", (("<|tool_call_start|>", "<|tool_call_end|>"), ("<|tool_list_start|>",))),
    ("qwen3_coder", (("<tool_call>\\n<function=",), ("<tool_call>\n<function=",))),
    ("kimi_k2", (("<|tool_calls_section_begin|>",),)),
    ("mistral", (("[TOOL_CALLS]",),)),
    ("harmony", (("<|channel|>commentary", "<|call|>"),)),
    ("inkling", (("<|content_invoke_tool_json|>",),)),
    ("json_tools", (("<tool_call>", "tool_call.name"), ("<tool_call>", "tools"))),
)


def infer_tool_parser_name(chat_template: Optional[str]) -> Optional[str]:
    """Return the tool-call format name for a chat template, or None.

    Pure-Python mirror of ``mlx_vlm.tools.registry._infer_tool_parser`` so
    capability discovery never forces an MLX import.
    """
    if not chat_template or not isinstance(chat_template, str):
        return None
    for name, marker_groups in _TEMPLATE_TOOL_SPECS:
        if any(all(m in chat_template for m in group) for group in marker_groups):
            return name
    return None


def anthropic_tools_to_openai(tools: Optional[Sequence[Any]]) -> Optional[list[dict[str, Any]]]:
    """Convert Anthropic ``ToolDefinition`` list to OpenAI function-tool dicts."""
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        d = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
        fn: dict[str, Any] = {
            "name": d["name"],
            "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
        }
        if d.get("description"):
            fn["description"] = d["description"]
        out.append({"type": "function", "function": fn})
    return out


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Convert Anthropic ``tool_choice`` to OpenAI Chat Completions vocabulary."""
    if tool_choice is None:
        return None
    d = tool_choice.model_dump(exclude_none=True) if hasattr(tool_choice, "model_dump") else tool_choice
    if isinstance(d, str):
        return d
    if not isinstance(d, dict):
        return None
    kind = d.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        name = d.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


def openai_tool_call_to_anthropic(call: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI-shaped tool_call dict into an Anthropic ``tool_use`` dict."""
    call_id = call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"
    if "function" in call and isinstance(call["function"], dict):
        fn = call["function"]
        name = str(fn.get("name") or "").strip()
        raw_args = fn.get("arguments", {})
    else:
        name = str(call.get("name") or "").strip()
        raw_args = call.get("input", call.get("arguments", {}))

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        parsed = raw_args
    else:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {"value": parsed}

    return {
        "type": "tool_use",
        "id": str(call_id),
        "name": name,
        "input": parsed,
    }


def normalize_tool_calls_for_jinja(tool_calls: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Ensure ``tool_calls[].function.arguments`` are dicts for HF/MLX Jinja templates."""
    if not tool_calls:
        return tool_calls
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_copy = dict(tc)
        fn = tc_copy.get("function")
        if isinstance(fn, dict):
            fn_copy = dict(fn)
            args = fn_copy.get("arguments", {})
            if isinstance(args, str):
                try:
                    fn_copy["arguments"] = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    fn_copy["arguments"] = {}
            tc_copy["function"] = fn_copy
        out.append(tc_copy)
    return out


def normalize_tool_calls_for_openai_wire(tool_calls: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """Ensure ``tool_calls[].function.arguments`` are JSON strings for llama-server."""
    if not tool_calls:
        return tool_calls
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_copy = dict(tc)
        fn = tc_copy.get("function")
        if isinstance(fn, dict):
            fn_copy = dict(fn)
            args = fn_copy.get("arguments", {})
            if not isinstance(args, str):
                fn_copy["arguments"] = json.dumps(args, ensure_ascii=False)
            tc_copy["function"] = fn_copy
        out.append(tc_copy)
    return out


def _with_tool_choice_instruction(messages: list[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    copied = [dict(m) for m in messages]
    if copied and copied[0].get("role") == "system" and isinstance(copied[0].get("content"), str):
        copied[0]["content"] = f"{copied[0]['content']}\n\n{instruction}".strip()
        return copied
    copied.insert(0, {"role": "system", "content": instruction})
    return copied


def prepare_tool_choice_policy(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: Any,
) -> tuple[list[dict[str, Any]], Optional[list[dict[str, Any]]]]:
    """Enforce OpenAI-vocabulary ``tool_choice`` on MLX (mirrors mlx_vlm.tools.policy).

    Raises ``InvalidGenerationRequest`` (400) for invalid or mismatched choices.
    """
    available = list(tools or [])
    if tool_choice is None:
        return messages, available or None

    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return messages, None
        if tool_choice == "auto":
            return messages, available or None
        if tool_choice == "required":
            if not available:
                raise InvalidGenerationRequest("tool_choice 'any'/'required' requires at least one tool.")
            instruction = (
                "You must call one or more of the available functions to answer the "
                "user's request. Do not answer directly without calling a function."
            )
            return _with_tool_choice_instruction(messages, instruction), available
        raise InvalidGenerationRequest(f"Invalid tool_choice: {tool_choice!r}")

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise InvalidGenerationRequest("Specific tool_choice requires function.name.")
        selected = [
            t for t in available
            if isinstance(t.get("function"), dict) and t["function"].get("name") == name
        ]
        if not selected:
            raise InvalidGenerationRequest(f"tool_choice references unknown tool {name!r}.")
        instruction = (
            f"You must call the {name!r} function to answer the user's request. "
            "Do not call any other function and do not answer directly."
        )
        return _with_tool_choice_instruction(messages, instruction), selected

    raise InvalidGenerationRequest(f"Invalid tool_choice: {tool_choice!r}")


class ToolCallStreamFilter:
    """Buffers tool-call spans out of a token stream while streaming prose live.

    Once ``start_marker`` appears in the post-thinking stream, text is buffered
    until ``end_marker`` (or flush at EOF) and handed to ``extract_fn`` (which
    delegates to ``mlx_vlm.tools.extractor.process_tool_calls``). Thinking and
    non-tool prose stream with only ``len(start_marker) - 1`` chars of lookahead.
    """

    def __init__(
        self,
        start_marker: str,
        end_marker: str,
        extract_fn: Callable[[str], tuple[list[dict[str, Any]], str]],
    ):
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.extract_fn = extract_fn
        self._hold = max(len(start_marker) - 1, 0)
        self._buf = ""
        self._in_tool = False

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed text; returns ``(safe_prose_to_emit, completed_tool_calls)``."""
        if not text:
            return "", []
        self._buf += text
        emit_parts: list[str] = []
        calls: list[dict[str, Any]] = []

        while self._buf:
            if not self._in_tool:
                idx = self._buf.find(self.start_marker)
                if idx >= 0:
                    if idx > 0:
                        emit_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx:]
                    self._in_tool = True
                    continue
                if self._hold and len(self._buf) > self._hold:
                    cut = len(self._buf) - self._hold
                    emit_parts.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                elif not self._hold:
                    emit_parts.append(self._buf)
                    self._buf = ""
                break
            else:
                if not self.end_marker:
                    # Line-delimited end marker: wait for newline or EOF.
                    nl = self._buf.find("\n", len(self.start_marker))
                    if nl < 0:
                        break
                    span = self._buf[: nl + 1]
                    self._buf = self._buf[nl + 1 :]
                else:
                    end_idx = self._buf.find(self.end_marker, len(self.start_marker))
                    if end_idx < 0:
                        break
                    span_end = end_idx + len(self.end_marker)
                    span = self._buf[:span_end]
                    self._buf = self._buf[span_end:]
                parsed_calls, rem = self.extract_fn(span)
                if parsed_calls:
                    calls.extend(parsed_calls)
                elif rem:
                    emit_parts.append(rem)
                self._in_tool = False

        return "".join(emit_parts), calls

    def flush(self) -> tuple[str, list[dict[str, Any]]]:
        """Flush buffered tail at end of generation."""
        if not self._buf:
            return "", []
        tail = self._buf
        self._buf = ""
        if self._in_tool or self.start_marker in tail:
            self._in_tool = False
            parsed_calls, rem = self.extract_fn(tail)
            if parsed_calls:
                return rem, parsed_calls
            return tail, []
        return tail, []
