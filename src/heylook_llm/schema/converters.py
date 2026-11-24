# src/heylook_llm/schema/converters.py
#
# Conversion between the Messages wire (schema.MessageCreateRequest /
# schema.MessageResponse) and the INTERNAL request every provider takes
# (config.ChatRequest, which still speaks OpenAI's vocabulary: content_parts,
# finish_reason). The conversion lives in the route layer so the provider
# interface never changes with the wire. The OpenAI chat route that once
# passed ChatRequest through untouched was removed in v1.79.66; ChatRequest
# stays because the providers are written against it.
#
# Flow:
#   client -> /v1/messages -> MessageCreateRequest
#     -> to_chat_request() -> ChatRequest -> provider -> chunks
#     -> from_openai_response_dict() -> MessageResponse -> client

import json
import uuid
from typing import Dict, List, Optional

from heylook_llm.config import ChatMessage, ChatRequest
from heylook_llm.schema.content_blocks import (
    AudioBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.schema.responses import (
    MessageResponse,
    PerformanceInfo,
    StopReason,
    Usage,
)
from heylook_llm.tool_calling import (
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_tool_call_to_anthropic,
)


def thinking_switch(thinking) -> Optional[bool]:
    """The thinking switch from either wire form: a bool as given, the
    object form's ``type`` as a bool, absent as None (the model's default)."""
    if thinking is None or isinstance(thinking, bool):
        return thinking
    return None if thinking.type is None else thinking.type == "enabled"


def to_chat_request(request: MessageCreateRequest) -> ChatRequest:
    """Convert a new-format MessageCreateRequest to the existing ChatRequest.

    This is the key bridge: the provider layer only speaks ChatRequest,
    so we translate before handing off to the router.
    """
    chat_messages: List[ChatMessage] = []

    # Inject system prompt as a system message (OpenAI format)
    if request.system:
        chat_messages.append(ChatMessage(role="system", content=request.system))

    # Convert each Message to ChatMessage
    for msg in request.messages:
        if isinstance(msg.content, str):
            chat_messages.append(ChatMessage(role=msg.role, content=msg.content))
        else:
            # Convert content blocks to OpenAI content_parts, tool_calls, and
            # tool-role result messages. Thinking blocks leave the content list
            # for ChatMessage.thinking so providers hand them to the template
            # as reasoning.
            content_parts = []
            thinking_parts: List[str] = []
            tool_calls: List[Dict] = []
            tool_results: List[ChatMessage] = []
            for block in msg.content:
                if isinstance(block, ThinkingBlock):
                    if block.thinking:
                        thinking_parts.append(block.thinking)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, ensure_ascii=False),
                        },
                    })
                elif isinstance(block, ToolResultBlock):
                    if isinstance(block.content, str):
                        res_content = block.content
                    else:
                        res_parts = []
                        for part in block.content:
                            if isinstance(part, TextBlock):
                                res_parts.append(part.text)
                        res_content = "\n".join(res_parts)
                    tool_results.append(ChatMessage(
                        role="tool",
                        tool_call_id=block.tool_use_id,
                        content=res_content,
                    ))
                elif isinstance(block, TextBlock):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.source_type == "base64" and block.data:
                        url = f"data:{block.media_type};base64,{block.data}"
                    elif block.url:
                        url = block.url
                    else:
                        continue
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })
                elif isinstance(block, AudioBlock):
                    # OpenAI wire shape: RAW base64 in input_audio.data (a
                    # data: URI here is rejected downstream), or a URL.
                    input_audio: dict = {}
                    if block.source_type == "base64" and block.data:
                        input_audio["data"] = block.data
                        if block.media_type and "/" in block.media_type:
                            input_audio["format"] = block.media_type.split("/", 1)[1]
                    elif block.url:
                        input_audio["url"] = block.url
                    else:
                        continue
                    content_parts.append({
                        "type": "input_audio",
                        "input_audio": input_audio,
                    })

            if tool_results:
                chat_messages.extend(tool_results)
                if content_parts:
                    chat_messages.append(ChatMessage(role=msg.role, content=content_parts))
            else:
                # Plain assistant text + tool_calls: keep content as "" if empty
                # when tool_calls are present so Jinja templates don't iterate an
                # empty list.
                msg_content = content_parts if content_parts else ("" if tool_calls else content_parts)
                chat_messages.append(ChatMessage(
                    role=msg.role,
                    content=msg_content,
                    thinking=("".join(thinking_parts) or None) if msg.role == "assistant" else None,
                    tool_calls=(tool_calls or None) if msg.role == "assistant" else None,
                ))

    return ChatRequest(
        model=request.model,
        messages=chat_messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        stream=request.stream,
        seed=request.seed,
        enable_thinking=thinking_switch(request.thinking),
        thinking_budget_tokens=getattr(request.thinking, "budget_tokens", None),
        reasoning_effort=request.reasoning_effort,
        response_schema=request.response_schema(),
        tools=anthropic_tools_to_openai(request.tools),
        tool_choice=anthropic_tool_choice_to_openai(request.tool_choice),
    )


# OpenAI finish_reason -> Anthropic stop_reason. ONE copy on purpose: this
# lived as an inline if/elif here AND as a raw passthrough in
# messages_api.py's streaming path, so the two wires disagreed about the same
# generation -- the streaming arm reported the provider's "length" while the
# non-streaming arm reported it too, and neither was a value the Messages
# spec defines. Providers speak OpenAI's vocabulary because the internal
# ChatRequest does; the boundary that renames it is here.
STOP_REASON_FROM_FINISH_REASON: Dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_calls": "tool_use",
    # Already-Anthropic values pass through, so a provider that learns the
    # richer vocabulary needs no change here.
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
}


def to_stop_reason(finish_reason: Optional[str]) -> StopReason:
    """Map a provider finish_reason onto the Messages stop_reason vocabulary.

    Unknown and absent both mean "the model stopped on its own": that is what
    an absent finish_reason has always meant on this path, and inventing a
    new value for an unrecognised one would put a non-spec string back on the
    wire -- the exact defect this function exists to remove.
    """
    default: StopReason = "end_turn"
    return STOP_REASON_FROM_FINISH_REASON.get(finish_reason or "stop", default)


def from_openai_response_dict(
    response_dict: Dict,
    metadata: Optional[Dict[str, str]] = None,
) -> MessageResponse:
    """Convert a provider-shaped (OpenAI-vocabulary) response dict to a
    MessageResponse. The dict is what the non-streaming Messages handler
    assembles from provider chunks; it never touches the wire itself.
    """
    content_blocks = []

    # Extract text and thinking from choices
    choices = response_dict.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})

        # Thinking content
        thinking = message.get("thinking")
        if thinking:
            content_blocks.append(ThinkingBlock(text=thinking))

        # Main text content
        text = message.get("content", "")
        if text:
            content_blocks.append(TextBlock(text=text))

        # Tool calls -> Anthropic ToolUseBlock(s)
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            anth = openai_tool_call_to_anthropic(tc)
            content_blocks.append(ToolUseBlock(
                id=anth["id"],
                name=anth["name"],
                input=anth["input"],
            ))

    # Usage
    usage_dict = response_dict.get("usage", {})
    # `input_tokens` / `cache_read_input_tokens` are already in Anthropic's
    # sense when the caller computed them (messages_api, via
    # perf_collector.usage_counts); a bare OpenAI-shaped dict's prompt_tokens
    # is taken as-is, with nothing known about reuse.
    usage = Usage(
        input_tokens=usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)),
        cache_read_input_tokens=usage_dict.get("cache_read_input_tokens"),
        output_tokens=usage_dict.get("completion_tokens", 0),
        thinking_tokens=usage_dict.get("thinking_tokens"),
        content_tokens=usage_dict.get("content_tokens"),
    )

    # Performance
    performance = None
    perf_dict = response_dict.get("performance")
    if perf_dict:
        # `.get(key)` throughout, never `.get(key, 0)`: the rates defaulted to
        # 0 while the model declared them required, which made a DROPPED rate
        # indistinguishable from a measured zero -- and that is exactly what
        # made `test_rates_and_duration_are_present` unable to fail (v1.79.51).
        # Absent now means None, which the model declares and a client can see.
        # Validated WHOLE, never re-listed: build_performance already emits
        # only PerformanceInfo's declared fields, and a field list here was a
        # hand copy that silently dropped every field added after it (the W5
        # cache/speculative reports, found by their route test).
        performance = PerformanceInfo.model_validate(perf_dict)

    # Determine stop reason
    raw_finish = choices[0].get("finish_reason") if choices else None
    if any(isinstance(b, ToolUseBlock) for b in content_blocks) and raw_finish in (None, "stop", "end_turn"):
        raw_finish = "tool_calls"
    finish_reason = to_stop_reason(raw_finish)

    return MessageResponse(
        id=f"msg_{uuid.uuid4().hex[:16]}",
        model=response_dict.get("model", "unknown"),
        content=content_blocks,
        stop_reason=finish_reason,
        usage=usage,
        performance=performance,
        metadata=metadata,
    )


