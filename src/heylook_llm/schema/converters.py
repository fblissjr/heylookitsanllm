# src/heylook_llm/schema/converters.py
#
# Bidirectional conversion between the existing OpenAI-compatible format
# (config.ChatRequest / config.ChatCompletionResponse) and the new Messages
# API format (schema.MessageCreateRequest / schema.MessageResponse).
#
# The conversion layer lives here so that:
# 1. The new /v1/messages endpoint can convert to ChatRequest before calling providers
# 2. The old /v1/chat/completions endpoint continues to work unchanged
# 3. Provider interface stays the same -- conversion is in the route layer
#
# Flow for new endpoint:
#   Frontend -> /v1/messages -> MessageCreateRequest
#     -> to_chat_request() -> ChatRequest -> provider -> chunks
#     -> from_chat_completion_response() -> MessageResponse -> Frontend
#
# Flow for old endpoint (unchanged):
#   Frontend -> /v1/chat/completions -> ChatRequest -> provider -> chunks
#     -> ChatCompletionResponse -> Frontend

import uuid
from typing import Dict, List, Optional

from heylook_llm.config import ChatMessage, ChatRequest
from heylook_llm.schema.content_blocks import (
    ImageBlock,
    LogprobsBlock,
    TextBlock,
    ThinkingBlock,
)
from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.schema.responses import MessageResponse, PerformanceInfo, Usage


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
            # Convert content blocks to OpenAI content_parts
            content_parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
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
            chat_messages.append(ChatMessage(role=msg.role, content=content_parts))

    # Map stream_options
    stream_options = None
    if request.stream_options:
        stream_options = {"include_usage": request.stream_options.include_usage}

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
        include_performance=request.include_performance,
        seed=request.seed,
        enable_thinking=request.thinking,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs,
        stream_options=stream_options,
    )


def from_openai_response_dict(
    response_dict: Dict,
    metadata: Optional[Dict[str, str]] = None,
) -> MessageResponse:
    """Convert an OpenAI-format response dict to a MessageResponse.

    Works with the dict format currently returned by non-streaming handlers
    in api.py (the raw dict, not ChatCompletionResponse).
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

        # Logprobs
        logprobs_data = choice.get("logprobs")
        if logprobs_data and logprobs_data.get("content"):
            content_blocks.append(LogprobsBlock(tokens=logprobs_data["content"]))

    # Usage
    usage_dict = response_dict.get("usage", {})
    usage = Usage(
        input_tokens=usage_dict.get("prompt_tokens", 0),
        output_tokens=usage_dict.get("completion_tokens", 0),
        thinking_tokens=usage_dict.get("thinking_tokens"),
        content_tokens=usage_dict.get("content_tokens"),
    )

    # Performance
    performance = None
    perf_dict = response_dict.get("performance")
    if perf_dict:
        performance = PerformanceInfo(
            prompt_tps=perf_dict.get("prompt_tps", 0),
            generation_tps=perf_dict.get("generation_tps", 0),
            peak_memory_gb=perf_dict.get("peak_memory_gb"),
            thinking_duration_ms=perf_dict.get("thinking_duration_ms"),
            content_duration_ms=perf_dict.get("content_duration_ms"),
            total_duration_ms=perf_dict.get("total_duration_ms"),
        )

    # Determine stop reason
    finish_reason = "stop"
    if choices:
        fr = choices[0].get("finish_reason", "stop")
        if fr == "length":
            finish_reason = "length"
        elif fr in ("stop", None):
            finish_reason = "stop"
        else:
            finish_reason = "stop"

    return MessageResponse(
        id=f"msg_{uuid.uuid4().hex[:16]}",
        model=response_dict.get("model", "unknown"),
        content=content_blocks,
        stop_reason=finish_reason,
        usage=usage,
        performance=performance,
        metadata=metadata,
    )


def to_openai_messages(request: MessageCreateRequest) -> List[Dict]:
    """Convert MessageCreateRequest messages to OpenAI-format message dicts.

    Useful for logging, debugging, or passing to external APIs.
    """
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})

    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.source_type == "base64" and block.data:
                        url = f"data:{block.media_type};base64,{block.data}"
                    elif block.url:
                        url = block.url
                    else:
                        continue
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": msg.role, "content": parts})

    return messages
