# src/heylook_llm/schema/__init__.py
#
# New API schema module (Anthropic Messages-inspired).
# Purely additive -- does not replace existing config.py models.
# Converters bridge between old (OpenAI) and new formats.

from heylook_llm.schema.content_blocks import (
    TextBlock,
    ImageBlock,
    ThinkingBlock,
    LogprobsBlock,
    HiddenStatesBlock,
    InputContentBlock,
    OutputContentBlock,
)
from heylook_llm.schema.messages import (
    Message,
    MessageCreateRequest,
)
from heylook_llm.schema.responses import (
    Usage,
    PerformanceInfo,
    MessageResponse,
)
from heylook_llm.schema.streaming import (
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    StreamEvent,
)
from heylook_llm.schema.batch import (
    BatchRequest,
    BatchResult,
    BatchResponse,
)
from heylook_llm.schema.system import (
    SystemCapabilities,
    SystemPerformance,
)

__all__ = [
    # Content blocks
    "TextBlock",
    "ImageBlock",
    "ThinkingBlock",
    "LogprobsBlock",
    "HiddenStatesBlock",
    "InputContentBlock",
    "OutputContentBlock",
    # Messages
    "Message",
    "MessageCreateRequest",
    # Responses
    "Usage",
    "PerformanceInfo",
    "MessageResponse",
    # Streaming
    "MessageStartEvent",
    "ContentBlockStartEvent",
    "ContentBlockDeltaEvent",
    "ContentBlockStopEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    "StreamEvent",
    # Batch
    "BatchRequest",
    "BatchResult",
    "BatchResponse",
    # System
    "SystemCapabilities",
    "SystemPerformance",
]
