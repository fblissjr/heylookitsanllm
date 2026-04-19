"""Format-aware reasoning parser.

Reads format from the model's own ``chat_template`` + ``tokenizer_config.json``
(plus ``tokenizer.json``) via ``ModelTemplateInfo`` rather than probing the
tokenizer's runtime API. Some tokenizers mark control tokens
``"special": true`` in ``added_tokens_decoder`` but don't always advertise
them through ``tokenizer.all_special_tokens``, so the template + tokenizer
files on disk are the reliable source of truth.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Protocol, Tuple


ParserDelta = Tuple[str, str]  # ("content" | "thinking", text)


class ReasoningParser(Protocol):
    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]: ...
    def flush(self) -> List[ParserDelta]: ...
    def reset(self) -> None: ...


def _compile_strip_pattern(strip_tokens: frozenset[str]) -> Optional[re.Pattern]:
    """Alternation regex over the declared specials, sorted longest-first so
    ``<|endoftext|>`` wins over a substring match inside it. Returns None for
    the empty set -- callers check and skip the sub()."""
    if not strip_tokens:
        return None
    ordered = sorted(strip_tokens, key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in ordered))


def _strip_specials(text: str, pattern: Optional[re.Pattern]) -> str:
    if pattern is None or not text:
        return text
    return pattern.sub("", text)


class PassThroughParser:
    """No reasoning structure. Text -> content. Strips declared specials
    as a defense against fast-detokenizer leaks."""

    def __init__(self, strip_tokens: frozenset[str] = frozenset()):
        self._strip_tokens = strip_tokens
        self._strip_pattern = _compile_strip_pattern(strip_tokens)

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]:
        if not text:
            return []
        cleaned = _strip_specials(text, self._strip_pattern)
        return [("content", cleaned)] if cleaned else []

    def flush(self) -> List[ParserDelta]:
        return []

    def reset(self) -> None:
        pass


# Structural control tokens for the OpenAI harmony multi-channel format.
# Presence of ``<|channel|>`` + ``<|message|>`` literals in the chat
# template triggers harmony parsing (see ``ModelTemplateInfo.has_harmony_structure``).
_HARMONY_CONTROL_TOKENS = (
    "<|start|>",
    "<|end|>",
    "<|channel|>",
    "<|message|>",
    "<|return|>",
    "<|call|>",
)
_HARMONY_MAX_TOKEN_LEN = max(len(t) for t in _HARMONY_CONTROL_TOKENS)
_HARMONY_CONTROL_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _HARMONY_CONTROL_TOKENS)
)
_HARMONY_ANALYSIS_CHANNELS = frozenset({"analysis", "commentary"})


class HarmonyChannelParser:
    """OpenAI harmony multi-channel parser.

    State machine over a rolling buffer. ``preamble`` exists separately from
    ``outside`` as a safety net for models that emit free text before the
    first ``<|start|>`` -- that text routes to content rather than being
    discarded. ``strip_tokens`` covers fast-detokenizer leaks of
    non-structural specials that slip into a message payload.
    """

    def __init__(self, strip_tokens: frozenset[str] = frozenset()):
        self._strip_tokens = strip_tokens
        self._strip_pattern = _compile_strip_pattern(strip_tokens)
        self._buffer = ""
        self._state = "preamble"
        self._channel_name_buf = ""
        self._current_channel: Optional[str] = None

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]:
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> List[ParserDelta]:
        return self._drain(final=True)

    def reset(self) -> None:
        self._buffer = ""
        self._state = "preamble"
        self._channel_name_buf = ""
        self._current_channel = None

    def _drain(self, final: bool) -> List[ParserDelta]:
        out: List[ParserDelta] = []
        progress = True
        while progress:
            progress = False

            if self._state == "preamble" or self._state == "outside":
                idx, token = self._find_next_control_token()
                if idx is not None and token is not None:
                    if self._state == "preamble" and idx > 0:
                        out.append(("content", self._buffer[:idx]))
                    self._buffer = self._buffer[idx + len(token):]
                    self._consume_control_token(token)
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        if self._state == "preamble":
                            out.append(("content", self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

            elif self._state == "in_channel":
                msg_idx = self._buffer.find("<|message|>")
                if msg_idx != -1:
                    self._channel_name_buf += self._buffer[:msg_idx]
                    self._current_channel = self._channel_name_buf.strip()
                    self._channel_name_buf = ""
                    self._buffer = self._buffer[msg_idx + len("<|message|>"):]
                    self._state = "in_message"
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        self._channel_name_buf += self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        progress = True

            elif self._state == "in_message":
                idx, token = self._find_next_control_token()
                if idx is not None and token is not None:
                    if idx > 0:
                        out.append(self._route_text(self._buffer[:idx]))
                    self._buffer = self._buffer[idx + len(token):]
                    self._consume_control_token(token)
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        out.append(self._route_text(self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

        if final and self._buffer:
            leftover = self._strip_partial_control(self._buffer)
            if leftover:
                if self._state == "preamble":
                    out.append(("content", leftover))
                elif self._state == "in_message":
                    out.append(self._route_text(leftover))
            self._buffer = ""

        return [
            (kind, cleaned)
            for (kind, text) in out
            for cleaned in [_strip_specials(text, self._strip_pattern)]
            if cleaned
        ]

    def _find_next_control_token(self) -> Tuple[Optional[int], Optional[str]]:
        m = _HARMONY_CONTROL_PATTERN.search(self._buffer)
        if m is None:
            return None, None
        return m.start(), m.group()

    def _consume_control_token(self, token: str) -> None:
        if token == "<|channel|>":
            self._channel_name_buf = ""
            self._state = "in_channel"
        elif token == "<|message|>":
            if self._current_channel is None:
                self._current_channel = "final"
            self._state = "in_message"
        else:  # <|start|>, <|end|>, <|return|>, <|call|>
            self._state = "outside"

    def _safe_prefix_len(self, final: bool) -> int:
        if final:
            return len(self._buffer)
        limit = _HARMONY_MAX_TOKEN_LEN - 1
        scan_start = max(0, len(self._buffer) - limit)
        for i in range(scan_start, len(self._buffer)):
            if self._buffer[i] == "<":
                return i
        return len(self._buffer)

    @staticmethod
    def _strip_partial_control(text: str) -> str:
        idx = text.rfind("<")
        if idx == -1:
            return text
        tail = text[idx:]
        if tail.startswith("<|") and not tail.endswith("|>"):
            return text[:idx]
        if tail == "<":
            return text[:idx]
        return text

    def _route_text(self, text: str) -> ParserDelta:
        if self._current_channel in _HARMONY_ANALYSIS_CHANNELS:
            return ("thinking", text)
        return ("content", text)


def select_reasoning_parser(template_info: Any = None) -> ReasoningParser:
    """Pick a parser from ``ModelTemplateInfo``. ``None`` returns a
    pass-through so shutdown paths + unit tests don't need a full load."""
    if template_info is None:
        return PassThroughParser()

    strip_tokens = getattr(template_info, "special_tokens", frozenset()) or frozenset()

    if getattr(template_info, "has_harmony_structure", False):
        return HarmonyChannelParser(strip_tokens=strip_tokens)

    if getattr(template_info, "has_thinking_markers", False):
        from heylook_llm.thinking_parser import HybridThinkingParser
        return HybridThinkingParser()

    return PassThroughParser(strip_tokens=strip_tokens)


def parse_reasoning(
    text: str, parser: ReasoningParser
) -> Tuple[str, Optional[str]]:
    """Non-streaming wrapper: returns ``(content, thinking_or_None)``.
    ``thinking`` stays ``None`` when the parser emitted nothing to that
    channel -- preserves the ``parse_thinking_content`` contract callers
    (non-streaming API handler) rely on."""
    parser.reset()
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    for kind, chunk in parser.process_chunk(text):
        (content_parts if kind == "content" else thinking_parts).append(chunk)
    for kind, chunk in parser.flush():
        (content_parts if kind == "content" else thinking_parts).append(chunk)
    content = "".join(content_parts)
    thinking = "".join(thinking_parts) if thinking_parts else None
    return content, thinking
