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
from functools import lru_cache
from typing import Any, List, Optional, Protocol, Tuple


ParserDelta = Tuple[str, str]  # ("content" | "thinking", text)


class ReasoningParser(Protocol):
    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]: ...
    def flush(self) -> List[ParserDelta]: ...
    def reset(self) -> None: ...


@lru_cache(maxsize=16)
def _compile_strip_pattern(strip_tokens: frozenset[str]) -> Optional[re.Pattern]:
    """Alternation regex over the declared specials, sorted longest-first so
    ``<|endoftext|>`` wins over a substring match inside it. Returns None for
    the empty set -- callers check and skip the sub().

    Cached: parsers are instantiated per request, and for models like Mistral
    (~1000 reserved tokens) building this alternation isn't free. The compiled
    pattern is stateless, so sharing it across parser instances is safe --
    only parser BUFFERS must be per-request."""
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


# Gemma-4 canonical channel format: `<|channel>NAME\n BODY <channel|>` emitted
# inline in the model turn. Single-token delimiters (no <|message|>); the
# `thought` channel is reasoning, text outside channels is content.
_GEMMA_CONTROL_TOKENS = ("<|channel>", "<channel|>")
_GEMMA_MAX_TOKEN_LEN = max(len(t) for t in _GEMMA_CONTROL_TOKENS)
_GEMMA_CONTROL_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _GEMMA_CONTROL_TOKENS)
)
_GEMMA_THINKING_CHANNELS = frozenset({"thought"})


class GemmaChannelParser:
    """Gemma-4 channel parser.

    Simpler state machine than harmony: content is the default state (the
    model turn starts as visible text unless the model opens a channel), the
    channel name runs from ``<|channel>`` to the first newline, and
    ``<channel|>`` returns to content. Unknown channel bodies route to
    content, ``thought`` routes to thinking.
    """

    def __init__(self, strip_tokens: frozenset[str] = frozenset()):
        self._strip_pattern = _compile_strip_pattern(strip_tokens)
        self._buffer = ""
        self._state = "content"
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
        self._state = "content"
        self._channel_name_buf = ""
        self._current_channel = None

    def _drain(self, final: bool) -> List[ParserDelta]:
        if final:
            # the loop below emits the WHOLE buffer when final -- drop a
            # trailing partial control token first or an abort landing
            # mid-token flushes literal garbage (harmony's strip only knows
            # its own "<|" shapes; gemma's close token starts "<c")
            self._buffer = self._strip_partial_gemma_control(self._buffer)
        out: List[ParserDelta] = []
        progress = True
        while progress:
            progress = False

            if self._state == "content":
                m = _GEMMA_CONTROL_PATTERN.search(self._buffer)
                if m is not None:
                    if m.start() > 0:
                        out.append(("content", self._buffer[:m.start()]))
                    token = m.group()
                    self._buffer = self._buffer[m.end():]
                    if token == "<|channel>":
                        self._channel_name_buf = ""
                        self._state = "in_name"
                    # a stray <channel|> in content is dropped (structural noise)
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        out.append(("content", self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

            elif self._state == "in_name":
                nl = self._buffer.find("\n")
                m = _GEMMA_CONTROL_PATTERN.search(self._buffer)
                if nl != -1 and (m is None or nl < m.start()):
                    self._channel_name_buf += self._buffer[:nl]
                    self._current_channel = self._channel_name_buf.strip()
                    self._buffer = self._buffer[nl + 1:]
                    self._state = "in_body"
                    progress = True
                elif m is not None:
                    # channel closed (or reopened) before any newline: empty body
                    self._channel_name_buf += self._buffer[:m.start()]
                    self._current_channel = self._channel_name_buf.strip()
                    token = m.group()
                    self._buffer = self._buffer[m.end():]
                    if token == "<|channel>":
                        self._channel_name_buf = ""
                    else:
                        self._state = "content"
                        self._current_channel = None
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        self._channel_name_buf += self._buffer[:safe_len]
                        self._buffer = self._buffer[safe_len:]
                        progress = True

            elif self._state == "in_body":
                m = _GEMMA_CONTROL_PATTERN.search(self._buffer)
                if m is not None:
                    if m.start() > 0:
                        out.append(self._route_text(self._buffer[:m.start()]))
                    token = m.group()
                    self._buffer = self._buffer[m.end():]
                    if token == "<channel|>":
                        self._state = "content"
                        self._current_channel = None
                    else:
                        self._channel_name_buf = ""
                        self._state = "in_name"
                    progress = True
                else:
                    safe_len = self._safe_prefix_len(final)
                    if safe_len > 0:
                        out.append(self._route_text(self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

        if final and self._buffer:
            # buffer already partial-stripped above
            if self._state == "content":
                out.append(("content", self._buffer))
            elif self._state == "in_body":
                out.append(self._route_text(self._buffer))
            # in_name leftovers are structural -- dropped
            self._buffer = ""

        return [
            (kind, cleaned)
            for (kind, text) in out
            for cleaned in [_strip_specials(text, self._strip_pattern)]
            if cleaned
        ]

    def _safe_prefix_len(self, final: bool) -> int:
        if final:
            return len(self._buffer)
        limit = _GEMMA_MAX_TOKEN_LEN - 1
        scan_start = max(0, len(self._buffer) - limit)
        for i in range(scan_start, len(self._buffer)):
            if self._buffer[i] == "<":
                return i
        return len(self._buffer)

    @staticmethod
    def _strip_partial_gemma_control(text: str) -> str:
        """Drop a trailing PARTIAL gemma control token on final flush.

        Harmony's version only knows ``<|``-prefixed shapes; gemma's close
        token ``<channel|>`` starts ``<c``, so an abort landing mid-close
        would otherwise flush literal garbage like ``<chan``.
        """
        idx = text.rfind("<")
        if idx == -1:
            return text
        tail = text[idx:]
        if tail in _GEMMA_CONTROL_TOKENS:
            return text  # complete token; the drain loop handles it
        if any(t.startswith(tail) for t in _GEMMA_CONTROL_TOKENS):
            return text[:idx]
        return text

    def _route_text(self, text: str) -> ParserDelta:
        if self._current_channel in _GEMMA_THINKING_CHANNELS:
            return ("thinking", text)
        return ("content", text)


def resolve_enable_thinking(request_value, config) -> bool:
    """THE single request-vs-config resolution for the thinking flag, used
    by BOTH template application (mlx_provider) and parser arming (below).
    One implementation on purpose: a prompt built thinking-ON with a
    content-state parser (or vice versa) misroutes the whole reasoning
    trace, so the two sides may never drift."""
    if request_value is not None:
        return bool(request_value)
    return bool(config.get("enable_thinking", True)) if isinstance(config, dict) else False


def effective_thinking_flag(request_enable_thinking, provider) -> bool:
    """Provider-object adapter over ``resolve_enable_thinking`` for the API
    layers (provider may be None on shutdown paths)."""
    return resolve_enable_thinking(
        request_enable_thinking, getattr(provider, "config", None)
    )


def select_reasoning_parser(
    template_info: Any = None, *, thinking_enabled: bool | None = None
) -> ReasoningParser:
    """Pick a parser from ``ModelTemplateInfo``. ``None`` returns a
    pass-through so shutdown paths + unit tests don't need a full load.

    ``thinking_enabled`` is the request's EFFECTIVE thinking flag (request
    value falling back to the model-config default). It matters for
    templates that pre-fill an unclosed ``<think>`` into the generation
    prompt (``prefills_thinking``): the model's output then starts inside
    the block, so the parser must start in thinking state.
    """
    if template_info is None:
        return PassThroughParser()

    strip_tokens = getattr(template_info, "special_tokens", frozenset()) or frozenset()

    if getattr(template_info, "has_harmony_structure", False):
        return HarmonyChannelParser(strip_tokens=strip_tokens)

    if getattr(template_info, "has_gemma_channel_structure", False):
        return GemmaChannelParser(strip_tokens=strip_tokens)

    if getattr(template_info, "has_thinking_markers", False):
        from heylook_llm.thinking_parser import HybridThinkingParser
        return HybridThinkingParser(
            initial_thinking=bool(thinking_enabled)
            and getattr(template_info, "prefills_thinking", False),
            strip_tokens=strip_tokens,
        )

    return PassThroughParser(strip_tokens=strip_tokens)


def parse_reasoning(
    text: str, parser: ReasoningParser
) -> Tuple[str, Optional[str]]:
    """Non-streaming wrapper: returns ``(content, thinking_or_None)``.
    ``thinking`` stays ``None`` when the parser emitted nothing to that
    channel -- the non-streaming API handlers rely on thinking staying
    ``None`` when absent."""
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
