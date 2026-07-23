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


# Declared-specials stripping is ONE implementation for all four parsers:
# they route text, ``StripSpecials`` filters it (composed by
# ``select_reasoning_parser``). Design + history: docs/parser_strip_unification.md.
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


@lru_cache(maxsize=16)
def _partial_prefixes(strip_tokens: frozenset[str]) -> Tuple[frozenset[str], int]:
    """Every PROPER prefix of every declared special, plus the longest
    special's length -- the holdback's lookup table.

    A text tail that is a proper prefix of some special may still grow into
    that special on the next delta, so it must be held back rather than
    emitted. Membership in a precomputed set keeps the per-delta cost at
    ``O(max_token_len)`` lookups even for Mistral-sized token sets; cached
    for the same reason as the strip pattern (both are stateless)."""
    prefixes = {t[:i] for t in strip_tokens for i in range(1, len(t))}
    return frozenset(prefixes), max((len(t) for t in strip_tokens), default=0)


def _strip_specials(text: str, pattern: Optional[re.Pattern]) -> str:
    if pattern is None or not text:
        return text
    return pattern.sub("", text)


class StripSpecials:
    """Declared-specials filter every routing parser composes.

    Wraps a ``ReasoningParser`` and removes ``strip_tokens`` from the text it
    routes -- the defense against fast-detokenizer leaks of control tokens
    into user-visible output. Stripping happens AFTER routing, so the inner
    parser sees the raw stream its state machine was written for.

    The rolling per-kind holdback is the reason this is a wrapper and not a
    per-parser ``sub()``: an inner parser's own buffering can emit
    ``<|reserved_2`` and ``00000|>`` in separate deltas, and a per-delta
    sub() misses both halves. The held-back tail is the longest suffix that
    is still a proper prefix of some declared special -- sized by the STRIP
    SET, so it is correct no matter how long a special is relative to the
    inner parser's own structural tokens.
    """

    def __init__(self, inner: ReasoningParser, strip_tokens: frozenset[str]):
        self.inner = inner
        self._pattern = _compile_strip_pattern(strip_tokens)
        self._prefixes, self._max_len = _partial_prefixes(strip_tokens)
        self._pend_kind: Optional[str] = None
        self._pend = ""

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]:
        return self._clean(self.inner.process_chunk(text, token_id))

    def flush(self) -> List[ParserDelta]:
        return self._clean(self.inner.flush(), final=True)

    def reset(self) -> None:
        self.inner.reset()
        self._pend = ""
        self._pend_kind = None

    def _holdback_len(self, buf: str) -> int:
        """Length of the trailing run of ``buf`` that could still grow into a
        declared special (longest such suffix, 0 if none)."""
        for i in range(max(0, len(buf) - (self._max_len - 1)), len(buf)):
            if buf[i:] in self._prefixes:
                return len(buf) - i
        return 0

    def _flush_pend(self, out: List[ParserDelta]) -> None:
        if self._pend and self._pend_kind is not None:
            cleaned = _strip_specials(self._pend, self._pattern)
            if cleaned:
                out.append((self._pend_kind, cleaned))
        self._pend = ""
        self._pend_kind = None

    def _clean(
        self, deltas: List[ParserDelta], final: bool = False
    ) -> List[ParserDelta]:
        if self._pattern is None:
            return [d for d in deltas if d[1]]
        out: List[ParserDelta] = []
        for kind, text in deltas:
            # a pending tail belongs to the kind it was routed under: flush it
            # before the stream switches channels so ordering is preserved
            if self._pend_kind is not None and kind != self._pend_kind:
                self._flush_pend(out)
            self._pend_kind = kind
            # sub() removes only COMPLETE tokens, so carrying stripped text
            # forward is safe; partials survive into the held tail below
            buf = _strip_specials(self._pend + text, self._pattern)
            keep = self._holdback_len(buf)
            self._pend = buf[len(buf) - keep:] if keep else ""
            emit = buf[: len(buf) - keep] if keep else buf
            if emit:
                out.append((kind, emit))
        if final:
            self._flush_pend(out)
        return out


def _safe_prefix_len(buffer: str, max_token_len: int, final: bool) -> int:
    """How much of ``buffer`` can be emitted without splitting a STRUCTURAL
    control token (a different concern from declared-specials stripping:
    these tokens drive the state machine, so they may never be handed to a
    routing branch half-formed). Everything is safe on the final drain."""
    if final:
        return len(buffer)
    scan_start = max(0, len(buffer) - (max_token_len - 1))
    for i in range(scan_start, len(buffer)):
        if buffer[i] == "<":
            return i
    return len(buffer)


def _strip_partial_token(text: str, control_tokens: Tuple[str, ...]) -> str:
    """Drop a trailing PARTIAL structural control token before a final drain.

    The final drain emits the whole buffer, so an abort landing mid-token
    would otherwise flush literal garbage (``<|chan`` for harmony, ``<chan``
    for gemma's close token). Membership-based rather than shape-based so it
    generalizes to any control-token set; a COMPLETE trailing token is left
    alone for the drain loop to consume."""
    idx = text.rfind("<")
    if idx == -1:
        return text
    tail = text[idx:]
    if tail in control_tokens:
        return text
    if any(t.startswith(tail) for t in control_tokens):
        return text[:idx]
    return text


class PassThroughParser:
    """No reasoning structure. Text -> content."""

    def process_chunk(
        self, text: str, token_id: Optional[int] = None
    ) -> List[ParserDelta]:
        return [("content", text)] if text else []

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
    discarded. Non-structural specials leaking into a message payload are
    handled by ``StripSpecials``, not here.
    """

    def __init__(self):
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
        if final:
            # the loop below emits the WHOLE buffer when final -- drop a
            # trailing partial control token first, or an abort landing
            # mid-token flushes literal garbage like "<|chan"
            self._buffer = _strip_partial_token(self._buffer, _HARMONY_CONTROL_TOKENS)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _HARMONY_MAX_TOKEN_LEN, final)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _HARMONY_MAX_TOKEN_LEN, final)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _HARMONY_MAX_TOKEN_LEN, final)
                    if safe_len > 0:
                        out.append(self._route_text(self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

        # The buffer itself is always empty after a final drain (every state's
        # else-branch consumes buffer[:len(buffer)] when final). The channel
        # NAME buffer is not: a turn that ended between <|channel|> and
        # <|message|> left model text in it, and dropping that is text loss --
        # see GemmaChannelParser._drain for the live case that proved it.
        if final and self._state == "in_channel" and self._channel_name_buf:
            out.append(("content", self._channel_name_buf))
            self._channel_name_buf = ""

        return [d for d in out if d[1]]

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

    def __init__(self):
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
            # same reason as harmony: the loop below emits the WHOLE buffer
            # when final, so a trailing partial (gemma's close token starts
            # "<c") has to go before the drain, not after it
            self._buffer = _strip_partial_token(self._buffer, _GEMMA_CONTROL_TOKENS)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _GEMMA_MAX_TOKEN_LEN, final)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _GEMMA_MAX_TOKEN_LEN, final)
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
                    safe_len = _safe_prefix_len(
                        self._buffer, _GEMMA_MAX_TOKEN_LEN, final)
                    if safe_len > 0:
                        out.append(self._route_text(self._buffer[:safe_len]))
                        self._buffer = self._buffer[safe_len:]
                        progress = True

        # The buffer is always empty after a final drain, but the channel-NAME
        # buffer may not be, and it is NOT safe to treat as structural: gemma-4
        # sometimes emits a spurious `<|channel>` mid-answer and then just keeps
        # answering (live 2026-07-23: `['<|channel>', ' to', ' the', ' movies',
        # '!', '<turn|>']`). Everything after the open token landed here and was
        # dropped, so the user got an EMPTY reply on a normal 6-token stop --
        # the "immediate empty-EOS" long blamed on model behavior. At end of
        # turn, unrouted model text goes to content. The cost is that an abort
        # inside a LEGIT header surfaces a fragment ("thou"); losing a whole
        # answer is the worse failure.
        if final and self._state == "in_name" and self._channel_name_buf:
            out.append(("content", self._channel_name_buf))
            self._channel_name_buf = ""

        return [d for d in out if d[1]]

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

    Declared specials are stripped by ONE wrapper (``StripSpecials``) that
    every routing parser composes -- a model that declares none gets the
    bare parser, since the wrapper exists only to strip.
    """
    if template_info is None:
        return PassThroughParser()

    strip_tokens = frozenset(
        getattr(template_info, "special_tokens", frozenset()) or frozenset()
    )

    parser: ReasoningParser
    if getattr(template_info, "has_harmony_structure", False):
        parser = HarmonyChannelParser()
    elif getattr(template_info, "has_gemma_channel_structure", False):
        parser = GemmaChannelParser()
    elif getattr(template_info, "has_thinking_markers", False):
        from heylook_llm.thinking_parser import HybridThinkingParser
        parser = HybridThinkingParser(
            initial_thinking=bool(thinking_enabled)
            and getattr(template_info, "prefills_thinking", False),
        )
    else:
        parser = PassThroughParser()

    return StripSpecials(parser, strip_tokens) if strip_tokens else parser


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
