# src/heylook_llm/stop_sequences.py
"""Client stop sequences (`/v1/messages` `stop_sequences`), applied to the
reply text on the way out, identically on every engine.

Applied here rather than handed to an engine because the engines disagree
about what they would match: llama-server's `stop` runs over the raw
generated text, reasoning included, and MLX has no string stop at all. A
stop sequence is a statement about the REPLY, so it is matched against the
parsed reply text only; thinking never ends a generation.

Streaming holds back only the tail of the text that could still become a
match (the longest suffix that is a proper prefix of some sequence), so a
sequence split across chunks is caught and nothing after it is ever sent.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Bounds on the request field, enforced by MessageCreateRequest.
MAX_STOP_SEQUENCES = 16
MAX_STOP_SEQUENCE_CHARS = 256


def earliest(text: str, sequences: Iterable[str]) -> Optional[tuple[int, str]]:
    """(index, sequence) of the first stop sequence in ``text``, or None.
    The earliest start wins; at the same start, the longer sequence."""
    best: Optional[tuple[int, str]] = None
    for seq in sequences:
        at = text.find(seq)
        if at < 0:
            continue
        if best is None or at < best[0] or (at == best[0] and len(seq) > len(best[1])):
            best = (at, seq)
    return best


def truncate(text: str, sequences: Iterable[str]) -> tuple[str, Optional[str]]:
    """``text`` cut before its first stop sequence, and that sequence."""
    hit = earliest(text, sequences)
    return (text, None) if hit is None else (text[:hit[0]], hit[1])


class StopSequenceFilter:
    """Incremental form of :func:`truncate` for a stream of reply text."""

    def __init__(self, sequences: Iterable[str]):
        self.sequences = [s for s in sequences if s]
        self.pending = ""
        self.matched: Optional[str] = None

    def feed(self, text: str) -> str:
        """The part of ``text`` that is safe to send now."""
        if self.matched is not None:
            return ""
        buf = self.pending + text
        hit = earliest(buf, self.sequences)
        if hit is not None:
            self.matched = hit[1]
            self.pending = ""
            return buf[:hit[0]]
        keep = 0
        for k in range(min(len(buf), max((len(s) for s in self.sequences), default=1) - 1), 0, -1):
            tail = buf[-k:]
            if any(s.startswith(tail) for s in self.sequences):
                keep = k
                break
        self.pending = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        """Release the held tail: the reply ended without a match."""
        out, self.pending = self.pending, ""
        return out if self.matched is None else ""
