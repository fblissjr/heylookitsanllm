"""Why a gguf request reused what it did (plan W5, gguf "why").

llama-server reports how much of a prompt it reused (`cache_n`), never where
that came from or why it stopped short. Two things heylook can see on its own
side fill part of that gap, and this module holds both, in memory, for ONE
llama-server process:

- a fingerprint per request: one hash per wire message plus one for the
  request settings that change the rendered prompt (chat_template_kwargs).
  Hashes only; no prompt text is kept.
- the process's own log lines about its host-RAM prompt cache, read from
  the subprocess pipe (never a file, so observability_level=off still writes
  nothing): an entry skipped for exceeding the budget, entries evicted, the
  budget shrinking after a failed allocation, the server going to sleep.
  At llama-server's default verbosity these are the only cache lines it
  prints; where a restore came from is trace-level and invisible here.

From those, `explain` labels a request that reused less than it could have.
A label heylook can be sure of is plain (`cold`, `no_common_prefix`); one it
infers is `probable_*`, because the log cannot confirm it. The durable fix is
upstream: a cache source in llama-server's `timings`.

Stdlib only: the gguf provider imports no MLX.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

# Log lines, keyed on llama-server's own format strings (tools/server/
# server-task.cpp and server-context.cpp). `search`, not `match`: the log
# prefix ("srv  update: ", timestamps, colour) varies with build flags.
# Pinned against the build tree by test_cache_log_patterns_match_the_build.
LOG_PATTERNS = (
    ("budget_skipped", re.compile(r"prompt state size [\d.]+ MiB exceeds cache size limit")),
    ("evicted", re.compile(r"removing oldest entry \(size = [\d.]+ MiB\)")),
    ("budget_reduced", re.compile(r"cache size limit reduced to [\d.]+ MiB")),
    ("slept", re.compile(r"server is entering sleeping state")),
)

# A fully extended conversation should reuse about the whole previous prompt
# (a checkpointed model rolls back to its last checkpoint, which sits near the
# end of the prompt). Reusing less than this share of it is "stopped short".
SHORT_FRACTION = 0.5

# Recent requests remembered per process. The RAM cache holds a handful of
# conversations; a request matching none of the remembered ones is reported
# as no_common_prefix, which is what llama-server would conclude too.
HISTORY = 32


def classify_line(line: str) -> Optional[str]:
    for kind, pattern in LOG_PATTERNS:
        if pattern.search(line):
            return kind
    return None


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]


def fingerprint(payload: dict) -> tuple[tuple[str, ...], str]:
    """(one hash per wire message, a hash of the render-affecting settings)."""
    messages = tuple(_digest(m) for m in payload.get("messages") or ())
    return messages, _digest(payload.get("chat_template_kwargs") or {})


@dataclass(frozen=True, slots=True)
class _Seen:
    messages: tuple[str, ...]
    head: str
    prompt_tokens: int
    events_after: int  # the last event sequence number when this request finished


def _common(a: tuple, b: tuple) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class CacheWitness:
    """Per-process memory of recent requests and cache log events.

    `note_line` runs on the log-pump thread, `explain`/`record` on the
    request thread, so every access takes the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: deque[_Seen] = deque(maxlen=HISTORY)
        # (sequence number, kind); bounded, so a long-lived process cannot
        # grow it. An event old enough to fall out is older than any
        # remembered request it could explain.
        self._events: deque[tuple[int, str]] = deque(maxlen=HISTORY * 8)
        self._seq = 0

    def note_line(self, line: str) -> None:
        kind = classify_line(line)
        if kind is not None:
            with self._lock:
                self._seq += 1
                self._events.append((self._seq, kind))

    def explain(self, messages: tuple[str, ...], head: str, report):
        """`report` with `cause` and `reason` set when it reused less than a
        remembered request says it could have; unchanged otherwise."""
        with self._lock:
            seen = list(self._seen)
            events = list(self._events)

        def since(s: _Seen) -> set[str]:
            return {kind for seq, kind in events if seq > s.events_after}

        cached = report.cached_tokens
        if not seen:
            if cached:
                return report
            return replace(report, cause="cold", reason=(
                "the first request since this llama-server process started"))

        last = seen[-1]
        if "slept" in since(last):
            if cached:
                return report
            return replace(report, cause="cold", reason=(
                "llama-server slept since the last request and dropped its caches"))

        # The remembered request this one extends furthest; the latest on a tie.
        best, depth = None, 0
        for s in seen:
            k = _common(s.messages, messages)
            if k >= depth and k:
                best, depth = s, k
        if best is None:
            if cached:
                return report
            return replace(report, cause="no_common_prefix", reason=(
                "no recent request shares this one's first message"))

        # Only a request this one fully extends says how much should have
        # been reusable (its whole prompt); a partial share (a common system
        # prompt) is too small a claim to call anything a shortfall.
        if depth < len(best.messages) or cached >= SHORT_FRACTION * best.prompt_tokens:
            return report
        head_note = ("" if best.head == head else
                     " (the request's template settings also changed since then)")
        if best is last:
            return replace(report, cause="probable_template_diverged", reason=(
                f"the slot held the previous turn ({best.prompt_tokens} tokens) "
                f"and this request repeats its messages, but reused only "
                f"{cached}: the re-rendered prompt probably differs early"
                f"{head_note}"))
        after = since(best)
        if "budget_skipped" in after:
            return replace(report, cause="probable_budget_skipped", reason=(
                "this conversation's state was probably not kept: a prompt "
                "state larger than the host-RAM prompt-cache budget was "
                "skipped when another conversation took the slot"))
        if "evicted" in after or "budget_reduced" in after:
            return replace(report, cause="probable_evicted", reason=(
                "this conversation's saved state was probably evicted from "
                "the host-RAM prompt cache to make room for another"))
        return replace(report, cause="probable_template_diverged", reason=(
            f"an earlier request with these messages ({best.prompt_tokens} "
            f"tokens) should have been restorable, and the log shows no "
            f"eviction or skip since; the re-rendered prompt probably differs "
            f"early{head_note}"))

    def record(self, messages: tuple[str, ...], head: str, prompt_tokens: int) -> None:
        with self._lock:
            self._seen.append(_Seen(messages, head, prompt_tokens, self._seq))
