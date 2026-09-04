# src/heylook_llm/request_registry.py
"""In-flight request id -> the generation's cooperative abort signal.

Exists so a client can cancel a NON-STREAMING request. A streaming request is
already cancellable by hanging up: the server is writing chunks, so it notices
the peer is gone. A non-streaming request writes nothing until the generation
finishes, so an abandoned client is never detected and the run continues to
completion -- and on a server that serialises generation, it blocks everything
queued behind it. (A consuming client's timings motivated this; they are NOT
recorded here. They were uncontrolled for model, quant, context and machine,
and this repo's rule is that no performance numbers live in tracked files --
the mechanism is what justifies the code, and the mechanism does not need
them.)

This registry does NOT close that measurement's case on its own, and it is
worth being precise about which half it closes. It gives a client an explicit
way to say "stop that one". A client that HANGS UP rather than calling
``DELETE`` still leaves the generation running -- detecting that is disconnect
polling, a separate mechanism, deliberately not built (owner call 2026-08-30).

The abort plumbing itself is not new: ``AbortEvent`` already reaches both
engines (``generation_core.py`` breaks its decode loop on it, and the gguf
provider's chunk pump takes one). All that was missing was a way to name a
running request from outside it.

WHY A SET PER ID, not one event. The id is CLIENT-SUPPLIED, so uniqueness is
not ours to assume: two concurrent requests can carry the same
``X-Request-ID`` through a retry, a buggy client, or a shared correlation id.
A plain ``dict[str, AbortEvent]`` would let the second registration orphan the
first -- leaving a running generation nobody can name, which is precisely the
condition this module exists to remove. Cancelling an id therefore cancels
every generation currently registered under it, and the count is returned so
the route can tell "cancelled" from "no such request".

LIFETIME is bounded by liveness, not by a clock. Entries are removed in a
``finally``, so the map holds in-flight requests only and there is no TTL
sweeper to get wrong. An id becomes unknown the moment its request finishes,
which is why cancelling a completed request is a 404 rather than a silent
success -- the client learns it was too late, instead of believing it stopped
something.
"""

from __future__ import annotations

import re
import threading
import uuid

from .providers.abort import AbortEvent

# Client ids reach logs and the JSONL telemetry streams, so they are bounded
# and charset-restricted before use: a header carrying a newline could forge a
# log line, and an unbounded one could bloat every event it appears in. This
# is the ONE place the rule lives -- every route that takes a client id calls
# it (/v1/messages, the conversation generate route), and a second copy is how
# two routes would come to disagree about what a request id may contain.
# `fullmatch`, and no `$`. Python's `$` ALSO matches just before a trailing
# newline, so `re.match(r"^...$", "abc\n")` is truthy -- an id ending in a
# newline sailed through the guard whose whole job is stopping a forged log
# line. The parametrized test covered an INTERIOR newline and passed. Only the
# trailing case was open, which is the one an HTTP client is most likely to
# produce by accident.
_ID_SAFE = re.compile(r"[A-Za-z0-9._:-]{1,128}")

# The rule as TEXT, for error messages that must tell a client what is allowed.
# Derived from the compiled pattern rather than retyped: a hand-written
# "[A-Za-z0-9._:-], max 128" beside the regex is a second copy that goes stale
# the first time the charset changes, and it goes stale SILENTLY because no
# test reads an error string.
REQUEST_ID_PATTERN = _ID_SAFE.pattern


def is_valid_request_id(value: str) -> bool:
    """Whether a string is a usable request id.

    Exists so the CANCEL route asks the same question the resolver asks
    rather than growing its own regex. It uses ``fullmatch`` for the reason
    the comment above gives, and a caller that re-implemented the charset
    would eventually disagree with the end that assigns it -- which is the
    defect this repo names "a hand-copied constant list".
    """
    return bool(_ID_SAFE.fullmatch(value))


def resolve_request_id(header_value: str | None, *, prefix: str) -> str:
    """The id to track this request under.

    A usable client-supplied ``X-Request-ID`` is honoured verbatim -- that is
    the whole point, since the client must be able to name the request in a
    later DELETE, and an id the server rewrote would not match. Anything
    missing or malformed gets a generated one, which is still a valid id for
    every other purpose (logs, correlation) and simply cannot be cancelled by
    a client that never chose it.
    """
    if header_value and is_valid_request_id(header_value):
        return header_value
    return f"{prefix}-{uuid.uuid4()}"


class RequestRegistry:
    """Thread-safe map of live request ids to their abort signals.

    Touched from the event loop (register/unregister/cancel on route handlers)
    and read by generation threads only through the ``AbortEvent`` itself,
    which carries its own synchronization.
    """

    __slots__ = ("_lock", "_live")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, set[AbortEvent]] = {}

    def register(self, request_id: str, abort_event: AbortEvent) -> None:
        if not request_id:
            return
        with self._lock:
            self._live.setdefault(request_id, set()).add(abort_event)

    def unregister(self, request_id: str, abort_event: AbortEvent) -> None:
        """Drop one registration. Never raises: this runs in a ``finally`` on
        paths that may already be unwinding from an error, and a bookkeeping
        failure there would replace the real exception with a useless one."""
        if not request_id:
            return
        with self._lock:
            events = self._live.get(request_id)
            if events is None:
                return
            events.discard(abort_event)
            if not events:
                del self._live[request_id]

    def cancel(self, request_id: str) -> int:
        """Signal every generation registered under ``request_id``.

        Returns how many were signalled -- 0 means no such live request. The
        entries are NOT removed here: each request removes its own in its
        ``finally``, so the generation gets to observe the flag, run its
        cleanup, and release the generation gate on its own terms. Deleting
        the entry here would make a second cancel of a still-running request
        report 404 while it was demonstrably still running.
        """
        with self._lock:
            events = list(self._live.get(request_id, ()))
        for event in events:
            event.set()
        return len(events)

    def live_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._live)

    def __len__(self) -> int:
        with self._lock:
            return len(self._live)


# Process-global, like the generation gate: the DELETE route and the request
# being cancelled are different requests on the same server, so the handle has
# to outlive either one's scope.
_registry = RequestRegistry()


def get_request_registry() -> RequestRegistry:
    return _registry


class track_request:
    """Context manager registering ``abort_event`` under ``request_id``.

    A context manager rather than two bare calls because the unregister has to
    happen on every exit path including an exception, and this is wrapped
    around route bodies that raise HTTPException as ordinary control flow.
    """

    __slots__ = ("request_id", "abort_event")

    def __init__(self, request_id: str, abort_event: AbortEvent) -> None:
        self.request_id = request_id
        self.abort_event = abort_event

    def __enter__(self) -> "track_request":
        _registry.register(self.request_id, self.abort_event)
        return self

    def __exit__(self, *exc_info) -> None:
        _registry.unregister(self.request_id, self.abort_event)


async def tracked_stream(agen, request_id: str, abort_event: AbortEvent):
    """Wrap a streaming body so it is registered for exactly its own lifetime.

    A streaming response's body outlives the route function, so a ``with``
    block around the ``return`` would unregister before the first token. And
    the bodies themselves are long generators whose error paths ``return``
    rather than fall through a ``finally``, so hand-placed enter/exit calls
    inside them are easy to get wrong and easy to break later. Wrapping is the
    version that cannot: the ``with`` exits when this generator is exhausted
    OR closed, which covers a completed stream, a client hang-up, and an
    exception identically.
    """
    with track_request(request_id, abort_event):
        try:
            async for item in agen:
                yield item
        finally:
            # Close the WRAPPED generator before the registration drops.
            # Starlette never calls aclose on a response body_iterator (it just
            # `async for`s it), and an `async for` interrupted by GeneratorExit
            # does not close the iterable either -- so without this the id
            # disappeared while the real generator was still suspended, still
            # holding the generation gate and the executor lease. A DELETE in
            # that window answered 404 on a demonstrably live run, which is the
            # opposite of the "the client learns it was too late" contract this
            # module documents.
            await agen.aclose()
