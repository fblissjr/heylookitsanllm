# src/heylook_llm/providers/abort.py
"""The per-request signal channel between the API layer and a generation.

One object per request, created by the route and handed to the provider's
generator. Two signals ride it, in opposite directions:

- ABORT flows down (API -> generation): ``set()`` asks the generation to
  stop at its next check. A threading.Event underneath, so it is safe
  across the event loop and the pinned generation thread.
- PREFILL PROGRESS flows up (generation -> API): the engine reports how
  much of the prompt it has processed and the streaming layer turns each
  change into a control frame on the wire (v1.79.65). BOTH engines report
  through this one slot -- mlx-lm via its ``prompt_progress_callback``
  (which fires inside the first ``next()``, where nothing can be yielded),
  llama-server via the ``prompt_progress`` frames ``return_progress`` adds
  to its stream -- so the routes spell progress once per protocol instead
  of once per engine, and a third engine would report the same way. The
  slot is a single tuple, and a tuple assignment is atomic in CPython, so
  it needs no lock; readers see either the old pair or the new one.

The class keeps its original name. Every route and provider already passes
an ``AbortEvent`` under that name, and renaming it would touch all of them
to say the same thing.
"""

import threading


class AbortEvent:
    """Cooperative abort signal, plus the upward prefill-progress slot.

    Usage:
        abort = AbortEvent()
        abort.clear()          # reset before new generation
        ...
        abort.set()            # signal abort from another thread
        ...
        if abort.is_set():     # check inside generation loop
            break
    """

    __slots__ = ("_event", "_prefill")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._prefill: tuple[int, int] | None = None

    def set(self) -> None:
        """Signal abort."""
        self._event.set()

    def clear(self) -> None:
        """Reset for a new generation."""
        self._event.clear()
        self._prefill = None

    def is_set(self) -> bool:
        """Check if abort has been signaled."""
        return self._event.is_set()

    def set_prefill_progress(self, processed: int, total: int) -> None:
        """Report prefill progress: ``processed`` of ``total`` prompt tokens.

        Counts the work THIS request runs -- a reused cached prefix is
        excluded from both numbers on every engine, so the ratio means the
        same thing whichever engine produced it.
        """
        self._prefill = (int(processed), int(total))

    def prefill_progress(self) -> tuple[int, int] | None:
        """The last reported (processed, total), or None before any report."""
        return self._prefill

    def __repr__(self) -> str:
        state = "set" if self.is_set() else "clear"
        return f"AbortEvent({state})"
