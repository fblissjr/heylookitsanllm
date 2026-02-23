# src/heylook_llm/providers/abort.py
"""Cooperative abort signal for generation cancellation.

Provides a thread-safe mechanism to signal in-flight generation to stop.
Uses threading.Event internally which is safe for cross-thread signaling.
"""

import threading


class AbortEvent:
    """Cooperative abort signal for generation cancellation.

    Thread-safe. Can be shared across the async API layer and the
    synchronous MLX generation thread.

    Usage:
        abort = AbortEvent()
        abort.clear()          # reset before new generation
        ...
        abort.set()            # signal abort from another thread
        ...
        if abort.is_set():     # check inside generation loop
            break
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        """Signal abort."""
        self._event.set()

    def clear(self) -> None:
        """Reset for a new generation."""
        self._event.clear()

    def is_set(self) -> bool:
        """Check if abort has been signaled."""
        return self._event.is_set()

    def __repr__(self) -> str:
        state = "set" if self.is_set() else "clear"
        return f"AbortEvent({state})"
