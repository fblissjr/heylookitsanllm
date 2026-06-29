# src/heylook_llm/providers/common/generation_gate.py
"""FIFO admission gate for model generation.

A single GPU with one loaded model (max_loaded_models=1) and a shared KV cache
means only one generation can run at a time. This gate serializes generations
**in arrival order** -- there is no preemption. Concurrent requests (the Batch
applet, the batch-labeler client, multiple frontends) queue and each complete,
instead of the newest aborting the in-flight one.

Backpressure: ``check_capacity()`` lets an HTTP entry point reject early with
``ModelBusyError`` (-> 503) when too many requests are already queued, instead
of letting the queue grow without bound. Internal orchestration (batch, RLM)
skips the capacity check and simply queues.

``acquire()`` / ``release()`` are a matched pair but need not run on the same
thread -- the streaming layer acquires on the thread that first drives the
generator and releases from the generator's ``finally`` on the pinned worker
thread. The implementation therefore does not rely on lock ownership.
"""
import collections
import threading

__all__ = ["GenerationGate", "ModelBusyError", "GenerationCancelled"]


class GenerationCancelled(Exception):
    """Raised inside ``acquire(cancel_check=...)`` when the caller's request was
    cancelled (e.g. client disconnected) before its FIFO turn arrived. The
    caller should abandon the generation; nothing was acquired."""


class ModelBusyError(RuntimeError):
    """Raised when the generation queue is full.

    Subclasses ``RuntimeError`` and embeds the literal ``MODEL_BUSY`` in its
    message so the API layer's existing ``"MODEL_BUSY" in str(e)`` checks map it
    to HTTP 503.
    """


class GenerationGate:
    """Strict-FIFO, single-slot admission gate with bounded queue depth.

    Args:
        max_waiting: Maximum number of requests allowed to queue *behind* the
            one actively generating. ``check_capacity()`` raises once this many
            are already waiting. ``0`` means single-flight (reject any overlap).
            The blocking ``acquire()`` itself is unbounded -- the cap is only
            enforced at the ``check_capacity()`` admission point, so internal
            callers that skip it always queue rather than fail.
    """

    def __init__(self, max_waiting: int = 8):
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        self.max_waiting = max_waiting
        self._cv = threading.Condition()
        self._queue: collections.deque[int] = collections.deque()
        self._busy = False
        self._waiting = 0
        self._ticket_seq = 0

    @property
    def waiting(self) -> int:
        """Number of callers currently blocked in ``acquire()`` (queued)."""
        with self._cv:
            return self._waiting

    @property
    def busy(self) -> bool:
        """True while a generation holds the slot."""
        with self._cv:
            return self._busy

    def snapshot(self) -> dict:
        """Return a consistent point-in-time view of the queue.

        Keys: ``active`` (0/1, one generation runs at a time), ``waiting``
        (queued behind it), ``max_waiting`` (depth cap), ``capacity``
        (``1 + max_waiting``, total requests the gate admits before 503).
        Used for backpressure headers and observability.
        """
        with self._cv:
            return {
                "active": 1 if self._busy else 0,
                "waiting": self._waiting,
                "max_waiting": self.max_waiting,
                "capacity": 1 + self.max_waiting,
            }

    def check_capacity(self) -> None:
        """Raise :class:`ModelBusyError` if the system is already full.

        Full means ``active + waiting >= 1 + max_waiting`` (one running plus
        ``max_waiting`` queued behind it). Accounting for the active holder is
        what makes ``max_waiting=0`` behave as single-flight: an *idle* gate
        (nothing active) still admits the first request, but a second request
        while one is running is rejected.

        Non-blocking and advisory: it reflects state at call time. A small
        overshoot is possible if several callers check simultaneously near the
        cap -- acceptable for a soft backpressure limit.
        """
        with self._cv:
            in_system = (1 if self._busy else 0) + self._waiting
            if in_system >= 1 + self.max_waiting:
                raise ModelBusyError(
                    f"MODEL_BUSY: {self._waiting} request(s) queued behind an "
                    f"active generation (max_waiting={self.max_waiting})"
                )

    def acquire(self, cancel_check=None) -> None:
        """Block until it is this caller's turn to generate (strict FIFO).

        If *cancel_check* is provided, it is polled while waiting; when it
        returns True the caller gives up its place in the queue and
        :class:`GenerationCancelled` is raised. This lets a request whose client
        has already disconnected leave the queue instead of waiting its turn to
        do (now-pointless) work. Without *cancel_check* the wait is unbounded
        (internal callers: batch, RLM).
        """
        with self._cv:
            ticket = self._ticket_seq
            self._ticket_seq += 1
            self._queue.append(ticket)
            self._waiting += 1
            try:
                while self._busy or self._queue[0] != ticket:
                    if cancel_check is not None and cancel_check():
                        raise GenerationCancelled()
                    # Poll when cancellable so the cancel_check is re-evaluated
                    # even without a release/notify; block otherwise.
                    self._cv.wait(timeout=0.25 if cancel_check is not None else None)
            except BaseException:
                # Don't poison the queue if waiting is interrupted/cancelled.
                self._waiting -= 1
                try:
                    self._queue.remove(ticket)
                except ValueError:
                    pass
                self._cv.notify_all()
                raise
            self._queue.popleft()
            self._busy = True
            self._waiting -= 1

    def release(self) -> None:
        """Release the slot and wake the next waiter. Safe from any thread."""
        with self._cv:
            self._busy = False
            self._cv.notify_all()
