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

__all__ = ["GenerationGate", "ModelBusyError"]


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

    def check_capacity(self) -> None:
        """Raise :class:`ModelBusyError` if the queue is already full.

        Non-blocking and advisory: it reflects the queue depth at call time. A
        small overshoot is possible if several callers check simultaneously near
        the cap -- acceptable for a soft backpressure limit.
        """
        with self._cv:
            if self._waiting >= self.max_waiting:
                raise ModelBusyError(
                    f"MODEL_BUSY: {self._waiting} request(s) already queued "
                    f"(max_waiting={self.max_waiting})"
                )

    def acquire(self) -> None:
        """Block until it is this caller's turn to generate (strict FIFO)."""
        with self._cv:
            ticket = self._ticket_seq
            self._ticket_seq += 1
            self._queue.append(ticket)
            self._waiting += 1
            try:
                while self._busy or self._queue[0] != ticket:
                    self._cv.wait()
            except BaseException:
                # Don't poison the queue if waiting is interrupted.
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
