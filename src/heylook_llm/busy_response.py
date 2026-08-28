"""The 503 a MODEL_BUSY raise becomes, in one place.

Three endpoints turn ``MODEL_BUSY`` into a 503 -- ``/v1/chat/completions``,
``/v1/messages`` and ``POST /v1/conversations/{id}/generate`` -- and each had
its own hand-written copy of the body and headers. That is this repo's named
defect class (see CLAUDE.md: a hand-copied constant list is a defect with a
delay), and it had already drifted: two copies said "Retry shortly.", one said
"Please retry in a moment.", and none of them said what the server had actually
raised.

That last part is the bug this module fixes. ``MODEL_BUSY`` has TWO causes and
they are not the same situation:

- the generation QUEUE is full (the gate refuses a fourth waiter) -- transient,
  frees in seconds as requests drain in FIFO order;
- eviction is BLOCKED (``_evict_lru_model``): every loaded model is generating,
  so there is no slot to load the requested one into. The raise names the
  models and says "Stop the generation or wait for it to finish", and every
  call site replaced that with "The generation queue is full", which is not
  true and points at nothing the user can act on.

The WIRE SHAPE is deliberately unchanged -- same 503, same
``code: "model_overloaded"``, same ``Retry-After: 1``. v3's `streamTypedSSE`
retries on exactly that pair, at most ``MAX_BUSY_RETRIES = 3`` times, so the
client gives up after about three seconds either way; lengthening the wait
would make the user sit longer for the same outcome, and changing the code
would silently disable the retry. Only the message changes, and only when the
server had something better to say.
"""
from __future__ import annotations

from typing import Optional

from fastapi.responses import JSONResponse

# The queue-full wording, which is the right answer for the gate's own refusal
# and was the wrong one for everything else.
QUEUE_FULL_MESSAGE = "The generation queue is full. Retry shortly."


def model_busy_response(exc: BaseException, provider=None) -> JSONResponse:
    """The 503 for a MODEL_BUSY raise.

    ``exc``: the raised error. Its message is used verbatim when it carries
    more than the bare marker -- the eviction-blocked case names which models
    are generating and what to do about it, and that is strictly more useful
    than a fixed sentence about a queue.
    ``provider``: the loaded provider, when the caller has one, for the
    rate-limit headers. ``None`` is fine: eviction can be blocked before any
    provider is bound, which is exactly when the old copies' `provider` guard
    was load-bearing and undocumented.
    """
    detail = str(exc).removeprefix("MODEL_BUSY:").removeprefix("MODEL_BUSY").strip()
    message = detail if detail else QUEUE_FULL_MESSAGE
    capacity: Optional[int] = None
    if provider is not None:
        try:
            capacity = (provider.generation_queue_stats() or {}).get("capacity")
        except Exception:  # a provider mid-teardown must not turn 503 into 500
            capacity = None
    return JSONResponse(
        status_code=503,
        content={"error": {
            "message": message,
            "type": "server_error",
            "code": "model_overloaded",
        }},
        headers={
            "Retry-After": "1",
            # Total in-flight + queued requests the server admits.
            "X-RateLimit-Limit": str(capacity) if capacity else "1",
            "X-RateLimit-Remaining": "0",
        },
    )
