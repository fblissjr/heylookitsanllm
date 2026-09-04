"""The 503 a MODEL_BUSY raise becomes, in one place.

THE RULE IS ABOUT THE TRIGGER, NOT ABOUT WHO CALLS THIS (v1.79.57):

    Every ``router.get_provider(...)`` reachable from a route SHOULD answer
    MODEL_BUSY through this module.

SHOULD, not DOES -- and the weaker verb is the honest one. That sentence has
an enumerable population (``get_provider`` call sites, readable from source),
so the remainder after subtracting the compliant ones IS the answer to "who
should have called this and did not". ``tests/unit/
test_model_busy_reaches_the_handler.py`` asks it for the sites it can see.

KNOWN NON-COMPLIANT, so this docstring cannot be read as a guarantee either:
``rlm`` answers a bare 503 non-streaming and an in-band ``rlm_error``
streaming, out of scope by owner decision, recorded in ``docs/project/TODO.md``
(the batch processor's broad handler, the other known case, left with the
OpenAI chat route in v1.79.66). The
first draft of this paragraph asserted the rule as fact -- which would have
made it the very thing it replaced, a census read as a construction guarantee,
one release after that failure mode was named here.

The docstring used to open with a COUNT of endpoints instead, and that is the
thing worth remembering. v1.79.53 corrected the count from three to four; a
sweep the same day found SIX MORE routes answering backpressure with a 500, a
400, and in one case a 200 with the busy sentence stringified into a data
field. The count was accurate and useless. **An enumerated caller list reads
as a description of the mechanism when it is a record of who remembered.**
Having one speller guarantees the callers AGREE; it guarantees nothing about
who calls -- and a module whose whole purpose is being the one speller is
structurally unable to tell you who is not using it.

So the mechanism no longer depends on remembering. ``api.py`` registers an
app-level ``exception_handler(ModelBusyError)``, and the dispatch is on the
TYPE (both causes already raised ``ModelBusyError``; the four
``"MODEL_BUSY" in str(e)`` checks were four copies of a magic string). A route
that does nothing now answers correctly. The only way to get it wrong is to
SWALLOW the exception in a broad handler, which is a single local property and
is what the test above enforces.

The wording drift this module originally fixed is still worth knowing: two
copies said "Retry shortly.", one said "Please retry in a moment.", and none
of them said what the server had actually raised.

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
