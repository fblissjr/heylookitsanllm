# src/heylook_llm/requests_api.py
"""Cancel an in-flight request by the id the client already sends.

The gap this closes: a STREAMING request is cancellable by hanging up (the
server is writing chunks, so it notices the peer is gone), while a
NON-STREAMING one writes nothing until the generation finishes and therefore
never notices. Measured on this server at 1.79.42: a 73.1s generation whose
client aborted at 5.0s still ran to completion and left the next request
waiting 57.9s behind it.

Scope, stated plainly because it is easy to over-read: this makes an
abandoned run STOPPABLE, not self-stopping. A client that hangs up without
calling DELETE still leaves the generation running -- noticing that is
disconnect polling, a different mechanism, deliberately not built (owner call
2026-08-30, on the grounds that an explicit endpoint cannot mistake a proxy
hiccup for a departed client and kill a live generation).

`/v1/requests` is a top-level resource on purpose. The existing
`DELETE /v1/conversations/{id}/generate` cancels a CONVERSATION's run and is
addressed by conversation, so it cannot name a plain `/v1/messages` call --
which is exactly the traffic that needed cancelling.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from heylook_llm.auth import require_api_key
from heylook_llm.request_registry import get_request_registry

requests_router = APIRouter(tags=["Requests"], dependencies=[Depends(require_api_key)])


@requests_router.delete(
    "/v1/requests/{request_id}",
    summary="Cancel an In-Flight Request",
    description="""
Cancel a generation that is still running, addressed by its request id.

**The id is the one YOU sent.** Pass `X-Request-ID` on the original request
and cancel by that exact value. A request that arrived without the header got
a server-generated id which the client never learns in time to use on the
non-streaming path -- so on that path, sending the header is the precondition
for being able to cancel at all.

**What cancelling does.** It sets the generation's cooperative abort flag. The
decode loop checks it between tokens and stops, then unwinds normally --
releasing the generation gate, and for a partial run on a conversation,
persisting what was produced. It is not a kill: a generation blocked in a
single long operation (prompt prefill on a large context) stops at the next
token boundary, not instantly.

**404 means the request is not running.** Most often it already finished --
ids are tracked only while in flight, so a completed request is unknown rather
than silently "cancelled". That distinction is the point: a client is told it
was too late instead of believing it stopped something.

**Ids are not assumed unique.** They are client-supplied, so if two in-flight
requests share one, cancelling it cancels both, and `cancelled` reports how
many. That is deliberate -- the alternative lets one registration hide
another, leaving a running generation nothing can name.
""",
    response_description="How many in-flight generations were signalled",
    responses={
        404: {"description": "No in-flight request with that id"},
    },
)
async def cancel_request(request_id: str):
    cancelled = get_request_registry().cancel(request_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=(f"No in-flight request with id '{request_id}'. It may have "
                    f"already finished, or it was sent without an X-Request-ID "
                    f"header and is tracked under a server-generated id."),
        )
    # Deliberately logged: a cancelled generation shows up downstream as a
    # short or truncated answer, and this is the line that explains why.
    logging.info(f"[CANCEL] {request_id}: signalled {cancelled} generation(s)")
    return {"cancelled": cancelled, "request_id": request_id}
