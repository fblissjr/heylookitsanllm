# src/heylook_llm/model_ops_api.py
"""Model lifecycle operations available to INFERENCE clients.

``POST /v1/models/{id}/load`` is here, not under ``/v1/admin``, because
gating it on the admin token protected nothing. Loading is exactly what an
inference request already does: ``/v1/messages`` calls
``router.get_provider(model_id)`` on the way in, so any client that can
generate can already trigger a multi-GB load and
(at ``max_loaded_models=1``) an eviction, just by naming a model in the
body. The admin gate only stopped a client from doing EXPLICITLY and
OBSERVABLY what it could already do implicitly. It is gated on
``require_api_key`` -- whoever may generate may load -- while ``unload`` and
``reload`` stay admin, because those stop a model out from under other
clients, and ``GET /v1/admin/models`` stays admin because it discloses
``model_path`` and the full per-model config.

WHY A CLIENT WANTS THIS. The load happens BEFORE the response begins: the
route resolves the provider and only then returns a ``StreamingResponse`` or
starts the blocking consume. So during a cold load there is nothing on the
wire at all -- no headers, no ``message_start``, no keepalive -- on either
wire, streaming or not. A non-streaming client sees one long opaque POST and
cannot tell a loading model from a hung server. Calling this first moves
that wait into a request the client can label, and costs nothing it was not
already going to pay: without ``warm`` this IS ``get_provider``, the same
call, so a resident model answers in a round trip.

``warm=true`` additionally runs a 1-token generation through the normal
generation path, which pays the first-forward-pass (Metal kernel JIT) cost.
That takes the process-global FIFO generation gate, so it can QUEUE behind
another request's long run -- fine for a startup or model-switch readiness
call (which is what the harnesses use it for), wrong for a per-request
pre-flight. Bare ``/load`` never touches the gate.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from heylook_llm.auth import require_api_key
from heylook_llm.busy_response import model_busy_response
from heylook_llm.providers.common.generation_gate import ModelBusyError

logger = logging.getLogger(__name__)

model_ops_router = APIRouter(
    prefix="/v1/models",
    tags=["Models"],
    dependencies=[Depends(require_api_key)],
)


@model_ops_router.post(
    "/{model_id:path}/load",
    summary="Load Model",
    description=(
        "Load a model into the LRU cache, optionally warming it. This is the "
        "same `router.get_provider` an inference request performs on its way "
        "in, so calling it first does not add work -- it MOVES the load out "
        "of an opaque generate call, which writes nothing to the connection "
        "while a multi-GB load runs, into a request a client can show "
        "progress against. A resident model answers in a round trip. "
        "`warm=true` also runs a 1-token generation (pays Metal kernel JIT) "
        "and therefore takes the FIFO generation gate, so it can queue behind "
        "another request's long run: use it for startup/model-switch "
        "readiness, not as a per-request pre-flight. Returns 200 with "
        "`warmed: false` + `warm_error` if only the warm generation failed -- "
        "the model is loaded and usable either way. 400 for an unknown or "
        "disabled id, the same answer the generate call would give. **503** "
        "with `Retry-After` when the model cannot be made room for because "
        "another is generating -- the same backpressure envelope the "
        "inference routes return, so a client classifies it the same way. A "
        "500 here is a genuine load failure and nothing else."
    ),
)
async def load_model(model_id: str, request: Request, warm: bool = False):
    router = request.app.state.router_instance
    return await load_and_warm(router, model_id, warm)


async def load_and_warm(router, model_id: str, warm: bool):
    """Returns the load result dict, or a 503 JSONResponse when the model
    is busy -- FastAPI passes a returned Response straight through, and
    both callers (`/v1/models/{id}/load`, admin `/reload`) return this
    verbatim, so both inherit the shared backpressure envelope."""
    # The one load(+warm) body -- shared by this route and admin's /reload,
    # so neither the warm contract nor the busy answer can fork between them.
    try:
        import asyncio
        provider = await asyncio.to_thread(router.get_provider, model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # MODEL_BUSY is BACKPRESSURE, not a broken model. It reached the
        # generic handler below and came back as a 500 carrying the same
        # sentence /v1/messages returns as a 503 -- so the same transient,
        # self-clearing condition had two status codes across two routes, and
        # the only thing separating this 500 from the genuine "that model
        # exists and failed to load" was a substring. A consuming client
        # classified on status, sent a wait down its unknown-model branch and
        # told the user to refresh the roster (measured against 1.79.52 and
        # reported 2026-08-31).
        #
        # busy_response.py exists precisely so this answer has ONE speller;
        # its own docstring names the three endpoints that use it, and .48
        # added a fourth route that did not. `provider` is deliberately not
        # passed: get_provider is what raised, so there is none to ask for a
        # queue capacity -- the helper's None branch covers exactly that.
        if isinstance(e, ModelBusyError):
            return model_busy_response(e)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    result: dict = {"status": "loaded", "model_id": model_id}
    if warm:
        from heylook_llm.config import ChatMessage, ChatRequest

        warm_request = ChatRequest(
            model=model_id,
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=1,
            stream=False,
        )

        def _consume() -> None:
            gen = provider.create_chat_completion(warm_request)
            try:
                for _ in gen:
                    pass
            finally:
                gen.close()

        start = time.time()
        try:
            await asyncio.to_thread(_consume)
            result["warmed"] = True
            result["warm_ms"] = int((time.time() - start) * 1000)
        except Exception as e:
            result["warmed"] = False
            result["warm_error"] = str(e)[:500]
    return result
