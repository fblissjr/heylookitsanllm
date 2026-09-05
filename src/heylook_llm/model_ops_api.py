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
from heylook_llm.capabilities import derived_model_facts
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


# GET /v1/models is DISCOVERY, not an operation: it was mounted on the app with
# no API-key dependency (v3 and external clients resolve ids from it before
# they have anything to authenticate), so it keeps its own router rather than
# inheriting the load route's require_api_key.
models_router = APIRouter(tags=["Models"])


@models_router.get("/v1/models",
    summary="List Available Models",
    description="""
List all language models currently available on this server.

**Use this endpoint to:**
- Discover which models are loaded and ready for inference
- Verify a specific model is available before making requests
- Get model IDs for use in completion requests

**Returns:**
- Model IDs. These are INSTALL-LOCAL -- the registry is override-only, so the
  roster is whatever sits under the scanned folders on this machine. Resolve
  them here at runtime rather than hardcoding one (this description named a
  concrete id for years after that model was gone).
- `provider`, `modalities`, and `capabilities` per row. Gate features on
  `capabilities` (what this server will SERVE) rather than `modalities` (what
  the checkpoint author declared) -- they differ on purpose.
- The OpenAI list shape (`object: "list"`, `data: [...]`), which is what the bundled frontend and external clients read
- Only shows models marked as `enabled: true` in models.toml
    """,
    response_description="List of available models (OpenAI list shape, heylook fields per row)",
)
def list_models(request: Request):
    """Get the list of available models (OpenAI list shape) with capabilities.

    Plain ``def`` (FastAPI threadpool), NOT ``async def``: deriving
    capabilities reads each model dir's ``config.json`` -- the template probes
    have always done so, and since v1.79.43 the vision capability resolves
    through the loader router, which stats the dir too. That is the same
    per-row filesystem cost that moved the two admin read routes off the event
    loop; the reads are mtime/lru cached and cheap on a warm local disk, and
    unbounded on a slow or network-mounted one.
    """
    router = request.app.state.router_instance
    models_data = []

    for model_id in router.list_available_models():
        model_entry = {
            "id": model_id,
            "object": "model",
            "owned_by": "user",
        }

        # Add capabilities and provider if available from config
        model_config = router.app_config.get_model_config(model_id)
        if model_config:
            model_entry["provider"] = model_config.provider

            # modalities = full author-declared DESCRIPTION (text/vision/audio/
            # video); capabilities below stays gated to what the server actually
            # SERVES (image input today) -- description != served.
            modalities = getattr(model_config.config, "modalities", None)
            if modalities:
                model_entry["modalities"] = modalities

            # Capabilities, the thinking default and the context window come
            # from the ONE derivation the admin row also reads
            # (capabilities.derived_model_facts), so a page reading either
            # list gates on the same capabilities, labels its "model default"
            # thinking choice with the value generation will use, and can
            # size a prompt against the ceiling the provider enforces instead
            # of learning it from a 400. `context_length` is null when the
            # files do not say.
            facts = derived_model_facts(model_config)
            if facts.capabilities:
                model_entry["capabilities"] = facts.capabilities
            model_entry["thinking_default"] = facts.thinking_default
            model_entry["context_length"] = facts.context_length

        models_data.append(model_entry)

    return {"object": "list", "data": models_data}
