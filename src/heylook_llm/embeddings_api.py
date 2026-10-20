"""Non-streaming multimodal embeddings on the existing model lifecycle."""
import asyncio
import time

from fastapi import APIRouter, HTTPException, Request, Response

from .embedding_models import embedding_info
from .capabilities import config_dict
from .providers.abort import AbortEvent
from .providers.base import InvalidGenerationRequest
from .providers.common.generation_gate import GenerationCancelled, ModelBusyError
from .request_registry import resolve_request_id, track_request
from .schema.embeddings import EmbeddingRequest, EmbeddingResult
from .streaming_utils import _executor_pool

embeddings_router = APIRouter(prefix="/v1", tags=["Embeddings"])


@embeddings_router.post("/embeddings", response_model=EmbeddingResult)
async def create_embeddings(body: EmbeddingRequest, request: Request, response: Response):
    router = request.app.state.router_instance
    config = router.app_config.get_model_config(body.model)
    if config is None:
        raise HTTPException(status_code=404, detail="Unknown embedding model")
    info = embedding_info(config.provider, config_dict(config.config))
    if info is None:
        raise HTTPException(status_code=400, detail="Model does not support embeddings")
    if body.dimensions is not None and body.dimensions not in info["dimensions"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dimensions; choose {info['dimensions']}")
    for index, item in enumerate(body.input):
        if item.prompt_name is not None and item.prompt_name not in info["prompts"]:
            raise HTTPException(status_code=400, detail=f"input[{index}]: unknown prompt_name")
        if any(block.type not in info["modalities"] for block in item.content):
            raise HTTPException(status_code=400, detail=f"input[{index}]: unsupported modality")

    request_id = resolve_request_id(request.headers.get("X-Request-ID"), prefix="emb")
    response.headers["X-Request-ID"] = request_id
    abort = AbortEvent()
    executor = _executor_pool.acquire()
    started = time.perf_counter()

    def work():
        with track_request(request_id, abort):
            provider = router.get_provider(body.model)
            # Reserve activity under the same lock eviction uses, before waiting
            # for the inference gate. Never run an already-evicted provider.
            with router.cache_lock:
                if router.providers.get(body.model) is not provider:
                    raise ModelBusyError("Model was evicted before embedding admission; retry")
                active = provider.generation_active()
                active.__enter__()
            try:
                result = provider.embed(body, abort)
                result.request_id = request_id
                result.elapsed_ms = (time.perf_counter() - started) * 1000
                from .observability import record_event
                record_event("embedding", tier="metrics", min_level="minimal", source="embeddings",
                             fields={"model": body.model, "items": len(body.input),
                                     "dimensions": result.dimensions, "elapsed_ms": result.elapsed_ms})
                return result
            finally:
                active.__exit__(None, None, None)

    future = asyncio.get_running_loop().run_in_executor(executor, work)
    # Worker ownership survives HTTP-task cancellation. A callback returns the
    # executor only when work (and all media cleanup) has actually completed.
    future.add_done_callback(lambda done: _executor_pool.release(executor))
    try:
        return await asyncio.shield(future)
    except InvalidGenerationRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except asyncio.CancelledError:
        abort.set()
        raise
