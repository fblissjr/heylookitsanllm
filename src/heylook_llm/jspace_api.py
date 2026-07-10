# src/heylook_llm/jspace_api.py
"""J-space (Jacobian lens) analysis API.

Post-hoc read-out of a model's verbalizable "workspace": per-layer, which
vocabulary tokens a residual is disposed toward before the model answers, plus
optional hallucination-risk. Lenses are fitted offline and placed under
``HEYLOOK_JSPACE_DIR/<model_id>/`` (see jspace/registry.py). Feature/lens math:
src/heylook_llm/jspace/. Design + verifier plan: docs/jspace_integration_plan.md.
"""
import asyncio
import logging

import mlx.core as mx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from heylook_llm.jspace.analyze import analyze as run_analyze
from heylook_llm.jspace.registry import LensRegistry
from heylook_llm.providers.common.generation_core import _get_generation_stream
from heylook_llm.streaming_utils import _executor_pool

logger = logging.getLogger(__name__)

jspace_router = APIRouter(prefix="/v1/jspace", tags=["JSpace"])


def _registry(request: Request) -> LensRegistry:
    reg = getattr(request.app.state, "jspace_registry", None)
    if reg is None:
        reg = LensRegistry.from_env()
        request.app.state.jspace_registry = reg
    return reg


class AnalyzeRequest(BaseModel):
    model: str
    messages: list[dict] | None = None
    prompt: str | None = None
    max_answer_tokens: int = 8
    top_k: int = 8
    heatmap: bool = False
    # >0: each heatmap cell also carries its top-k tokens. Bounded: the cost is
    # band_layers x positions x k decoded tokens, computed on the pinned MLX
    # executor thread under the process-global generation gate -- an unbounded k
    # (clamped only to vocab downstream) would let one request wedge all inference.
    heatmap_top_k: int = Field(0, ge=0, le=64)
    chat: bool = False   # False = raw completion (crisp viz); True = chat template (risk)


@jspace_router.get(
    "/models",
    summary="List models with a j-space lens",
    description="Served model ids that have a fitted+converted Jacobian lens available.",
)
async def jspace_models(request: Request):
    reg = _registry(request)
    models = reg.available()
    return {"models": models,
            "meta": {m: reg.provenance(m) for m in models},
            "base_dir": str(reg.base_dir) if reg.base_dir else None}


@jspace_router.post(
    "/analyze",
    summary="Analyze the model's verbalizable workspace",
    description="Format the prompt, greedily generate a short answer, and read the "
                "Jacobian-lens workspace: per-band-layer top-k silent tokens at the "
                "answer-onset, optional layer x position heatmap, workspace features, "
                "and (if a normalizer+router are configured) a hallucination-risk score.",
)
async def jspace_analyze(request: Request, body: AnalyzeRequest):
    reg = _registry(request)
    if not reg.has(body.model):
        raise HTTPException(
            status_code=404,
            detail=f"no j-space lens for model {body.model!r} (available: {reg.available()})")

    router_instance = request.app.state.router_instance
    try:
        provider = router_instance.get_provider(body.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not load {body.model!r}: {e}")
    if getattr(provider, "model", None) is None:
        raise HTTPException(status_code=400,
                            detail=f"{body.model!r} has no MLX residual stack to read")

    messages = body.messages
    if not messages:
        if not body.prompt:
            raise HTTPException(status_code=422, detail="provide 'messages' or 'prompt'")
        messages = [{"role": "user", "content": body.prompt}]

    # Load lens/normalizer/router here (broken lens files -> a clean error, not a
    # bare 500 from deep in the pipeline).
    try:
        lens = reg.get(body.model)
        normalizer = reg.normalizer(body.model)
        router = reg.router(body.model) if normalizer is not None else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load lens for {body.model!r}: {e}")

    # Pin the model (no LRU-evict / idle-unload mid-analyze) and run the forwards
    # under the process-global FIFO generation gate, so analyze serializes with
    # generation and with other analyze calls -- no concurrent Metal work, no
    # racing mutation of the shared block list.
    pinned = False
    try:
        router_instance.pin_model(body.model)
        pinned = True
    except Exception:
        pass  # best-effort: a not-currently-loaded/unpinnable model still runs

    # Drive the MLX forwards on a PINNED, reused executor thread (a dying MLX
    # thread aborts the process -- never use starlette's ephemeral threadpool)
    # inside the thread-local generation stream, exactly like generation.
    loop = asyncio.get_running_loop()
    executor = _executor_pool.acquire()
    try:
        return await loop.run_in_executor(
            executor, _gated_analyze, provider, lens, messages,
            body.max_answer_tokens, body.top_k, body.heatmap, body.heatmap_top_k,
            body.chat, router, normalizer)
    except Exception as e:
        logger.exception("jspace analyze failed")
        raise HTTPException(status_code=500, detail=f"analyze failed: {e}")
    finally:
        _executor_pool.release(executor)   # reuse; the call returned (not wedged)
        if pinned:
            router_instance.unpin_model(body.model)


def _gated_analyze(provider, lens, messages, max_answer_tokens, top_k, heatmap,
                   heatmap_top_k, chat, router, normalizer):
    """Runs on a pinned mlx-stream executor thread. Enters the thread-local
    generation stream (MLX streams are thread-bound -- a fresh thread has none)
    and holds the process-global FIFO generation gate so all Metal work
    serializes with generation and other analyze calls."""
    gate = getattr(provider, "_gen_gate", None)
    gen_stream = _get_generation_stream()

    def _work():
        with mx.stream(gen_stream):
            return run_analyze(
                provider, lens, messages, max_answer_tokens=max_answer_tokens,
                top_k=top_k, heatmap=heatmap, heatmap_top_k=heatmap_top_k, chat=chat,
                router=router, normalizer=normalizer)

    if gate is None:
        return _work()
    gate.acquire()
    try:
        return _work()
    finally:
        gate.release()
