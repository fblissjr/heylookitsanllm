# src/heylook_llm/jspace_api.py
"""J-space (Jacobian lens) analysis API.

Post-hoc read-out of a model's verbalizable "workspace": per-layer, which
vocabulary tokens a residual is disposed toward before the model answers, plus
optional hallucination-risk. Lenses are fitted offline and placed under
``HEYLOOK_JSPACE_DIR/<model_id>/`` (see jspace/registry.py). Feature/lens math:
src/heylook_llm/jspace/. Design + verifier plan: docs/jspace_integration_plan.md.
"""
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from heylook_llm.jspace.analyze import analyze as run_analyze
from heylook_llm.jspace.registry import LensRegistry

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
    chat: bool = False   # False = raw completion (crisp viz); True = chat template (risk)


@jspace_router.get(
    "/models",
    summary="List models with a j-space lens",
    description="Served model ids that have a fitted+converted Jacobian lens available.",
)
async def jspace_models(request: Request):
    reg = _registry(request)
    return {"models": reg.available(),
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

    lens = reg.get(body.model)
    normalizer = reg.normalizer(body.model)
    router = reg.router(body.model) if normalizer is not None else None
    try:
        # MLX compute is blocking + Metal-bound; run off the event loop.
        return await run_in_threadpool(
            run_analyze, provider, lens, messages,
            max_answer_tokens=body.max_answer_tokens, top_k=body.top_k,
            heatmap=body.heatmap, chat=body.chat, router=router, normalizer=normalizer)
    except Exception as e:
        logger.exception("jspace analyze failed")
        raise HTTPException(status_code=500, detail=f"analyze failed: {e}")
