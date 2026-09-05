# src/heylook_llm/request_guards.py
"""Route-boundary guards shared by the inference routes.

Lived in api.py until v1.79.67. The Messages and conversation-generate
routes are imported BY api.py, so they could only reach a guard defined
there through a lazy in-function import; a module of its own ends the
cycle and makes the guard an ordinary import.
"""
from fastapi import HTTPException


def validate_request_sampler(sampler: str | None) -> None:
    """Reject unknown sampler names at the route boundary.

    The deep SamplerNotFound raise happens inside the provider's
    _apply_model_defaults, which runs lazily on first generator advance --
    past the route's guarded stage -- so it escapes as a bare 500. Failing
    here turns a typo'd name into an immediate 400 and skips the model load.
    """
    if not sampler:
        return
    from heylook_llm.samplers import get_sampler_registry
    registry = get_sampler_registry()
    if sampler not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"sampler '{sampler}' not found; known: {registry.list_names()}",
        )
