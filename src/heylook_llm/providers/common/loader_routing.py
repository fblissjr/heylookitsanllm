"""Is an MLX model served with vision? One resolver, two readers.

Every MLX model loads with mlx-vlm (plan W10), so this no longer picks a
library. It decides whether a model is SERVED WITH VISION: the template path
it renders on (``MLXProvider.is_vlm``) and the ``vision`` capability
``/v1/models`` advertises, which the provider's image guard reads through the
same answer, so the two cannot disagree.

The rule is library-aware: a model is served with vision only if it declares
vision AND mlx-vlm actually registers its ``model_type``; otherwise it is
served as text. That degrades a vision model mlx-vlm can't run as a VLM to
text instead of crashing at load.

Description lives in the registry; this resolution is deliberately separate --
modality detection has no library dependency, this does.

Until v2.0.89 the answer was a string, ``"mlx-vlm"`` or ``"mlx-lm"``, from
when those were two libraries; the second spelling outlived the dependency.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional


_logged_degradations: set[str] = set()


@lru_cache(maxsize=None)
def mlx_vlm_supports(model_type: str) -> bool:
    """Whether mlx-vlm registers a dedicated model class for ``model_type`` (i.e.
    can load it as a real VLM): lower-case, apply MODEL_REMAPPING, then try to
    import the module. Any failure (mlx-vlm absent, unknown type) is a clean False.

    We intentionally do NOT call ``mlx_vlm.utils.get_model_and_args`` here: it
    falls back to a ``text_only`` module (and resolves speculator/dflash aliases)
    rather than raising, so "it resolved" does not mean "loadable as a VLM" -- the
    direct module-import probe is the honest gate signal this router needs.

    Cached: what mlx-vlm registers cannot change inside one process, and this is
    now called per ROW by ``GET /v1/admin/models`` (not just once per load), where
    an uncached miss re-pays a failed import -- a filesystem search -- on every
    request. A hit is already free via ``sys.modules``; the cache is for the
    misses.

    It assumes mlx-vlm's AVAILABILITY is fixed for the process, which is true in
    production and not in a test that mocks the mlx tree partway through: one
    early probe under a mock would pin ``False`` for the session. Related scar,
    same shape: mocking ``mlx_vlm.generate.diffusion`` made ``_detect_diffusion``'s
    absent-dependency branch untestable. Nothing hits it today -- on Apple
    hardware ``mlx_mocks`` skips the patch entirely -- but clear the cache rather
    than debug it cold."""
    if not model_type:
        return False
    try:
        import importlib

        from mlx_vlm.utils import MODEL_REMAPPING  # type: ignore[import-not-found]

        mt = MODEL_REMAPPING.get(model_type.lower(), model_type.lower())
        importlib.import_module(f"mlx_vlm.models.{mt}")
        return True
    except Exception:
        return False


def read_model_type(model_path: str) -> Optional[str]:
    """The ``model_type`` from a model dir's config.json, or None. Defensive --
    a missing/odd config (draft/MTP heads, sparse checkpoints) yields None."""
    from ...modality_detect import read_model_config_json  # shared, mtime-cached
    mt = (read_model_config_json(Path(model_path)) or {}).get("model_type")
    return mt if isinstance(mt, str) else None


def _modalities_of(config: dict) -> list:
    """Modalities from a config dict. Normally present (validated `model_dump`),
    but the provider also accepts a raw dict (tests, back-compat callers), so
    fall back to deriving from the legacy `vision` bool -- the same rule as
    `MLXModelConfig._resolve_modalities`, kept in sync deliberately."""
    return config.get("modalities") or (["text", "vision"] if config.get("vision") else ["text"])


def resolve_serves_vision(
    config: dict,
    model_type_getter: Callable[[], Optional[str]],
    *,
    vlm_supports: Callable[[str], bool] = mlx_vlm_supports,
) -> bool:
    """Whether this MLX model is served with vision.

    ``config``: the model's config dict (``modalities``/``vision``).
    Usually a validated ``model_dump()``, but the provider accepts raw dicts too,
    so modalities are read via :func:`_modalities_of`.
    ``model_type_getter``: lazy -- called only when a vision model must probe
    the mlx-vlm registry.

    The ``loader`` field that could force an answer was retired in plan W10
    stage 3.
    """
    # non-vision -> served as text.
    if "vision" not in _modalities_of(config):
        return False
    # vision: keep the historical vision->mlx-vlm default UNLESS we can POSITIVELY
    # prove mlx-vlm lacks the model_type. Uncertainty (config.json unreadable ->
    # model_type None) trusts the vision declaration rather than silently
    # degrading a working VLM.
    model_type = model_type_getter()
    if model_type is None:
        return True
    if vlm_supports(model_type):
        return True
    # Once per model_type per process. This used to run only at LOAD; it now runs
    # per row of every `GET /v1/admin/models`, and an unconditional INFO there
    # would repeat the same sentence forever without ever saying anything new.
    if model_type not in _logged_degradations:
        _logged_degradations.add(model_type)
        logging.info(
            "model_type %r declares vision but mlx-vlm has no vision model for "
            "it; serving it as text", model_type)
    return False


def serves_vision_for_config(provider: str, config: dict) -> Optional[bool]:
    """Whether an MLX model WOULD be served with vision, resolved WITHOUT
    loading it.

    ``MLXProvider.is_vlm`` is the same answer read off a live provider, and
    exists only for a resident model. This is the unloaded-model form, for
    callers that have a config and no process: the capability report on
    ``/v1/models`` and the admin row, and through them the live harnesses,
    which split MLX into text and vision arms.

    ``None`` for anything but ``"mlx"``: a gguf model's vision is its
    projector, answered from its own config.

    Pure over the config plus one mtime-cached read of the model dir's
    ``config.json`` -- no import of the model, no MLX. It agrees with the loaded
    provider by CONSTRUCTION: both call :func:`resolve_serves_vision` with the
    same two inputs.

    THAT AGREEMENT HAS A PRECONDITION, and it used to go unsaid: the config must
    carry a capability declaration. Validation always supplies one, but
    ``merge_discovered`` returns raw dicts and derivation happens later, so a
    config taken straight from the merge declares nothing -- and the ``auto``
    rule reads the declaration. Answering from its absence reports every
    model as text-only, silently. That is now refused rather than answered;
    see the guard below.
    """
    if provider != "mlx":
        return None
    # REFUSE AN UNRESOLVED DESCRIPTION rather than answer from its absence.
    #
    # The rule reads `modalities`, and `MLXModelConfig._resolve_modalities`
    # always populates it -- so `None` here does not mean "text-only", it means
    # this config never went through validation. Answering anyway returns
    # False (text) for EVERY model including vision ones, with no exception and no
    # log line: a confident wrong answer indistinguishable from a real one.
    #
    # Two sessions hit exactly that on 2026-09-08 by passing `merge_discovered`
    # output straight in (it returns raw dicts; derivation happens at
    # validation). It matters beyond a bad count: anything comparing a served
    # set before and after an edit calls this per model, so an unvalidated
    # snapshot on either side reports engine changes that never happened.
    #
    # Raise rather than warn -- this runs per row of GET /v1/admin/models, where a warning
    # is either noise or filtered, and no legitimate caller reaches it (the
    # production caller passes a validated config, and every existing test
    # states `modalities` explicitly).
    # NOT `_modalities_of(config) is None` -- that helper never returns None, it
    # falls back to the legacy `vision` bool and yields ["text"]. Writing the
    # check that way would have made this guard dead code that reads as a
    # guard, which is the failure this whole change is about. The honest test
    # is that NEITHER declaration is present: no `modalities`, no `vision`.
    declared = config.get("modalities") is not None or "vision" in config
    if not declared:
        raise ValueError(
            "serves_vision_for_config was given a config whose `modalities` "
            "is unresolved, which happens when the config has not been through "
            "MLXModelConfig validation (merge_discovered returns raw dicts). "
            "Answering would report every model as text-only. Validate first -- "
            "AppConfig(**merged) -- and pass that config."
        )
    return resolve_serves_vision(
        config, lambda: read_model_type(config.get("model_path", "") or ""))
