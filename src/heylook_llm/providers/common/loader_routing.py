"""Effective-loader resolution: which mlx engine actually loads a model.

The registry stores a model's DESCRIPTION (``modalities``) and a ROUTING hint
(``loader``); this turns them into the concrete engine -- ``"mlx-vlm"`` or
``"mlx-lm"`` -- the provider loads with, and from which ``is_vlm`` derives.

The ``"auto"`` rule is library-aware: a model routes to mlx-vlm only if it
declares vision AND mlx-vlm actually registers its ``model_type``; otherwise it
falls to mlx-lm. That degrades a vision model mlx-vlm can't load to the text
loader instead of crashing at load (the failure mode that motivated the split;
see plan Phase 6 refinement 2026-07-11). An explicit ``loader`` forces the
engine (e.g. run a dual-capable VLM as text via ``"mlx-lm"``).

Description lives in the registry; this routing is deliberately separate --
detection (model_importer.detect_modalities) has no library dependency, this
does.
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


def resolve_effective_loader(
    config: dict,
    model_type_getter: Callable[[], Optional[str]],
    *,
    vlm_supports: Callable[[str], bool] = mlx_vlm_supports,
) -> str:
    """Resolve to ``"mlx-vlm"`` or ``"mlx-lm"``.

    ``config``: the model's config dict (``loader`` + ``modalities``/``vision``).
    Usually a validated ``model_dump()``, but the provider accepts raw dicts too,
    so modalities are read via :func:`_modalities_of`.
    ``model_type_getter``: lazy -- called only when ``auto`` must probe the
    mlx-vlm registry (skipped for explicit loaders and non-vision models).
    """
    loader = config.get("loader", "auto")
    if loader != "auto":
        return loader                      # explicit engine (Literal: mlx-vlm | mlx-lm)

    # auto: non-vision -> text loader.
    if "vision" not in _modalities_of(config):
        return "mlx-lm"
    # vision: keep the historical vision->mlx-vlm default UNLESS we can POSITIVELY
    # prove mlx-vlm lacks the model_type. Uncertainty (config.json unreadable ->
    # model_type None) trusts the vision declaration rather than silently
    # degrading a working VLM.
    model_type = model_type_getter()
    if model_type is None:
        return "mlx-vlm"
    if vlm_supports(model_type):
        return "mlx-vlm"
    # Once per model_type per process. This used to run only at LOAD; it now runs
    # per row of every `GET /v1/admin/models`, and an unconditional INFO there
    # would repeat the same sentence forever without ever saying anything new.
    if model_type not in _logged_degradations:
        _logged_degradations.add(model_type)
        logging.info(
            "loader=auto: model_type %r declares vision but mlx-vlm has no loader for "
            "it; routing to mlx-lm (text)", model_type)
    return "mlx-lm"


def effective_loader_for_config(provider: str, config: dict) -> Optional[str]:
    """The engine a model WOULD load with, resolved WITHOUT loading it.

    ``MLXProvider.effective_loader`` is the same answer read off a live provider,
    and is therefore null for every model that is not resident. This is the
    unloaded-model form, for callers that have a config and no process: the admin
    listing, and through it the live smoke harness, whose whole premise is that an
    arm names an ENGINE rather than a provider Literal.

    ``None`` for anything but ``"mlx"``: the question is *which mlx library*, and
    it has no answer for a gguf subprocess or an embedding model. gguf is one
    engine, already named by ``provider``.

    Pure over the config plus one mtime-cached read of the model dir's
    ``config.json`` -- no import of the model, no MLX. It agrees with the loaded
    provider by CONSTRUCTION: both call :func:`resolve_effective_loader` with the
    same two inputs.
    """
    if provider != "mlx":
        return None
    return resolve_effective_loader(
        config, lambda: read_model_type(config.get("model_path", "") or ""))
