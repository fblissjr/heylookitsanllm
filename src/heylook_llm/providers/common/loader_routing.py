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

import json
import logging
from pathlib import Path
from typing import Callable, Optional


def mlx_vlm_supports(model_type: str) -> bool:
    """Whether mlx-vlm registers a dedicated model class for ``model_type`` (i.e.
    can load it as a real VLM): lower-case, apply MODEL_REMAPPING, then try to
    import the module. Any failure (mlx-vlm absent, unknown type) is a clean False.

    We intentionally do NOT call ``mlx_vlm.utils.get_model_and_args`` here: it
    falls back to a ``text_only`` module (and resolves speculator/dflash aliases)
    rather than raising, so "it resolved" does not mean "loadable as a VLM" -- the
    direct module-import probe is the honest gate signal this router needs."""
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
    try:
        with open(Path(model_path) / "config.json") as f:
            mt = json.load(f).get("model_type")
        return mt if isinstance(mt, str) else None
    except Exception:
        return None


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
    logging.info(
        "loader=auto: model_type %r declares vision but mlx-vlm has no loader for "
        "it; routing to mlx-lm (text)", model_type)
    return "mlx-lm"
