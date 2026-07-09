"""Lens registry: maps a served model id to its converted j-space lens.

Lens conversion (PyTorch .pt -> mx safetensors) is an offline dev-time step
(needs torch), so the server only *loads* pre-converted lenses. Layout:

    <base_dir>/<model_id>/lens.safetensors
    <base_dir>/<model_id>/lens.sidecar.json
    <base_dir>/<model_id>/normalizer.json     (optional; per-model z-score stats)

``<base_dir>`` defaults to the git-tracked ``adapters/jspace`` at the repo root
(contents gitignored, like ``modelzoo/``); override with ``HEYLOOK_JSPACE_DIR``.
The directory name is the served model id (symlink/alias if a lens is shared
across quantizations). Populate it with ``scripts/jspace_convert_lens.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .features import FeatureNormalizer
from .lens import JSpaceLens


class LensRegistry:
    def __init__(self, base_dir: str | os.PathLike | None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self._lens_cache: dict[str, JSpaceLens] = {}

    @classmethod
    def from_env(cls) -> "LensRegistry":
        """``HEYLOOK_JSPACE_DIR`` override, else the git-tracked ``adapters/jspace``
        at the repo root (registry.py -> jspace -> heylook_llm -> src -> repo)."""
        base = os.environ.get("HEYLOOK_JSPACE_DIR")
        if not base:
            base = Path(__file__).resolve().parents[3] / "adapters" / "jspace"
        return cls(base)

    def _dir(self, model_id: str) -> Path | None:
        return self.base_dir / model_id if self.base_dir else None

    def has(self, model_id: str) -> bool:
        d = self._dir(model_id)
        return bool(d and (d / "lens.safetensors").is_file())

    def available(self) -> list[str]:
        if not self.base_dir or not self.base_dir.is_dir():
            return []
        return sorted(p.name for p in self.base_dir.iterdir()
                      if (p / "lens.safetensors").is_file())

    def get(self, model_id: str) -> JSpaceLens:
        if model_id in self._lens_cache:
            return self._lens_cache[model_id]
        d = self._dir(model_id)
        if not d or not (d / "lens.safetensors").is_file():
            raise KeyError(model_id)
        lens = JSpaceLens.from_files(d / "lens.safetensors", d / "lens.sidecar.json")
        self._lens_cache[model_id] = lens
        return lens

    def normalizer(self, model_id: str) -> FeatureNormalizer | None:
        """Optional per-model feature z-scoring stats for the risk router."""
        d = self._dir(model_id)
        if not d or not (d / "normalizer.json").is_file():
            return None
        spec = json.loads((d / "normalizer.json").read_text())
        return FeatureNormalizer(mean=spec["mean"], std=spec["std"])

    def router(self, model_id: str, *, variant: str = "combined"):
        """Optional hallucination-risk classifier (solarkyle-style spec)."""
        d = self._dir(model_id)
        if not d or not (d / "router.json").is_file():
            return None
        from .features import HallucinationRouter
        return HallucinationRouter.from_file(d / "router.json", variant=variant)
