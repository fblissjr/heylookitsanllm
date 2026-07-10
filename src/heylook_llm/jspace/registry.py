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
        """``HEYLOOK_JSPACE_DIR`` override, else the git-tracked ``adapters/jspace``.
        Tries the src-layout repo root (registry.py -> jspace -> heylook_llm -> src
        -> repo); if that doesn't exist (e.g. a non-editable/wheel install), falls
        back to ``<cwd>/adapters/jspace`` so a server started from the repo root
        still finds lenses."""
        base = os.environ.get("HEYLOOK_JSPACE_DIR")
        if not base:
            candidate = Path(__file__).resolve().parents[3] / "adapters" / "jspace"
            if not candidate.is_dir():
                cwd_candidate = Path.cwd() / "adapters" / "jspace"
                if cwd_candidate.is_dir():
                    candidate = cwd_candidate
            base = candidate
        return cls(base)

    def _dir(self, model_id: str) -> Path | None:
        return self.base_dir / model_id if self.base_dir else None

    @staticmethod
    def _is_lens_dir(d: Path) -> bool:
        # Require BOTH files: a lens.safetensors without its sidecar (e.g. a
        # crashed/partial convert) must not pass has() then 500 in get().
        return (d / "lens.safetensors").is_file() and (d / "lens.sidecar.json").is_file()

    def has(self, model_id: str) -> bool:
        d = self._dir(model_id)
        return bool(d and self._is_lens_dir(d))

    def available(self) -> list[str]:
        if not self.base_dir or not self.base_dir.is_dir():
            return []
        return sorted(p.name for p in self.base_dir.iterdir()
                      if p.is_dir() and self._is_lens_dir(p))

    def get(self, model_id: str) -> JSpaceLens:
        if model_id in self._lens_cache:
            return self._lens_cache[model_id]
        d = self._dir(model_id)
        if not d or not self._is_lens_dir(d):
            raise KeyError(model_id)
        lens = JSpaceLens.from_files(d / "lens.safetensors", d / "lens.sidecar.json")
        self._lens_cache[model_id] = lens
        return lens

    def provenance(self, model_id: str) -> dict | None:
        """Lens provenance for the UI. ``provisional`` = no own-fit stamp (empty
        or missing ``hf_model_name`` in the sidecar -- e.g. a converted
        third-party lens of unknown origin). Deliberately does NOT return the
        model name itself -- the UI only needs the flag."""
        d = self._dir(model_id)
        if not d or not self._is_lens_dir(d):
            return None
        try:
            side = json.loads((d / "lens.sidecar.json").read_text())
        except Exception:
            return {"provisional": True}
        return {
            "provisional": not side.get("hf_model_name"),
            "fit_date": side.get("fit_date"),
            "fit_source": side.get("fit_source"),
            "n_prompts": side.get("n_prompts"),
        }

    def normalizer(self, model_id: str) -> FeatureNormalizer | None:
        """Optional per-model feature z-scoring stats for the risk router."""
        d = self._dir(model_id)
        if not d or not (d / "normalizer.json").is_file():
            return None
        spec = json.loads((d / "normalizer.json").read_text())
        return FeatureNormalizer(mean=spec["mean"], std=spec["std"])

    def router(self, model_id: str, *, variant: str | None = None):
        """Optional hallucination-risk classifier (solarkyle-style spec). When
        ``variant`` is None, prefers 'combined', else 'workspace_only', else the
        first defined variant -- never KeyErrors on a spec missing 'combined'."""
        d = self._dir(model_id)
        if not d or not (d / "router.json").is_file():
            return None
        from .features import HallucinationRouter
        spec = json.loads((d / "router.json").read_text())
        models = spec.get("models", {})
        if variant is None:
            variant = next((v for v in ("combined", "workspace_only") if v in models),
                           next(iter(models), None))
        if variant is None or variant not in models:
            return None
        return HallucinationRouter(spec, variant=variant)
