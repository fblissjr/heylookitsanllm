# src/edge_llm/providers/mlx_unified/__init__.py
from .text_backend import LmBackend
from .vision_backend import VlmBackend

_BACKENDS = {
    "text":  LmBackend(),
    "vlm":   VlmBackend(),
}

def load(model_cfg):
    """Return the right backend instance for this model."""
    return _BACKENDS["vlm" if model_cfg.get("vision") else "text"]
