"""Unit tests for the j-space LensRegistry."""
import json

import numpy as np
from safetensors.numpy import save_file

from heylook_llm.jspace.registry import LensRegistry


def _make_lens(dir_path, source_layers=(0, 1), d=8):
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file({str(l): np.eye(d, dtype=np.float32) for l in source_layers},
              str(dir_path / "lens.safetensors"))
    (dir_path / "lens.sidecar.json").write_text(json.dumps(
        {"source_layers": list(source_layers), "d_model": d,
         "final_logit_softcapping": None}))


def test_empty_registry(tmp_path):
    reg = LensRegistry(None)
    assert reg.available() == []
    assert reg.has("anything") is False


def test_available_and_get(tmp_path):
    _make_lens(tmp_path / "model-a")
    _make_lens(tmp_path / "model-b")
    (tmp_path / "not-a-lens").mkdir()            # no lens.safetensors -> ignored
    reg = LensRegistry(tmp_path)
    assert reg.available() == ["model-a", "model-b"]
    assert reg.has("model-a") and not reg.has("not-a-lens")
    lens = reg.get("model-a")
    assert lens.source_layers == [0, 1] and lens.d_model == 8
    assert reg.get("model-a") is lens            # cached


def test_missing_get_raises(tmp_path):
    reg = LensRegistry(tmp_path)
    try:
        reg.get("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_normalizer_and_router_optional(tmp_path):
    d = tmp_path / "model-a"
    _make_lens(d)
    reg = LensRegistry(tmp_path)
    assert reg.normalizer("model-a") is None     # no normalizer.json
    assert reg.router("model-a") is None
    (d / "normalizer.json").write_text(json.dumps({"mean": {"x": 1.0}, "std": {"x": 2.0}}))
    (d / "router.json").write_text(json.dumps(
        {"models": {"combined": {"features": ["x"], "weights": [1.0], "bias": 0.0}}}))
    norm = reg.normalizer("model-a")
    assert norm is not None and norm.mean["x"] == 1.0
    router = reg.router("model-a")
    assert router is not None and router.features == ["x"]
