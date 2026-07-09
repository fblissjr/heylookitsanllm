"""Unit tests for the j-space (Jacobian lens) module.

Uses tiny random-weight mlx-lm models (no downloads) to verify the
architecture adapter, residual capture, and the reference invariant
`unembed(last_block_output) == model_logits` across two unembed shapes:
gpt2 (LayerNorm final norm, tied head, no softcap) and gemma2 (RMSNorm,
sqrt(d) embed scaling, tied head, final_logit_softcapping).

The full fitted-lens parity (V1/V2, cos ~1.0 vs the genuine jlens) lives in
the opt-in spike harness under coderef/jspace_scratch/; here we test the
mechanics deterministically and download-free.
"""
import mlx.core as mx
import numpy as np
import pytest

from heylook_llm.jspace import JSpaceLens, ModelAdapter, capture_residuals


def _gpt2_tiny():
    from mlx_lm.models.gpt2 import Model, ModelArgs
    args = ModelArgs(model_type="gpt2", n_ctx=64, n_embd=32, n_head=4, n_layer=3,
                     n_positions=64, layer_norm_epsilon=1e-5, vocab_size=50)
    m = Model(args)
    mx.eval(m.parameters())
    return m


def _gemma2_tiny():
    from mlx_lm.models.gemma2 import Model, ModelArgs
    args = ModelArgs(model_type="gemma2", hidden_size=32, num_hidden_layers=3,
                     intermediate_size=64, num_attention_heads=4, head_dim=8,
                     rms_norm_eps=1e-6, vocab_size=50, num_key_value_heads=2)
    m = Model(args)
    mx.eval(m.parameters())
    return m


@pytest.fixture(autouse=True)
def _seed():
    mx.random.seed(0)


@pytest.mark.parametrize("builder,expect_softcap", [(_gpt2_tiny, None), (_gemma2_tiny, 30.0)])
def test_adapter_resolves(builder, expect_softcap):
    model = builder()
    ad = ModelAdapter(model)
    assert ad.n_layers == 3
    assert len(ad.layers) == 3
    assert ad.softcap == expect_softcap


def test_adapter_resolves_multimodal_nesting():
    """gemma-4 VLM shape: text stack under model.language_model.model, softcap
    on language_model. The adapter must drill through the wrapper."""
    class _Embed:
        def as_linear(self, x):
            return x

    class _Inner:                      # Gemma4TextModel-like
        def __init__(self, layers):
            self.norm = lambda x: x
            self.embed_tokens = _Embed()
            self.layers = layers

    class _LM:                         # mlx-vlm LanguageModel-like
        def __init__(self, layers):
            self.model = _Inner(layers)
            self.final_logit_softcapping = 30.0

    class _VLM:
        def __init__(self):
            self.language_model = _LM([object(), object(), object(), object()])

        @property
        def layers(self):
            return self.language_model.model.layers

    vlm = _VLM()
    ad = ModelAdapter(vlm)
    assert ad.inner is vlm.language_model.model     # drilled to the text decoder
    assert ad.n_layers == 4
    assert ad.softcap == 30.0                        # found on language_model


@pytest.mark.parametrize("builder", [_gpt2_tiny, _gemma2_tiny])
def test_capture_shapes(builder):
    model = builder()
    ids = [1, 2, 3, 4, 5]
    res = capture_residuals(model, ids, layers=[0, 2])
    assert set(res.keys()) == {0, 2}
    for arr in res.values():
        assert arr.shape == (len(ids), 32)   # [L, d_model]


@pytest.mark.parametrize("builder", [_gpt2_tiny, _gemma2_tiny])
def test_unembed_invariant(builder):
    """unembed(last block output) must reproduce the model's real logits."""
    model = builder()
    ad = ModelAdapter(model)
    ids = [1, 2, 3, 4, 5]
    inputs = mx.array([ids])
    logits = model(inputs)[0]                              # [L, vocab]
    res = capture_residuals(model, ids, layers=[ad.n_layers - 1], adapter=ad)
    recon = ad.unembed(res[ad.n_layers - 1])              # [L, vocab]
    assert np.allclose(np.asarray(recon), np.asarray(logits), atol=1e-3, rtol=1e-3)


def test_lens_apply_identity_matches_unembed():
    """With identity J, lens.apply == unembed(residual) at the chosen positions."""
    model = _gpt2_tiny()
    ad = ModelAdapter(model)
    ids = [1, 2, 3, 4, 5]
    res = capture_residuals(model, ids, layers=[0, 1])
    eye = mx.eye(32, dtype=mx.float32)
    lens = JSpaceLens(jacobians={0: eye, 1: eye}, source_layers=[0, 1], d_model=32)
    out = lens.apply(ad, res, positions=[-1])
    for l in (0, 1):
        expect = ad.unembed(res[l][[-1]])
        assert np.allclose(np.asarray(out[l]), np.asarray(expect), atol=1e-4)


def test_capture_pipeline_fresh_slice_layers():
    """Regression (#1): pipeline-parallel models (Qwen3.5, deepseek, glm4_moe)
    expose .layers as a property returning a FRESH slice each access, and the
    forward iterates that slice. Capture must mutate the underlying list, not the
    snapshot, or it silently records nothing."""
    d = 8

    class Blk:
        def __call__(self, x, *a, **k):
            return x + 1.0

    class Embed:
        def __init__(self):
            self.weight = mx.zeros((5, d))

        def __call__(self, ids):
            return mx.zeros((ids.shape[0], ids.shape[1], d))

        def as_linear(self, x):
            return x

    class Inner:                                  # mimics PipelineMixin
        def __init__(self):
            self.norm = lambda x: x
            self.embed_tokens = Embed()
            self.layers = [Blk(), Blk(), Blk(), Blk()]   # the REAL list

        @property
        def pipeline_layers(self):
            return self.layers[0:None]            # fresh slice every access

        def __call__(self, inputs):
            x = self.embed_tokens(inputs)
            for layer in self.pipeline_layers:    # iterate the fresh slice
                x = layer(x)
            return self.norm(x)

    class Model:
        def __init__(self):
            self.model = Inner()

        @property
        def layers(self):
            return self.model.pipeline_layers     # fresh slice (the bug trigger)

    m = Model()
    ad = ModelAdapter(m)
    assert ad.n_layers == 4
    res = capture_residuals(m, [1, 2, 3], layers=[0, 2], adapter=ad)
    assert set(res.keys()) == {0, 2}
    for arr in res.values():
        assert arr.shape == (3, d)


def test_lens_from_files(tmp_path):
    from safetensors.numpy import save_file
    import json
    d = 32
    save_file({"0": np.eye(d, dtype=np.float32), "2": np.eye(d, dtype=np.float32)},
              str(tmp_path / "lens.safetensors"))
    (tmp_path / "lens.sidecar.json").write_text(json.dumps(
        {"source_layers": [0, 2], "d_model": d, "final_logit_softcapping": 30.0}))
    lens = JSpaceLens.from_files(tmp_path / "lens.safetensors", tmp_path / "lens.sidecar.json")
    assert lens.source_layers == [0, 2]
    assert lens.d_model == d
    r = mx.random.normal((4, d))
    assert lens.transport(r, 0).shape == (4, d)
