"""Unit tests for j-space workspace features + the hallucination router.

Feature math mirrors solarkyle/jspace `probe_uncertainty.py` (readout from lens
logits) and `analyze_router.py` (the 10-feature router vector). Tested with
synthetic lens vectors so no model/download is needed. The offline AUC
reproduction against the shipped TriviaQA traces (V4) lives in the opt-in
harness under coderef/jspace_scratch/.
"""
import json

import numpy as np

from heylook_llm.jspace.features import (
    FeatureNormalizer,
    HallucinationRouter,
    band_layers,
    baseline_features,
    router_feature_vector,
    workspace_readout,
)


def test_band_layers_middle_half():
    # 42-layer model, all layers fitted -> band = [10, 31)
    band = band_layers(42, list(range(41)))
    assert band[0] == 10 and band[-1] == 30
    # only fitted layers survive
    band2 = band_layers(42, [10, 12, 14, 30])
    assert band2 == [10, 12, 14, 30]


# Distinct logits so ranks are unambiguous. _buried: answer far from top
# (rank vocab-1, top1 != answer). _ignited: answer is top-1 (rank 0).
def _buried(vocab, ans):
    v = np.arange(vocab, dtype=float)
    v[ans] = -100.0
    return v


def _ignited(vocab, ans):
    v = np.arange(vocab, dtype=float)
    v[ans] = 1000.0
    return v


def test_workspace_readout_ignition_and_agreement():
    vocab, ans = 20, 7
    lens = {0: _buried(vocab, ans), 1: _buried(vocab, ans),
            2: _ignited(vocab, ans), 3: _ignited(vocab, ans)}
    r = workspace_readout(lens, answer_token_id=ans, hedge_ids=[5])
    assert r["band_agreement"] == 0.5                 # top-1==answer in 2 of 4
    assert r["ignition_frac"] == 0.5                  # rank(answer)<=10 in 2 of 4
    assert r["ignition_depth"] == 0.5                 # first ignites at band index 2 -> 2/4
    assert len(r["layer_entropies"]) == 4
    assert r["mean_log_rank_answer"] > 0


def test_workspace_readout_never_ignites():
    vocab, ans = 20, 7
    lens = {0: _buried(vocab, ans), 1: _buried(vocab, ans)}
    r = workspace_readout(lens, answer_token_id=ans, hedge_ids=[5])
    assert r["ignition_frac"] == 0.0
    assert r["ignition_depth"] == 1.0                 # 1.0 = never


def test_router_feature_vector_from_entropy_trajectory():
    e = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]                # linear, slope 1
    readout = {"layer_entropies": e, "ignition_frac": 0.3, "ignition_depth": 0.4,
               "mean_log_rank_answer": 2.0, "band_agreement": 0.6,
               "best_hedge_rank_log": 1.5}
    f = router_feature_vector(readout)
    assert f["ws_mean_entropy"] == 3.5
    assert f["ws_max_entropy"] == 6.0
    assert np.isclose(f["ws_late_entropy"], np.mean(e[4:]))   # last third (n=6 -> idx 4:)
    assert np.isclose(f["ws_entropy_slope"], 1.0)
    assert f["ws_ignition_frac"] == 0.3 and f["ws_hedge_rank"] == 1.5


def test_baseline_features():
    b = baseline_features([-0.1, -2.0, -0.5])
    assert b["bl_first_token_logprob"] == -0.1
    assert b["bl_min_logprob"] == -2.0
    assert b["bl_answer_len"] == 3
    assert np.isclose(b["bl_mean_logprob"], np.mean([-0.1, -2.0, -0.5]))


def test_router_score_matches_manual_sigmoid():
    spec = {"models": {"workspace_only": {
        "features": ["a", "b"], "weights": [0.5, -1.0], "bias": 0.25}}}
    router = HallucinationRouter(spec, variant="workspace_only")
    norm = FeatureNormalizer(mean={"a": 0.0, "b": 0.0}, std={"a": 1.0, "b": 1.0})
    risk = router.score({"a": 2.0, "b": 1.0}, norm)
    expect = 1 / (1 + np.exp(-(0.5 * 2 + -1.0 * 1 + 0.25)))
    assert np.isclose(risk, expect)


def test_normalizer_fit_transform_roundtrip():
    rows = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 20.0}, {"a": 5.0, "b": 30.0}]
    norm = FeatureNormalizer.fit(rows, ["a", "b"])
    z = norm.transform({"a": 3.0, "b": 20.0}, ["a", "b"])   # the mean row -> ~0
    assert np.allclose(z, [0.0, 0.0], atol=1e-6)


def test_router_from_file(tmp_path):
    spec = {"models": {"combined": {"features": ["a"], "weights": [1.0], "bias": 0.0}}}
    p = tmp_path / "router.json"
    p.write_text(json.dumps(spec))
    router = HallucinationRouter.from_file(p, variant="combined")
    assert router.features == ["a"]
