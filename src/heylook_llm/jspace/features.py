"""Workspace features + the hallucination-risk router.

Reproduces the feature math from solarkyle/jspace (Apache-2.0):
``probe_uncertainty.py`` (readout from lens logits at the answer-onset position)
and ``analyze_router.py`` (the 10-feature router vector + logistic regression).

The router predicts P(answer is WRONG) from workspace features (optionally plus
output-confidence baselines). Features must be z-scored per model over your own
traffic before scoring -- that per-model normalization is the whole transfer
trick (train on one model, apply zero-shot to others). See
docs/jspace_integration_plan.md (Phase 4 / V4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Middle ~half of the network -- the paper reads the workspace "band", not
# single layers.
BAND_LO_FRAC, BAND_HI_FRAC = 0.25, 0.75
IGNITION_TOPK = 10

# First sub-token of each hedge word; a low lens-rank for any of these signals
# the workspace is hedging. (From probe_uncertainty.HEDGE_WORDS.)
HEDGE_WORDS = [
    " guess", " maybe", " unsure", " unknown", " perhaps", " possibly",
    " unclear", " uncertain", "?", " hmm", " Hmm", " probably",
]

WORKSPACE_FEATURES = [
    "ws_mean_entropy", "ws_max_entropy", "ws_late_entropy", "ws_entropy_slope",
    "ws_entropy_std", "ws_ignition_frac", "ws_ignition_depth", "ws_mean_log_rank",
    "ws_band_agreement", "ws_hedge_rank",
]
BASELINE_FEATURES = ["bl_first_token_logprob", "bl_mean_logprob", "bl_min_logprob",
                     "bl_answer_len"]


def band_layers(n_layers: int, source_layers, *, lo=BAND_LO_FRAC, hi=BAND_HI_FRAC):
    """The workspace band: fitted layers in the middle ``[lo, hi)`` of the stack."""
    src = set(int(l) for l in source_layers)
    return [l for l in range(int(n_layers * lo), int(n_layers * hi)) if l in src]


def hedge_token_ids(encode) -> list[int]:
    """First-token ids of the hedge words. ``encode`` maps text -> list[int]
    (e.g. a tokenizer's ``.encode`` with specials off)."""
    ids = set()
    for w in HEDGE_WORDS:
        toks = encode(w)
        if toks:
            ids.add(int(toks[0]))
    return sorted(ids)


def _entropy(logits: np.ndarray) -> float:
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def workspace_readout(lens_vectors: dict, answer_token_id: int,
                      hedge_ids) -> dict:
    """Read workspace stats from per-band-layer lens logit vectors.

    Args:
        lens_vectors: ``{band_layer: 1-D logits over vocab}`` at the answer-onset
            (final prompt) position. Iterated in ascending layer order.
        answer_token_id: The token the model actually generated first.
        hedge_ids: Token ids for the hedge words.

    Returns:
        dict with ``layer_entropies`` (list, band order) and the scalars
        ``ignition_frac``, ``ignition_depth`` (band-relative; 1.0 = never),
        ``mean_log_rank_answer``, ``band_agreement``, ``best_hedge_rank_log``,
        ``mean_entropy``.
    """
    hedge_ids = list(hedge_ids)
    ranks_ans, ranks_hedge, entropies, top1s = [], [], [], []
    for layer in sorted(lens_vectors):
        logits = np.asarray(lens_vectors[layer], dtype=np.float64).ravel()
        vocab = len(logits)
        order = np.argsort(-logits)
        rank_of = np.empty(vocab, dtype=np.int64)
        rank_of[order] = np.arange(vocab)
        # A token id can exceed the head's vocab (tokenizer padded larger than the
        # lens/head rows) -> treat any out-of-range id as the worst rank (`vocab`),
        # and tolerate an empty hedge set, rather than IndexError/ValueError.
        ranks_ans.append(int(rank_of[answer_token_id])
                         if 0 <= answer_token_id < vocab else vocab)
        valid_hedge = [int(rank_of[t]) for t in hedge_ids if 0 <= t < vocab]
        ranks_hedge.append(min(valid_hedge) if valid_hedge else vocab)
        entropies.append(_entropy(logits))
        top1s.append(int(order[0]))

    ranks_arr = np.array(ranks_ans)
    ignited = np.nonzero(ranks_arr <= IGNITION_TOPK)[0]
    n_band = len(ranks_ans)
    return {
        "layer_entropies": [float(e) for e in entropies],
        "ignition_frac": float((ranks_arr <= IGNITION_TOPK).mean()),
        "ignition_depth": float(ignited[0] / n_band) if len(ignited) else 1.0,
        "mean_log_rank_answer": float(np.log1p(ranks_arr).mean()),
        "band_agreement": float(np.mean(np.array(top1s) == answer_token_id)),
        "best_hedge_rank_log": float(np.log1p(min(ranks_hedge))),
        "mean_entropy": float(np.mean(entropies)),
    }


def router_feature_vector(readout: dict) -> dict:
    """Map a :func:`workspace_readout` to the 10 named router features
    (5 derived from the entropy trajectory + 5 scalars)."""
    e = np.asarray(readout["layer_entropies"], dtype=np.float64)
    n = len(e)
    slope = float(np.polyfit(np.arange(n), e, 1)[0]) if n >= 2 else 0.0
    return {
        "ws_mean_entropy": float(e.mean()),
        "ws_max_entropy": float(e.max()),
        "ws_late_entropy": float(e[2 * n // 3:].mean()),
        "ws_entropy_slope": slope,
        "ws_entropy_std": float(e.std()),
        "ws_ignition_frac": float(readout["ignition_frac"]),
        "ws_ignition_depth": float(readout["ignition_depth"]),
        "ws_mean_log_rank": float(readout["mean_log_rank_answer"]),
        "ws_band_agreement": float(readout["band_agreement"]),
        "ws_hedge_rank": float(readout["best_hedge_rank_log"]),
    }


def baseline_features(step_logprobs) -> dict:
    """Output-confidence baselines from the generated answer's per-token logprobs."""
    lp = np.asarray(step_logprobs, dtype=np.float64)
    return {
        "bl_first_token_logprob": float(lp[0]),
        "bl_mean_logprob": float(lp.mean()),
        "bl_min_logprob": float(lp.min()),
        "bl_answer_len": int(len(lp)),
    }


class FeatureNormalizer:
    """Per-feature z-scoring stats. Fit over your own traffic per model (the
    transfer trick); a single request cannot be z-scored on its own."""

    def __init__(self, mean: dict, std: dict) -> None:
        self.mean = dict(mean)
        self.std = dict(std)

    @classmethod
    def fit(cls, rows, features) -> "FeatureNormalizer":
        mean, std = {}, {}
        for f in features:
            col = np.asarray([r[f] for r in rows], dtype=np.float64)
            mean[f] = float(col.mean())
            std[f] = float(col.std())
        return cls(mean, std)

    def transform(self, feats: dict, features) -> np.ndarray:
        return np.asarray(
            [(feats[f] - self.mean[f]) / (self.std[f] + 1e-9) for f in features],
            dtype=np.float64)


class HallucinationRouter:
    """Logistic-regression hallucination-risk classifier (predicts P(wrong)).

    Loaded from a solarkyle-style spec: ``{"models": {variant: {"features",
    "weights", "bias"}}}``. Variants: ``workspace_only`` (10 feats) or
    ``combined`` (14, adds output-confidence baselines).
    """

    def __init__(self, spec: dict, *, variant: str = "workspace_only") -> None:
        m = spec["models"][variant]
        self.variant = variant
        self.features = list(m["features"])
        self.weights = np.asarray(m["weights"], dtype=np.float64)
        self.bias = float(m["bias"])

    @classmethod
    def from_file(cls, path, *, variant: str = "workspace_only") -> "HallucinationRouter":
        spec = json.loads(Path(path).read_text())
        return cls(spec, variant=variant)

    def score(self, feats: dict, normalizer: FeatureNormalizer) -> float:
        """P(answer is wrong): ``sigmoid(w . z + b)`` with per-model z-scored feats."""
        z = normalizer.transform(feats, self.features)
        return float(1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias))))
