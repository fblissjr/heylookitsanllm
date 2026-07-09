"""The post-hoc j-space analyze pipeline.

Given a chat prompt and a loaded model + fitted lens: format the prompt exactly
as the server does (chat template + BOS -- both load-bearing for gemma), greedily
generate a short answer, capture residual-stream activations, and read the
Jacobian-lens "workspace" out of them: per-band-layer top-k silent tokens at the
answer-onset position, an optional layer x position heatmap, workspace features,
and (if a per-model normalizer is configured) a hallucination-risk score.

Compute is O(band_layers x positions x vocab) unembeds + a few no-cache greedy
forwards -- fine for an analysis endpoint, NOT a hot path.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import features as F
from .capture import ModelAdapter, capture_residuals

_DEFAULT_HEATMAP_POSITIONS = 24


def _tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _encode_no_special(tok, text):
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        return tok.encode(text)


def format_prompt(model, processor, is_vlm: bool, messages: list[dict]) -> list[int]:
    """Formatted + tokenized input ids, matching MLXProvider._apply_template
    (chat template then tokenizer.encode -- includes gemma's <bos>)."""
    tok = _tokenizer(processor)
    if is_vlm:
        from mlx_vlm.prompt_utils import apply_chat_template as vlm_tpl
        prompt = vlm_tpl(processor, model.config, messages, num_images=0)
    else:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    return list(tok.encode(prompt) if isinstance(prompt, str) else prompt)


def _eos_ids(tok) -> set[int]:
    eos = set()
    e = getattr(tok, "eos_token_id", None)
    if isinstance(e, int):
        eos.add(e)
    try:
        eot = tok.convert_tokens_to_ids("<end_of_turn>")   # gemma turn end
        if isinstance(eot, int) and eot >= 0:
            eos.add(eot)
    except Exception:
        pass
    return eos


def greedy_generate(adapter: ModelAdapter, ids, max_tokens: int, eos_ids):
    """No-cache greedy decode. Returns (gen_ids, per-token logprobs)."""
    ids = list(ids)
    gen, logps = [], []
    for _ in range(max_tokens):
        lg = adapter.logits(mx.array([ids]))[0, -1]
        nxt = int(mx.argmax(lg).item())
        logp = float((lg[nxt] - mx.logsumexp(lg)).item())
        if nxt in eos_ids:
            break
        gen.append(nxt)
        logps.append(logp)
        ids.append(nxt)
    return gen, logps


def analyze(provider, lens, messages: list[dict], *, max_answer_tokens: int = 8,
            top_k: int = 8, heatmap: bool = False,
            heatmap_positions: int = _DEFAULT_HEATMAP_POSITIONS,
            router=None, normalizer: F.FeatureNormalizer | None = None) -> dict:
    model, processor, is_vlm = provider.model, provider.processor, provider.is_vlm
    tok = _tokenizer(processor)
    ad = ModelAdapter(model)
    ids = format_prompt(model, processor, is_vlm, messages)

    gen_ids, step_logprobs = greedy_generate(ad, ids, max_answer_tokens, _eos_ids(tok))
    if gen_ids:
        first_answer_id = gen_ids[0]
    else:
        first_answer_id = int(mx.argmax(ad.logits(mx.array([ids]))[0, -1]).item())
    answer = tok.decode(gen_ids).strip() if gen_ids else ""

    band = F.band_layers(ad.n_layers, lens.source_layers)
    if not band:
        raise ValueError("lens has no fitted layers in the workspace band")
    residuals = capture_residuals(model, ids, band, adapter=ad)

    # Answer-onset workspace (final prompt position): top-k silent tokens per band layer.
    onset = lens.apply(ad, residuals, positions=[-1], layers=band)
    onset_vectors = {l: np.asarray(onset[l][0], dtype=np.float64) for l in band}
    onset_strip = []
    for l in band:
        v = onset_vectors[l]
        idx = np.argsort(-v)[:top_k]
        onset_strip.append({
            "layer": int(l),
            "entropy": F._entropy(v),
            "top_k": [{"token": tok.decode([int(t)]), "logit": float(v[t])} for t in idx],
        })

    # Optional layer x position heatmap (top-1 token + entropy per cell).
    grid = None
    positions = None
    if heatmap:
        seq = len(ids)
        start = max(0, seq - heatmap_positions)
        positions = list(range(start, seq))
        cells = lens.apply(ad, residuals, positions=positions, layers=band)
        grid = []
        for l in band:
            arr = np.asarray(cells[l], dtype=np.float64)     # [P, vocab]
            row = [{"token": tok.decode([int(arr[p].argmax())]),
                    "entropy": F._entropy(arr[p])} for p in range(arr.shape[0])]
            grid.append({"layer": int(l), "cells": row})

    # Workspace features + optional hallucination risk.
    hedge = F.hedge_token_ids(lambda w: _encode_no_special(tok, w))
    readout = F.workspace_readout(onset_vectors, first_answer_id, hedge)
    feats = F.router_feature_vector(readout)
    if step_logprobs:
        feats.update(F.baseline_features(step_logprobs))
    risk = None
    if router is not None and normalizer is not None and all(f in feats for f in router.features):
        try:
            risk = router.score(feats, normalizer)
        except Exception:
            risk = None

    return {
        "model": getattr(provider, "model_id", None),
        "answer": answer,
        "first_answer_token": tok.decode([first_answer_id]),
        "prompt_tokens": [tok.decode([i]) for i in ids],
        "band_layers": [int(l) for l in band],
        "onset_strip": onset_strip,
        "heatmap": grid,
        "heatmap_positions": positions,
        "features": feats,
        "risk": risk,
    }
