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
from ..providers.common.stop_tokens import resolve_stop_tokens

_DEFAULT_HEATMAP_POSITIONS = 24


def _tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _encode_no_special(tok, text):
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        return tok.encode(text)


def _message_text(m: dict) -> str:
    """Text of a message whose ``content`` is a str OR OpenAI-style content
    blocks (list of {type, text, ...}); non-text blocks (images) are dropped --
    j-space is text-only."""
    content = m.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type", "text") == "text")
    return ""


def format_prompt(model, processor, is_vlm: bool, messages: list[dict],
                  *, chat: bool = False) -> list[int]:
    """Tokenized input ids.

    chat=True: full chat template (matches MLXProvider._apply_template) -- the
    right way to prompt an instruct model, but the final position is the
    generation-prompt boundary (formatting tokens), so the workspace read-out
    there is dominated by format, not content. Good for the risk features
    (which use the answer token's rank), poor for the top-k visualization.

    chat=False (default, viz-first): raw completion -- ``<bos>`` + the joined
    message text, so the final position is a real content token and the
    workspace surfaces sensible "silent words" (e.g. "...city of" -> Paris)."""
    tok = _tokenizer(processor)
    # Coerce content to plain text (drop image blocks) -- j-space is text-only,
    # and it keeps the raw-completion join + chat template from choking on the
    # OpenAI content-block message shape.
    norm = [{"role": m.get("role", "user"), "content": _message_text(m)} for m in messages]
    if chat:
        if is_vlm:
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_tpl
            prompt = vlm_tpl(processor, model.config, norm, num_images=0)
        else:
            prompt = tok.apply_chat_template(norm, tokenize=False,
                                             add_generation_prompt=True)
        return list(tok.encode(prompt) if isinstance(prompt, str) else prompt)

    text = "\n".join(m["content"] for m in norm if m["content"])
    ids = list(_encode_no_special(tok, text))
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):     # gemma needs the BOS sink
        ids = [bos] + ids
    return ids


def _eos_ids(tok) -> set[int]:
    """Stop tokens: reuse the shared resolver (handles eos_token_ids PLURAL) and
    add gemma's <end_of_turn>."""
    eos = set(resolve_stop_tokens(tok))
    try:
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            eos.add(eot)
    except Exception:
        pass
    return eos


def greedy_generate(adapter: ModelAdapter, ids, max_tokens: int, eos_ids):
    """No-cache greedy decode. Returns (first_token, gen_ids, per-token logprobs).

    ``first_token`` is the model's first predicted token even when it is a stop
    token (so the answer-onset workspace read-out reflects the model's real
    disposition -- 'it wanted to stop' -- rather than silently re-deriving it)."""
    ids = list(ids)
    gen, logps = [], []
    first_token = None
    for step in range(max_tokens):
        lg = adapter.logits(mx.array([ids]))[0, -1]
        nxt = int(mx.argmax(lg).item())
        if step == 0:
            first_token = nxt
        if nxt in eos_ids:
            break
        logps.append(float((lg[nxt] - mx.logsumexp(lg)).item()))
        gen.append(nxt)
        ids.append(nxt)
    return first_token, gen, logps


def analyze(provider, lens, messages: list[dict], *, max_answer_tokens: int = 8,
            top_k: int = 8, heatmap: bool = False, chat: bool = False,
            heatmap_positions: int = _DEFAULT_HEATMAP_POSITIONS,
            heatmap_top_k: int = 0,
            router=None, normalizer: F.FeatureNormalizer | None = None) -> dict:
    model, processor, is_vlm = provider.model, provider.processor, provider.is_vlm
    tok = _tokenizer(processor)
    ad = ModelAdapter(model)
    ids = format_prompt(model, processor, is_vlm, messages, chat=chat)

    first_token, gen_ids, step_logprobs = greedy_generate(
        ad, ids, max_answer_tokens, _eos_ids(tok))
    # first_token is the model's real first prediction (even if it's a stop token,
    # i.e. an empty answer). Fallback only for the degenerate max_answer_tokens<1.
    first_answer_id = (first_token if first_token is not None
                       else int(mx.argmax(ad.logits(mx.array([ids]))[0, -1]).item()))
    answer = tok.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""

    band = F.band_layers(ad.n_layers, lens.source_layers)
    if not band:
        raise ValueError("lens has no fitted layers in the workspace band")
    residuals = capture_residuals(model, ids, band, adapter=ad)

    dm = residuals[band[0]].shape[-1]
    if dm != lens.d_model:
        raise ValueError(
            f"lens d_model {lens.d_model} != model residual width {dm} "
            f"(wrong lens for this model?)")

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
        k = max(0, min(heatmap_top_k, int(cells[band[0]].shape[-1])))
        for l in band:
            cl = cells[l]                                    # mx.array [P, vocab]
            # Reduce on-device: bring back only per-position top-1 id + entropy
            # (+ the top-k ids/scores when asked), not the full [P, vocab]
            # logits (~vocab*P doubles otherwise).
            top1 = np.asarray(mx.argmax(cl, axis=-1))
            logp = cl - mx.logsumexp(cl, axis=-1, keepdims=True)
            ent = np.asarray(-mx.sum(mx.exp(logp) * logp, axis=-1))
            kids = kscores = None
            if k > 0:
                part = mx.argpartition(cl, kth=-k, axis=-1)[:, -k:]     # [P, k] unsorted
                kids = np.asarray(part)
                kscores = np.asarray(mx.take_along_axis(cl, part, axis=-1))
            row = []
            for p in range(top1.shape[0]):
                cell = {"token": tok.decode([int(top1[p])]), "entropy": float(ent[p])}
                if kids is not None and kscores is not None:
                    pk_ids, pk_scores = kids[p], kscores[p]
                    order = np.argsort(-pk_scores)
                    cell["top_k"] = [{"token": tok.decode([int(pk_ids[j])]),
                                      "logit": float(pk_scores[j])} for j in order]
                row.append(cell)
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
