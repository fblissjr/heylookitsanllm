"""Architecture adapter + residual-stream capture for the Jacobian lens.

The lens needs three things from a model, and they live in slightly different
places per architecture: the residual blocks (to read block outputs), the final
pre-unembed norm, and the unembedding head (+ any final-logit softcap). This
module resolves them generically for mlx-lm ``Model`` objects and captures block
outputs via a temporary wrapper around each block's ``__call__`` (mlx has no
forward-hook API), mirroring the reference jlens ``ActivationRecorder``.

The recorded activation at layer ``l`` is the *output* of block ``l`` (the
residual stream after the block) -- the same convention the lens was fit on.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

_NORM_ATTRS = ("norm", "ln_f", "final_layernorm", "final_layer_norm")
_EMBED_ATTRS = ("embed_tokens", "wte", "embed_in")
# The underlying, mutable block-list attribute on the text decoder. This is the
# real list the forward's loop reads (directly, or via a `pipeline_layers` slice
# of it) -- NOT the top ``model.layers`` property, which for pipeline-parallel
# models (Qwen3.5, deepseek, glm4_moe) returns a FRESH slice each access, so
# mutating that snapshot would never reach the forward. See capture_residuals.
_LIST_ATTRS = ("layers", "h")


def _holder_chain(model: Any) -> list:
    """Candidate modules that may hold the text decoder / norm / head / softcap,
    walking the common nestings: ``model`` -> ``.model`` and ``.language_model``
    -> ``.model`` (mlx-vlm multimodal wrappers put the text stack under
    ``language_model.model``)."""
    holders, seen = [], set()

    def add(obj):
        if obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            holders.append(obj)

    add(model)
    add(getattr(model, "language_model", None))
    for h in list(holders):
        add(getattr(h, "model", None))
    # one more hop for language_model.model
    for h in list(holders):
        add(getattr(getattr(h, "language_model", None), "model", None))
    return holders


class ModelAdapter:
    """Resolves the lens-relevant submodules of an mlx-lm / mlx-vlm ``Model``.

    Exposes ``layers`` (residual blocks), ``final_norm`` / ``head`` / ``unembed``
    (matching the model's real logit path, so gemma soft-cap and tied embeddings
    are correct by construction), and ``softcap``. Handles multimodal wrappers
    (e.g. gemma-4 VLM: text stack under ``model.language_model.model``).
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        holders = _holder_chain(model)

        # The text decoder is the first holder exposing BOTH a final norm and an
        # input embedding.
        self.inner = next(
            (h for h in holders
             if self._find(h, _NORM_ATTRS) is not None
             and self._find(h, _EMBED_ATTRS) is not None),
            None)
        if self.inner is None:
            raise ValueError(
                f"could not locate the text decoder in {type(model).__name__} "
                f"(walked {[type(h).__name__ for h in holders]})")

        # The REAL underlying block list on the text decoder (see _LIST_ATTRS).
        self._blocks = self._find(self.inner, _LIST_ATTRS)
        if self._blocks is None:
            raise ValueError(
                f"could not locate the residual block list on {type(self.inner).__name__} "
                f"(tried {_LIST_ATTRS})")
        self._norm = self._find(self.inner, _NORM_ATTRS)

        # A factory for a fresh per-layer cache list, if the model exposes one.
        # Hybrid models (mlx-vlm qwen3.5: KVCache + ArraysCache) REQUIRE a real
        # cache -- their full-attention block dereferences ``cache.offset`` with
        # no None-guard, so a cache-less forward crashes (AttributeError on
        # NoneType). Non-hybrid models (gemma, gpt2) tolerate cache=None, but
        # handing them a fresh empty cache is the normal generation path and
        # produces identical residuals. See _resolve_make_cache / fresh_cache.
        self._make_cache = self._resolve_make_cache(holders)

        head_mod = next((hm for hm in (getattr(h, "lm_head", None) for h in holders)
                         if hm is not None and hasattr(hm, "weight")), None)
        if head_mod is not None:
            self._head = head_mod                     # untied unembedding
        else:
            self._head = self._find(self.inner, _EMBED_ATTRS).as_linear  # tied

        self.softcap: float | None = self._find_softcap(holders)

    @staticmethod
    def _find(obj: Any, attrs: tuple[str, ...]):
        for a in attrs:
            found = getattr(obj, a, None)
            if found is not None:
                return found
        return None

    def _resolve_make_cache(self, holders: list):
        """The first holder ``make_cache`` that yields a per-layer list matching
        this model's block count, or None. Length-matching guards against a
        wrapper whose ``make_cache`` returns something else (e.g. a vision cache).
        The probe result is discarded -- fresh caches start empty and cheap."""
        n = self.n_layers
        for h in holders:
            mk = getattr(h, "make_cache", None)
            if not callable(mk):
                continue
            try:
                probe = mk()
            except Exception:
                continue
            if isinstance(probe, (list, tuple)) and len(probe) == n:
                return mk
        return None

    def fresh_cache(self):
        """A new, empty per-layer cache for a single standalone (full-sequence)
        forward, or None when the model tolerates a cache-less forward. Each
        analyze forward re-prefills the whole sequence, so a throwaway fresh
        cache (offset 0) reproduces no-cache semantics -- required for the
        hybrid mlx-vlm path (see _make_cache)."""
        return self._make_cache() if self._make_cache is not None else None

    def run_inner(self, input_ids: mx.array) -> mx.array:
        """The inner (pre-head) forward with a fresh cache when the model needs
        one. The cache is passed as a KWARG only when non-None: architectures
        differ in positional order (mlx-vlm qwen3.5 puts inputs_embeds second),
        and a cache-less path must not force a ``cache=`` arg on forwards that
        don't declare one."""
        cache = self.fresh_cache()
        return self.inner(input_ids) if cache is None else self.inner(input_ids, cache=cache)

    @staticmethod
    def _find_softcap(holders: list) -> float | None:
        for h in holders:
            for src in (h, getattr(h, "args", None), getattr(h, "config", None)):
                cap = getattr(src, "final_logit_softcapping", None)
                if cap:
                    return float(cap)
        return None

    @property
    def layers(self):
        return self._blocks

    @property
    def n_layers(self) -> int:
        return len(self._blocks)

    def final_norm(self, x: mx.array) -> mx.array:
        return self._norm(x)

    def head(self, x: mx.array) -> mx.array:
        return self._head(x)

    def unembed(self, x: mx.array) -> mx.array:
        """Map a PRE-norm residual ``[..., d_model]`` to logits:
        softcap(head(final_norm(x)))."""
        logits = self._head(self._norm(x))
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    def logits(self, input_ids: mx.array) -> mx.array:
        """The model's real logits for ``input_ids`` ``[1, L]``: runs the text
        forward (already applies the final norm) then head + softcap. Used for
        greedy answer generation in the analyze pipeline."""
        normed = self.run_inner(input_ids)            # post-final-norm hidden
        out = self._head(normed)
        if self.softcap:
            out = mx.tanh(out / self.softcap) * self.softcap
        return out


class _Recorder:
    """Wraps a block: delegates the call, stashes the (batch-stripped) output.

    Attribute access is proxied to the wrapped block so the surrounding forward
    can still read block attributes (e.g. gemma-4's ``layer.layer_type`` in mask
    construction) while the block is temporarily swapped out.
    """

    __slots__ = ("_mod", "_store", "_idx")

    def __init__(self, mod, store, idx):
        self._mod, self._store, self._idx = mod, store, idx

    def __call__(self, *args, **kwargs):
        out = self._mod(*args, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        self._store[self._idx] = tensor[0]            # drop batch -> [L, d_model]
        return out

    def __getattr__(self, name):                      # _mod is a slot -> no recursion
        return getattr(self._mod, name)


def capture_residuals(model, input_ids, layers, *, adapter: ModelAdapter | None = None):
    """Run ``model`` on ``input_ids`` once and return block-output residuals.

    Args:
        model: An mlx-lm ``Model``.
        input_ids: A 1-D sequence of token ids (no batch dim).
        layers: Block indices to capture (the lens's ``source_layers``).
        adapter: Optional pre-built :class:`ModelAdapter` (avoids re-resolving).

    Returns:
        ``{layer_index: mx.array[seq_len, d_model]}`` for each requested layer.
    """
    ad = adapter or ModelAdapter(model)
    blocks = ad.layers
    want = sorted(set(int(l) for l in layers))
    store: dict[int, mx.array] = {}

    originals = {i: blocks[i] for i in want}
    try:
        for i in want:
            blocks[i] = _Recorder(originals[i], store, i)
        # Fresh empty cache when the model needs one (hybrid mlx-vlm attention
        # requires a real cache); see ModelAdapter.run_inner. Head is skipped.
        ad.run_inner(mx.array([list(input_ids)]))
        mx.eval(list(store.values()))
    finally:
        for i, mod in originals.items():
            blocks[i] = mod
    missing = [i for i in want if i not in store]
    if missing:
        # The forward never called our recorders: adapter.layers is not the object
        # the forward iterates (would be a silently-empty read-out otherwise).
        raise RuntimeError(
            f"jspace capture recorded nothing for layers {missing} on "
            f"{type(model).__name__}; the forward iterates a different block list "
            f"than adapter.layers")
    return store
