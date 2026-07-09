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


class ModelAdapter:
    """Resolves the lens-relevant submodules of an mlx-lm ``Model``.

    Exposes ``layers`` (residual blocks), ``final_norm`` / ``head`` / ``unembed``
    (matching the model's real logit path, so gemma soft-cap and tied embeddings
    are correct by construction), and ``softcap``.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.inner = getattr(model, "model", model)   # mlx-lm: Model.model holds blocks

        self._layers = model.layers                   # all mlx-lm Models expose this
        self._norm = self._resolve(self.inner, _NORM_ATTRS, "final norm")

        head_mod = getattr(model, "lm_head", None)
        if head_mod is not None and hasattr(head_mod, "weight"):
            self._head = head_mod                     # untied unembedding
        else:
            embed = self._resolve(self.inner, _EMBED_ATTRS, "input embedding")
            self._head = embed.as_linear              # tied unembedding

        cap = getattr(model, "final_logit_softcapping", None)
        if cap is None:
            cap = getattr(getattr(model, "args", None), "final_logit_softcapping", None)
        self.softcap: float | None = float(cap) if cap else None   # 0/None -> None

    @staticmethod
    def _resolve(obj: Any, attrs: tuple[str, ...], what: str) -> Any:
        for a in attrs:
            found = getattr(obj, a, None)
            if found is not None:
                return found
        raise ValueError(f"could not locate the {what} on {type(obj).__name__} "
                         f"(tried {attrs})")

    @property
    def layers(self):
        return self._layers

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def final_norm(self, x: mx.array) -> mx.array:
        return self._norm(x)

    def head(self, x: mx.array) -> mx.array:
        return self._head(x)

    def unembed(self, x: mx.array) -> mx.array:
        """Map a residual ``[..., d_model]`` to logits: softcap(head(final_norm(x)))."""
        logits = self._head(self._norm(x))
        if self.softcap:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits


class _Recorder:
    """Wraps a block: delegates the call, stashes the (batch-stripped) output."""

    __slots__ = ("_mod", "_store", "_idx")

    def __init__(self, mod, store, idx):
        self._mod, self._store, self._idx = mod, store, idx

    def __call__(self, *args, **kwargs):
        out = self._mod(*args, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        self._store[self._idx] = tensor[0]            # drop batch -> [L, d_model]
        return out


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
        ad.inner(mx.array([list(input_ids)]))         # inner forward; head skipped
        mx.eval(list(store.values()))
    finally:
        for i, mod in originals.items():
            blocks[i] = mod
    return store
