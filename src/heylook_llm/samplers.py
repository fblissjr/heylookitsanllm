"""The sampler cascade: what decode settings a request actually runs with.

ONE function does the work -- ``resolve_effective_sampling`` -- and its whole
job is to answer "the request said nothing, so what?" in a defined order.

The answer is, in order: the model's OWN published settings, then anything
models.toml says about that model, then whatever the request states outright.
Only where all three are silent does a hardcoded fallback apply, and that
fallback is two numbers.

The bundled sampler REGISTRY that used to sit here -- five TOMLs under
``data/samplers/`` loaded by a ``SamplerRegistry``, reachable as
``ChatRequest.sampler`` and models.toml ``default_sampler`` -- was removed in
v2.0.30. It shipped generic guesses that applied the same values to every
model, which is the opposite of what the vendor layer does; three of its five
entries had no consumer at all, and the frontend never touched any of it.
Named bundles that a USER wants still exist, as the DuckDB ``/v1/presets``
system, which is editable and is what v3's preset bar drives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# THE FLOOR IS TWO OPINIONS, A SAFETY STOP, AND FOUR OFF-SWITCHES -- kept
# apart because they are not the same kind of thing and rot differently.
#
# Only these two are a judgement about how to sample, and they apply ONLY
# where the model's own metadata is silent. The vendor layer below overlays a
# model's published values, and that is the answer that should normally win:
# a per-model number the author chose beats a global number we chose.
# Owner ruling 2026-09-01 (v1.79.60) -- a narrowed distribution flattens
# generative prose, and low temperature was judged worse on real output. The
# 0.7/1.0 before it was a chat-sane guess; the 0.1/512 before that made freshly
# imported models near-greedy and truncated long answers mid-sentence.
FALLBACK_TEMPERATURE = 1.0
FALLBACK_TOP_P = 0.95

# NOT taste. llama-server's own `n_predict` default is UNLIMITED, so a request
# naming no cap generates until the context runs out. A stop, not a preference.
DEFAULT_MAX_TOKENS = 4096

# Each of these means "this knob is OFF", not "we prefer this value" -- and
# they are load-bearing for a reason that is easy to miss: the ENGINE's own
# defaults are not neutral. llama.cpp ships `top_k = 40` (common/common.h) and
# applies it to any request that omits the key. Dropping these would hand each
# engine its own taste back and let the two diverge on identical input.
KNOBS_OFF = {
    'top_k': 0,
    'min_p': 0.0,
    'repetition_penalty': 1.0,
    'presence_penalty': 0.0,
}

GLOBAL_SAMPLER_FLOOR = {
    'temperature': FALLBACK_TEMPERATURE,
    'top_p': FALLBACK_TOP_P,
    'max_tokens': DEFAULT_MAX_TOKENS,
    **KNOBS_OFF,
}

# Anti-loop overlay applied whenever thinking is ON, both engines, every
# thinking-capable model.
#
# UNMEASURED, and worth knowing before trusting it: the value came from a
# "Qwen3-style" bundle in July 2026 and became automatic in the same commit
# that slimmed that bundle away, on the strength of one gemma MoE repetition
# loop. It is the one survivor of a set whose other values were replaced by
# the vendor layer, and it survived only because no vendor ships a
# `presence_penalty` -- it is not an HF generation_config field at all.
# Qwen's own published guidance for a THINKING model is 0.0; 1.5 is what they
# recommend for the non-thinking variant. Nothing here has been measured on
# this hardware. See CLAUDE.md on why an unmeasured sampling default is the
# kind of thing this repo otherwise refuses to generalise.
THINKING_PRESENCE_PENALTY = 1.5


VENDOR_SAMPLING_KEYS = ('temperature', 'top_p', 'top_k')


# Model-config / request keys the cascade resolves. Providers whose config
# class lacks a key (GGUFModelConfig has only max_tokens/default_sampler)
# simply never contribute it; unknown keys in the result are ignored by the
# consumer (gguf's payload map picks only what llama-server understands).
EFFECTIVE_SAMPLER_KEYS = (
    'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
    'repetition_penalty', 'repetition_context_size', 'presence_penalty',
    'enable_thinking', 'reasoning_effort', 'vision_tokens',
)
REQUEST_SAMPLER_FIELDS = EFFECTIVE_SAMPLER_KEYS + ('seed',)


def resolve_effective_sampling(request: Any, model_config: dict,
                               vendor: dict | None = None, *,
                               thinking_capable: bool = False) -> dict[str, Any]:
    """THE effective-request cascade, shared by every provider.

    ``thinking_capable`` is whether the MODEL can think at all (the same
    answer as the served ``thinking`` capability: gguf's ``supports_thinking``,
    MLX's template probe). It is the LAST fallback for the thinking switch --
    see the resolution order at ``thinking_active`` below -- and the caller
    passes it because the cascade has no file or template access of its own.

    Layers, later overriding earlier (each only for fields it sets):
      1.  Global floor (``GLOBAL_SAMPLER_FLOOR``).
      1b. Vendor layer -- the model's own generation_config.json values,
          passed by the caller: MLX from generation_config.json
          (``load_vendor_sampling``), gguf from the GGUF header's
          ``general.sampling.*`` (``gguf_metadata.vendor_sampling``, v2.0.22).
      2.  Thinking anti-loop overlay (``THINKING_PRESENCE_PENALTY``), keyed
          on the EFFECTIVE switch: request.enable_thinking when present,
          else the model config flag, else the capability.
      3.  Model sampler fields from models.toml.
      4.  Request explicit fields -- always win.

    Four layers, not the six this had until v2.0.30: the two named-sampler
    layers (models.toml ``default_sampler`` and ``ChatRequest.sampler``) are
    gone with the bundled registry. A user who wants a named bundle uses the
    ``/v1/presets`` system, which the client expands into explicit fields --
    so those arrive at layer 4 and need no layer of their own.
    """
    merged = dict(GLOBAL_SAMPLER_FLOOR)
    if vendor:
        merged.update(vendor)

    # The thinking switch, resolved in ONE order: the request's explicit
    # value, else the model's models.toml `enable_thinking`, else whether the
    # model CAN think. That last fallback is v1.79.62: from v1.50.0 unset
    # meant OFF on both engines, chosen because the two engines disagreed on
    # unset (gguf omitted the kwarg and got the template's own default,
    # thinking-ON for gemma-4/Qwen3.6; MLX resolved it to False) and because
    # the UI could send only true-or-absent, so "unset = off" was the only
    # way to have an off switch at all. The UI now sends an explicit false,
    # so that reason is gone -- and a thinking model that silently does not
    # think unless someone finds the models.toml flag was the standing
    # complaint (2026-09-04). A model that cannot think still resolves to
    # OFF, so the thinking sampler overlay below never fires on one.
    #
    # Materialized ALWAYS, not only when the overlay fires: gguf's payload
    # builder only sends chat_template_kwargs for a non-None value, so an
    # absent key is not "no opinion" downstream -- both engines must read the
    # same resolved bool, and `thinking_default()` reports this same answer.
    request_thinking = getattr(request, 'enable_thinking', None)
    config_thinking = model_config.get('enable_thinking')
    thinking_active = bool(
        request_thinking if request_thinking is not None
        else config_thinking if config_thinking is not None
        else thinking_capable
    )
    merged['enable_thinking'] = thinking_active
    if thinking_active:
        merged['presence_penalty'] = THINKING_PRESENCE_PENALTY

    merged.update({k: v for k, v in model_config.items()
                   if k in EFFECTIVE_SAMPLER_KEYS and v is not None})

    for field in REQUEST_SAMPLER_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            merged[field] = value
    return merged



class _NoRequest:
    """The empty request: every cascade field absent (getattr -> None)."""

    def __getattr__(self, name):  # pragma: no cover - trivial
        return None


def thinking_default(model_config: dict, *, thinking_capable: bool) -> bool:
    """What thinking resolves to when a request says NOTHING about it.

    The cascade's own answer for an empty request -- a `default_sampler`
    naming the thinking sampler, the models.toml `enable_thinking` flag and
    the capability fallback all count, exactly as they do at generation
    time. Reported on the admin row (`thinking_default`) so a UI can label
    its "model default" choice with the value it actually means instead of
    leaving the user to find out by generating.
    """
    return bool(resolve_effective_sampling(
        _NoRequest(), model_config, thinking_capable=thinking_capable,
    ).get('enable_thinking'))


class _ThinkingRequest(_NoRequest):
    """An empty request that states the thinking switch and nothing else."""

    def __init__(self, enable_thinking: bool):
        self.enable_thinking = enable_thinking


def sampler_defaults(model_config: dict, *, thinking_capable: bool,
                     vendor: dict | None = None) -> dict[str, dict[str, Any]]:
    """What EVERY sampler key resolves to when a request says nothing.

    Sibling of :func:`thinking_default` and bound by the same rule: this is
    the cascade's own answer, run for real, never a re-derivation. Reported
    on the admin row and ``/v1/models`` so the settings panel can show a
    blank field's actual value instead of the word "auto" -- a user who has
    to generate to find out what temperature they are running is the
    complaint this closes.

    Keyed by the THINKING SWITCH, ``{"off": {...}, "on": {...}}``, because
    the anti-loop overlay fires off that switch (thinking.toml sets
    presence_penalty) while the panel's thinking control is live and
    independent. Reporting one state's numbers while the user has selected
    the other would put a WRONG number on screen, which is worse than the
    "auto" it replaces. Two dict merges and no I/O, so the honest shape is
    also the cheap one.

    ``vendor`` must be passed exactly as the model's own PROVIDER passes it
    at generation time -- MLX from the model dir's generation_config.json
    (``load_vendor_sampling``), gguf from the GGUF header's
    ``general.sampling.*`` (``gguf_metadata.vendor_sampling``, v2.0.22).
    temperature/top_p/top_k are precisely the vendor keys, so omitting it for
    an engine that has one reports the global floor for every model that
    overrides it -- the models where the number matters most.
    ``capabilities._vendor_sampling_pairs`` is the one place that pairing
    lives; it drifted once already, within a commit.
    """
    out: dict[str, dict[str, Any]] = {}
    for state, active in (("off", False), ("on", True)):
        merged = resolve_effective_sampling(
            _ThinkingRequest(active), model_config, vendor,
            thinking_capable=thinking_capable,
        )
        out[state] = {k: merged[k] for k in REQUEST_SAMPLER_FIELDS if k in merged}
    return out


def load_vendor_sampling(model_path: str) -> dict[str, Any]:
    """Sampling defaults from ``<model_path>/generation_config.json``.

    Best-effort by design: a missing file, malformed JSON, or non-numeric
    values yield {}/are dropped -- a broken vendor file must never block a
    model load.
    """
    try:
        with open(Path(model_path) / 'generation_config.json', 'rb') as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in VENDOR_SAMPLING_KEYS:
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out
