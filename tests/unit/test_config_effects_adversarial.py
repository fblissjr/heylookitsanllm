"""Adversarial guards on the `effect` classification.

`test_config_effects.py` checks the mechanism does the right thing on today's
data. This file checks it cannot be made to do the WRONG thing -- specifically
the failure shape that has now bitten this design three times: something
degrades quietly toward "no reload required", and every existing check still
passes.

Prior instances, for pattern recognition:
  1. A misspelt effect got its own bucket and vanished from the reload set.
  2. `load_time_only` meant both "immovable" and "applies instantly".
  3. (this file) A hand-written table silently overrode the derived one.
"""
import typing

import pytest
from pydantic import BaseModel, Field as PField

import heylook_llm.model_service as ms
from heylook_llm.config import (
    EFFECT_IDENTITY,
    ModelConfig,
    PROVIDER_CONFIG_CLASSES,
    configurable_fields,
    field_effect,
    fields_by_effect,
    invalid_effects,
    reload_required_fields,
)

PROVIDERS = sorted(PROVIDER_CONFIG_CLASSES)


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_derived_reload_set_beats_the_hand_written_runtime_list(provider):
    """A reload-required field must never be reported as "runtime".

    `get_field_reload_info` fills one dict from two sources: the DERIVED
    reload set and the hand-written `RUNTIME_CHANGEABLE_FIELDS`. Whichever
    loop runs last wins. If the hand-written list wins, the entire point of
    deriving the reload set is undone for any field named in both -- the UI is
    told a spawn-time flag is a live knob, the user changes it, and the
    process keeps the old value.

    The two sets happen not to overlap today, so this cannot be caught by
    inspecting current data; it is a latent trap for the next person who adds
    a field to either side.
    """
    reload_fields = ms.reload_required_for(provider)
    original = ms.RUNTIME_CHANGEABLE_FIELDS
    # Force the collision that current data does not happen to contain.
    collide = sorted(reload_fields)[0]
    svc = ms.ModelService.__new__(ms.ModelService)
    ms.RUNTIME_CHANGEABLE_FIELDS = frozenset(original | {collide})
    try:
        info = ms.ModelService.get_field_reload_info(svc, provider)
        assert info[collide] == "reload_required", (
            f"{provider}.{collide} is reload-required but was reported as "
            f"{info[collide]!r} -- the hand-written RUNTIME_CHANGEABLE_FIELDS "
            f"overrode the derived source of truth"
        )
    finally:
        ms.RUNTIME_CHANGEABLE_FIELDS = original


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_no_field_is_both_reload_required_and_runtime_changeable(provider):
    """The two sets must stay disjoint, not merely resolve in a good order.

    Ordering makes a contradiction harmless; this makes it visible. If a field
    lands in both, one of the two declarations is simply wrong and should be
    corrected at the source rather than papered over by loop order.
    """
    overlap = ms.reload_required_for(provider) & ms.RUNTIME_CHANGEABLE_FIELDS
    assert not overlap, (
        f"{provider}: {sorted(overlap)} are declared requires_reload/identity "
        f"in config.py AND listed in RUNTIME_CHANGEABLE_FIELDS. Fix the "
        f"declaration or drop it from the hand-written list."
    )


@pytest.mark.unit
def test_provider_registry_and_the_provider_literal_stay_in_sync():
    """`ModelConfig.provider` is a hand-maintained Literal beside a registry.

    Adding a provider means editing both, and nothing enforced it. Drift here
    is the same class of bug the effect metadata was introduced to kill: two
    places asserting the same fact, one of them silently stale. A provider in
    the registry but missing from the Literal cannot be configured at all; one
    in the Literal but missing from the registry gets `reload_required_for`'s
    union fallback instead of its own answer.
    """
    literal = set(typing.get_args(ModelConfig.model_fields["provider"].annotation))
    registry = set(PROVIDER_CONFIG_CLASSES)
    assert literal == registry, (
        f"ModelConfig.provider Literal {sorted(literal)} != "
        f"PROVIDER_CONFIG_CLASSES {sorted(registry)}"
    )


# Over-reporting costs a needless prompt; under-reporting serves stale.
# `reload_required_for` keys off the raw provider string from the toml entry.
# - unknown: a missing or misspelt provider (None, '', typo, case drift) must
#   not come back with an EMPTY reload set, which would report every edit as
#   free; it gets the union.
# - registry: every real key is a plain lowercase identifier resolving to its
#   own non-empty set inside the union. A key that were not normalised would
#   make the silent downgrade to the union the NORMAL path, not the exception.
_PROVIDER_SPELLING_ROWS = [
    pytest.param((None, "", "ggf", "MLX", "does-not-exist"), False,
                 id="unknown_provider_falls_back_to_the_conservative_union"),
    pytest.param(tuple(PROVIDERS), True,
                 id="provider_spelling_matches_registry_keys_exactly"),
]


@pytest.mark.unit
@pytest.mark.parametrize("spellings, is_registry_key", _PROVIDER_SPELLING_ROWS)
def test_provider_spelling_resolves_its_reload_set(spellings, is_registry_key):
    union = ms.RELOAD_REQUIRED_FIELDS
    for key in spellings:
        got = ms.reload_required_for(key)
        if is_registry_key:
            assert key == key.lower().strip(), f"registry key {key!r} is not normalised"
            assert got, f"{key!r} resolved to an empty set"
            # The union genuinely covers every provider's own answer.
            assert got <= union
        else:
            assert got == union, (
                f"provider={key!r} should fall back to the union, not {sorted(got)}"
            )


def _mutate(schema):  # pydantic calls this instead of merging a dict
    schema["effect"] = "requires_reload"


# "We don't know" must never become "no reload required". A field whose effect
# is misspelt, set through a callable json_schema_extra, or missing lands in
# the None (unclassified) bucket, stays out of the reload set, and is exactly
# what the import guard reports (the only thing that can stop it, since a class
# registered after import reaches the derived helpers unguarded).
# - misspelt: an earlier version bucketed by the raw string, so
#   "requires-reload" (hyphen) created its own bucket, left the unclassified
#   set empty, passed every completeness check and dropped the field out of
#   the reload set. The guard reports it as bogus.
# - callable: pydantic allows a CALLABLE json_schema_extra that `_extra` cannot
#   read; the declaration LOOKS annotated at a glance. It is *missing*, not
#   *bogus*, so the guard reports it as undeclared (invalid_effects is empty).
# - missing: no json_schema_extra at all; likewise undeclared.
_UNCLASSIFIED_ROWS = [
    pytest.param({"json_schema_extra": {"effect": "requires-reload"}},
                 "requires-reload", {"ctx_size": "requires-reload"},
                 id="a_misspelt_effect_is_unclassified_not_a_new_category"),
    pytest.param({"json_schema_extra": _mutate}, None, {},
                 id="json_schema_extra_as_a_callable_is_unclassified_not_silently_ok"),
    pytest.param({}, None, {},
                 id="an_unclassified_field_never_counts_as_reload_required"),
]


@pytest.mark.unit
@pytest.mark.parametrize("field_kwargs, raw_effect, expected_invalid", _UNCLASSIFIED_ROWS)
def test_an_unknown_effect_is_unclassified(field_kwargs, raw_effect, expected_invalid):
    class Probe(BaseModel):
        model_path: str = PField(json_schema_extra={"effect": EFFECT_IDENTITY})
        ctx_size: int = PField(default=0, **field_kwargs)

    # The raw declaration as read (None when unreadable or absent).
    assert field_effect(Probe.model_fields["ctx_size"]) == raw_effect
    by = fields_by_effect(Probe)
    assert by[None] == {"ctx_size"}, "an unknown effect must land as unclassified"
    assert "requires-reload" not in by, "a typo must not invent a bucket"
    assert "ctx_size" not in reload_required_fields(Probe)
    assert invalid_effects(Probe) == expected_invalid


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_any_field_with_an_arg_spelling_requires_a_reload(provider):
    """If it is emitted into the spawn argv, changing it needs a respawn.

    This is the STRUCTURAL version of the existing check that lists known gguf
    spawn flags by hand. A hand-written list only defends fields someone
    remembered to add to it; `arg` is declared on the field itself, so this
    catches a newly-added argv field that gets misclassified as `per_request`
    -- the exact silent failure (change reported as live, process keeps the
    old value) that the whole classification exists to prevent.

    The one legitimate exception is a field the process can genuinely vary per
    request despite having a flag spelling; there are none today, and adding
    one should be a deliberate edit here rather than a quiet reclassification.
    """
    cls = PROVIDER_CONFIG_CLASSES[provider]
    reload_fields = reload_required_fields(cls)
    offenders = {}
    for name, f in cls.model_fields.items():
        extra = getattr(f, "json_schema_extra", None)
        arg = extra.get("arg") if isinstance(extra, dict) else None
        if arg and name not in reload_fields:
            offenders[name] = (arg, field_effect(f))
    assert not offenders, (
        f"{provider}: fields carry a spawn-argv `arg` but are not "
        f"reload-required: {offenders}. A flag baked into the process at "
        f"spawn cannot take effect without a respawn."
    )


# The gguf import allowlist widened from a hand-written tuple to a derived set
# (configurable_fields; "every field but model_path, each validating with its
# default" is test_config_effects.py::test_configurable_fields_exclude_identity_only).
# "Derived" also means nobody chose the contents, so these rows pin what it
# must still admit, as a visible diff:
# - previously_allowed: the old hand-written tuple; losing one means a field
#   that used to import silently stopped importing.
# - once_dropped: the five fields the derivation FIXED; each used to be
#   silently dropped on import. (`default_sampler` was a sixth until v2.0.30
#   removed the named-sampler system entirely.)
# Deliberately NOT an exact frozen list of the difference. The original version
# pinned one, and adding four legitimate fields (spec_draft_p_min, n_cpu_moe,
# cpu_moe, override_tensor) broke it with no safety gained -- a hand-maintained
# list that must be edited whenever a field is added is the very pattern this
# change removed.
_ALLOWLIST_ROWS = [
    pytest.param(
        ("mmproj_path", "draft_model_path", "spec_type", "spec_draft_n_max",
         "ctx_size", "n_gpu_layers", "server_binary", "host", "port",
         "startup_timeout_s", "extra_args", "max_tokens", "supports_thinking",
         "modalities"),
        id="previously_allowed",
    ),
    pytest.param(
        ("cache_ram_mb", "enable_thinking", "load_mode",
         "n_gpu_layers_draft", "sleep_idle_seconds"),
        id="once_dropped",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("fields", _ALLOWLIST_ROWS)
def test_gguf_import_allowlist_keeps_every_named_field(fields):
    now = configurable_fields(PROVIDER_CONFIG_CLASSES["gguf"])
    missing = sorted(set(fields) - now)
    assert not missing, (
        f"the derived gguf import allowlist DROPPED {missing} -- a field that "
        f"used to import (or was fixed to) silently stopped importing"
    )
