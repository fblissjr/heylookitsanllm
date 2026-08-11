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
    EFFECT_CLASSES,
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


@pytest.mark.unit
def test_unknown_provider_falls_back_to_the_conservative_union():
    """Over-reporting costs a needless prompt; under-reporting serves stale.

    A `models.toml` entry with a missing or misspelt provider must not come
    back with an EMPTY reload set, which would report every edit as free.
    """
    union = ms.RELOAD_REQUIRED_FIELDS
    for bogus in (None, "", "ggf", "MLX", "does-not-exist"):
        assert ms.reload_required_for(bogus) == union, (
            f"provider={bogus!r} should fall back to the union, not "
            f"{sorted(ms.reload_required_for(bogus))}"
        )
    # And the union genuinely covers every provider's own answer.
    for provider in PROVIDERS:
        assert ms.reload_required_for(provider) <= union


@pytest.mark.unit
def test_provider_spelling_matches_registry_keys_exactly():
    """`reload_required_for` keys off the raw string from the toml entry.

    Case or separator drift silently downgrades to the union, which is safe,
    but a registry key that is not a plain lowercase identifier would make
    that silent downgrade the NORMAL path rather than the exception.
    """
    for key in PROVIDER_CONFIG_CLASSES:
        assert key == key.lower().strip(), f"registry key {key!r} is not normalised"
        assert ms.reload_required_for(key), f"{key!r} resolved to an empty set"


@pytest.mark.unit
def test_json_schema_extra_as_a_callable_is_unclassified_not_silently_ok():
    """Pydantic allows a CALLABLE json_schema_extra; `_extra` cannot read it.

    The safe outcome is "unclassified" (loud, caught by the import guard), not
    a field that quietly claims no effect. Pinning it because the failure would
    otherwise be invisible: the declaration LOOKS annotated at a glance.
    """
    def _mutate(schema):  # pydantic calls this instead of merging a dict
        schema["effect"] = "requires_reload"

    class CallableExtra(BaseModel):
        model_path: str = PField(json_schema_extra={"effect": EFFECT_IDENTITY})
        ctx_size: int = PField(default=0, json_schema_extra=_mutate)

    assert field_effect(CallableExtra.model_fields["ctx_size"]) is None
    assert "ctx_size" in fields_by_effect(CallableExtra)[None]
    # It is *missing*, not *bogus* -- so the guard reports it as undeclared.
    assert invalid_effects(CallableExtra) == {}


@pytest.mark.unit
def test_an_unclassified_field_never_counts_as_reload_required():
    """Whatever else happens, "we don't know" must not become "no reload".

    A class the import guard never saw (registered after import) can still
    reach the derived helpers. This pins the one property that matters: an
    unclassified field is absent from the reload set, so it can only ever be
    caught by the guard -- never silently blessed as safe to change live.
    """
    class Unguarded(BaseModel):
        model_path: str = PField(json_schema_extra={"effect": EFFECT_IDENTITY})
        ctx_size: int = PField(default=0)

    by = fields_by_effect(Unguarded)
    assert by[None] == {"ctx_size"}
    assert "ctx_size" not in reload_required_fields(Unguarded)
    # Which is exactly why the import guard must be the thing that stops it.
    assert invalid_effects(Unguarded) == {}


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_effect_values_are_the_module_constants_not_loose_strings(provider):
    """Every declared effect is one of the exported constants.

    Belt-and-braces on the typo class: `fields_by_effect` now buckets an
    unknown value under None, but this names the offender directly.
    """
    cls = PROVIDER_CONFIG_CLASSES[provider]
    for name, f in cls.model_fields.items():
        effect = field_effect(f)
        assert effect in EFFECT_CLASSES, (
            f"{provider}.{name}: effect={effect!r} not in {sorted(EFFECT_CLASSES)}"
        )


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


@pytest.mark.unit
def test_gguf_import_allowlist_rejects_identity_and_accepts_nothing_unknown():
    """The allowlist widened from a hand-written tuple to a derived set.

    Wider is the point (five fields were being dropped), but it must not have
    become "anything at all": every key it yields has to be a real field the
    config class will validate, and `model_path` must stay excluded because
    the import path sets it separately.
    """
    cls = PROVIDER_CONFIG_CLASSES["gguf"]
    settable = configurable_fields(cls)
    assert "model_path" not in settable
    assert settable <= set(cls.model_fields)
    # Every settable key must actually be accepted by the model (extra="forbid"
    # would reject an invented one).
    for name in settable:
        cls(model_path="/tmp/x.gguf", **{name: cls.model_fields[name].default})


@pytest.mark.unit
def test_allowlist_widening_added_exactly_the_dropped_fields():
    """The derived allowlist must be a superset, and a KNOWN one.

    Deriving it fixed five silently-dropped fields, but "derived" also means
    nobody chose the contents. This pins what the change actually admits, so a
    future field lands here as a visible diff rather than appearing in written
    config unnoticed. `default_sampler` is included deliberately: it is now
    settable via the config block, where previously only the explicit importer
    argument could set it (and that argument still wins, being stamped after).
    """
    previously_allowed = {
        "mmproj_path", "draft_model_path", "spec_type", "spec_draft_n_max",
        "ctx_size", "n_gpu_layers", "server_binary", "host", "port",
        "startup_timeout_s", "extra_args", "max_tokens", "supports_thinking",
        "modalities",
    }
    now = configurable_fields(PROVIDER_CONFIG_CLASSES["gguf"])
    assert previously_allowed <= now, (
        f"the derived allowlist DROPPED {sorted(previously_allowed - now)} -- "
        f"a field that used to import silently stopped importing"
    )
    # The five fields the derivation FIXED, plus default_sampler. These are the
    # regression -- each one used to be silently dropped on import.
    for field in ("cache_ram_mb", "enable_thinking", "load_mode",
                  "n_gpu_layers_draft", "sleep_idle_seconds", "default_sampler"):
        assert field in now, f"`{field}` was dropped on import before; it must stay settable"

    # Deliberately NOT an exact frozen list of the difference. The original
    # version pinned one, and adding four legitimate fields (spec_draft_p_min,
    # n_cpu_moe, cpu_moe, override_tensor) broke it with no safety gained --
    # a hand-maintained list of field names that must be edited whenever a
    # field is added is the very pattern this whole change removed. The real
    # invariants are: nothing is dropped (above), and the allowlist is exactly
    # the non-identity fields of the class (below), which stays true forever.
    cls = PROVIDER_CONFIG_CLASSES["gguf"]
    assert now == frozenset(cls.model_fields) - {"model_path"}
