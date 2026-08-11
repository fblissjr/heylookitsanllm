"""Every provider-config field must declare WHEN a change takes effect.

This is the guard that makes the `effect` metadata self-maintaining. Three
real drifts motivated it, all of which had been silently wrong in production:

1. `RELOAD_REQUIRED_FIELDS` was a single MLX-shaped frozenset naming no gguf
   load-time field at all, so changing `ctx_size` on a loaded gguf model
   reported "no reload required" and the server kept serving the old argv.
2. It still listed `supports_thinking`, removed from MLXModelConfig in v1.46.0.
3. The gguf import allowlist had drifted from the config class and silently
   dropped five fields (`n_gpu_layers_draft`, `cache_ram_mb`, `load_mode`,
   `sleep_idle_seconds`, `enable_thinking`).

All three shared a cause: the fact lived somewhere other than the field it
described. These tests fail if a new field is added without classifying it,
which is the only thing that keeps that from happening again.
"""
import pytest

from heylook_llm.config import (
    EFFECT_APPLIES_LIVE,
    EFFECT_CLASSES,
    EFFECT_DESCRIPTIVE,
    EFFECT_IDENTITY,
    EFFECT_LOAD_TIME_ONLY,
    EFFECT_PER_REQUEST,
    EFFECT_REQUIRES_RELOAD,
    PROVIDER_CONFIG_CLASSES,
    _validate_effect_declarations,
    configurable_fields,
    field_effect,
    invalid_effects,
    fields_by_effect,
    reload_required_fields,
)

# Iterating the registry means a NEW PROVIDER is covered automatically -- the
# registry is already the declared single source of truth for which exist.
PROVIDERS = sorted(PROVIDER_CONFIG_CLASSES)


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_field_declares_an_effect(provider):
    cls = PROVIDER_CONFIG_CLASSES[provider]
    undeclared = sorted(
        name for name, f in cls.model_fields.items() if field_effect(f) is None
    )
    assert not undeclared, (
        f"{cls.__name__} fields with no `effect`: {undeclared}. Add "
        f'json_schema_extra={{"effect": ...}} to the field declaration -- one '
        f"of {sorted(EFFECT_CLASSES)}."
    )


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_declared_effects_are_valid(provider):
    cls = PROVIDER_CONFIG_CLASSES[provider]
    assert not invalid_effects(cls), (
        f"{cls.__name__} has unknown effect classes: {invalid_effects(cls)}"
    )


@pytest.mark.unit
def test_a_misspelt_effect_is_unclassified_not_a_new_category():
    """Regression: a typo must not silently mean "no reload required".

    An earlier version bucketed by the raw string, so `"requires-reload"`
    (hyphen) created its own bucket, left the unclassified set empty, passed
    every completeness check, and dropped the field out of the reload set --
    reintroducing the precise bug this metadata exists to prevent.
    """
    from pydantic import BaseModel, Field as PField

    class Typo(BaseModel):
        model_path: str = PField(json_schema_extra={"effect": EFFECT_IDENTITY})
        ctx_size: int = PField(
            default=0, json_schema_extra={"effect": "requires-reload"}
        )

    by = fields_by_effect(Typo)
    assert "ctx_size" in by[None], "a misspelt effect must land as unclassified"
    assert "requires-reload" not in by, "a typo must not invent a bucket"
    assert invalid_effects(Typo) == {"ctx_size": "requires-reload"}


@pytest.mark.unit
def test_bad_declarations_fail_at_import_not_just_under_test():
    """The guard must fire outside a test run, or a typo ships."""
    from pydantic import BaseModel, Field as PField
    import heylook_llm.config as cfg

    class Unclassified(BaseModel):
        model_path: str = PField(json_schema_extra={"effect": EFFECT_IDENTITY})
        mystery: int = PField(default=0)  # no effect at all

    original = dict(cfg.PROVIDER_CONFIG_CLASSES)
    cfg.PROVIDER_CONFIG_CLASSES["_probe"] = Unclassified
    try:
        with pytest.raises(RuntimeError, match="mystery"):
            _validate_effect_declarations()
    finally:
        cfg.PROVIDER_CONFIG_CLASSES.clear()
        cfg.PROVIDER_CONFIG_CLASSES.update(original)
    # And the real registry is clean.
    _validate_effect_declarations()


@pytest.mark.unit
def test_applies_live_is_distinct_from_load_time_only():
    """"Cannot be changed" and "changes freely" need opposite affordances.

    max_queue_depth is process-wide (first provider created wins), so not even
    reloading this model changes it. unload_after_idle_seconds is re-read by
    the router on each idle sweep. Collapsing both into one bucket would leave
    the UI unable to tell a disabled control from a freely editable one.
    """
    mlx = PROVIDER_CONFIG_CLASSES["mlx"]
    by = fields_by_effect(mlx)
    assert "max_queue_depth" in by[EFFECT_LOAD_TIME_ONLY]
    assert "unload_after_idle_seconds" in by[EFFECT_APPLIES_LIVE]
    # Neither is reload-required: one cannot be fixed by a reload, the other
    # needs no reload at all.
    assert not (by[EFFECT_APPLIES_LIVE] & reload_required_fields(mlx))


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_effects_partition_the_fields_exactly(provider):
    """No field unclassified, none counted twice, nothing invented."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    by = fields_by_effect(cls)
    buckets = [names for effect, names in by.items() if effect is not None]
    union = frozenset().union(*buckets) if buckets else frozenset()
    total = sum(len(b) for b in buckets)

    assert union == frozenset(cls.model_fields), (
        f"{cls.__name__}: classified set != model_fields "
        f"(missing {sorted(frozenset(cls.model_fields) - union)}, "
        f"extra {sorted(union - frozenset(cls.model_fields))})"
    )
    assert total == len(union), f"{cls.__name__}: a field is in two buckets"


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_exactly_one_identity_field(provider):
    """`model_path` is what makes an entry the model it is."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    identity = fields_by_effect(cls).get(EFFECT_IDENTITY, frozenset())
    assert identity == {"model_path"}, (
        f"{cls.__name__}: expected model_path as the sole identity field, "
        f"got {sorted(identity)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_configurable_fields_exclude_identity_only(provider):
    cls = PROVIDER_CONFIG_CLASSES[provider]
    configurable = configurable_fields(cls)
    assert "model_path" not in configurable
    assert configurable == frozenset(cls.model_fields) - {"model_path"}


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_reload_set_covers_identity_and_reload(provider):
    """Swapping the weights is the strongest form of "needs a reload"."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    by = fields_by_effect(cls)
    expected = by.get(EFFECT_REQUIRES_RELOAD, frozenset()) | by.get(
        EFFECT_IDENTITY, frozenset()
    )
    assert reload_required_fields(cls) == expected
    # A per_request field must never demand a reload -- that is the whole
    # distinction, and getting it wrong makes the UI confirm needlessly.
    assert not (reload_required_fields(cls) & by.get(EFFECT_PER_REQUEST, frozenset()))


@pytest.mark.unit
def test_gguf_load_time_fields_are_reload_required():
    """The regression that motivated all of this.

    Every one of these is baked into the llama-server argv at spawn, so
    changing it on a loaded model cannot take effect without a respawn. The
    old hand-written set contained NONE of them.
    """
    reload_fields = reload_required_fields(PROVIDER_CONFIG_CLASSES["gguf"])
    for field in (
        "ctx_size", "n_gpu_layers", "n_gpu_layers_draft", "spec_type",
        "spec_draft_n_max", "mmproj_path", "draft_model_path", "extra_args",
        "cache_ram_mb", "load_mode", "sleep_idle_seconds",
    ):
        assert field in reload_fields, (
            f"gguf `{field}` is a spawn-time argv flag but is not marked "
            f"requires_reload -- changing it would report no reload needed "
            f"while the process keeps the old value"
        )


@pytest.mark.unit
def test_gguf_import_allowlist_covers_every_settable_field():
    """The import path must not silently drop fields the config accepts."""
    settable = configurable_fields(PROVIDER_CONFIG_CLASSES["gguf"])
    for field in (
        "n_gpu_layers_draft", "cache_ram_mb", "load_mode",
        "sleep_idle_seconds", "enable_thinking",
    ):
        assert field in settable, f"gguf import would drop `{field}`"


@pytest.mark.unit
def test_descriptive_fields_are_not_reload_required():
    """`modalities`/`supports_thinking` on gguf describe what we serve it as;
    nothing about the running process depends on them.

    Note the deliberate asymmetry: MLX marks `modalities` requires_reload
    because there it selects the engine (mlx-vlm vs mlx-lm). Same field name,
    different effect per provider -- which a single shared frozenset could
    never express.
    """
    gguf = PROVIDER_CONFIG_CLASSES["gguf"]
    descriptive = fields_by_effect(gguf).get(EFFECT_DESCRIPTIVE, frozenset())
    assert descriptive == {"modalities", "supports_thinking"}
    assert not (descriptive & reload_required_fields(gguf))

    mlx = PROVIDER_CONFIG_CLASSES["mlx"]
    assert "modalities" in reload_required_fields(mlx)


@pytest.mark.unit
def test_gguf_arg_spellings_are_unique_and_plausible():
    """`arg` carries the llama-server spelling for the argv builder."""
    cls = PROVIDER_CONFIG_CLASSES["gguf"]
    args = {}
    for name, field in cls.model_fields.items():
        extra = getattr(field, "json_schema_extra", None) or {}
        arg = extra.get("arg")
        if arg is None:
            continue
        assert str(arg).startswith("-"), f"{name}: `arg` {arg!r} is not a flag"
        assert arg not in args, f"{name} and {args[arg]} both claim {arg}"
        args[arg] = name
    # Only load_time_only/descriptive/per_request fields may lack an `arg`;
    # anything reload-required is by definition part of the spawn argv.
    reload_fields = fields_by_effect(cls).get(EFFECT_REQUIRES_RELOAD, frozenset())
    argless = sorted(
        n for n in reload_fields
        if not (getattr(cls.model_fields[n], "json_schema_extra", None) or {}).get("arg")
    )
    # Legitimate reasons a reload-required field carries no `arg`: extra_args IS
    # argv but has no single flag; host/port/server_binary/startup_timeout_s are
    # spawn PLUMBING (binary is argv[0], port is the resolved free port rather
    # than the field, the timeout never reaches llama). They are
    # requires_reload because a reload genuinely changes them -- the property
    # that matters -- not because they map to a flag.
    assert argless == ["extra_args", "host", "port", "server_binary",
                       "startup_timeout_s"], (
        f"gguf reload-required fields with no `arg` spelling: {argless}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("provider", PROVIDERS)
def test_load_time_only_fields_are_not_offered_as_reload(provider):
    """Infrastructure is not a tuning control; it must not prompt a reload."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    load_time = fields_by_effect(cls).get(EFFECT_LOAD_TIME_ONLY, frozenset())
    assert not (load_time & reload_required_fields(cls))
    live = fields_by_effect(cls).get(EFFECT_APPLIES_LIVE, frozenset())
    assert not (live & reload_required_fields(cls))
