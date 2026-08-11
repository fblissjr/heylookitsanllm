"""`GET /v1/admin/model-options` — every settable default, per provider.

This route is the first consumer that can distinguish all the effect classes.
Both in-process consumers collapse them: `reload_required_fields` flattens six
classes to a binary, and the import allowlist to "not identity". So until
something reads `effect` per field, a misclassification is invisible -- the
completeness test checks that a class is DECLARED, not that it is right.

It is derived from the provider config classes, so a new field appears here
without anyone editing the route. These tests pin the properties a UI depends
on, not a snapshot of the fields, which would just be another hand-maintained
list to drift.
"""
import pytest

from heylook_llm.admin_api import _field_options
from heylook_llm.config import (
    EFFECT_CLASSES,
    EFFECT_LOAD_TIME_ONLY,
    PROVIDER_CONFIG_CLASSES,
    configurable_fields,
    fields_by_effect,
)

PROVIDERS = sorted(PROVIDER_CONFIG_CLASSES)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_settable_field_is_offered(provider):
    """A field the config accepts but the schema omits is a default the user
    cannot set through the UI -- invisible, and the reason nobody would know
    the knob exists."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    offered = {f["name"] for f in _field_options(cls)}
    assert offered == configurable_fields(cls)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_identity_is_never_offered(provider):
    """`model_path` is what makes an entry the model it is; editing it in a
    'defaults' form would silently repoint the entry at other weights."""
    assert "model_path" not in {
        f["name"] for f in _field_options(PROVIDER_CONFIG_CLASSES[provider])
    }


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_field_carries_a_valid_effect(provider):
    """Effect is what the UI keys its affordance off. A field arriving with a
    null or unknown effect leaves the client guessing, and guessing wrong here
    means telling someone a change took effect when the process kept the old
    value."""
    for f in _field_options(PROVIDER_CONFIG_CLASSES[provider]):
        assert f["effect"] in EFFECT_CLASSES, (
            f"{provider}.{f['name']} has effect={f['effect']!r}"
        )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_field_has_a_renderable_type(provider):
    """Without a type the client cannot pick a control."""
    for f in _field_options(PROVIDER_CONFIG_CLASSES[provider]):
        assert f["type"] in {"integer", "number", "string", "boolean", "array", "object"}, (
            f"{provider}.{f['name']} has type={f['type']!r}"
        )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_load_time_only_fields_explain_themselves(provider):
    """`load_time_only` renders DISABLED, and a disabled control with no reason
    is just a dead input. It is also the one thing the class genuinely cannot
    imply: max_queue_depth is fixed because it is process-wide, `port` for a
    completely unrelated reason."""
    cls = PROVIDER_CONFIG_CLASSES[provider]
    frozen = fields_by_effect(cls).get(EFFECT_LOAD_TIME_ONLY, frozenset())
    for f in _field_options(cls):
        if f["name"] in frozen:
            assert f.get("reason"), (
                f"{provider}.{f['name']} is load_time_only with no `reason`"
            )


def test_constraints_survive_into_the_schema():
    """Bounds and enums must reach the client, or it renders an input that
    happily submits a value the server will reject."""
    gguf = {f["name"]: f for f in _field_options(PROVIDER_CONFIG_CLASSES["gguf"])}
    assert gguf["ctx_size"]["minimum"] == 512
    assert gguf["spec_draft_p_min"]["minimum"] == 0.0
    assert gguf["spec_draft_p_min"]["maximum"] == 1.0
    assert gguf["n_cpu_moe"]["minimum"] == 0
    assert set(gguf["load_mode"]["enum"]) == {
        "none", "mmap", "mlock", "mmap+mlock", "dio"
    }


def test_optional_types_are_flattened_not_anyof():
    """Pydantic wraps Optional[T] as anyOf [T, null]. A client should not have
    to understand that encoding to render a number input."""
    for provider in PROVIDERS:
        for f in _field_options(PROVIDER_CONFIG_CLASSES[provider]):
            assert "anyOf" not in f, f"{provider}.{f['name']} leaked an anyOf"


def test_bare_flags_are_marked_as_such():
    """`-cmoe` takes no value. A client that renders it as a text field would
    produce an argv llama-server refuses to start on."""
    gguf = {f["name"]: f for f in _field_options(PROVIDER_CONFIG_CLASSES["gguf"])}
    assert gguf["cpu_moe"]["shape"] == "flag"
    assert gguf["cpu_moe"]["type"] == "boolean"


def test_the_newly_exposed_perf_levers_are_present():
    """spec_draft_p_min is the difference between +1% and +15.7% on gemma-4 12B
    and was unreachable before; the offload set is the KV-headroom escape."""
    gguf = {f["name"]: f for f in _field_options(PROVIDER_CONFIG_CLASSES["gguf"])}
    for name in ("spec_draft_p_min", "n_cpu_moe", "cpu_moe", "override_tensor"):
        assert name in gguf
        assert gguf[name]["effect"] == "requires_reload"
