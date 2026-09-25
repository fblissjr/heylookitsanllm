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
    field_effect,
    fields_by_effect,
)

PROVIDERS = sorted(PROVIDER_CONFIG_CLASSES)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_offered_field_is_settable_typed_and_classified(provider):
    """One pass over every offered field, per provider:

    - offered == configurable_fields: a field the config accepts but the
      schema omits is a default the user cannot set through the UI --
      invisible, and the reason nobody would know the knob exists.
    - `model_path` is never offered: it is what makes an entry the model it
      is; editing it in a 'defaults' form would silently repoint the entry at
      other weights.
    - a valid effect, and the one the field declares: effect is what the UI
      keys its affordance off. A null, unknown or mis-relayed effect leaves
      the client guessing, and guessing wrong means telling someone a change
      took effect when the process kept the old value. (Every gguf field with
      an argv `arg` is requires_reload by
      test_config_effects_adversarial.py::test_any_field_with_an_arg_spelling_requires_a_reload.)
    - a renderable type: without one the client cannot pick a control.
    - `load_time_only` explains itself: it renders DISABLED, and a disabled
      control with no reason is just a dead input. It is also the one thing
      the class genuinely cannot imply: max_queue_depth is fixed because it is
      process-wide, `port` for a completely unrelated reason.
    """
    cls = PROVIDER_CONFIG_CLASSES[provider]
    options = _field_options(cls)
    offered = {f["name"] for f in options}
    assert offered == configurable_fields(cls)
    assert "model_path" not in offered

    frozen = fields_by_effect(cls).get(EFFECT_LOAD_TIME_ONLY, frozenset())
    for f in options:
        assert f["effect"] in EFFECT_CLASSES, (
            f"{provider}.{f['name']} has effect={f['effect']!r}"
        )
        assert f["effect"] == field_effect(cls.model_fields[f["name"]]), (
            f"{provider}.{f['name']}: the route relays a different effect than declared"
        )
        assert f["type"] in {"integer", "number", "string", "boolean", "array", "object"}, (
            f"{provider}.{f['name']} has type={f['type']!r}"
        )
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
