"""`GET /v1/admin/model-options` — every settable default, per provider.

This route is the first consumer that can distinguish all the effect classes.
Both in-process consumers collapse them: `reload_required_fields` flattens six
classes to a binary, and the import allowlist to "not identity". So until
something reads `effect` per field, a misclassification is invisible -- the
completeness test checks that a class is DECLARED, not that it is right.

It is derived from the provider config classes, so a new field appears here
without anyone editing the route. These tests read the route's body and pin
the properties a UI depends on, not a snapshot of the fields, which would just
be another hand-maintained list to drift.
"""
import pytest

from heylook_llm.config import EFFECT_CLASSES, EFFECT_LOAD_TIME_ONLY, PROVIDER_CONFIG_CLASSES

PROVIDERS = sorted(PROVIDER_CONFIG_CLASSES)
RENDERABLE = {"integer", "number", "string", "boolean", "array", "object"}


def _fields(client, provider):
    res = client.get("/v1/admin/model-options")
    assert res.status_code == 200
    return {f["name"]: f for f in res.json()["providers"][provider]["fields"]}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_offered_field_is_renderable_and_classified(client, provider):
    """One pass over every field the route offers, per provider:

    - `model_path` is never offered: it is what makes an entry the model it
      is; editing it in a 'defaults' form would silently repoint the entry at
      other weights.
    - a valid effect: effect is what the UI keys its affordance off. A null
      or unknown effect leaves the client guessing, and guessing wrong means
      telling someone a change took effect when the process kept the old
      value. (Every gguf field with an argv `arg` is requires_reload by
      test_config_effects_adversarial.py::test_any_field_with_an_arg_spelling_requires_a_reload.)
    - a renderable type, never pydantic's `anyOf [T, null]` wrapping of
      Optional[T]: a client should not have to understand that encoding to
      render a number input, and without a type it cannot pick a control.
    - `load_time_only` explains itself: it renders DISABLED, and a disabled
      control with no reason is just a dead input. It is also the one thing
      the class genuinely cannot imply: max_queue_depth is fixed because it is
      process-wide, `port` for a completely unrelated reason.
    """
    fields = _fields(client, provider)
    assert fields
    assert "model_path" not in fields
    for name, f in fields.items():
        assert f["effect"] in EFFECT_CLASSES, f"{provider}.{name} has effect={f['effect']!r}"
        assert "anyOf" not in f, f"{provider}.{name} leaked an anyOf"
        assert f["type"] in RENDERABLE, f"{provider}.{name} has type={f['type']!r}"
        if f["effect"] == EFFECT_LOAD_TIME_ONLY:
            assert f.get("reason"), f"{provider}.{name} is load_time_only with no `reason`"


def test_gguf_constraints_and_bare_flags_reach_the_client(client):
    """Bounds and enums must reach the client, or it renders an input that
    happily submits a value the server will reject. `-cmoe` takes no value:
    a client that renders it as a text field would produce an argv
    llama-server refuses to start on."""
    gguf = _fields(client, "gguf")
    assert gguf["ctx_size"]["minimum"] == 512
    assert gguf["spec_draft_p_min"]["minimum"] == 0.0
    assert gguf["spec_draft_p_min"]["maximum"] == 1.0
    assert gguf["n_cpu_moe"]["minimum"] == 0
    assert set(gguf["load_mode"]["enum"]) == {
        "none", "mmap", "mlock", "mmap+mlock", "dio"
    }
    assert gguf["cpu_moe"]["shape"] == "flag"
    assert gguf["cpu_moe"]["type"] == "boolean"
