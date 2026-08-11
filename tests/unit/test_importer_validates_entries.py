"""A bad `--override` must fail the IMPORT, not the next server start.

`heylookllm import --override <key>=<value>` is free-form by design: it has to
reach provider config fields the importer does not enumerate. But the importer
wrote models.toml directly and validated nothing, so one mistyped key
(`ctx_sze=8192`) produced a successful-looking import and a server that then
refused to start -- the failure surfacing at config load, far from the command
that caused it, with no indication which of N imported entries was at fault.

Same lesson as the derived reload set and import allowlist: the config class
is the only thing that knows what is valid, so ask it.
"""
import pytest

from heylook_llm.model_importer import ModelImporter


def _gguf_entry(**config):
    return {
        "id": "m",
        "provider": "gguf",
        "enabled": True,
        "config": {"model_path": "/tmp/fake.gguf", **config},
    }


@pytest.mark.unit
def test_a_mistyped_override_key_is_rejected_at_import():
    """The typo case: `ctx_sze` instead of `ctx_size`."""
    importer = ModelImporter()
    with pytest.raises(ValueError) as excinfo:
        importer._validate([_gguf_entry(ctx_sze=8192)])
    message = str(excinfo.value)
    assert "ctx_sze" in message, "the error must name the offending key"
    assert "'m'" in message, "the error must name which entry is at fault"
    # And it must tell you what WOULD have been valid, from the schema.
    assert "ctx_size" in message, "the error should list settable keys"


@pytest.mark.unit
def test_a_wrongly_typed_override_value_is_rejected():
    """`--override` coerces to int/float/bool by shape, so a value can arrive
    with a type the field does not accept."""
    importer = ModelImporter()
    with pytest.raises(ValueError, match="'m'"):
        importer._validate([_gguf_entry(ctx_size="not-a-number")])


@pytest.mark.unit
def test_an_out_of_range_value_is_rejected():
    """Constraints count too: ctx_size carries ge=512."""
    importer = ModelImporter()
    with pytest.raises(ValueError):
        importer._validate([_gguf_entry(ctx_size=1)])


@pytest.mark.unit
def test_valid_entries_pass_through_unchanged():
    """Validation must be a gate, not a transform -- the caller writes what it
    gets back, so any mutation here would silently alter models.toml."""
    importer = ModelImporter()
    entries = [_gguf_entry(ctx_size=8192)]
    assert importer._validate(entries) is entries


@pytest.mark.unit
def test_every_provider_can_round_trip_a_minimal_entry():
    """Guard against the validator rejecting things it should accept -- a gate
    that fails valid input is worse than no gate, because it blocks import."""
    importer = ModelImporter()
    for provider, config in (
        ("gguf", {"model_path": "/tmp/m.gguf"}),
        ("mlx", {"model_path": "/tmp/m"}),
        ("mlx_embedding", {"model_path": "/tmp/m"}),
    ):
        entry = {"id": f"m-{provider}", "provider": provider,
                 "enabled": True, "config": config}
        assert importer._validate([entry]) == [entry]
