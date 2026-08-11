"""Clearing a model config field back to its default must work, and persist.

`None` is the way every optional load option spells "inherit the default"
(`ctx_size`, `cache_ram_mb`, `load_mode`, `n_gpu_layers_draft`...), so a
reset-to-default control in the UI sends `{"config": {"ctx_size": null}}`.

That path was broken end to end, and every layer looked fine on its own:

  1. `model_dump(exclude_none=True)` does NOT strip nested nulls -- it applies
     to the request model's own fields, not values inside a plain `Dict` -- so
     the null reaches the service.
  2. `ModelConfig(**model)` validates it happily: `Optional[int] = None` is
     legal to pydantic.
  3. `tomli_w` then raises `TypeError: Object of type 'NoneType' is not TOML
     serializable`, because TOML has no null literal.
  4. The route caught only `ValueError`, so it surfaced as a 500.

The importer already knew this ("Skip None -- TOML has no null literal"); the
update path never learned it. The fix is to treat an explicit null as "remove
the key", since absent IS how the default is spelled on disk.
"""
import pytest

from heylook_llm.model_service import ModelService


@pytest.fixture
def service(tmp_path):
    """A ModelService over a throwaway models.toml with one gguf entry."""
    toml = tmp_path / "models.toml"
    toml.write_text(
        'default_model = "m"\n'
        "max_loaded_models = 1\n\n"
        "[[models]]\n"
        'id = "m"\n'
        'provider = "gguf"\n'
        "enabled = true\n\n"
        "[models.config]\n"
        'model_path = "/tmp/fake.gguf"\n'
        "ctx_size = 8192\n"
    )
    return ModelService(str(toml))


def test_clearing_a_field_removes_it_and_persists(service):
    """The whole chain: null -> key removed -> file rewritten -> still gone."""
    _, reload_fields = service.update_config("m", {"config": {"ctx_size": None}})

    config = service.get_config("m")
    assert config is not None
    assert config.config.model_dump().get("ctx_size") is None, (
        "cleared field should read back unset"
    )

    # It must survive the TOML round-trip -- this is where it used to blow up.
    reread = ModelService(service.config_path).get_config("m")
    assert reread is not None
    assert reread.config.model_dump().get("ctx_size") is None
    assert "ctx_size" not in reread.config.model_dump(exclude_unset=True), (
        "an unset field must be ABSENT on disk, not stored as a null"
    )

    # And clearing a spawn-time flag is a reload-required change, not a no-op.
    assert "ctx_size" in reload_fields


def test_clearing_an_already_unset_field_is_not_a_change(service):
    """Idempotence: no spurious 'needs a reload' for a field that was unset."""
    service.update_config("m", {"config": {"ctx_size": None}})
    _, reload_fields = service.update_config("m", {"config": {"ctx_size": None}})
    assert "ctx_size" not in reload_fields


def test_setting_a_value_still_works_and_is_reload_required(service):
    """The fix must not break the ordinary set-a-value path."""
    _, reload_fields = service.update_config("m", {"config": {"ctx_size": 4096}})
    config = service.get_config("m")
    assert config is not None
    assert config.config.model_dump().get("ctx_size") == 4096
    assert "ctx_size" in reload_fields


def test_no_none_ever_reaches_the_toml_writer(service):
    """Belt and braces: whatever else changes, nothing writes a null value.

    Asserted structurally rather than by catching TypeError, so this keeps
    holding if the writer is ever swapped for one that silently drops nulls
    instead of raising.
    """
    service.update_config(
        "m", {"config": {"ctx_size": None, "n_gpu_layers": 999}}
    )
    text = service.config_path.read_text() if hasattr(service.config_path, "read_text") \
        else open(service.config_path).read()
    assert "ctx_size" not in text
    assert "n_gpu_layers = 999" in text
