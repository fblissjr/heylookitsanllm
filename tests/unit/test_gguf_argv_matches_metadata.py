"""The `arg` spelling on a config field must be the flag actually emitted.

`GGUFModelConfig` fields carry `json_schema_extra={"arg": "--ctx-size"}`, and
`LlamaServerProvider._build_args` writes the real argv. Nothing tied the two
together, so they could drift silently -- and drift here is invisible: the
metadata is what a UI and any future table-driven emitter read, while the
builder is what the process actually gets. They would disagree about what the
model is configured with, and nothing would say so.

This is the third leg of the same drift the `effect` classification closed for
the reload set and the import allowlist.
"""
import pytest

from heylook_llm.config import PROVIDER_CONFIG_CLASSES
from heylook_llm.providers.llama_server_provider import LlamaServerProvider

# A value per field that is (a) valid for the type and (b) TRUTHY, so the
# builder's `if cfg.get(...)` guards all fire.
SAMPLE_VALUES = {
    "ctx_size": 4096,
    "mmproj_path": "/tmp/mmproj.gguf",
    "draft_model_path": "/tmp/draft.gguf",
    "spec_type": "draft-mtp",
    "spec_draft_n_max": 2,
    "n_gpu_layers": 42,
    "n_gpu_layers_draft": 7,
    "cache_ram_mb": 512,
    "sleep_idle_seconds": 30,
    "load_mode": "mlock",
}


def _declared_args() -> dict[str, str]:
    """{field name -> declared flag} for every gguf field carrying one."""
    cls = PROVIDER_CONFIG_CLASSES["gguf"]
    out = {}
    for name, field in cls.model_fields.items():
        extra = getattr(field, "json_schema_extra", None) or {}
        if extra.get("arg"):
            out[name] = str(extra["arg"])
    return out


def _build_argv(config: dict) -> list[str]:
    """Build argv without constructing a real provider (no subprocess, no I/O)."""
    provider = LlamaServerProvider.__new__(LlamaServerProvider)
    provider.config = config  # type: ignore[attr-defined]
    from pathlib import Path

    return provider._build_args(Path("/tmp/llama-server"), 8080)


@pytest.mark.unit
def test_every_sampled_field_has_a_sample_value():
    """Guard the guard: a new `arg` field must get a sample here, or the
    coverage test below would silently stop covering it."""
    missing = sorted(set(_declared_args()) - set(SAMPLE_VALUES) - {"model_path"})
    assert not missing, (
        f"add a SAMPLE_VALUES entry for {missing} so the argv coverage test "
        f"actually exercises the flag"
    )


@pytest.mark.unit
@pytest.mark.parametrize("field,flag", sorted(_declared_args().items()))
def test_declared_flag_is_the_one_actually_emitted(field, flag):
    """Set one field; the flag it declares must appear in the argv.

    Catches both halves of a drift: the builder omitting a configured field,
    and the metadata naming a different spelling than the builder writes (an
    alias counts as different -- a derived emitter would produce a different
    command line).
    """
    if field == "model_path":
        pytest.skip("identity; emitted as -m and never optional")
    config = {"model_path": "/tmp/model.gguf", field: SAMPLE_VALUES[field]}
    argv = _build_argv(config)
    assert flag in argv, (
        f"`{field}` declares arg={flag!r} but _build_args emitted "
        f"{[a for a in argv if a.startswith('-')]!r}. Either the builder "
        f"does not emit it, or the two disagree on the spelling."
    )


@pytest.mark.unit
def test_the_value_lands_next_to_the_flag():
    """A flag present with the wrong value is as broken as one missing."""
    argv = _build_argv({"model_path": "/tmp/m.gguf", "ctx_size": 4096})
    assert argv[argv.index("--ctx-size") + 1] == "4096"


@pytest.mark.unit
def test_zero_is_emitted_not_swallowed():
    """0 is MEANINGFUL for -ngld (keep the drafter off the GPU) and -cram
    (disable the prompt cache), so a truthiness check would drop it. The
    builder uses `is not None` for exactly this; pin it."""
    argv = _build_argv(
        {"model_path": "/tmp/m.gguf", "n_gpu_layers_draft": 0, "cache_ram_mb": 0}
    )
    assert argv[argv.index("-ngld") + 1] == "0"
    assert argv[argv.index("-cram") + 1] == "0"


@pytest.mark.unit
def test_extra_args_are_appended_last():
    """Raw passthrough must come after the managed flags, so a deliberate
    override wins (llama.cpp takes the last occurrence)."""
    argv = _build_argv(
        {"model_path": "/tmp/m.gguf", "ctx_size": 4096,
         "extra_args": ["--ctx-size", "9999"]}
    )
    assert argv[-2:] == ["--ctx-size", "9999"]
