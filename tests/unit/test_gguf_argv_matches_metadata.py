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
import tempfile
from pathlib import Path

import pytest

from heylook_llm.config import PROVIDER_CONFIG_CLASSES
from heylook_llm.providers.llama_server_provider import LlamaServerProvider

# The model lives in a directory this module owns (two levels deep, so the
# folder above it is ours too): _build_args probes the model's folder for a
# chat_template.jinja sidecar, and under a shared /tmp a stray file there would
# add a --chat-template-file to every case below.
MODEL = str(Path(tempfile.mkdtemp(prefix="argv-drift-")) / "repo" / "m.gguf")

# A value per field that is (a) valid for the type and (b) TRUTHY, so the
# builder's `if cfg.get(...)` guards all fire.
SAMPLE_VALUES = {
    "ctx_size": 4096,
    "flash_attn": "off",
    "mmproj_path": "/tmp/mmproj.gguf",
    "chat_template_path": "/tmp/chat_template.jinja",
    "draft_model_path": "/tmp/draft.gguf",
    "spec_type": "draft-mtp",
    "spec_draft_n_max": 2,
    "n_gpu_layers": 42,
    "n_gpu_layers_draft": 7,
    "cache_ram_mb": 512,
    "sleep_idle_seconds": 30,
    "load_mode": "mlock",
    "spec_draft_p_min": 0.9,
    "spec_draft_n_min": 3,
    "n_cpu_moe": 8,
    "cpu_moe": True,  # bare flag -- no value follows it in argv
    "override_tensor": "exps=CPU",
    "n_cpu_moe_draft": 4,
    "cpu_moe_draft": True,  # bare flag, like cpu_moe
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "n_ubatch": 1024,
    "n_batch": 4096,
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


def _declared_rows():
    """One row per declared `arg` field: set that field alone, expect its
    declared flag followed by the sample value (a bare flag: the flag alone)."""
    rows = []
    for field, flag in sorted(_declared_args().items()):
        if field == "model_path":
            rows.append(pytest.param(
                {}, [], id=f"{field}-{flag}",
                marks=pytest.mark.skip(reason="identity; emitted as -m and never optional")))
            continue
        value = SAMPLE_VALUES[field]
        rows.append(pytest.param(
            {field: value}, [(flag, None if isinstance(value, bool) else str(value))],
            id=f"{field}-{flag}"))
    return rows


# Set a field; the flag it declares must appear in the argv, followed by the
# value (a flag present with the wrong value is as broken as one missing).
# Catches both halves of a drift: the builder omitting a configured field, and
# the metadata naming a different spelling than the builder writes (an alias
# counts as different -- a derived emitter would produce a different command
# line). The two hand-written rows set several memory/lifecycle knobs at once
# and a chat template override.
@pytest.mark.unit
@pytest.mark.parametrize("config, expected", _declared_rows() + [
    pytest.param({"n_gpu_layers_draft": 0, "cache_ram_mb": 32768,
                  "sleep_idle_seconds": 120, "load_mode": "mmap+mlock"},
                 [("-ngld", "0"), ("-cram", "32768"),
                  ("--sleep-idle-seconds", "120"), ("-lm", "mmap+mlock")],
                 id="memory_and_lifecycle_flags"),
    pytest.param({"chat_template_path": "/tmp/qwen38-official.jinja"},
                 [("--chat-template-file", "/tmp/qwen38-official.jinja")],
                 id="chat_template_override"),
])
def test_declared_flag_is_emitted_with_its_value(config, expected):
    argv = _build_argv({"model_path": MODEL, **config})
    pairs = list(zip(argv, argv[1:]))
    for flag, value in expected:
        assert flag in argv, (
            f"{config} declares arg={flag!r} but _build_args emitted "
            f"{[a for a in argv if a.startswith('-')]!r}. Either the builder "
            f"does not emit it, or the two disagree on the spelling."
        )
        if value is not None:
            assert (flag, value) in pairs, (
                f"{flag} is emitted but not followed by {value!r}")


# An explicit falsy value that is a real setting is emitted, never swallowed
# by a truthiness check (the builder uses `is not None`; these went through
# `if cfg.get(x)` once).
# - -ngld 0: the drafter entirely off the GPU.
# - -cram 0: disable the prompt cache; -cram -1: unlimited.
# - --spec-draft-p-min 0.0: keep every draft, AND llama.cpp's default, so
#   truthiness would make an explicit 0.0 indistinguishable from unset.
# - -ncmoe 0: offload no layers, an explicit choice distinct from unset.
@pytest.mark.unit
@pytest.mark.parametrize("field, flag, value", [
    ("n_gpu_layers_draft", "-ngld", 0),
    ("cache_ram_mb", "-cram", 0),
    ("cache_ram_mb", "-cram", -1),
    ("spec_draft_p_min", "--spec-draft-p-min", 0.0),
    ("n_cpu_moe", "-ncmoe", 0),
], ids=["ngld_zero", "cram_zero", "cram_minus_one", "p_min_zero", "ncmoe_zero"])
def test_explicit_falsy_value_is_emitted_not_swallowed(field, flag, value):
    argv = _build_argv({"model_path": MODEL, field: value})
    assert argv[argv.index(flag) + 1] == str(value)


@pytest.mark.unit
def test_extra_args_are_appended_last():
    """Raw passthrough must come after the managed flags, so a deliberate
    override wins (llama.cpp takes the last occurrence)."""
    argv = _build_argv(
        {"model_path": MODEL, "ctx_size": 4096,
         "extra_args": ["--ctx-size", "9999"]}
    )
    assert argv[-2:] == ["--ctx-size", "9999"]


# `-cmoe` is a bare, presence-signalled flag. True adds exactly one token:
# llama.cpp takes NO argument after it, and a value would be read as a
# positional and fail at spawn -- a load failure, not a misconfiguration. The
# length delta is checked rather than the following token, because the flag
# can land last in argv and a value starting with "-" would slip past a token
# check. False adds nothing, or 'expert offload off' would silently mean 'all
# experts on CPU'.
@pytest.mark.unit
@pytest.mark.parametrize("value, added", [(True, 1), (False, 0)], ids=["true_bare_flag", "false_emits_nothing"])
def test_cpu_moe_is_a_bare_flag(value, added):
    base = _build_argv({"model_path": MODEL})
    argv = _build_argv({"model_path": MODEL, "cpu_moe": value})
    assert len(argv) == len(base) + added, (
        f"-cmoe should add exactly {added} token(s); added {len(argv) - len(base)}: "
        f"{[a for a in argv if a not in base]}"
    )
    assert ("-cmoe" in argv) is value


@pytest.mark.unit
def test_n_max_bound_and_emitter_agree():
    """`spec_draft_n_max` is the one flag in its block gated on truthiness
    rather than `is not None`, which is safe ONLY because `ge=1` makes 0
    unreachable through validation -- the two tests are then equivalent over
    every valid value.

    That safety is a coupling between a constraint and an emitter sitting in
    different files, with nothing holding them together. If the bound is ever
    relaxed to allow 0, the emitter silently starts dropping a valid setting;
    this fails first and says so. The raw-dict path matters here: provider unit
    tests and gguf_probe both build config dicts that bypass validation, so an
    out-of-range 0 can reach the emitter in exactly the places used to measure.
    """
    from heylook_llm.config import PROVIDER_CONFIG_CLASSES

    field = PROVIDER_CONFIG_CLASSES["gguf"].model_fields["spec_draft_n_max"]
    lower_bounds = [
        getattr(m, "ge", None) or getattr(m, "gt", None)
        for m in field.metadata
        if hasattr(m, "ge") or hasattr(m, "gt")
    ]
    assert lower_bounds and min(b for b in lower_bounds if b is not None) >= 1, (
        "spec_draft_n_max no longer excludes 0, so _build_args' truthiness "
        "check now silently drops a valid setting -- switch that line to "
        "`is not None` like the rest of the block"
    )


@pytest.mark.unit
def test_binary_fallback_matches_where_build_llama_builds():
    """The provider's last-resort binary path and the path
    `scripts/build_llama.py` builds to must be the same directory.

    They live in different files with no import between them (the script is a
    standalone PEP 723 tool, deliberately not importable from the package), so
    nothing but this test stops them drifting. If they drift, the fallback
    silently stops finding a binary that was just built -- and the failure
    looks like "no binary configured" rather than "these two disagree".
    """
    import re
    from pathlib import Path as P
    from heylook_llm.providers.llama_server_provider import LlamaServerProvider

    fallback = LlamaServerProvider.DEFAULT_BUILD
    assert fallback == P.home() / ".heylook" / "llama.cpp" / "build" / "bin" / "llama-server"

    script = (P(__file__).resolve().parents[2] / "scripts" / "build_llama.py").read_text()
    m = re.search(r'HOME_SUBDIR\s*=\s*\(([^)]*)\)', script)
    assert m, "build_llama.py no longer declares HOME_SUBDIR"
    parts = [p.strip().strip('"\'') for p in m.group(1).split(",") if p.strip()]
    assert list(fallback.parent.parent.parent.parts[-len(parts):]) == parts, (
        f"provider falls back to {fallback}, but build_llama.py builds to "
        f"{'/'.join(parts)}/build/bin/llama-server"
    )
