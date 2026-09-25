# tests/unit/test_gguf_sidecar_template.py
"""Sidecar chat-template discovery for the gguf provider (v1.79.43).

llama-server otherwise uses the template EMBEDDED IN THE GGUF, which is
whatever the quantizer baked in -- and this repo has measured two publishers
shipping materially different templates for identical weights. A
``chat_template.jinja`` sitting beside the weights is the readable answer, so
it now wins over the embedded one by default.

The precedence being pinned here is a three-way ladder, and the order matters
in both directions: an explicit ``chat_template_path`` is someone naming a
file on purpose and must outrank a file that merely happens to be in the
directory, while the embedded template has to stay REACHABLE without deleting
anything from a downloaded snapshot dir.
"""

import pytest


def _provider(model_path, **config):
    """A provider instance WITHOUT loading anything -- ``__init__`` does not
    spawn, so template resolution is testable with no llama-server."""
    from heylook_llm.providers.llama_server_provider import LlamaServerProvider

    return LlamaServerProvider(
        "m", {"model_path": str(model_path), **config}, verbose=False)


def _gguf_dir(tmp_path, *, sidecar: bool):
    (tmp_path / "model-Q8_0.gguf").write_bytes(b"GGUF")
    if sidecar:
        (tmp_path / "chat_template.jinja").write_text("{{ 'sidecar' }}")
    return tmp_path / "model-Q8_0.gguf"


class _Contains(str):
    """An expected origin that only has to appear IN the reported one."""

    def __eq__(self, other):
        return str(self) in other

    __hash__ = str.__hash__


_SIDECAR = "chat_template.jinja"
_CHOSEN = "chosen.jinja"


def _expected_path(tmp_path, name):
    return None if name is None else str(tmp_path / name)


# Precedence: explicit path > sidecar > embedded.
# - explicit_outranks_sidecar: naming a file is a stronger statement than a
#   file being present; silently preferring the directory would override an
#   operator's deliberate choice with an incidental one.
# - opt_out_restores_embedded: why the opt-out is a field rather than "delete
#   the file": the embedded template is a legitimate choice, and a downloaded
#   snapshot dir is not somewhere to vandalize to get the default back.
# - discovery_scoped_to_own_dir: a template one level up belongs to whatever
#   else lives up there. Split GGUF shards all sit in the model dir, so the
#   narrow probe loses nothing.
@pytest.mark.unit
@pytest.mark.parametrize("layout, config, expected_path, expected_origin", [
    ("sidecar", {}, _SIDECAR, "sidecar"),
    ("none", {}, None, _Contains("embedded")),
    ("sidecar", {"chat_template_path": _CHOSEN}, _CHOSEN, "configured"),
    ("sidecar", {"use_sidecar_chat_template": False}, None, _Contains("embedded")),
    ("parent", {}, None, _Contains("embedded")),
], ids=["sidecar_beside_weights_used", "no_sidecar_falls_to_embedded", "explicit_outranks_sidecar",
        "opt_out_restores_embedded", "discovery_scoped_to_own_dir"])
def test_sidecar_template_precedence(tmp_path, layout, config, expected_path, expected_origin):
    if layout == "parent":
        (tmp_path / _SIDECAR).write_text("{{ 'parent' }}")
        nested = tmp_path / "quant"
        nested.mkdir()
        model = _gguf_dir(nested, sidecar=False)
    else:
        model = _gguf_dir(tmp_path, sidecar=layout == "sidecar")
    if "chat_template_path" in config:
        (tmp_path / _CHOSEN).write_text("{{ 'chosen' }}")
        config = {**config, "chat_template_path": str(tmp_path / _CHOSEN)}
    path, origin = _provider(model, **config)._resolve_chat_template()
    assert path == _expected_path(tmp_path, expected_path)
    assert expected_origin == origin


# A sidecar may not cost the model its projector: the guard that stopped this
# feature shipping broken. Found live 2026-08-30: every sidecar-carrying gguf
# model on this machine was MULTIMODAL, and `unsloth_Muse-Glimmer-30B-GGUF`
# ships a sidecar with the bare words "image" and "video" and no media
# control tokens. Unguarded, the default would have loaded that model's
# projector and rendered every prompt through a template that can never
# reference an image -- a vision model quietly answering as a text one.
# - projector_accepts_media_aware: the guard must not become a blanket ban;
#   real sidecars that DO carry the markers are the ones the owner asked for.
# - text_model_takes_media_blind: the guard keys on what would be LOST, not
#   on the template alone.
# - bare_word_image_is_not_media: the precise shape of the real defect; a
#   substring test for "image" would have waved Muse-Glimmer through. Markers
#   are what a projector can actually bind to.
# - explicit_path_never_second_guessed: the guard exists because discovery is
#   implicit; naming a file is not, and the warning tells operators to use it
#   as the override.
_TEXT_ONLY = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
_WITH_MEDIA = ("{% for m in messages %}{% if m.image %}"
               "<|vision_start|><|image_pad|><|vision_end|>{% endif %}"
               "{{ m['content'] }}{% endfor %}")
_BARE_WORD_IMAGE = ("{# describe the image or video #}{% for m in messages %}"
                    "{{ m['content'] }}{% endfor %}")


@pytest.mark.unit
@pytest.mark.parametrize("template, projector, explicit, expected_path, expected_origin", [
    (_TEXT_ONLY, True, False, None, _Contains("sidecar skipped")),
    (_WITH_MEDIA, True, False, _SIDECAR, "sidecar"),
    (_TEXT_ONLY, False, False, _SIDECAR, "sidecar"),
    (_BARE_WORD_IMAGE, True, False, None, _Contains("sidecar skipped")),
    (_TEXT_ONLY, True, True, _CHOSEN, "configured"),
], ids=["projector_refuses_media_blind", "projector_accepts_media_aware", "text_model_takes_media_blind",
        "bare_word_image_is_not_media", "explicit_path_never_second_guessed"])
def test_a_sidecar_may_not_cost_the_model_its_projector(
        tmp_path, template, projector, explicit, expected_path, expected_origin):
    (tmp_path / "model-Q8_0.gguf").write_bytes(b"GGUF")
    (tmp_path / _SIDECAR).write_text(template)
    config = {}
    if projector:
        config.update(mmproj_path=str(tmp_path / "mmproj.gguf"), modalities=["text", "vision"])
    if explicit:
        (tmp_path / _CHOSEN).write_text(template)
        config["chat_template_path"] = str(tmp_path / _CHOSEN)
    path, origin = _provider(tmp_path / "model-Q8_0.gguf", **config)._resolve_chat_template()
    assert path == _expected_path(tmp_path, expected_path)
    assert expected_origin == origin


# Discovery degrades to a clean miss.
# - nonexistent_model_path: `_build_args` is exercised with paths that do not
#   exist (the argv/metadata drift test), so a filesystem probe on that path
#   must be a clean miss rather than an error.
# - dir_named_chat_template_jinja: `is_file`, not `exists` -- handing
#   llama-server a directory would turn a quiet fallthrough into a spawn
#   failure.
@pytest.mark.unit
@pytest.mark.parametrize("case", ["nonexistent_model_path", "dir_named_chat_template_jinja"])
def test_sidecar_discovery_degrades_quietly(tmp_path, case):
    if case == "nonexistent_model_path":
        model = "/no/such/dir/model.gguf"
    else:
        model = _gguf_dir(tmp_path, sidecar=False)
        (tmp_path / _SIDECAR).mkdir()
    path, origin = _provider(model)._resolve_chat_template()
    assert path is None
    assert "embedded" in origin


@pytest.mark.unit
class TestSidecarReachesTheCommandLine:
    def test_the_discovered_template_is_emitted_as_chat_template_file(self, tmp_path):
        """The resolution above is only worth anything if it reaches argv --
        and `--chat-template-file`, never the `--chat-template` sibling, which
        takes template TEXT rather than a path."""
        from pathlib import Path

        model = _gguf_dir(tmp_path, sidecar=True)
        args = _provider(model)._build_args(Path("/bin/llama-server"), 8080)
        assert "--chat-template-file" in args
        assert args[args.index("--chat-template-file") + 1] == str(
            tmp_path / "chat_template.jinja")

    # Absent must mean ABSENT: the default MUST stay "whatever the quantizer
    # baked in", and an empty or placeholder value would be llama-server's
    # problem to interpret. Rows: a real dir with no sidecar, and a model path
    # that does not exist (the provider tests' `/fake/model.gguf`).
    @pytest.mark.parametrize("where", ["dir_without_sidecar", "nonexistent_model_path"])
    def test_no_sidecar_emits_no_template_flag_at_all(self, tmp_path, where):
        from pathlib import Path

        if where == "dir_without_sidecar":
            model = _gguf_dir(tmp_path, sidecar=False)
        else:
            model = "/fake/model.gguf"
        args = _provider(model)._build_args(Path("/bin/llama-server"), 8080)
        assert "--chat-template-file" not in args


@pytest.mark.unit
class TestTheFieldIsClassified:

    def test_the_provider_reads_the_default_off_the_field(self):
        """No hand-copied literal. The provider also accepts RAW dicts (the
        argv/metadata drift test builds one, and so do the provider unit
        tests), and those hit the fallback -- so a copied `True` would mean
        flipping the field's default left every raw-dict caller on the old
        behaviour with the suite still green."""
        from unittest.mock import patch

        from heylook_llm.config import GGUFModelConfig
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider

        field = GGUFModelConfig.model_fields["use_sidecar_chat_template"]
        assert LlamaServerProvider._sidecar_default() is bool(field.default)
        with patch.object(field, "default", False):
            assert LlamaServerProvider._sidecar_default() is False
