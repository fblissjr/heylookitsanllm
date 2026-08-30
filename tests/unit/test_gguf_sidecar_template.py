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


@pytest.mark.unit
class TestSidecarTemplatePrecedence:
    def test_a_sidecar_beside_the_weights_is_used(self, tmp_path):
        model = _gguf_dir(tmp_path, sidecar=True)
        path, origin = _provider(model)._resolve_chat_template()
        assert path == str(tmp_path / "chat_template.jinja")
        assert origin == "sidecar"

    def test_no_sidecar_falls_through_to_the_embedded_template(self, tmp_path):
        model = _gguf_dir(tmp_path, sidecar=False)
        path, origin = _provider(model)._resolve_chat_template()
        assert path is None
        assert "embedded" in origin

    def test_an_explicit_path_outranks_a_sidecar(self, tmp_path):
        """Naming a file is a stronger statement than a file being present.
        Silently preferring the directory would override an operator's
        deliberate choice with an incidental one."""
        model = _gguf_dir(tmp_path, sidecar=True)
        chosen = tmp_path / "chosen.jinja"
        chosen.write_text("{{ 'chosen' }}")
        path, origin = _provider(model, chat_template_path=str(chosen))._resolve_chat_template()
        assert path == str(chosen)
        assert origin == "configured"

    def test_the_embedded_template_stays_reachable_with_the_sidecar_on_disk(self, tmp_path):
        """The reason the opt-out is a field rather than "delete the file":
        the embedded template is a legitimate choice, and a downloaded
        snapshot dir is not somewhere to vandalize to get the documented
        default back."""
        model = _gguf_dir(tmp_path, sidecar=True)
        path, origin = _provider(model, use_sidecar_chat_template=False)._resolve_chat_template()
        assert path is None
        assert "embedded" in origin

    def test_discovery_is_scoped_to_the_model_file_s_own_directory(self, tmp_path):
        """A template one level up belongs to whatever else lives up there.
        Split GGUF shards all sit in the model dir, so this is the same folder
        either way and the narrow probe loses nothing."""
        (tmp_path / "chat_template.jinja").write_text("{{ 'parent' }}")
        nested = tmp_path / "quant"
        nested.mkdir()
        model = _gguf_dir(nested, sidecar=False)
        assert _provider(model)._resolve_chat_template()[0] is None


@pytest.mark.unit
class TestSidecarDiscoveryDegradesQuietly:
    def test_a_nonexistent_model_path_finds_nothing_and_does_not_raise(self):
        """`_build_args` is exercised with paths that do not exist (the
        argv/metadata drift test), so a filesystem probe added to that path
        must be a clean miss rather than an error."""
        path, origin = _provider("/no/such/dir/model.gguf")._resolve_chat_template()
        assert path is None
        assert "embedded" in origin

    def test_a_directory_named_chat_template_jinja_is_not_a_template(self, tmp_path):
        """`is_file`, not `exists` -- handing llama-server a directory would
        turn a quiet fallthrough into a spawn failure."""
        model = _gguf_dir(tmp_path, sidecar=False)
        (tmp_path / "chat_template.jinja").mkdir()
        assert _provider(model)._resolve_chat_template()[0] is None


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

    def test_no_sidecar_emits_no_template_flag_at_all(self, tmp_path):
        """Absent must mean ABSENT: passing an empty or placeholder value would
        be llama-server's problem to interpret, and the documented default is
        that it reads the template out of the GGUF itself."""
        from pathlib import Path

        model = _gguf_dir(tmp_path, sidecar=False)
        args = _provider(model)._build_args(Path("/bin/llama-server"), 8080)
        assert "--chat-template-file" not in args


@pytest.mark.unit
class TestTheFieldIsClassified:
    def test_use_sidecar_chat_template_declares_its_effect(self):
        """Every provider-config field declares when a change takes effect;
        the reload set, the import allowlist and /v1/admin/model-options all
        DERIVE from that metadata, so an unclassified field is invisible to
        three surfaces at once. Spawn-time, like every other template lever
        here -- llama-server reads it at exec."""
        from heylook_llm.config import EFFECT_REQUIRES_RELOAD, GGUFModelConfig

        extra = GGUFModelConfig.model_fields["use_sidecar_chat_template"].json_schema_extra
        assert isinstance(extra, dict)
        assert extra.get("effect") == EFFECT_REQUIRES_RELOAD
