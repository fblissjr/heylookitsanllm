"""The operator's chat-template override: does it win, and is it safe to undo.

Four claims, one test each. They are the ones that decide whether the template
editor is usable or actively harmful:

1. The override outranks what the vendor shipped -- on BOTH engines. A rung
   added to one ladder and not the other is the shape this repo keeps getting
   bitten by, so the engines are one parameterized case rather than two tests
   that could drift apart.
2. `use_sidecar_chat_template = false` does NOT suppress it. That flag chooses
   between the publisher's sidecar and the embedded template; if it also
   silenced the override, the editor would write a file nothing reads with no
   error anywhere.
3. Reverting restores the vendor template AND leaves the vendor file byte-
   identical. This is the whole reason the override has its own filename: on
   MLX the vendor `chat_template.jinja` is usually the only copy in existence.
4. Validation refuses a template that would brick a model, BEFORE writing.
"""

import pytest

from heylook_llm import chat_template_files as ctf
from heylook_llm.providers.common.template_info import HEYLOOK_TEMPLATE_FILENAME

VENDOR = "VENDOR{% for m in messages %}{{ m.content }}{% endfor %}<end_of_turn>"
OVERRIDE = "OVERRIDE{% for m in messages %}{{ m.content }}{% endfor %}<end_of_turn>"


def _gguf_dir(tmp_path):
    """A model folder shaped the way gguf's ladder expects: weights are a FILE."""
    d = tmp_path / "gguf"
    d.mkdir()
    weights = d / "model.gguf"
    weights.write_bytes(b"")  # only stat'd -- the ladder never parses it
    (d / "chat_template.jinja").write_text(VENDOR)
    return str(weights)


def _mlx_dir(tmp_path):
    """A model folder shaped the way MLX expects: model_path IS the directory."""
    d = tmp_path / "mlx"
    d.mkdir()
    (d / "chat_template.jinja").write_text(VENDOR)
    return str(d)


@pytest.fixture(params=["gguf", "mlx"])
def model(request, tmp_path):
    """(provider, model_path) for each engine, with a vendor template present."""
    provider = request.param
    path = _gguf_dir(tmp_path) if provider == "gguf" else _mlx_dir(tmp_path)
    return provider, path


def _view(provider, path, **config):
    return ctf.view("m", provider, {"model_path": path, **config})


class TestOverrideWins:
    def test_override_outranks_the_vendor_template(self, model):
        provider, path = model
        assert _view(provider, path).template.startswith("VENDOR")

        ctf.write_override(path, OVERRIDE, provider=provider, config={})

        view = _view(provider, path)
        assert view.template.startswith("OVERRIDE"), (
            f"{provider}: override on disk did not win the ladder "
            f"(origin={view.origin})"
        )
        assert view.override_present is True
        assert view.inert_reason is None

    def test_sidecar_flag_does_not_suppress_the_override(self, model):
        """`use_sidecar_chat_template=false` governs the PUBLISHER's sidecar.

        Applying it to the operator's own file would make the editor write
        somewhere the loader does not look -- silently, which is the one
        failure an editor must not have. gguf owns this flag; MLX has no
        equivalent, and asserting on both is what keeps the rule from being
        quietly engine-specific.
        """
        provider, path = model
        ctf.write_override(path, OVERRIDE, provider=provider, config={})

        view = _view(provider, path, use_sidecar_chat_template=False)

        assert view.template.startswith("OVERRIDE"), (
            f"{provider}: use_sidecar_chat_template=false suppressed the "
            f"operator's own override (origin={view.origin})"
        )

    def test_revert_restores_the_vendor_template_and_never_touched_it(self, model):
        """The vendor file must survive an edit unmodified.

        On MLX `chat_template.jinja` is typically the ONLY copy of the model's
        template, so an editor that wrote through to it would destroy the
        original with nothing to fall back to. Reverting is deleting our own
        file, and that is only safe if theirs was never written.
        """
        provider, path = model
        vendor_file = ctf.model_directory(path) / "chat_template.jinja"
        before = vendor_file.read_bytes()

        ctf.write_override(path, OVERRIDE, provider=provider, config={})
        assert vendor_file.read_bytes() == before, "the vendor template was modified"

        assert ctf.remove_override(path) is True
        view = _view(provider, path)
        assert view.template.startswith("VENDOR")
        assert view.override_present is False
        assert vendor_file.read_bytes() == before


class TestValidation:
    @pytest.mark.parametrize("body, why", [
        ("{% for m in messages %}{{ m.content }}", "unclosed block"),
        ("{# only a comment #}", "renders nothing"),
        ("   ", "empty"),
    ])
    def test_a_template_that_would_brick_the_model_is_refused_before_writing(
        self, tmp_path, body, why
    ):
        """Refusal has to happen before the write, not at the next load.

        llama-server turns a raised jinja exception into a 500, so a bad
        template on disk makes every request to that model fail with nothing
        naming the cause -- and the model is already unloadable by then.
        """
        path = _gguf_dir(tmp_path)
        override = ctf.model_directory(path) / HEYLOOK_TEMPLATE_FILENAME

        with pytest.raises(ctf.TemplateWriteRefused):
            ctf.write_override(path, body, provider="gguf", config={})

        assert not override.exists(), f"refused template ({why}) still hit disk"

    def test_a_working_template_is_accepted(self, tmp_path):
        path = _gguf_dir(tmp_path)
        ctf.write_override(path, OVERRIDE, provider="gguf", config={})
        assert _view("gguf", path).template.startswith("OVERRIDE")
