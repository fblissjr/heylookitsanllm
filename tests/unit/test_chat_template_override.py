"""The operator's chat-template override: does it win, and is it safe to undo.

The claims that decide whether the template editor is usable or actively
harmful:

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
5. A rejected override does not cost the model its working template -- the
   stop-less fallback must skip the source that FAILED, not a fixed list.
6. The install reaches the object the vision path actually reads.
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


class TestInstallTargetsTheProcessor:
    """The override must reach the object the VISION path reads.

    mlx-vlm's `get_chat_template` picks its template holder in order --
    `processor` when `processor.chat_template` is set, only then
    `processor.tokenizer` -- and transformers fills the processor's from a
    `chat_template.json` in the model dir. Verified live against a real
    Qwen3-VL processor (2026-09-08): with the tokenizer alone targeted the
    rendered prompt came back as the VENDOR's template while the install
    reported success; with the processor targeted it came back as ours.

    That live check needs a multi-GB model, so what is pinned here is the
    property it established -- force writes to the processor too -- which is
    the half that can regress in this repo.
    """

    def _info(self):
        from heylook_llm.providers.common.template_info import ModelTemplateInfo
        return ModelTemplateInfo(chat_template=OVERRIDE, special_tokens=frozenset(),
                                 template_source="heylook_override")

    def test_force_writes_the_template_to_the_processor_too(self):
        from heylook_llm.providers.common.template_info import install_chat_template

        class Obj:
            chat_template = "VENDOR"
        processor, tokenizer = Obj(), Obj()

        assert install_chat_template(tokenizer, self._info(), force=True,
                                     processor=processor) is True
        assert tokenizer.chat_template == OVERRIDE
        assert processor.chat_template == OVERRIDE, (
            "the processor kept the vendor template -- the vision path would "
            "render with it while the install reported success"
        )

    def test_auto_leaves_the_processor_alone(self):
        """Auto's contract is fill-a-missing-template, never stomp one.

        A VLM whose processor holds the vendor template while its tokenizer
        holds none is exactly the shape auto must not touch: extending auto to
        the processor would rewrite the vendor template of every such model at
        load.
        """
        from heylook_llm.providers.common.template_info import install_chat_template

        class Obj:
            chat_template: object = None
        processor, tokenizer = Obj(), Obj()
        processor.chat_template = "VENDOR"

        install_chat_template(tokenizer, self._info(), force=False, processor=processor)
        assert processor.chat_template == "VENDOR"


class TestStopLessFallback:
    """A rejected override must not cost the model its working template.

    `read_template_info` refuses a template that renders none of the model's
    stop tokens and walks the OTHER sources for a usable one. That fallback
    list was hand-written as (tokenizer_config, chat_template_json), which was
    correct only while `chat_template.jinja` was the TOP auto rung -- omitting
    the winner was the point. Inserting the override above it turned the
    omission into a bug: the vendor jinja sitting in the same directory was
    never retried, so a stop-less override left NOTHING installed.
    """

    def test_a_stopless_override_falls_back_to_the_vendor_template(self, tmp_path):
        from heylook_llm.providers.common.template_info import read_template_info

        d = tmp_path / "mlx"
        d.mkdir()
        (d / "chat_template.jinja").write_text(VENDOR)  # carries <end_of_turn>
        (d / "tokenizer_config.json").write_text(
            '{"eos_token": "<end_of_turn>", "added_tokens_decoder": '
            '{"1": {"content": "<end_of_turn>", "special": true}}}')
        # renders text but never a stop token -> the loader must refuse it
        (d / HEYLOOK_TEMPLATE_FILENAME).write_text(
            "OVERRIDE{% for m in messages %}{{ m.content }}{% endfor %}")

        info = read_template_info(d, None)

        assert info.chat_template.startswith("VENDOR"), (
            f"the vendor template was not recovered (source={info.template_source}) "
            "-- a rejected override cost the model its only working template"
        )

    def test_a_stopless_vendor_template_still_falls_back_past_itself(self, tmp_path):
        """The fallback skips the source that FAILED, not a fixed list.

        Guards the derivation rather than the one case above: whichever rung
        loses must be the one excluded, so adding a rung cannot strand it.
        """
        from heylook_llm.providers.common.template_info import read_template_info

        d = tmp_path / "mlx"
        d.mkdir()
        (d / "chat_template.jinja").write_text(
            "VENDOR{% for m in messages %}{{ m.content }}{% endfor %}")  # stop-less
        (d / "tokenizer_config.json").write_text(
            '{"eos_token": "<end_of_turn>", "chat_token": null, '
            '"chat_template": "EMBEDDED{{ messages[0].content }}<end_of_turn>", '
            '"added_tokens_decoder": {"1": {"content": "<end_of_turn>", "special": true}}}')

        info = read_template_info(d, None)
        assert info.chat_template.startswith("EMBEDDED"), info.template_source


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

    def test_refusing_one_conversation_shape_is_not_a_broken_template(self):
        """Raising is how a template says "not that shape", not "I am broken".

        Real ones do it: Qwen's official template raises on two leading system
        messages, others refuse a trailing assistant turn. Refusing to SAVE
        such a template would block valid work -- while a template that raises
        on EVERY shape really would brick the model at its next load, so that
        one has to be refused. The distinction is the whole reason validation
        probes more than one shape.
        """
        picky = ("{% if messages[-1].role == 'user' and messages|length > 2 %}"
                 "{{ raise_exception('no trailing user turn') }}{% endif %}"
                 "{% for m in messages %}{{ m.content }}{% endfor %}")
        ctf.validate(picky, provider="gguf", config={})  # must not raise

        with pytest.raises(ctf.TemplateWriteRefused, match="every shape"):
            ctf.validate("{{ raise_exception('always') }}", provider="gguf", config={})
