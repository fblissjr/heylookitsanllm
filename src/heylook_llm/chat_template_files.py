"""Reading and writing the operator's chat-template override.

The override is ONE file, ``chat_template.heylook.jinja``, sitting in the
model's own folder, discovered at load by the same ladders that already find a
publisher's sidecar. Deliberately NOT a models.toml value: nothing here writes
config, so there is no path to materialize a discovered entry, nothing can
drift between a stored path and a file, and revert is deleting one file.

WHY A DISTINCT FILENAME, not the vendor's ``chat_template.jinja``: on MLX that
file is usually the ONLY copy of the model's template (measured 2026-09-08
across the model dirs here: none carried an embedded ``tokenizer_config``
template, and all but one had ``chat_template.jinja`` as the sole source), so
editing it in place destroys the original with nothing to fall back to. On
gguf the embedded header template is an immutable original, so a sidecar there
is additive -- but one filename for both engines beats a rule that changes
meaning per engine. It also survives a re-download: huggingface_hub prunes
nothing, so a file outside the repo manifest is left alone while
``chat_template.jinja`` is refreshed.

WHAT THIS MODULE DOES NOT DO: resolve the ladder itself. Both engines already
have one, and each is called here rather than re-implemented -- a preview that
walks its own ladder agrees on the day it is written and silently diverges
after, which is the failure this repo keeps naming.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .providers.common.template_info import (
    HEYLOOK_TEMPLATE_FILENAME,
    HEYLOOK_OVERRIDE,
    is_explicit_source,
    read_template_info,
)

logger = logging.getLogger(__name__)

# Providers that render a chat template at all: DERIVED from the provider
# roster, not hand-listed. `PROVIDER_CONFIG_CLASSES` is this repo's single
# source of truth for the provider Literal, and a hand-copied sibling list is
# what the reload set, the import allowlist and /v1/admin/model-options were
# all changed to stop being. Only embeddings are excluded, and that exclusion
# is the thing worth stating: they generate nothing, so no template applies.
def _template_providers() -> frozenset:
    from .config import PROVIDER_CONFIG_CLASSES
    return frozenset(PROVIDER_CONFIG_CLASSES) - {"mlx_embedding"}


TEMPLATE_PROVIDERS = _template_providers()


class TemplateWriteRefused(ValueError):
    """A template was rejected before anything touched disk."""


@dataclass(frozen=True)
class TemplateView:
    """What is in force for one model, and whether an edit would reach it."""

    model_id: str
    provider: str
    supported: bool
    template: Optional[str] = None
    origin: str = "unknown"
    override_present: bool = False
    override_path: Optional[str] = None
    writable: bool = False
    # Set when the override exists (or would be written) but the resolved
    # template did NOT come from it. An editor whose writes go nowhere is
    # worse than no editor, so this is a first-class field rather than a note.
    inert_reason: Optional[str] = None
    # The override file's OWN body, independent of whether it won. `template`
    # is what the model will RENDER with; this is what the editor must show,
    # and they differ exactly when an override exists but lost the ladder.
    # Painting the winner's body into the editor made the operator's own file
    # unreadable from the surface that wrote it -- a rejected template showed
    # as an empty box with Save disabled and no way to repair it.
    override_template: Optional[str] = None
    # True when the model is RESIDENT and rendering with something other than
    # what is on disk now -- i.e. someone edited the template since it loaded.
    # None when the model is not loaded, which is not the same as False and
    # must not be rendered as "up to date".
    stale: Optional[bool] = None
    notes: list[str] = field(default_factory=list)


def model_directory(model_path: str) -> Optional[Path]:
    """The folder a model's template files live in, or None.

    Derived from what is ON DISK rather than from the provider name, because
    the engines differ in what ``model_path`` MEANS -- gguf names a FILE (the
    .gguf, whose shards all live in one directory), MLX names a DIRECTORY (the
    HF snapshot) -- while wanting the identical thing done. A path that
    resolves to neither is a broken entry on either engine, and None says so
    once instead of twice.
    """
    if not model_path:
        return None
    try:
        p = Path(model_path).expanduser()
        if p.is_dir():
            return p
        if p.is_file():
            return p.parent
    except (OSError, ValueError):
        return None
    return None


def override_path(model_path: str) -> Optional[Path]:
    """Where this model's override lives, whether or not it exists yet."""
    directory = model_directory(model_path)
    return None if directory is None else directory / HEYLOOK_TEMPLATE_FILENAME


def view(model_id: str, provider: str, config: dict,
         loaded_template: Optional[str] = None) -> TemplateView:
    """Resolve what this model's template is and where it came from.

    Reads files only -- never loads a model and never talks to a running one,
    so it answers for models that are not resident. That is the point: the
    thing you most want to see before loading a model is the prompt format it
    will load with.
    """
    if provider not in TEMPLATE_PROVIDERS:
        return TemplateView(
            model_id=model_id, provider=provider, supported=False,
            origin="not applicable",
            notes=[f"The {provider} provider does not render a chat template."],
        )

    model_path = str(config.get("model_path") or "")
    path = override_path(model_path)
    present = bool(path and path.is_file())
    directory = model_directory(model_path)
    writable = _is_writable(directory)

    if provider == "gguf":
        template, origin, inert = _gguf_view(model_id, config, present)
    else:
        template, origin, inert = _mlx_view(config, present)

    # Read straight off disk rather than from the ladder: the ladder answers
    # "what wins", and when the override lost, its body is precisely what the
    # ladder did not return.
    override_body = _read(path) if (path and present) else None

    notes: list[str] = []
    if directory is None:
        notes.append(
            "model_path does not resolve to a file or directory, so there is "
            "nowhere to write an override."
        )
    elif not writable:
        notes.append(f"{directory} is not writable, so an override cannot be saved.")

    return TemplateView(
        model_id=model_id, provider=provider, supported=True,
        template=template, origin=origin,
        override_present=present,
        override_path=str(path) if path else None,
        writable=writable,
        inert_reason=inert,
        override_template=override_body,
        stale=None if loaded_template is None else loaded_template != template,
        notes=notes,
    )


def _gguf_view(model_id: str, config: dict,
               present: bool) -> tuple[Optional[str], str, Optional[str]]:
    """Resolve through the gguf provider's OWN ladder, then read the winner."""
    from .providers.llama_server_provider import LlamaServerProvider

    resolved, origin = LlamaServerProvider.resolve_chat_template(
        config, model_id, log=False)
    if resolved:
        body = _read(Path(resolved))
    else:
        # No file won, so the template is the one baked into the GGUF header.
        from . import gguf_metadata
        body = gguf_metadata.chat_template(Path(str(config.get("model_path") or "")))

    inert = None
    if present and origin != HEYLOOK_OVERRIDE:
        if config.get("chat_template_path"):
            # VERIFIED cause: the field is right there in the config.
            inert = (
                "chat_template_path names an explicit file, which outranks the "
                "override. Clear that field for edits here to take effect."
            )
        else:
            # NOT verified, so NOT asserted. The earlier version named the
            # media guard here by elimination and was confidently wrong
            # whenever an override lost for any third reason -- a model_path
            # naming a directory, for instance, makes the sidecar probe return
            # None with the media guard never having run. State what is known
            # (it lost, and to what) and point at the log for why.
            inert = (
                f"The override is on disk but not in force -- the model "
                f"resolves to its {origin} template instead. The load log "
                f"names the reason; a template with no media markers is "
                f"refused on a model served with a projector."
            )
    return body, origin, inert


def _mlx_view(config: dict, present: bool) -> tuple[Optional[str], str, Optional[str]]:
    """Resolve through template_info's ladder -- the one load() walks."""
    directory = model_directory(str(config.get("model_path") or ""))
    if directory is None:
        return None, "unresolvable model_path", None

    source = config.get("chat_template_source")
    info = read_template_info(directory, source)
    origin = info.template_source or "unknown"

    inert = None
    if present and origin != HEYLOOK_OVERRIDE:
        if is_explicit_source(source):
            inert = (
                f"chat_template_source is set to {source!r}, which outranks the "
                "override. Clear it for edits here to take effect."
            )
        else:
            # Same rule as gguf: report what is KNOWN, name the likely cause
            # without asserting it. `none(stopless)` is the one origin that
            # does state its own cause, so it gets the specific message.
            if origin.startswith("none("):
                inert = (
                    "The override renders none of the model's stop tokens, so "
                    "it was refused -- it would generate to the token cap. No "
                    "file template was usable; the loader's built-in one stands."
                )
            else:
                inert = (
                    f"The override is on disk but not in force -- the model "
                    f"resolves to its {origin} template instead. The load log "
                    f"names the reason."
                )
    return (info.chat_template or None), origin, inert


def write_override(model_path: str, body: str, *, provider: str,
                   config: dict) -> "tuple[Path, list[str]]":
    """Validate ``body``, then write it as this model's override.

    Returns the path written and the conversation shapes the template REFUSES,
    for the caller to disclose. A refused shape is not an error -- see
    ``validate`` -- but it is what the operator needs to know before the
    model's next load.

    Validation happens BEFORE anything touches disk. This is the one place a
    bad write bricks a model: llama-server turns a raised jinja exception into
    a 500, so an unparseable template makes every request to that model fail
    with nothing naming the cause.
    """
    path = override_path(model_path)
    if path is None:
        raise TemplateWriteRefused(
            f"model_path {model_path!r} does not resolve to a file or "
            "directory, so there is nowhere to write an override."
        )
    if not body or not body.strip():
        raise TemplateWriteRefused(
            "Refusing to write an empty template. To go back to the model's "
            "own template, delete the override instead."
        )
    # The shapes this template refuses. NOT grounds to refuse the write --
    # a template legitimately declining a shape is normal -- but the caller
    # must be able to say so, or the operator learns about it from a 500.
    refused_shapes = validate(body, provider=provider, config=config)
    # ATOMIC: write a temp file beside it, then rename. A plain write_text
    # truncates and then fills, so a load racing the write can read a
    # zero-length or half-written template -- on gguf that is a jinja parse
    # error inside llama-server, i.e. a 500 on every request with nothing
    # naming the cause. That is the exact bricking validate() exists to
    # prevent, and doing the write non-atomically would reintroduce it below
    # the guard. os.replace is atomic within a filesystem, which a sibling
    # temp file guarantees.
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise TemplateWriteRefused(f"could not write {path}: {exc}") from exc
    logger.info("[template] wrote chat template override %s", path)
    if refused_shapes:
        logger.warning(
            "[template] the override written to %s refuses these conversation "
            "shapes: %s. That is legal, but a shape the server actually sends "
            "will fail at generation rather than here.", path,
            "; ".join(refused_shapes))
    return path, refused_shapes


def remove_override(model_path: str) -> bool:
    """Delete the override. True if one was there.

    Revert is deleting our own file, never restoring a backup -- which is only
    safe because the vendor's template was never touched. See the module
    docstring for why the filenames differ.
    """
    path = override_path(model_path)
    if path is None or not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise TemplateWriteRefused(f"could not remove {path}: {exc}") from exc
    logger.info("[template] removed chat template override %s", path)
    return True


# Templates legitimately refuse a shape -- Qwen's official template raises on
# two leading system messages, several refuse a trailing assistant turn -- so a
# raise on ONE shape cannot mean the template is broken. But a template that
# raises on EVERY shape is broken, and passing it would brick the model at its
# next load, which is the outcome this function exists to prevent. So: refuse
# only when nothing renders.
#
# EVERY SHAPE IS TRIED, and that is the fix rather than the design. The loop
# used to stop at the first shape that rendered, which made every shape after
# the first a FALLBACK rather than a check -- and the plain user/assistant/user
# exchange is first. So a template that renders that and raises on any system
# message saved with a clean 200 and then failed at generation, since
# llama-server turns a raised jinja exception into a 500. That is not an exotic
# shape: `conversation_generate_api` prepends a system message whenever the
# document has a system prompt, which is the default in v3. The shapes below
# cover a system message, a trailing assistant turn (the continuation path) and
# two leading system messages (the case the publishers actually differ on).
_PROBE_SHAPES = (
    (
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
        {"role": "user", "content": "ping"},
    ),
    (
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "ping"},
    ),
    (
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ),
    (
        {"role": "system", "content": "be brief"},
        {"role": "system", "content": "and kind"},
        {"role": "user", "content": "ping"},
    ),
)


def validate(body: str, *, provider: str, config: dict) -> list[str]:
    """Raise TemplateWriteRefused unless ``body`` compiles AND renders.

    Returns the shapes this template REFUSES, as human-readable strings, for
    a caller to disclose. An empty list means it rendered every shape.

    Refusing some shapes is legal and not grounds to reject the write, but it
    is the single most useful thing to tell the operator: a template that
    raises on a system message saves cleanly and then makes every request from
    a document with a system prompt fail, since llama-server turns a raised
    jinja exception into a 500. Silence about that is what made the earlier
    version of this function feel safe while it was not.

    Compiling alone is not enough -- the errors that matter (an undefined
    variable, a bad filter, a call into something that is not there) only
    surface when the template is actually run over messages.

    The environment MIRRORS what the engines provide rather than being a bare
    jinja2 one: chat templates routinely call ``raise_exception``,
    ``strftime_now`` and ``tojson``, which transformers and llama.cpp inject.
    A bare environment would reject most real templates as broken, which is
    worse than no validation -- it would refuse valid work.
    """
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:  # pragma: no cover -- jinja2 ships with transformers
        logger.warning("[template] jinja2 unavailable; skipping validation")
        return

    def _raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: ""
    env.filters["tojson"] = lambda value, **kw: "{}"

    try:
        template = env.from_string(body)
    except Exception as exc:
        raise TemplateWriteRefused(
            f"template does not parse as jinja: {type(exc).__name__}: {exc}"
        ) from exc

    rendered = ""
    refusals: list[str] = []
    for shape in _PROBE_SHAPES:
        label = "+".join(m["role"] for m in shape)
        try:
            out = template.render(
                messages=[dict(m) for m in shape],
                add_generation_prompt=True,
                bos_token="", eos_token="",
                enable_thinking=False,
            )
        except jinja2.exceptions.TemplateError as exc:
            # This shape is refused. That may well be correct -- record it and
            # try the next one; only a template that refuses EVERY shape is
            # broken.
            refusals.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        except Exception as exc:
            # Not a template-authored refusal: a bad filter, a call into
            # something absent. That is broken on any shape.
            raise TemplateWriteRefused(
                f"template failed to render: {type(exc).__name__}: {exc}"
            ) from exc
        if out.strip() and not rendered:
            # No `break`. Stopping here made every later shape a FALLBACK
            # rather than a check, and the plainest shape is first -- so the
            # shapes that actually differ between publishers were never
            # reached for any template that rendered the plain one.
            rendered = out

    if not rendered:
        detail = (f" It raised on every shape tried: {'; '.join(refusals)}."
                  if refusals else "")
        raise TemplateWriteRefused(
            "template produced no prompt for any ordinary conversation -- the "
            "model would receive an empty prompt or fail at its next load."
            + detail
        )

    # The MLX half of the same promise. `read_template_info` REFUSES a
    # template that renders none of the model's stop tokens (it would generate
    # to the cap) and walks on to another source -- so without this check the
    # editor accepts, with a 200, a template the next load will throw away.
    # The gguf media guard below was enforced here from the start; this one
    # was implemented right next door and never called.
    if provider == "mlx":
        directory = model_directory(str(config.get("model_path") or ""))
        if directory is not None:
            from .providers.common.template_info import (
                _read_eos_tokens, _read_json, _template_can_stop)
            eos = _read_eos_tokens(directory, _read_json(directory / "tokenizer.json"),
                                   _read_json(directory / "tokenizer_config.json"))
            if not _template_can_stop(body, eos):
                raise TemplateWriteRefused(
                    "this template renders none of the model's stop tokens "
                    f"({', '.join(sorted(eos))}), so the model would generate "
                    "until the max_tokens cap. The loader refuses such a "
                    "template at load and falls back, so saving it would cost "
                    "the model its working template."
                )

    if provider == "gguf":
        from .providers.llama_server_provider import LlamaServerProvider
        if LlamaServerProvider._is_media_served(config) \
                and not LlamaServerProvider._template_handles_media(body):
            raise TemplateWriteRefused(
                "this model is served with a projector (mmproj/vision) but the "
                "template has no media markers, so saving it would load the "
                "vision tower and then render prompts that can never reference "
                "an image. The spawn-time guard would refuse it too."
            )

    return refusals


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _is_writable(directory: Optional[Path]) -> bool:
    if directory is None:
        return False
    import os
    return os.access(directory, os.W_OK)
