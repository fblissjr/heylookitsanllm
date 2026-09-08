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

# Providers that render a chat template at all. mlx_embedding has no chat
# surface, so the route answers "not applicable" rather than an empty editor.
TEMPLATE_PROVIDERS = frozenset({"mlx", "gguf"})


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
        stale=None if loaded_template is None else loaded_template != template,
        notes=notes,
    )


def _gguf_view(model_id: str, config: dict,
               present: bool) -> tuple[Optional[str], str, Optional[str]]:
    """Resolve through the gguf provider's OWN ladder, then read the winner."""
    from .providers.llama_server_provider import LlamaServerProvider

    resolved, origin = LlamaServerProvider.resolve_chat_template(config, model_id)
    if resolved:
        body = _read(Path(resolved))
    else:
        # No file won, so the template is the one baked into the GGUF header.
        from . import gguf_metadata
        body = gguf_metadata.chat_template(Path(str(config.get("model_path") or "")))

    inert = None
    if present and origin != HEYLOOK_OVERRIDE:
        if config.get("chat_template_path"):
            inert = (
                "chat_template_path names an explicit file, which outranks the "
                "override. Clear that field for edits here to take effect."
            )
        else:
            # The remaining way a present override loses: the media guard
            # refused it (a template with no media markers on a model served
            # with a projector). The ladder says so in its own phrase.
            inert = (
                f"The override is present but not in force ({origin}). "
                "A template with no media markers is refused on a model served "
                "with a projector."
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
            # The other way a present override loses: read_template_info
            # rejected it for rendering none of the model's stop tokens and
            # walked on to another source, which it logs loudly at load.
            inert = (
                f"The override is present but not in force ({origin}). A "
                "template that renders none of the model's stop tokens is "
                "refused, because it would generate to the token cap."
            )
    return (info.chat_template or None), origin, inert


def write_override(model_path: str, body: str, *, provider: str,
                   config: dict) -> Path:
    """Validate ``body``, then write it as this model's override.

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
    validate(body, provider=provider, config=config)
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise TemplateWriteRefused(f"could not write {path}: {exc}") from exc
    logger.info("[template] wrote chat template override %s (%d chars)",
                path, len(body))
    return path


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


# TWO conversation shapes, and needing two is the point. Templates
# legitimately refuse a shape -- Qwen's official template raises on two
# leading system messages, several refuse a trailing assistant turn -- so a
# raise on ONE shape cannot mean the template is broken. But a template that
# raises on EVERY shape is broken, and passing it would brick the model at its
# next load, which is the outcome this function exists to prevent. So: refuse
# only when nothing renders.
#
# The shapes differ in the two dimensions templates actually branch on -- a
# system message, and whether the last turn is the user's.
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
)


def validate(body: str, *, provider: str, config: dict) -> None:
    """Raise TemplateWriteRefused unless ``body`` compiles AND renders.

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
            refusals.append(f"{type(exc).__name__}: {exc}")
            continue
        except Exception as exc:
            # Not a template-authored refusal: a bad filter, a call into
            # something absent. That is broken on any shape.
            raise TemplateWriteRefused(
                f"template failed to render: {type(exc).__name__}: {exc}"
            ) from exc
        if out.strip():
            rendered = out
            break

    if not rendered:
        detail = (f" It raised on every shape tried: {'; '.join(refusals)}."
                  if refusals else "")
        raise TemplateWriteRefused(
            "template produced no prompt for any ordinary conversation -- the "
            "model would receive an empty prompt or fail at its next load."
            + detail
        )

    if provider == "gguf" and _is_media_served(config):
        from .providers.llama_server_provider import LlamaServerProvider
        if not LlamaServerProvider._template_handles_media(body):
            raise TemplateWriteRefused(
                "this model is served with a projector (mmproj/vision) but the "
                "template has no media markers, so saving it would load the "
                "vision tower and then render prompts that can never reference "
                "an image. The spawn-time guard would refuse it too."
            )


def _is_media_served(config: dict) -> bool:
    from .providers.llama_server_provider import LlamaServerProvider
    return LlamaServerProvider._is_media_served(config)


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
