"""Template + special-token info read from the model folder (C4.5).

Single source of truth for output-parsing decisions. Reads:

- ``chat_template.jinja`` (newer HF convention) OR the ``chat_template``
  field embedded in ``tokenizer_config.json`` OR ``chat_template.json``
  (processor-side convention). Policy controlled by
  ``MLXModelConfig.chat_template_source``.
- ``added_tokens_decoder`` from ``tokenizer_config.json`` -- every entry
  marked ``"special": true`` becomes part of the special-tokens set.

Downstream consumers (reasoning parser factory, harmony parser's strip
set, observability) read a single ``ModelTemplateInfo`` object instead
of querying the tokenizer / poking at HF internals. This keeps literal
strings like ``"<|channel|>"`` OUT of the detection code -- we learn the
set of special tokens from the model folder, and we detect format by
scanning the template source for structure signals (``<|channel|>``
with channel names == harmony; ``<think>`` == qwen3 thinking). Those
strings appear in our detection code only because they're the literal
tokens in the model's OWN template file, not because we invented them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import orjson


AUTO = "auto"
JINJA = "jinja"
TOKENIZER_CONFIG = "tokenizer_config"
CHAT_TEMPLATE_JSON = "chat_template_json"

def _template_can_stop(body: str, eos_tokens: "frozenset[str]") -> bool:
    """True unless we're CONFIDENT the template can't signal the model to stop.

    A chat template that never renders one of the model's stop tokens can't tell
    the model to stop -> runaway generation (found in the wild: a corrupted gemma
    jinja emitting ``<|turn>model`` with no ``<end_of_turn>`` -- coherent answer,
    then generated to the max_tokens cap). Rather than hardcode a marker list
    (fragile: misses new families, false-rejects exotic ones), we ask the MODEL
    what stops it (``_read_eos_tokens``) and check the template renders at least
    one. Conservative: an empty template, or a model whose stop tokens we can't
    determine, is NOT rejected (never break on uncertainty).
    """
    if not body or not eos_tokens:
        return True
    return any(tok and tok in body for tok in eos_tokens)


@dataclass(frozen=True)
class ModelTemplateInfo:
    """Parsed output of the chat_template + tokenizer_config pair.

    All fields are read-only after construction. Derived flags
    (``has_harmony_structure``, ``has_thinking_markers``) are computed
    once so factory code doesn't re-scan the template.
    """
    chat_template: str = ""
    special_tokens: frozenset[str] = frozenset()
    template_source: str = AUTO
    has_harmony_structure: bool = False
    has_thinking_markers: bool = False


# Signal patterns learned from real model templates (not hardcoded
# semantics -- these are the literal strings the templates USE).
_HARMONY_CHANNEL_PATTERN = re.compile(r"<\|channel\|>")
_HARMONY_MESSAGE_PATTERN = re.compile(r"<\|message\|>")
_THINK_OPEN_PATTERN = re.compile(r"<think>")
_THINK_CLOSE_PATTERN = re.compile(r"</think>")


def read_template_info(
    model_dir: Path, source: Optional[str]
) -> ModelTemplateInfo:
    """Read ``chat_template`` + ``added_tokens_decoder`` from the model dir.

    ``source`` values (from ``MLXModelConfig.chat_template_source``):
      - ``None`` / ``"auto"``: prefer ``chat_template.jinja`` if present,
        fall back to the embedded template, then to ``chat_template.json``
        (the processor-side convention some VLM conversions use).
      - ``"jinja"``: force-load ``chat_template.jinja``.
      - ``"tokenizer_config"``: force the embedded template.
      - absolute path to a ``.jinja`` file: load that file.

    Never raises -- malformed configs log warnings and fall back to empty
    strings, so model load survives a broken config file.
    """
    model_dir = Path(model_dir)
    special_tokens = _read_special_tokens(model_dir)
    eos_tokens = _read_eos_tokens(model_dir)
    template, template_source = _read_template(model_dir, source)

    # Robustness: reject a template that renders none of the model's OWN stop
    # tokens (it would run away). Walk the remaining file sources for a valid one;
    # if none, install NOTHING (empty template -> install_chat_template no-ops ->
    # the loader's built-in template stands) rather than force-install a broken file.
    if template and not _template_can_stop(template, eos_tokens):
        logging.warning(
            "chat template for %s (source=%s) renders none of the model's stop "
            "tokens %s -- it would generate until the max_tokens cap. Trying other sources.",
            model_dir.name, template_source, sorted(eos_tokens),
        )
        template, template_source = "", "none(stopless)"
        for alt in (TOKENIZER_CONFIG, CHAT_TEMPLATE_JSON):
            alt_body, alt_source = _read_template(model_dir, alt)
            if alt_body and _template_can_stop(alt_body, eos_tokens):
                template, template_source = alt_body, alt_source
                break
        if not template:
            logging.warning(
                "No file chat template with a stop token for %s -- installing none; "
                "the loader's built-in template will be used. Fix chat_template.jinja "
                "or set chat_template_source in models.toml.", model_dir.name,
            )

    has_harmony = bool(
        _HARMONY_CHANNEL_PATTERN.search(template)
        and _HARMONY_MESSAGE_PATTERN.search(template)
    )
    has_thinking = bool(
        _THINK_OPEN_PATTERN.search(template)
        and _THINK_CLOSE_PATTERN.search(template)
    )

    return ModelTemplateInfo(
        chat_template=template,
        special_tokens=special_tokens,
        template_source=template_source,
        has_harmony_structure=has_harmony,
        has_thinking_markers=has_thinking,
    )


def _read_template(model_dir: Path, source: Optional[str]) -> tuple[str, str]:
    """Return ``(template_body, source_label)``."""
    normalized = (source or "").strip().lower() or AUTO

    jinja_path = model_dir / "chat_template.jinja"
    config_path = model_dir / "tokenizer_config.json"

    if normalized == JINJA:
        body = _read_file(jinja_path)
        if body is not None:
            return body, JINJA
        logging.warning(
            "chat_template_source='jinja' requested but %s missing; "
            "falling back to auto", jinja_path,
        )
        normalized = AUTO

    if normalized == TOKENIZER_CONFIG:
        body = _read_embedded_template(config_path)
        if body is not None:
            return body, TOKENIZER_CONFIG
        logging.warning(
            "chat_template_source='tokenizer_config' but %s lacks a "
            "chat_template field; falling back to auto", config_path,
        )
        normalized = AUTO

    if normalized == CHAT_TEMPLATE_JSON:
        body = _read_embedded_template(model_dir / "chat_template.json")
        if body is not None:
            return body, CHAT_TEMPLATE_JSON
        logging.warning(
            "chat_template_source='chat_template_json' but %s has no usable "
            "chat_template.json; falling back to auto", model_dir,
        )
        normalized = AUTO

    # Absolute path override.
    if source and normalized not in (AUTO, JINJA, TOKENIZER_CONFIG):
        candidate = Path(source)
        if candidate.is_absolute() and candidate.is_file():
            body = _read_file(candidate)
            if body is not None:
                return body, str(candidate)
        logging.warning(
            "chat_template_source=%r not recognized; falling back to auto",
            source,
        )

    # Auto: jinja wins when both present.
    body = _read_file(jinja_path)
    if body is not None:
        return body, JINJA
    body = _read_embedded_template(config_path)
    if body is not None:
        return body, TOKENIZER_CONFIG
    # Last resort: processor-side chat_template.json. The tokenizer never
    # loads this file, so a model shipping ONLY it would otherwise render
    # template-less at the tokenizer level.
    body = _read_embedded_template(model_dir / "chat_template.json")
    if body is not None:
        return body, CHAT_TEMPLATE_JSON
    return "", AUTO


def detect_chat_template_source(model_dir) -> Optional[str]:
    """Import-time detection shared by BOTH import paths (CLI wizard and the
    /v1/admin import route): record ``"jinja"`` when the folder ships a
    ``chat_template.jinja``, else leave the source unset (auto).

    Expands ``~`` -- the admin API accepts arbitrary path strings.
    """
    model_dir = Path(model_dir).expanduser()
    jinja_path = model_dir / "chat_template.jinja"
    if jinja_path.is_file():
        # Only prefer the jinja if it actually renders a stop token. A broken/
        # corrupted jinja (no <end_of_turn>/<|im_end|>/...) would force-install a
        # stop-less template -> runaway generation. If it's stop-less, don't record
        # `jinja` -- leave the source unset (auto), and the load-time guard in
        # read_template_info also rejects it.
        body = _read_file(jinja_path)
        if body and _template_can_stop(body, _read_eos_tokens(model_dir)):
            return JINJA
        logging.warning(
            "chat_template.jinja in %s renders none of the model's stop tokens -- "
            "NOT recording it as chat_template_source (it would run away). "
            "Leaving source=auto.", model_dir.name,
        )
    return None


def is_explicit_source(source: Optional[str]) -> bool:
    """True when ``chat_template_source`` names an explicit template choice.

    ``None``/empty/``"auto"`` are NOT explicit: auto means fill-only-when-
    missing, never force-install (a force with auto's resolution could clobber
    a natively-loaded dict of named templates).
    """
    if not source:
        return False
    return source.strip().lower() not in ("", AUTO)


def missing_template_error(tokenizer, model_id: Optional[str] = None) -> Optional[ValueError]:
    """Return an actionable error if the tokenizer truly has no chat template,
    else None.

    Call from an ``except ValueError`` around ``apply_chat_template`` --
    transformers raises a raw ValueError there whose message would otherwise
    surface verbatim as the HTTP error detail. The decision is made from
    tokenizer STATE, not by matching upstream error prose (version-fragile):
    ``chat_template`` covers HF templates; ``has_chat_template`` covers
    mlx-lm's wrapper-level python templates (``chat_template_type``), which
    render fine while the HF attribute stays None.
    """
    if getattr(tokenizer, "chat_template", None):
        return None
    if getattr(tokenizer, "has_chat_template", False):
        return None
    who = f"Model '{model_id}'" if model_id else "This model"
    return ValueError(
        f"{who} has no chat template: none of chat_template.jinja, a "
        f"tokenizer_config.json 'chat_template' field, or chat_template.json "
        f"loaded from the model folder. Add one of those files or set "
        f"chat_template_source in models.toml."
    )


def install_chat_template(tokenizer, info: ModelTemplateInfo, *, force: bool) -> bool:
    """Attach the resolved template to a live tokenizer (and its inner
    ``_tokenizer`` for mlx-lm's TokenizerWrapper).

    ``force=True`` (explicit ``chat_template_source``): always overwrite --
    the registry entry is authoritative.
    ``force=False`` (auto): only fill in a MISSING tokenizer template.
    Covers models whose template lives somewhere AutoTokenizer doesn't look
    (e.g. only ``chat_template.json``) without stomping on what transformers
    loaded natively.

    Returns True if a template was installed. Never raises.
    """
    if tokenizer is None or not info.chat_template:
        return False
    if not force and getattr(tokenizer, "chat_template", None):
        return False
    inner = getattr(tokenizer, "_tokenizer", None)
    targets = [tokenizer] if inner is None or inner is tokenizer else [tokenizer, inner]
    installed = False
    for target in targets:
        try:
            target.chat_template = info.chat_template
            installed = True
        except (AttributeError, TypeError) as exc:
            logging.debug("could not install chat_template on %r: %s", type(target), exc)
    return installed


def _read_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return path.read_text()
    except OSError as exc:
        logging.warning("template_info: cannot read %s: %s", path, exc)
        return None


def _read_embedded_template(config_path: Path) -> Optional[str]:
    data = _read_json(config_path)
    if not isinstance(data, dict):
        return None
    template = data.get("chat_template")
    if isinstance(template, str) and template:
        return template
    return None


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        logging.warning("template_info: cannot parse %s: %s", path, exc)
        return None


def _read_eos_tokens(model_dir: Path) -> frozenset[str]:
    """The model's OWN declared stop tokens as strings -- NOT hardcoded.

    Unions ``eos_token`` (string) and every id in ``eos_token_id`` (from
    ``tokenizer_config.json`` and ``generation_config.json``, either of which may
    be a scalar or a list), resolving ids to strings via ``added_tokens_decoder``.
    This is what generation actually stops on, straight from the model's files, so
    a template is validated against the model's real stop set rather than a guess.
    Empty when we can't determine them (callers then don't reject -- see
    ``_template_can_stop``).
    """
    tcfg = _read_json(model_dir / "tokenizer_config.json")
    gcfg = _read_json(model_dir / "generation_config.json")
    tcfg = tcfg if isinstance(tcfg, dict) else {}
    gcfg = gcfg if isinstance(gcfg, dict) else {}

    out: set[str] = set()

    def _tok_str(v) -> Optional[str]:
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and isinstance(v.get("content"), str):
            return v["content"]
        return None

    eos_str = _tok_str(tcfg.get("eos_token"))
    if eos_str:
        out.add(eos_str)

    # id -> string map from added_tokens_decoder (id-keyed dict)
    id_to_str: dict[str, str] = {}
    decoder = tcfg.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for tid, spec in decoder.items():
            s = _tok_str(spec)
            if s:
                id_to_str[str(tid)] = s

    for src in (tcfg.get("eos_token_id"), gcfg.get("eos_token_id")):
        ids = src if isinstance(src, list) else ([src] if src is not None else [])
        for i in ids:
            s = id_to_str.get(str(i))
            if s:
                out.add(s)

    return frozenset(out)


def _read_special_tokens(model_dir: Path) -> frozenset[str]:
    """Union the special-token sets from both tokenizer files.

    - ``tokenizer.json`` (the fast-tokenizer's own state) has an
      ``added_tokens`` array -- authoritative for fast tokenizers.
    - ``tokenizer_config.json`` has ``added_tokens_decoder`` -- some
      models list specials only here, some list them in both.

    Only tokens with ``"special": true`` count. We never hardcode which
    tokens the model considers special -- the files are the source of
    truth.
    """
    specials: set[str] = set()

    # tokenizer.json's added_tokens array
    tok_json = _read_json(model_dir / "tokenizer.json")
    if isinstance(tok_json, dict):
        added = tok_json.get("added_tokens")
        if isinstance(added, list):
            for entry in added:
                if not isinstance(entry, dict):
                    continue
                if not entry.get("special", False):
                    continue
                content = entry.get("content")
                if isinstance(content, str) and content:
                    specials.add(content)

    # tokenizer_config.json's added_tokens_decoder (id-keyed dict)
    config = _read_json(model_dir / "tokenizer_config.json")
    if isinstance(config, dict):
        decoder = config.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            for entry in decoder.values():
                if not isinstance(entry, dict):
                    continue
                if not entry.get("special", False):
                    continue
                content = entry.get("content")
                if isinstance(content, str) and content:
                    specials.add(content)

    return frozenset(specials)
