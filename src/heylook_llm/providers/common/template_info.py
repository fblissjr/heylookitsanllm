"""Template + special-token info read from the model folder (C4.5).

Single source of truth for output-parsing decisions. Reads:

- ``chat_template.jinja`` (newer HF convention) OR the ``chat_template``
  field embedded in ``tokenizer_config.json``. Policy controlled by
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
    template, template_source = _read_template(model_dir, source)
    special_tokens = _read_special_tokens(model_dir)

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
