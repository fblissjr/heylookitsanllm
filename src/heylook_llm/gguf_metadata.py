"""Read a GGUF file's metadata header -- the facts the filename only implies.

Import-time GGUF handling used to infer everything from filenames: vision from
"is there an mmproj sidecar", modality from nothing at all (`model_importer`
carried a standing note that audio "would need reading the GGUF's own metadata
(out of scope here)"), and the speculative family from a `mtp-` prefix that
missed three of the four families llama.cpp actually resolves. The file says
all of it directly.

Only the KV header is read, and only its head: parsing stops as soon as every
requested key is found, so this never walks the tensor table. That matters --
the primary of a sharded model is a ~5 MB index shard but an mmproj or drafter
is multi-GB, and metadata sits at the front of all of them.

Deliberately stdlib-only (struct + a file handle), matching the gguf provider's
own no-extra-deps property. The format is a stable, versioned header: magic,
version, tensor count, KV count, then length-prefixed key/typed-value pairs.
The upstream `gguf` PyPI package reads the same bytes plus the whole tensor
index, which is exactly the part worth not paying for here.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Optional

_MAGIC = b"GGUF"

# gguf.constants.GGUFValueType
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARRAY, _U64, _I64, _F64 = range(13)

# type -> (struct format, byte width). ARRAY and STRING are length-prefixed and
# handled separately.
_SCALARS: dict[int, tuple[str, int]] = {
    _U8: ("<B", 1), _I8: ("<b", 1),
    _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4),
    _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# A cap on how much header we will read before giving up. Real GGUF headers are
# tens of KB; a tokenizer-heavy one can reach a few MB (the vocab lives in KV
# arrays). Past this we are almost certainly parsing garbage.
_MAX_HEADER_BYTES = 64 * 1024 * 1024


class GGUFMetadataError(Exception):
    """Unreadable or malformed GGUF header."""


def _read_exact(fp, n: int) -> bytes:
    buf = fp.read(n)
    if len(buf) != n:
        raise GGUFMetadataError("truncated GGUF header")
    return buf


def _read_scalar(fp, vtype: int) -> Any:
    fmt, width = _SCALARS[vtype]
    return struct.unpack(fmt, _read_exact(fp, width))[0]


def _read_string(fp) -> str:
    (length,) = struct.unpack("<Q", _read_exact(fp, 8))
    if length > _MAX_HEADER_BYTES:
        raise GGUFMetadataError(f"implausible GGUF string length {length}")
    return _read_exact(fp, length).decode("utf-8", errors="replace")


def _skip_value(fp, vtype: int) -> None:
    """Advance past a value without materialising it.

    The point of the whole module: token vocabularies are megabyte-scale KV
    arrays, and we never want them -- but we do have to step over them to reach
    the keys that come after.
    """
    if vtype == _STR:
        (length,) = struct.unpack("<Q", _read_exact(fp, 8))
        fp.seek(length, 1)
    elif vtype == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(fp, 4))
        (count,) = struct.unpack("<Q", _read_exact(fp, 8))
        if elem_type in _SCALARS:
            fp.seek(_SCALARS[elem_type][1] * count, 1)  # fixed stride: one seek
        else:
            for _ in range(count):
                _skip_value(fp, elem_type)
    elif vtype in _SCALARS:
        fp.seek(_SCALARS[vtype][1], 1)
    else:
        raise GGUFMetadataError(f"unknown GGUF value type {vtype}")


def _read_value(fp, vtype: int) -> Any:
    if vtype == _STR:
        return _read_string(fp)
    if vtype in _SCALARS:
        return _read_scalar(fp, vtype)
    if vtype == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(fp, 4))
        (count,) = struct.unpack("<Q", _read_exact(fp, 8))
        return [_read_value(fp, elem_type) for _ in range(count)]
    raise GGUFMetadataError(f"unknown GGUF value type {vtype}")


def read_metadata(path: Path, keys: set[str]) -> dict[str, Any]:
    """The requested KV entries from ``path``'s header. Missing keys are absent.

    Stops as soon as all of ``keys`` are found. Values not in ``keys`` are
    skipped without being decoded, so a multi-MB tokenizer array costs one
    seek rather than a list of a hundred thousand Python strings.

    Raises :class:`GGUFMetadataError` on a non-GGUF or malformed file; callers
    that prefer a default should use :func:`safe_read_metadata`.
    """
    found: dict[str, Any] = {}
    with open(path, "rb") as fp:
        if _read_exact(fp, 4) != _MAGIC:
            raise GGUFMetadataError(f"not a GGUF file: {path.name}")
        version, = struct.unpack("<I", _read_exact(fp, 4))
        if version not in (2, 3):
            # v1 predates the current header layout; a future v4 may move things.
            raise GGUFMetadataError(f"unsupported GGUF version {version} in {path.name}")
        _tensor_count, kv_count = struct.unpack("<QQ", _read_exact(fp, 16))
        if kv_count > 1_000_000:
            raise GGUFMetadataError(f"implausible GGUF kv_count {kv_count}")

        for _ in range(kv_count):
            key = _read_string(fp)
            (vtype,) = struct.unpack("<I", _read_exact(fp, 4))
            if key in keys:
                found[key] = _read_value(fp, vtype)
                if len(found) == len(keys):
                    return found
            else:
                _skip_value(fp, vtype)
    return found


def safe_read_metadata(path: Path, keys: set[str]) -> dict[str, Any]:
    """:func:`read_metadata`, but an unreadable file yields ``{}``.

    Import scans walk whatever is on disk, including partial downloads and
    files that merely end in ``.gguf``. One bad file must degrade that entry's
    detection, never abort the scan.
    """
    try:
        return read_metadata(path, keys)
    except (GGUFMetadataError, OSError, struct.error) as e:
        logging.debug(f"[GGUF] could not read metadata from {path.name}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------

_ARCH_KEY = "general.architecture"
_VISION_KEY = "clip.has_vision_encoder"
_AUDIO_KEY = "clip.has_audio_encoder"


def architecture(path: Path) -> Optional[str]:
    """``general.architecture`` (e.g. ``deepseek4``, ``clip``, ``dflash``)."""
    return safe_read_metadata(path, {_ARCH_KEY}).get(_ARCH_KEY)


def detect_modalities(primary: Path, mmproj: Optional[Path] = None) -> list[str]:
    """Author-declared modalities, ``text`` always first.

    The projector is the ground truth and it declares vision and audio
    SEPARATELY (``clip.has_vision_encoder`` / ``clip.has_audio_encoder``).
    Presence of an mmproj was previously read as "vision", which mislabels
    every omni projector: gemma-4's mmproj sets both flags, so its audio
    tower was invisible to a mmproj-presence check.
    """
    mods = ["text"]
    if mmproj is None:
        return mods
    meta = safe_read_metadata(mmproj, {_VISION_KEY, _AUDIO_KEY})
    if not meta:
        # Unreadable projector: fall back to the old presence heuristic rather
        # than silently dropping vision from a model that clearly has it.
        return ["text", "vision"]
    if meta.get(_VISION_KEY):
        mods.append("vision")
    if meta.get(_AUDIO_KEY):
        mods.append("audio")
    return mods


# Drafter filename prefix -> llama-server --spec-type. The prefix is llama.cpp's
# OWN sibling-resolution key (common/download.cpp find_best_sibling), which makes
# it the authoritative signal; `general.architecture` corroborates but does not
# discriminate -- both dflash- and dspark- sidecars carry arch "dflash", and only
# the name says which one has the extra Markov head.
_SPEC_TYPE_BY_PREFIX = (
    ("dspark-", "draft-dspark"),
    ("dflash-", "draft-dflash"),
    ("eagle3-", "draft-eagle3"),
    ("mtp-", "draft-mtp"),
)


def infer_spec_type(drafter: Path) -> Optional[str]:
    """The ``--spec-type`` this drafter REQUIRES, or None if unrecognised.

    A fact about the file, not a recommendation: whether speculative decoding
    pays off is measured per model (draft-accept rates vary widely), so the
    importer reports this rather than writing ``spec_type`` for you.
    """
    name = drafter.name.lower()
    for prefix, spec_type in _SPEC_TYPE_BY_PREFIX:
        if name.startswith(prefix):
            return spec_type
    return None
