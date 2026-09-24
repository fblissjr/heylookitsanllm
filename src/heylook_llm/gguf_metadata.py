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
import re
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


# clip.cpp reads the generic key first and falls back to the vision-specific
# one, which mixed-modality (omni) projectors carry instead.
_PROJ_TYPE_KEY = "clip.projector_type"
_VISION_PROJ_TYPE_KEY = "clip.vision.projector_type"


def vision_projector_type(mmproj: Path) -> Optional[str]:
    """The projector's vision type name as llama.cpp reads it (e.g.
    ``qwen3vl_merger``, ``gemma4v``, ``deepseek4v``), or None if unreadable."""
    meta = safe_read_metadata(mmproj, {_PROJ_TYPE_KEY, _VISION_PROJ_TYPE_KEY})
    return meta.get(_PROJ_TYPE_KEY) or meta.get(_VISION_PROJ_TYPE_KEY)


_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"


# The GGUF spec's "Recommended Sampler Parameters" block. Converters write it
# from the HF repo's generation_config.json (gguf-py/gguf/metadata.py), which
# is the SAME source the MLX side reads off disk -- so this is the gguf
# spelling of the existing vendor layer, not a new concept.
#
# Only the three keys the vendor layer takes are read. The spec also defines
# min_p, xtc_*, penalty_*, mirostat* and a sampler `sequence`; none of the
# files here carry them, because generation_config.json does not.
_VENDOR_SAMPLING_KEYS = {
    "temperature": "general.sampling.temp",
    "top_p": "general.sampling.top_p",
    "top_k": "general.sampling.top_k",
}


def vendor_sampling(primary: Path) -> dict[str, Any]:
    """The model's own recommended decode settings, from the GGUF header.

    llama.cpp reads these itself at load (`common_init_sampler_from_model`)
    for any key not set on the CLI -- but heylook sends every one of them
    explicitly on every request, so the baked-in values never survived to the
    sampler. That silently contradicted the file: a Qwen3.6 GGUF asks for
    top_k 20 and gemma-4 for 64, while heylook's floor sends 0.

    Best-effort like its MLX sibling: an unreadable header, a missing key or a
    non-numeric value yields {} / is dropped, because a vendor hint must never
    block a load.
    """
    found = safe_read_metadata(primary, set(_VENDOR_SAMPLING_KEYS.values()))
    out: dict[str, Any] = {}
    for our_key, gguf_key in _VENDOR_SAMPLING_KEYS.items():
        value = found.get(gguf_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # The header stores these as float32; widening to a Python float
            # exposes the representation error, so a top_p the publisher wrote
            # as a short decimal reads back with a long expansion tail.
            # Harmless to the sampler, NOT harmless on screen: the settings
            # panel prints this value as the placeholder of a `step=0.01`
            # field. The rounding RECOVERS the publisher's own decimal rather
            # than approximating it -- float32 round-trips any decimal written
            # within the precision it guarantees, and sampling values are
            # written well inside that (checked against generated
            # publisher-shaped decimals, none of which failed to recover).
            # Ints (top_k) pass through untouched.
            out[our_key] = round(value, 6) if isinstance(value, float) else value
    return out


def chat_template(primary: Path) -> Optional[str]:
    """The chat template EMBEDDED in the GGUF header, or None if it has none.

    This is the bottom rung of the gguf template ladder and the thing an
    override is an override OF, so a reader has to be able to see it without
    loading the model -- which is also why it goes through the header reader
    rather than llama-server's ``/props`` (that needs a running process, and a
    preview must never load one).

    None is a real answer: an MTP/drafter head legitimately carries no
    template.
    """
    return safe_read_metadata(primary, {_CHAT_TEMPLATE_KEY}).get(_CHAT_TEMPLATE_KEY) or None


def supports_thinking(primary: Path) -> Optional[bool]:
    """Whether the GGUF's embedded chat template references ``enable_thinking``.

    Same signal the MLX path uses (`template_info._ENABLE_THINKING_PATTERN`),
    applied to the template GGUF carries in its own metadata rather than to a
    `chat_template.jinja` on disk. The kwarg is the cross-model thinking
    mechanism -- transformers forwards extra apply_chat_template kwargs as
    template variables, so a template that mentions the variable is a template
    that can switch thinking -- which makes "mentions it" the capability
    signal on both engines.

    Returns None when there is no template to judge (an MTP/drafter head
    legitimately has none), so callers can leave `supports_thinking` unset
    rather than assert a false.
    """
    from .providers.common.template_info import _ENABLE_THINKING_PATTERN

    template = safe_read_metadata(primary, {_CHAT_TEMPLATE_KEY}).get(_CHAT_TEMPLATE_KEY)
    if not template:
        return None
    return bool(_ENABLE_THINKING_PATTERN.search(template))


def read_header(path: Path, keys: set[str]) -> tuple[dict[str, Any], list[str]]:
    """The requested KV entries AND every tensor name in ``path``'s header.

    Walks the whole KV section (skipping what is not asked for), then the
    tensor-info table, which follows it. No tensor data is read: a
    multi-hundred-GB split costs one header per file.
    """
    found: dict[str, Any] = {}
    with open(path, "rb") as fp:
        if _read_exact(fp, 4) != _MAGIC:
            raise GGUFMetadataError(f"not a GGUF file: {path.name}")
        version, = struct.unpack("<I", _read_exact(fp, 4))
        if version not in (2, 3):
            raise GGUFMetadataError(f"unsupported GGUF version {version} in {path.name}")
        tensor_count, kv_count = struct.unpack("<QQ", _read_exact(fp, 16))
        if kv_count > 1_000_000 or tensor_count > 10_000_000:
            raise GGUFMetadataError(f"implausible GGUF counts in {path.name}")
        for _ in range(kv_count):
            key = _read_string(fp)
            (vtype,) = struct.unpack("<I", _read_exact(fp, 4))
            if key in keys:
                found[key] = _read_value(fp, vtype)
            else:
                _skip_value(fp, vtype)
        names = []
        for _ in range(tensor_count):
            names.append(_read_string(fp))
            (n_dims,) = struct.unpack("<I", _read_exact(fp, 4))
            fp.seek(8 * n_dims + 4 + 8, 1)  # dims, ggml type, data offset
    return found, names


# llama.cpp split naming: `<prefix>-00001-of-00005.gguf`.
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def splits(first: Path) -> list[Path]:
    """Every split of the model whose first split (or only file) is ``first``,
    in order; ``[first]`` when it is not split."""
    m = SPLIT_RE.search(first.name)
    if m is None:
        return [first]
    prefix, total = first.name[: m.start()], m.group(2)
    return [first.with_name(f"{prefix}-{i:05d}-of-{total}.gguf") for i in range(1, int(total) + 1)]


# (resolved path, mtime_ns, size) per file -> spec type. A discovery scan asks
# this for every gguf model on every rescan; the answer changes only with the
# files.
_spec_type_cache: dict[tuple, Optional[str]] = {}


def spec_type_from_gguf(files: list[Path]) -> Optional[str]:
    """The speculative type a GGUF offers, by llama.cpp's own rule, or None.

    The rule is ``common_speculative_types_from_gguf`` in the build's
    common/speculative.cpp (pinned by test_spec_rule_matches_the_build):
    architecture ``dflash`` is draft-dspark when it carries
    ``markov_w1.weight`` and draft-dflash otherwise; any other architecture is
    draft-mtp when its LAST block has ``nextn.eh_proj.weight``. The same rule
    answers for a drafter file and for a target whose MTP head is built into
    its weights.

    ``files`` is every split of one model, first split first. llama.cpp reads
    only the first split (so a sharded drafter needs ``--spec-type``); this
    reads every split's tensor table, because a built-in head can sit in the
    last one. An unreadable file is None, never raised.
    """
    try:
        key = tuple((str(f.resolve()), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    except OSError:
        return None
    if key in _spec_type_cache:
        return _spec_type_cache[key]
    answer = None
    try:
        kv, names = read_header(files[0], {_ARCH_KEY})
        arch = kv.get(_ARCH_KEY)
        if arch == "dflash":
            answer = "draft-dspark" if "markov_w1.weight" in names else "draft-dflash"
        elif arch:
            kv, _ = read_header(files[0], {f"{arch}.block_count"})
            blocks = kv.get(f"{arch}.block_count")
            for f in files[1:]:
                names += read_header(f, set())[1]
            if isinstance(blocks, int) and f"blk.{blocks - 1}.nextn.eh_proj.weight" in names:
                answer = "draft-mtp"
    except (GGUFMetadataError, OSError, struct.error, IndexError) as e:
        logging.debug(f"[GGUF] could not read tensors from {files[0].name if files else '?'}: {e}")
    _spec_type_cache[key] = answer
    return answer


def model_names(primary: Path) -> set[str]:
    """The names a GGUF gives its own model, normalized for matching a drafter
    to its target: ``general.name``, ``general.basename`` and
    ``general.base_model.0.name``, lowercased with everything but letters and
    digits removed (publishers differ on case and separators for one model)."""
    keys = {"general.name", "general.basename", "general.base_model.0.name"}
    out = set()
    for v in safe_read_metadata(primary, keys).values():
        if isinstance(v, str) and (n := re.sub(r"[^a-z0-9]", "", v.lower())):
            out.add(n)
    return out


# ---------------------------------------------------------------------------
# Training context
# ---------------------------------------------------------------------------

# (resolved path, mtime_ns, size) -> context_length. The admin list route
# builds one row per served model on every request, and a GGUF header read
# is a file open plus a short KV walk -- small, but it is disk, per row, per
# request, on the route that was moved off the event loop because its per-row
# cost is real. The file's identity is the cache key, so a re-download or a
# re-quant at the same path is seen; the process never re-reads an unchanged
# file.
_context_length_cache: dict[tuple, Optional[int]] = {}


def context_length(primary: Path) -> Optional[int]:
    """The model's TRAINING context, ``<arch>.context_length``, or None.

    llama-server sizes its context from this value when no ``--ctx-size`` is
    passed (``-c 0`` = loaded from model, then ``--fit`` shrinks it to what
    device memory holds), so it is the ceiling a context-size control should
    offer -- asking for more than the model was trained on buys nothing.

    Two header reads, not one: the key is architecture-prefixed
    (``deepseek4.context_length``, ``gemma4.context_length``), so the
    architecture has to be known before the key can be named. Both stop at
    the first matching key, and ``general.architecture`` is among the first
    entries every converter writes. None for anything unreadable, missing,
    or not a positive integer -- a control that cannot learn the ceiling
    offers its generic steps rather than refusing to render.
    """
    try:
        st = primary.stat()
        key = (str(primary.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _context_length_cache:
        return _context_length_cache[key]
    arch = architecture(primary)
    value: Optional[int] = None
    if arch:
        ctx_key = f"{arch}.context_length"
        raw = safe_read_metadata(primary, {ctx_key}).get(ctx_key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            value = raw
    _context_length_cache[key] = value
    return value


def memory_kind(primary: Path) -> Optional[str]:
    """How the model's KV/state can be rolled back, read off the header:
    ``hybrid_recurrent`` (``<arch>.ssm.*`` keys: a recurrent state that
    cannot be truncated), ``sliding_window`` (``<arch>.attention
    .sliding_window``), else ``full_attention``. None when unreadable.

    llama.cpp decides checkpointing at load from the memory it builds, so
    this is the header's answer, not the runtime's; a model with neither key
    (a sparse-attention variant, say) may still checkpoint.
    """
    arch = architecture(primary)
    if not arch:
        return None
    meta = safe_read_metadata(primary, {f"{arch}.ssm.state_size",
                                        f"{arch}.ssm.conv_kernel",
                                        f"{arch}.attention.sliding_window"})
    if f"{arch}.ssm.state_size" in meta or f"{arch}.ssm.conv_kernel" in meta:
        return "hybrid_recurrent"
    if f"{arch}.attention.sliding_window" in meta:
        return "sliding_window"
    return "full_attention"
