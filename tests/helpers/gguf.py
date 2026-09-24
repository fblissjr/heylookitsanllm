# tests/helpers/gguf.py
"""Hand-built GGUF headers for tests.

ONE copy, because two files need it: `test_gguf_metadata.py` pins the parser
against these, and `test_llama_server_provider.py` needs a real header to prove
the vendor sampling layer reaches the request body. The provider test reached
across into the other test module for a release, which is not importable under
this suite's `helpers.`-rooted convention and is the hand-copied second copy
this repo derives away everywhere else.

Never the real model folders: their GGUFs are multi-GB.
"""
from __future__ import annotations

import struct

# GGUFValueType codes (gguf.constants)
U32, I32, F32, BOOL, STR, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10


def _str(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(vtype: int, value) -> bytes:
    if vtype == STR:
        return _str(value)
    if vtype == BOOL:
        return struct.pack("<?", value)
    if vtype == U32:
        return struct.pack("<I", value)
    if vtype == I32:
        return struct.pack("<i", value)
    if vtype == F32:
        return struct.pack("<f", value)
    if vtype == U64:
        return struct.pack("<Q", value)
    if vtype == ARRAY:
        elem_type, items = value
        body = b"".join(_value(elem_type, i) for i in items)
        return struct.pack("<IQ", elem_type, len(items)) + body
    raise AssertionError(f"unhandled type {vtype}")


def write_gguf(path, kvs, *, version=3, tensor_count=None, magic=b"GGUF", tensors=()):
    """Write a GGUF file that is header-only: valid KV section, then tensor
    INFOS for ``tensors`` (names; one F32 dim each) and no tensor data.

    kvs: list of (key, vtype, value). Order matters -- the reader walks
    sequentially, so tests can place a target key after a value it must skip.
    """
    out = bytearray(magic)
    out += struct.pack("<I", version)
    count = len(tensors) if tensor_count is None else tensor_count
    out += struct.pack("<QQ", count, len(kvs))
    for key, vtype, value in kvs:
        out += _str(key)
        out += struct.pack("<I", vtype)
        out += _value(vtype, value)
    for k, name in enumerate(tensors):
        out += _str(name) + struct.pack("<IQIQ", 1, 1, 0, 32 * k)
    path.write_bytes(bytes(out))
    return path
