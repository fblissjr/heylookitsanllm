# tests/unit/test_gguf_metadata.py
"""GGUF header reader (gguf_metadata.py).

Import-time GGUF handling used to infer facts from filenames. This module
reads them from the file, so these tests pin the parser against hand-built
headers -- never modelzoo/, whose real GGUFs are multi-GB and gitignored.

Claims (what breaks if a test is deleted):
- parser tests: the KV walk mis-steps a value type and every key AFTER the
  bad one is read from the wrong offset -- silent garbage, not an exception.
- skip tests: a tokenizer vocab (a megabyte-scale KV array) gets materialised
  as Python objects on every scan of every model, or is stepped over wrongly.
- modality tests: an omni projector's audio tower goes missing again, which
  is the bug that motivated the module.
- spec-type tests: a drafter family maps to the wrong --spec-type, which is a
  llama-server load failure rather than a degraded default.
"""
from __future__ import annotations

import struct

import pytest

from heylook_llm.gguf_metadata import (
    GGUFMetadataError,
    architecture,
    detect_modalities,
    infer_spec_type,
    read_metadata,
    safe_read_metadata,
)

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


def write_gguf(path, kvs, *, version=3, tensor_count=0, magic=b"GGUF"):
    """Write a GGUF file that is header-only: valid KV section, no tensors.

    kvs: list of (key, vtype, value). Order matters -- the reader walks
    sequentially, so tests can place a target key after a value it must skip.
    """
    out = bytearray(magic)
    out += struct.pack("<I", version)
    out += struct.pack("<QQ", tensor_count, len(kvs))
    for key, vtype, value in kvs:
        out += _str(key)
        out += struct.pack("<I", vtype)
        out += _value(vtype, value)
    path.write_bytes(bytes(out))
    return path


@pytest.mark.unit
class TestReadMetadata:
    def test_reads_scalars_strings_and_bools(self, tmp_path):
        f = write_gguf(tmp_path / "m.gguf", [
            ("general.architecture", STR, "deepseek4"),
            ("clip.has_vision_encoder", BOOL, True),
            ("clip.has_audio_encoder", BOOL, False),
            ("deepseek4.block_count", U32, 43),
        ])
        got = read_metadata(f, {
            "general.architecture", "clip.has_vision_encoder",
            "clip.has_audio_encoder", "deepseek4.block_count",
        })
        assert got == {
            "general.architecture": "deepseek4",
            "clip.has_vision_encoder": True,
            "clip.has_audio_encoder": False,
            "deepseek4.block_count": 43,
        }

    def test_absent_key_is_simply_missing(self, tmp_path):
        f = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
        assert read_metadata(f, {"general.architecture", "nope"}) == {
            "general.architecture": "gemma4"}

    @pytest.mark.parametrize("skipped_type,skipped_value", [
        (STR, "a" * 5000),                       # long string
        (ARRAY, (STR, [f"tok{i}" for i in range(500)])),   # string array (vocab)
        (ARRAY, (I32, list(range(2000)))),       # fixed-stride array (token types)
        (ARRAY, (F32, [0.5] * 1000)),            # float array (merges/scores)
        (F32, 1.25),
        (U64, 2 ** 40),
    ])
    def test_target_key_survives_a_skipped_value_of_any_type(
        self, tmp_path, skipped_type, skipped_value
    ):
        # The failure this guards is silent: mis-stepping one value shifts
        # every later read, so the parser returns plausible garbage instead
        # of raising. Putting the target AFTER the skipped value is the point.
        f = write_gguf(tmp_path / "m.gguf", [
            ("tokenizer.ggml.junk", skipped_type, skipped_value),
            ("general.architecture", STR, "qwen35moe"),
        ])
        assert read_metadata(f, {"general.architecture"}) == {
            "general.architecture": "qwen35moe"}

    def test_nested_array_of_arrays_is_stepped_over(self, tmp_path):
        f = write_gguf(tmp_path / "m.gguf", [
            ("weird.nested", ARRAY, (ARRAY, [(I32, [1, 2, 3]), (I32, [4, 5])])),
            ("general.architecture", STR, "clip"),
        ])
        assert architecture(f) == "clip"

    def test_stops_early_once_every_key_is_found(self, tmp_path):
        # A truncated tail proves the walk ended: the reader must not touch
        # bytes past the last requested key. Real files put multi-MB vocabs
        # there, and scanning a model directory should not pay for them.
        f = write_gguf(tmp_path / "m.gguf", [
            ("general.architecture", STR, "gemma4"),
            ("tokenizer.ggml.tokens", ARRAY, (STR, ["x"] * 10)),
        ])
        data = bytearray(f.read_bytes())
        f.write_bytes(bytes(data[: len(data) - 40]))   # lop off the vocab
        assert architecture(f) == "gemma4"

    @pytest.mark.parametrize("mutate,reason", [
        (lambda p: p.write_bytes(b"NOTG" + b"\x00" * 32), "bad magic"),
        (lambda p: p.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\x00" * 16), "v1"),
        (lambda p: p.write_bytes(b"GGUF" + struct.pack("<I", 3)), "truncated header"),
        (lambda p: p.write_bytes(b""), "empty"),
    ])
    def test_malformed_files_raise(self, tmp_path, mutate, reason):
        f = tmp_path / "bad.gguf"
        mutate(f)
        with pytest.raises(GGUFMetadataError):
            read_metadata(f, {"general.architecture"})

    def test_safe_read_swallows_everything(self, tmp_path):
        # Scans walk whatever is on disk -- partial downloads included. One
        # unreadable file must degrade that entry, never abort the scan.
        bad = tmp_path / "partial.gguf"
        bad.write_bytes(b"GGUF")
        assert safe_read_metadata(bad, {"general.architecture"}) == {}
        assert safe_read_metadata(tmp_path / "absent.gguf", {"x"}) == {}
        assert architecture(bad) is None

    def test_implausible_kv_count_is_rejected_not_looped(self, tmp_path):
        # A corrupt count must fail fast rather than spin for 2^64 iterations.
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 2 ** 40))
        with pytest.raises(GGUFMetadataError):
            read_metadata(f, {"general.architecture"})


@pytest.mark.unit
class TestDetectModalities:
    def test_no_projector_is_text_only(self, tmp_path):
        primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
        assert detect_modalities(primary, None) == ["text"]

    def test_vision_only_projector(self, tmp_path):
        primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "qwen35moe")])
        mm = write_gguf(tmp_path / "mmproj.gguf", [
            ("general.architecture", STR, "clip"),
            ("clip.has_vision_encoder", BOOL, True),
            ("clip.has_audio_encoder", BOOL, False),
        ])
        assert detect_modalities(primary, mm) == ["text", "vision"]

    def test_omni_projector_reports_audio(self, tmp_path):
        # THE bug this module exists for: gemma-4's mmproj sets both flags, and
        # "an mmproj exists -> vision" dropped the audio tower on the floor.
        primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
        mm = write_gguf(tmp_path / "mmproj.gguf", [
            ("general.architecture", STR, "clip"),
            ("clip.has_vision_encoder", BOOL, True),
            ("clip.has_audio_encoder", BOOL, True),
        ])
        assert detect_modalities(primary, mm) == ["text", "vision", "audio"]

    def test_unreadable_projector_falls_back_to_vision(self, tmp_path):
        # Degrade to the old presence heuristic rather than silently stripping
        # vision from a model that plainly has a projector.
        primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
        mm = tmp_path / "mmproj.gguf"
        mm.write_bytes(b"not a gguf at all")
        assert detect_modalities(primary, mm) == ["text", "vision"]

    def test_text_is_always_first(self, tmp_path):
        primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
        mm = write_gguf(tmp_path / "mmproj.gguf", [("clip.has_audio_encoder", BOOL, True)])
        assert detect_modalities(primary, mm)[0] == "text"


@pytest.mark.unit
class TestInferSpecType:
    @pytest.mark.parametrize("name,expected", [
        ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", "draft-dspark"),
        ("dflash-something.gguf", "draft-dflash"),
        ("eagle3-gpt-oss.gguf", "draft-eagle3"),
        ("mtp-gemma-4-12B-it.gguf", "draft-mtp"),
        ("MTP-Gemma-4-12B-it.gguf", "draft-mtp"),      # case-insensitive
    ])
    def test_prefix_maps_to_llama_server_spec_type(self, tmp_path, name, expected):
        assert infer_spec_type(tmp_path / name) == expected

    def test_dspark_wins_over_dflash_despite_sharing_an_architecture(self, tmp_path):
        # Both sidecar families report general.architecture == "dflash"; only
        # the NAME says which carries the extra Markov head, which is why the
        # prefix -- llama.cpp's own resolution key -- is the signal.
        assert infer_spec_type(tmp_path / "dspark-x.gguf") == "draft-dspark"
        assert infer_spec_type(tmp_path / "dflash-x.gguf") == "draft-dflash"

    def test_unrecognised_drafter_returns_none(self, tmp_path):
        assert infer_spec_type(tmp_path / "some-drafter.gguf") is None
