# tests/unit/test_gguf_metadata.py
"""GGUF header reader (gguf_metadata.py).

Import-time GGUF handling used to infer facts from filenames. This module
reads them from the file, so these tests pin the parser against hand-built
headers -- never the real model folders, whose GGUFs are multi-GB.

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
    context_length,
    detect_modalities,
    spec_type_from_gguf,
    splits,
    read_metadata,
    safe_read_metadata,
    supports_thinking,
)

from helpers.gguf import (  # noqa: E402
    U32, I32, F32, BOOL, STR, ARRAY, U64, write_gguf,
)


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


_UNREADABLE = object()  # a file that is not a gguf at all


# detect_modalities: no projector = text; the projector's own flags add
# vision/audio; text is always first.
# - omni_reports_audio: THE bug this module exists for: gemma-4's mmproj sets
#   both flags, and "an mmproj exists -> vision" dropped the audio tower.
# - unreadable_falls_back_to_vision: degrade to the old presence heuristic
#   rather than silently stripping vision from a model with a projector.
@pytest.mark.unit
@pytest.mark.parametrize("projector, expected", [
    (None, ["text"]),
    ([("general.architecture", STR, "clip"),
      ("clip.has_vision_encoder", BOOL, True),
      ("clip.has_audio_encoder", BOOL, False)], ["text", "vision"]),
    ([("general.architecture", STR, "clip"),
      ("clip.has_vision_encoder", BOOL, True),
      ("clip.has_audio_encoder", BOOL, True)], ["text", "vision", "audio"]),
    (_UNREADABLE, ["text", "vision"]),
    ([("clip.has_audio_encoder", BOOL, True)], ["text", "audio"]),
], ids=["no_projector_text_only", "vision_only_projector", "omni_reports_audio",
        "unreadable_falls_back_to_vision", "text_always_first"])
def test_detect_modalities(tmp_path, projector, expected):
    primary = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "gemma4")])
    if projector is None:
        mm = None
    elif projector is _UNREADABLE:
        mm = tmp_path / "mmproj.gguf"
        mm.write_bytes(b"not a gguf at all")
    else:
        mm = write_gguf(tmp_path / "mmproj.gguf", projector)
    got = detect_modalities(primary, mm)
    assert got[0] == "text"
    assert got == expected


# supports_thinking: the same signal the MLX path uses, read from GGUF's
# embedded template. It was a hand-set flag because GGUF metadata had
# "nothing cheap to probe"; the header read is that probe. One rule, two
# engines: every row with a template is also checked against template_info's
# own pattern, so the two paths cannot drift apart.
# - no_template_is_none_not_false: an MTP/drafter head legitimately carries no
#   chat template. None lets the importer leave the field unset; False would
#   assert a capability judgement it has no basis for.
# - word_boundary_not_substring: \benable_thinking\b; a template merely
#   containing the letters must not count.
@pytest.mark.unit
@pytest.mark.parametrize("kvs, expected", [
    ([("general.architecture", STR, "deepseek4"),
      ("tokenizer.chat_template", STR, "{% if enable_thinking %}<think>{% endif %}")], True),
    ([("tokenizer.chat_template", STR, "{{ messages[0].content }}")], False),
    ([("general.architecture", STR, "dflash")], None),
    ([("tokenizer.chat_template", STR, "{{ disable_thinkingness }}")], False),
    ([("tokenizer.chat_template", STR, "{%- if enable_thinking is defined %}x{%- endif %}")], True),
], ids=["mentions_enable_thinking", "template_without_it", "no_template_is_none_not_false",
        "word_boundary_not_substring", "matches_the_mlx_rule"])
def test_supports_thinking(tmp_path, kvs, expected):
    from heylook_llm.providers.common.template_info import _ENABLE_THINKING_PATTERN

    f = write_gguf(tmp_path / "m.gguf", kvs)
    got = supports_thinking(f)
    assert got is expected
    template = dict((k, v) for k, _, v in kvs).get("tokenizer.chat_template")
    if template is not None:
        assert got is bool(_ENABLE_THINKING_PATTERN.search(template))


def _model(path, arch, tensors, blocks=4, **names):
    kvs = [("general.architecture", STR, arch), (f"{arch}.block_count", U32, blocks)]
    kvs += [(f"general.{k}", STR, v) for k, v in names.items()]
    return write_gguf(path, kvs, tensors=tensors)


@pytest.mark.unit
class TestSpecTypeFromGguf:
    """llama.cpp's rule (common_speculative_types_from_gguf), from the header."""

    @pytest.mark.parametrize("arch,tensors,expected", [
        ("dflash", ["markov_w1.weight", "x"], "draft-dspark"),
        ("dflash", ["x"], "draft-dflash"),
        ("qwen35", ["blk.3.nextn.eh_proj.weight"], "draft-mtp"),   # last block
        ("qwen35", ["blk.2.nextn.eh_proj.weight"], None),          # not the last
        ("qwen35", ["blk.3.attn_q.weight"], None),
    ])
    def test_the_rule(self, tmp_path, arch, tensors, expected):
        assert spec_type_from_gguf([_model(tmp_path / "m.gguf", arch, tensors)]) == expected

    def test_a_head_in_a_later_split_is_found(self, tmp_path):
        """llama.cpp reads only the first split; a target's built-in head can
        sit in the last, so every split's tensor table is read."""
        first = _model(tmp_path / "m-00001-of-00002.gguf", "qwen35", ["blk.0.attn_q.weight"])
        write_gguf(tmp_path / "m-00002-of-00002.gguf", [], tensors=["blk.3.nextn.eh_proj.weight"])
        assert splits(first) == [first, tmp_path / "m-00002-of-00002.gguf"]
        assert spec_type_from_gguf(splits(first)) == "draft-mtp"

    def test_an_unreadable_file_is_none(self, tmp_path):
        (tmp_path / "junk.gguf").write_bytes(b"not a gguf")
        assert spec_type_from_gguf([tmp_path / "junk.gguf"]) is None


@pytest.mark.unit
class TestContextLength:
    """``<arch>.context_length`` -- the ceiling a context-size control offers.

    The key is architecture-prefixed, so the reader has to learn the
    architecture first; a file whose architecture it cannot read yields
    None rather than guessing a prefix.
    """

    @pytest.mark.parametrize("kvs, expected", [
        ([("general.architecture", STR, "deepseek4"),
          ("deepseek4.block_count", U32, 43),
          ("deepseek4.context_length", U32, 1048576)], 1048576),
        ([("general.architecture", STR, "gemma4"),
          ("qwen3.context_length", U32, 40960)], None),
        ([("deepseek4.context_length", U32, 1048576)], None),
        ([("general.architecture", STR, "gemma4")], None),
        (_UNREADABLE, None),
        ("bad_magic", None),
    ], ids=["arch_prefixed_key", "another_archs_key_is_not_this_models", "missing_arch",
            "missing_key", "unreadable_missing_file", "unreadable_bad_magic"])
    def test_reads_only_its_own_arch_key(self, tmp_path, kvs, expected):
        if kvs is _UNREADABLE:
            f = tmp_path / "missing.gguf"
        elif kvs == "bad_magic":
            f = write_gguf(tmp_path / "bad.gguf", [], magic=b"NOPE")
        else:
            f = write_gguf(tmp_path / "m.gguf", kvs)
        assert context_length(f) == expected

    def test_cache_keys_on_file_identity(self, tmp_path):
        f = write_gguf(tmp_path / "m.gguf", [
            ("general.architecture", STR, "gemma4"),
            ("gemma4.context_length", U32, 131072),
        ])
        assert context_length(f) == 131072
        # Same path, new content and size: a re-download at the same path is
        # a different file and must not answer from the cache.
        write_gguf(f, [
            ("general.architecture", STR, "gemma4"),
            ("gemma4.context_length", U32, 262144),
            ("gemma4.block_count", U32, 60),
        ])
        assert context_length(f) == 262144


@pytest.mark.unit
class TestVendorSampling:
    """The model's own recommended decode settings, from the GGUF header.

    heylook sends every sampler key on every request, so before v2.0.22 the
    values llama.cpp reads out of this block never survived to the sampler --
    the server sent top_k 0 at models whose own files ask for 20 or 64. These
    check that the layer arrives AND that it stays a layer: anything above it
    in the cascade still wins.
    """

    # The header's recommendation reaching the WIRE is pinned through the
    # provider, parse included: test_llama_server_provider.py
    # `TestPayload::test_the_vendor_layer_reaches_the_payload`.

    def test_an_explicit_value_still_beats_the_header(self, tmp_path):
        """It raises the floor; it does not overrule anyone.

        models.toml, a named sampler and the request all sit above it, and a
        vendor hint that could not be overridden would be a worse default than
        the one it replaced.
        """
        from heylook_llm.gguf_metadata import vendor_sampling
        from heylook_llm.samplers import resolve_effective_sampling

        f = write_gguf(tmp_path / "m.gguf", [
            ("general.architecture", STR, "qwen3"),
            ("general.sampling.top_k", I32, 20),
        ])
        merged = resolve_effective_sampling(
            _BareRequest(), {"model_path": str(f), "top_k": 7},
            vendor=vendor_sampling(f))
        assert merged["top_k"] == 7

    def test_a_model_with_no_recommendation_contributes_nothing(self, tmp_path):
        """Absent keys must yield {}, not zeros.

        A vendor layer that invented values would be worse than none: it would
        overwrite the floor with numbers no one chose.
        """
        from heylook_llm.gguf_metadata import vendor_sampling

        f = write_gguf(tmp_path / "m.gguf", [("general.architecture", STR, "qwen3")])
        assert vendor_sampling(f) == {}
        assert vendor_sampling(tmp_path / "missing.gguf") == {}


class _BareRequest:
    """A request that names nothing -- the case the cascade's defaults are for."""
    sampler = None
    enable_thinking = None

    def model_dump(self, **kwargs):
        return {}
