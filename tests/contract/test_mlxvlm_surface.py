# tests/contract/test_mlxvlm_surface.py
#
# Transcription of the 2026-07-06 library-drift audit (internal/log/log_2026-07-06.md,
# "cleanup: test suite + library-drift audit (v1.32.1)"; plan item:
# internal/backend/plan_2026-07.md Phase 1 #7) into executable tests.
#
# Purpose: pin the EXACT mlx-lm / mlx-vlm surface this server consumes -- private
# APIs, dataclass fields, and dynamically-set attribute conventions with no public
# contract -- so an aggressive `uv sync` upgrade of either library fails LOUDLY here
# instead of silently at runtime (wrong kwarg name swallowed by **kwargs, a renamed
# dataclass field defaulting via getattr(), etc.).
#
# Scope discipline (see CLAUDE.md "MLX / library gotchas" + the plan's Direction
# section on mlx-vlm bus-factor risk): import/inspect-level only. No model
# downloads, no network, no Metal-requiring calls (no thread-local GPU streams, no
# real vision-tower forward passes). Plain mx.array construction from Python lists
# IS used elsewhere in this suite unguarded (test_samplers.py etc.) so it
# appears here too where it gives a stronger pin than source-text
# matching alone.
#
# Import discipline: this file imports the REAL mlx_lm / mlx_vlm / heylook_llm.*
# modules at module level (collection time), matching the precedent set by
# the other real-MLX tests. This must
# NEVER go through the mock_mlx / mlx_mocks fixtures -- those replace mlx_lm/mlx_vlm
# with MagicMocks, which would make every assertion here vacuously true. Because
# collection happens before any fixture body runs, importing at module level (not
# inside a test function) guarantees these bindings refer to the real libraries even
# if a later-collected contract test file's session-scoped mlx_mocks fixture patches
# sys.modules afterward.
#
# heylook_llm.providers.mlx_provider itself is deliberately NOT imported here (or
# anywhere unmocked in this suite); its consumption sites are pinned via plain
# source-text reads instead.

import inspect
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# --- Real mlx_vlm surface -----------------------------------------------------
from mlx_vlm.utils import prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template, MODEL_CONFIG
from mlx_vlm.models.gemma4 import gemma4 as _gemma4_module



_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "heylook_llm"


def _source(*parts: str) -> str:
    """Read one of our own files for a source-text pin. ONE path walk: the
    second call site arrived with a verbatim copy of the first, parents walk
    included, which is the hand-copied-constant shape this repo keeps paying
    for elsewhere."""
    return _SRC_ROOT.joinpath(*parts).read_text()


def _mlx_provider_source() -> str:
    return _source("providers", "mlx_provider.py")


def _vlm_inputs_source() -> str:
    return _source("providers", "common", "vlm_inputs.py")


# ---------------------------------------------------------------------------
# Anti-contamination guard
# ---------------------------------------------------------------------------
#
# tests/contract/conftest.py's session-scoped `mlx_mocks` fixture patches
# sys.modules['mlx_lm']/['mlx_vlm'] (and submodules) with MagicMocks the first
# time any OTHER contract test file requests the `app`/`client` fixture, and
# does not revert until the whole session ends. Once that happens, a LAZY
# runtime import inside a real library function -- e.g. mlx_lm.utils._get_classes
# doing `importlib.import_module(f"mlx_lm.models.{model_type}")` -- resolves
# against the contaminated sys.modules at CALL time, not against the real
# modules this file bound at collection time. Snapshot the real entries here
# (this runs during collection, before any fixture body has executed) and
# reinstate them for the duration of every test in this file.
_REAL_MLX_SYS_MODULES = {
    name: mod for name, mod in sys.modules.items()
    if name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx_vlm"))
}


@pytest.fixture(autouse=True)
def _real_mlx_modules():
    with patch.dict(sys.modules, _REAL_MLX_SYS_MODULES):
        yield


# ---------------------------------------------------------------------------
# mlx_vlm.utils.prepare_inputs
# ---------------------------------------------------------------------------

class TestPrepareInputs:
    """Consumed at src/heylook_llm/providers/mlx_provider.py:384-397
    (VLMVisionStrategy.generate): calls vlm_prepare_inputs(processor, images=...,
    prompts=..., image_token_index=...), then reads inputs["input_ids"],
    inputs.get("pixel_values"), inputs.get("attention_mask"), and inputs.items()
    for model-specific extras (e.g. image_grid_thw)."""

    def test_signature_accepts_our_kwargs(self):
        # mlx_provider.py:384-389 calls prepare_inputs with these exact kwarg
        # names. If mlx-vlm renames any of them, this fails at import/signature
        # level instead of a swallowed-by-**kwargs silent no-op at runtime.
        sig = inspect.signature(prepare_inputs)
        for name in ("processor", "images", "prompts", "image_token_index"):
            assert name in sig.parameters, (
                f"mlx_vlm.utils.prepare_inputs lost its '{name}' parameter"
            )
            # Must be usable as a keyword (not positional-only) -- our call site
            # passes all of these by keyword.
            assert sig.parameters[name].kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )

    def test_returns_dict_with_input_ids_and_attention_mask(self):
        # Real (unmocked) call down the text-only branch (images=None) --
        # exactly what our vision strategy hits when a VLM request formats a
        # prompt with zero images resolved. No Metal: only mx.array(list) from
        # plain Python lists, the same pattern used unguarded elsewhere in this
        # suite (tests/unit/test_samplers.py).
        class _FakeTokenizerOutput:
            def __init__(self, input_ids, attention_mask):
                self.input_ids = input_ids
                self.attention_mask = attention_mask

        class _FakeTokenizer:
            pad_token = "<pad>"
            eos_token = "</s>"

            def __call__(self, prompts, **kwargs):
                return _FakeTokenizerOutput([[1, 2, 3]], [[1, 1, 1]])

        class _FakeProcessor:
            tokenizer = _FakeTokenizer()

        result = prepare_inputs(
            _FakeProcessor(), images=None, prompts="hello", image_token_index=None,
        )

        # This is exactly the dict-style access mlx_provider.py:391-397 relies on.
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result.get("pixel_values") is None  # .get() must not KeyError
        extras = {
            k: v for k, v in result.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        assert extras == {}
        assert hasattr(result["input_ids"], "shape")

    def test_vision_branch_still_produces_pixel_values_key(self):
        # Static pin for the image branch (not executed -- would require a real
        # image processor). If mlx-vlm stops naming the returned key
        # "pixel_values" in either code path (BaseImageProcessor branch or the
        # generic images->pixel_values rename), our
        # `inputs.get("pixel_values")` silently returns None forever and vision
        # requests would prefill with an empty vlm_kwargs. Source-text pin, not
        # signature-level, since the shape only appears deep inside conditional
        # branches keyed on processor type.
        src = inspect.getsource(prepare_inputs)
        assert '"pixel_values"' in src


# ---------------------------------------------------------------------------
# mlx_vlm.prompt_utils.apply_chat_template
# ---------------------------------------------------------------------------

class TestApplyChatTemplate:
    """Consumed at src/heylook_llm/providers/mlx_provider.py:108-166
    (vlm_apply_chat_template): calls mlx_vlm_apply_chat_template(processor,
    config, messages, num_images=num_images, return_messages=True), then
    flattens list-typed content by inspecting each dict's 'type' key."""

    def test_signature_accepts_our_kwargs(self):
        sig = inspect.signature(apply_chat_template)
        for name in ("processor", "config", "prompt", "num_images", "return_messages"):
            assert name in sig.parameters, (
                f"mlx_vlm.prompt_utils.apply_chat_template lost '{name}'"
            )

    def test_return_messages_shape_matches_our_flattening_logic(self):
        # Real call with a production-relevant model_type (qwen3_vl is one of
        # this server's actual vision models -- see CLAUDE.md / log_2026-07-06.md).
        # No processor is touched on the return_messages=True path (verified via
        # source read below), so passing None for it is safe.
        assert "qwen3_vl" in MODEL_CONFIG, (
            "qwen3_vl dropped from mlx_vlm.prompt_utils.MODEL_CONFIG -- "
            "vlm_apply_chat_template's docstring assumptions need re-checking"
        )
        messages = apply_chat_template(
            processor=None,
            config={"model_type": "qwen3_vl"},
            prompt=[{"role": "user", "content": "describe this image"}],
            num_images=1,
            return_messages=True,
        )
        assert isinstance(messages, list)
        msg = messages[0]
        assert msg["role"] == "user"
        # mlx_provider.py's Step 2 flattening (lines ~139-154) inspects each
        # list item's dict for a 'type' key of 'text'/'input_text' or
        # 'image'/'image_url'/'input_image'. Assert that shape still holds.
        assert isinstance(msg["content"], list)
        item_types = {item.get("type") for item in msg["content"] if isinstance(item, dict)}
        assert "image" in item_types
        assert "text" in item_types

    def test_our_call_site_still_passes_num_images_and_return_messages(self):
        # Source-text pin for the exact call in mlx_provider.py (own-code
        # traceability, not a library-compat check -- that's the test above).
        src = _mlx_provider_source()
        assert "num_images=num_images, return_messages=True" in src

    def test_the_media_attribution_call_site_is_pinned_too(self):
        """vlm_inputs.py is where per-message media attribution lives.

        OWN-CODE TRACEABILITY, like the test above it and NOT a library-compat
        check -- stated plainly because the module around it is a library-
        surface suite and the first version of this docstring read as one.
        vlm_inputs.py does not call mlx-vlm: it calls heylook's own
        `vlm_apply_chat_template` (mlx_provider.py), which is what reaches
        `mlx_vlm.prompt_utils`. So this watches OUR call keeping its kwargs.

        WHAT IT STILL DOES NOT COVER: the depth kwarg reaches mlx-vlm only
        through `**kwargs`, so the swallow case -- a library that quietly stops
        forwarding it to the template -- remains unpinned by anything here.
        """
        src = _vlm_inputs_source()
        # Plain substrings, matching the sibling pin above. A regex pass was
        # tried and reverted: it tolerated only spacing a formatter would never
        # emit for a kwarg, still failed the realistic reformat (a line wrap),
        # and left two adjacent pins of the same shape written differently.
        # Source-text pins ARE reformat-fragile; the message says so rather
        # than the check pretending otherwise.
        assert "num_images=len(images)" in src, (
            "vlm_inputs.py no longer passes num_images to the chat-template call "
            "as written -- a reformat here is a false alarm, a moved call is not"
        )
        assert "depth=depth" in src, (
            "vlm_inputs.py stopped forwarding the thinking depth to the template"
        )


# ---------------------------------------------------------------------------
# encode_image() / cached_image_features -- vision feature caching pattern
# ---------------------------------------------------------------------------

class TestEncodeImageCachedFeaturesPattern:
    """Consumed at src/heylook_llm/providers/mlx_provider.py:415-430
    (VLMVisionStrategy.generate) + providers/common/vision_feature_cache.py:
    hasattr(model, 'encode_image') gates computing model.encode_image(pixel_values)
    once, then passing it back in as the cached_image_features kwarg on later
    turns. Not every mlx-vlm model implements this pattern (most don't -- it's
    optional per-architecture); we pin that at least one SHIPPED model class
    still does, proving the pattern our cache relies on remains real."""

    def test_shipped_model_exposes_encode_image_method(self):
        Model = _gemma4_module.Model
        assert hasattr(Model, "encode_image")
        sig = inspect.signature(Model.encode_image)
        assert "pixel_values" in sig.parameters

    def test_shipped_model_get_input_embeddings_reads_cached_image_features(self):
        src = inspect.getsource(_gemma4_module.Model.get_input_embeddings)
        assert "cached_image_features" in src

    def test_our_call_sites_still_reference_the_pattern(self):
        src = _mlx_provider_source()
        assert "hasattr(model, 'encode_image')" in src
        assert "cached_image_features" in src


class TestVlmEngineSurface:
    """`providers/common/vlm_engine.py` drives mlx-vlm's BatchGenerator and
    APC the way mlx-vlm's own server does, leaning on names that are private
    or keyword-only upstream, on a SHA pin that moves. A pin bump that breaks
    them fails HERE, by name. Whether the path then generates the right
    tokens is `scripts/chain_probe.py` and the W10 spike harness, live."""

    def test_the_generator_takes_the_keywords_we_pass(self):
        from mlx_vlm.generate.ar import BatchGenerator

        init = inspect.signature(BatchGenerator.__init__).parameters
        for name in ("sampler", "compute_logprobs", "apc_manager", "greedy_sampling",
                     "max_tokens", "prefill_step_size"):
            assert name in init, name
        insert = inspect.signature(BatchGenerator.insert).parameters
        for name in ("prompts", "max_tokens", "prompt_kwargs", "logits_processors"):
            assert name in insert, name
        for method in ("next", "remove", "close"):
            assert callable(getattr(BatchGenerator, method)), method

    def test_the_prefill_progress_fields_still_exist(self):
        import importlib

        # `import mlx_vlm.generate.ar as ar` resolves against the package's
        # re-exported `generate` function, not the submodule.
        src = inspect.getsource(importlib.import_module("mlx_vlm.generate.ar"))
        for field in ("self._prompt_batch", "_processed_prompt_columns", "_inputs_embeds",
                      "_cached_tokens_per_row", "def _release_apc_meta_blocks", "self._apc_meta"):
            assert field in src, field

    def test_apc_takes_our_overrides_and_no_disk(self):
        from heylook_llm.providers.common.vlm_engine import (
            APC_CHECKPOINT_ENTRIES, APC_CHECKPOINT_INTERVAL_TOKENS, make_apc_manager)

        mgr = make_apc_manager()
        assert mgr.disk is None
        assert mgr.checkpoint_interval_tokens == APC_CHECKPOINT_INTERVAL_TOKENS
        assert mgr._exact_cache_max == APC_CHECKPOINT_ENTRIES

    def test_the_salt_helpers_take_what_we_pass(self):
        from mlx_vlm import apc as _apc
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer  # noqa: F401

        salt = inspect.signature(_apc.semantic_extra_hash).parameters
        for name in ("image_hash", "media", "model", "processor"):
            assert name in salt, name
        assert "pixel_values" in inspect.signature(_apc.hash_image_payload).parameters
