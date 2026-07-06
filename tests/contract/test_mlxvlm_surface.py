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
# IS used elsewhere in this suite unguarded (test_samplers.py, test_embedding_model.py,
# etc.) so it appears here too where it gives a stronger pin than source-text
# matching alone.
#
# Import discipline: this file imports the REAL mlx_lm / mlx_vlm / heylook_llm.*
# modules at module level (collection time), matching the precedent set by
# tests/unit/test_snapshot_thread_affinity.py (which imports
# heylook_llm.providers.common.cache_helpers unmocked, module-level). This must
# NEVER go through the mock_mlx / mlx_mocks fixtures -- those replace mlx_lm/mlx_vlm
# with MagicMocks, which would make every assertion here vacuously true. Because
# collection happens before any fixture body runs, importing at module level (not
# inside a test function) guarantees these bindings refer to the real libraries even
# if a later-collected contract test file's session-scoped mlx_mocks fixture patches
# sys.modules afterward.
#
# heylook_llm.providers.mlx_provider itself is deliberately NOT imported here (or
# anywhere unmocked in this suite): it creates a real MLX thread-local stream at
# module level (`generation_stream = mx.new_thread_local_stream(mx.default_device())`).
# Its consumption sites are pinned via plain source-text reads instead.

import dataclasses
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# --- Real mlx_vlm surface -----------------------------------------------------
from mlx_vlm.utils import prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template, MODEL_CONFIG
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.gemma4 import gemma4 as _gemma4_module
from mlx_vlm.models.qwen3_5 import language as _qwen3_5_language_module

# --- Real mlx_lm surface -------------------------------------------------------
from mlx_lm.generate import GenerationResponse
from mlx_lm.utils import _get_classes
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache, make_prompt_cache

# --- Real heylook_llm consumption-side helpers (safe to import unmocked: no
# Metal/GPU touch at import time, unlike mlx_provider.py itself) -------------
from heylook_llm.providers.common.model_wrappers import wrap_language_model
from heylook_llm.providers.common.generation_core import _reset_vlm_positions


_MLX_PROVIDER_SRC = Path(
    __file__
).parent.parent.parent / "src" / "heylook_llm" / "providers" / "mlx_provider.py"


def _mlx_provider_source() -> str:
    return _MLX_PROVIDER_SRC.read_text()


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
        # suite (tests/unit/test_samplers.py, tests/unit/test_embedding_model.py).
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


# ---------------------------------------------------------------------------
# LanguageModelOutput
# ---------------------------------------------------------------------------

class TestLanguageModelOutput:
    """Consumed at providers/common/model_wrappers.py:63-66
    (LanguageModelLogitsWrapper.__call__: getattr(result, 'logits', None)) and
    mlx_provider.py:452 (VLMVisionStrategy.generate:
    output.logits if hasattr(output, 'logits') else output). Both sites read
    ONLY the .logits field, defensively (getattr/hasattr), so pin that the
    field exists and is required -- not the rest of the dataclass shape."""

    def test_is_dataclass_with_required_logits_field(self):
        assert dataclasses.is_dataclass(LanguageModelOutput)
        fields = {f.name: f for f in dataclasses.fields(LanguageModelOutput)}
        assert "logits" in fields
        logits_field = fields["logits"]
        assert logits_field.default is dataclasses.MISSING
        assert logits_field.default_factory is dataclasses.MISSING

    def test_wrap_language_model_extracts_logits_via_getattr(self):
        # Real (unmocked) call into our own wrapper -- proves the consumption
        # site still works against a LanguageModelOutput-shaped result, not
        # just that the library dataclass shape is unchanged.
        class _FakeLanguageModel:
            def __call__(self, *args, **kwargs):
                return LanguageModelOutput(logits="LOGITS_SENTINEL")

        class _FakeVLM:
            language_model = _FakeLanguageModel()

        wrapper = wrap_language_model(_FakeVLM())
        assert wrapper(1, 2, foo="bar") == "LOGITS_SENTINEL"


# ---------------------------------------------------------------------------
# _position_ids / _rope_deltas mRoPE attribute convention
# ---------------------------------------------------------------------------

class TestPositionIdsRopeDeltasConvention:
    """Consumed at providers/common/generation_core.py:35-55
    (_reset_vlm_positions): Qwen3.5-style VLM language models cache
    `_position_ids`/`_rope_deltas` as plain instance attributes (no public
    accessor -- this is a private, dynamically-set convention with no type
    contract), and generation_core resets them to None between fresh
    generations to prevent stale mRoPE broadcast-shape mismatches."""

    def test_qwen3_5_language_model_still_sets_both_attributes(self):
        # Static source pin -- constructing a real LanguageModel needs a full
        # ModelArgs config graph, out of scope for an inspect-level test.
        src = inspect.getsource(_qwen3_5_language_module.LanguageModel.__init__)
        assert "_position_ids" in src
        assert "_rope_deltas" in src

    def test_reset_vlm_positions_clears_both_attrs_on_language_model(self):
        # Real (unmocked) call into our own reset function against the exact
        # attribute convention above.
        class _FakeLanguageModel:
            def __init__(self):
                self._position_ids = "stale_positions"
                self._rope_deltas = "stale_deltas"

        class _FakeWrappedVLM:
            def __init__(self):
                self.language_model = _FakeLanguageModel()

        wrapped = _FakeWrappedVLM()
        _reset_vlm_positions(wrapped)
        assert wrapped.language_model._position_ids is None
        assert wrapped.language_model._rope_deltas is None


# ---------------------------------------------------------------------------
# mlx-lm private-API touchpoints
# ---------------------------------------------------------------------------

class TestGetClasses:
    """Consumed at src/heylook_llm/models/embedding_model.py:75-94
    (load_backbone): mlx_lm.utils._get_classes(model_config) is a PRIVATE API
    (leading underscore, no compat guarantee) returning (ModelClass, ArgsClass);
    ArgsClass must be a dataclass so `ArgsClass.__dataclass_fields__` can filter
    the config dict before construction."""

    def test_get_classes_signature_takes_a_config_dict(self):
        sig = inspect.signature(_get_classes)
        assert "config" in sig.parameters

    def test_get_classes_returns_model_and_dataclass_args(self):
        ModelClass, ArgsClass = _get_classes({"model_type": "llama"})
        assert inspect.isclass(ModelClass)
        assert inspect.isclass(ArgsClass)
        assert hasattr(ArgsClass, "__dataclass_fields__"), (
            "embedding_model.load_backbone filters model_config keys via "
            "ArgsClass.__dataclass_fields__ -- ArgsClass must stay a dataclass"
        )


class TestCacheClasses:
    """Consumed at src/heylook_llm/providers/mlx_provider.py:15
    (`from mlx_lm.models.cache import make_prompt_cache`) and
    providers/common/cache_helpers.py:19
    (`from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache`).
    cache_helpers.make_cache/snapshot_kv/restore_kv_from_snapshot rely on every
    cache class exposing a settable `.state` property and an `.empty()` method."""

    @pytest.mark.parametrize("cache_factory", [
        lambda: KVCache(),
        lambda: QuantizedKVCache(),
        lambda: RotatingKVCache(max_size=8),
    ])
    def test_cache_class_exposes_settable_state_and_empty(self, cache_factory):
        cache = cache_factory()
        assert hasattr(cache, "empty")
        assert cache.empty() is True  # fresh cache: snapshot_kv's guard depends on this
        state_prop = type(cache).state
        assert isinstance(state_prop, property)
        assert state_prop.fset is not None, (
            "restore_kv_from_snapshot does `layer.state = state` -- the "
            "property must stay settable"
        )

    def test_make_prompt_cache_signature_and_shape(self):
        sig = inspect.signature(make_prompt_cache)
        assert "model" in sig.parameters
        assert "max_kv_size" in sig.parameters

        # Real (unmocked) call: a bare object with .layers (no make_cache
        # override) must fall back to plain KVCache per layer -- the branch
        # mlx_provider.py:438 (make_prompt_cache(self._cached_wrapper)) hits,
        # since LanguageModelLogitsWrapper has no make_cache method.
        class _FakeModel:
            layers = [object(), object(), object()]

        cache = make_prompt_cache(_FakeModel())
        assert len(cache) == 3
        assert all(isinstance(c, KVCache) for c in cache)


# ---------------------------------------------------------------------------
# mlx_lm.generate.GenerationResponse
# ---------------------------------------------------------------------------

class TestGenerationResponse:
    """Consumed throughout providers/common/generation_core.py (the sole
    lm_stream_generate call site, ~line 340) and read via getattr in
    api.py/messages_api.py for telemetry. Fields pinned per plan Phase 1 item 7:
    text, token, logprobs, from_draft, prompt_tokens, prompt_tps,
    generation_tokens, generation_tps, peak_memory, finish_reason."""

    EXPECTED_FIELDS = {
        "text", "token", "logprobs", "from_draft", "prompt_tokens", "prompt_tps",
        "generation_tokens", "generation_tps", "peak_memory", "finish_reason",
    }

    def test_has_exactly_the_consumed_fields(self):
        actual = {f.name for f in dataclasses.fields(GenerationResponse)}
        missing = self.EXPECTED_FIELDS - actual
        assert not missing, f"GenerationResponse dropped fields: {missing}"

    def test_is_nonslotted_dataclass_supporting_runtime_attribute_attachment(self):
        # generation_core.py:372-376 does
        # `response.cached_tokens = cached_count  # type: ignore[attr-defined]`
        # and `response.kv_cache_bytes = ...` on a live GenerationResponse
        # instance. A __slots__ class would raise AttributeError on that.
        assert "__slots__" not in GenerationResponse.__dict__

        response = GenerationResponse(
            text="hi", token=1, logprobs=None, from_draft=False,
            prompt_tokens=1, prompt_tps=0.0, generation_tokens=1,
            generation_tps=0.0, peak_memory=0.0, finish_reason=None,
        )
        response.cached_tokens = 5  # must not raise
        response.kv_cache_bytes = 1024  # must not raise
        assert response.cached_tokens == 5
        assert response.kv_cache_bytes == 1024

    def test_our_consumption_sites_still_read_documented_fields(self):
        # Source-text pin: generation_core.py's own-code traceability for the
        # subset it reads directly (response.token/.from_draft/.text); the
        # getattr-based telemetry fields (prompt_tokens, prompt_tps,
        # generation_tokens, generation_tps, peak_memory, finish_reason) are
        # read defensively via getattr(chunk, ..., default) in api.py /
        # messages_api.py and are covered by the field-set test above.
        gen_core_src = (
            Path(__file__).parent.parent.parent
            / "src" / "heylook_llm" / "providers" / "common" / "generation_core.py"
        ).read_text()
        assert "response.token" in gen_core_src
        assert "response.from_draft" in gen_core_src
        assert "response.text" in gen_core_src
