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
# Scope discipline (see .claude/rules/mlx.md + the plan's Direction
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
# heylook_llm.providers.mlx_provider is not imported at module level here. The
# one test that renders through it imports heylook's provider modules FRESH
# inside a sys.modules patch, so they bind the real libraries rather than the
# session's MagicMock tree; one consumption site (the vision feature cache
# handover) is still pinned by a source-text read, because its system check
# needs a live model (scripts/vlm_parity_probe.py).

import inspect
import re
import sys
from unittest.mock import patch

import pytest

# --- Real mlx_vlm surface -----------------------------------------------------
from mlx_vlm.utils import prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template, MODEL_CONFIG
from mlx_vlm.models.gemma4 import gemma4 as _gemma4_module


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
        # this server's actual vision models -- see .claude/rules/mlx.md / log_2026-07-06.md).
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

    def test_the_vision_path_renders_the_image_and_the_depth(self):
        """vlm_inputs.py -> heylook's vlm_apply_chat_template -> mlx-vlm's
        real message rebuild -> a template, with one image attached and a
        thinking depth chosen: the rendered prompt carries the image markup
        once and the depth's text. Replaces two source-text pins (the
        num_images and depth kwargs at the vlm_inputs.py call), and covers the
        case they left open: a depth that is passed but never reaches the
        template. The only stand-in is the image loader.

        heylook's MLX modules are imported FRESH inside a sys.modules patch,
        against the real mlx / mlx_vlm this file restores: other contract
        files import them under the session's MagicMock tree, and a module
        bound to that tree would render nothing real."""
        from PIL import Image

        from heylook_llm.chat_template_files import _engine_environment
        from heylook_llm.config import ChatMessage, ImageContentPart, ImageUrl, TextContentPart

        env = _engine_environment()
        body = ("{% for m in messages %}<{{ m['role'] }}>"
                "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
                "{% for c in m['content'] %}{% if c['type'] == 'image' %}<IMG>"
                "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}"
                "<assistant effort={{ reasoning_strength | default('unset') }}>{% endif %}")

        class Tok:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, **kw):
                return env.from_string(body).render(
                    messages=messages, add_generation_prompt=add_generation_prompt, **kw)

        class Proc:
            image_token = "<IMG>"
            tokenizer = Tok()

        class Cfg(dict):
            def __init__(self):
                super().__init__(model_type="qwen3_5")
                self.model_type = "qwen3_5"

        class Loader:
            def load_images_parallel(self, urls):
                return [Image.new("RGB", (8, 8), "red") for _ in urls]

        messages = [ChatMessage(role="user", content=[
            ImageContentPart(type="image_url", image_url=ImageUrl(url="https://example.test/a.png")),
            TextContentPart(type="text", text="what is this?")])]

        import heylook_llm

        # Both restored on exit: the module table, and the package attribute
        # the fresh import rebinds (mock.patch resolves dotted targets through
        # it, so a later patch would otherwise land on the fresh copy).
        with patch.dict(sys.modules), patch.dict(heylook_llm.__dict__):
            for name in [n for n in sys.modules if n.startswith("heylook_llm.providers")]:
                del sys.modules[name]
            from heylook_llm.providers.common.vlm_inputs import prepare_vlm_inputs_parallel
            from heylook_llm.providers.mlx_provider import vlm_apply_chat_template

            images, prompt, has_images, _ = prepare_vlm_inputs_parallel(
                messages, Proc(), Cfg(), Loader(), vlm_apply_chat_template,
                enable_thinking=True, depth={"reasoning_strength": "low"})

        assert has_images and len(images) == 1
        assert prompt.count("<IMG>") == 1, prompt
        assert "what is this?" in prompt
        assert "effort=low" in prompt, prompt


# ---------------------------------------------------------------------------
# vision feature caching: encode_image() / cached_image_features, and the
# vision_cache / _image_key kwargs for a model without encode_image()
# ---------------------------------------------------------------------------

class TestVisionFeatureCachePatterns:
    """VLMVisionStrategy.generate hands every model heylook's cache as
    mlx-vlm's ``vision_cache``/``_image_key`` kwargs, the way mlx-vlm's server
    does, and the model looks up and stores its own tower output. Until
    2026-09-25 heylook ran ``encode_image`` itself where a model had one, with
    pixels alone; the census below is why that went: every model with
    ``encode_image`` that caches at all reads the kwargs itself, and the
    pixels-only call was wrong for deepseek_v4, gemma4_unified and
    minimax_m3_vl. Pinned on the served families (qwen3_5 has no
    encode_image; gemma4 has one and still reads the kwargs)."""

    def test_served_families_read_the_cache_kwargs(self):
        import importlib
        for mod in ("mlx_vlm.models.qwen3_5.qwen3_5", "mlx_vlm.models.gemma4.gemma4"):
            src = inspect.getsource(importlib.import_module(mod).Model.get_input_embeddings)
            assert '"vision_cache"' in src and '"_image_key"' in src, mod


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

    def test_the_memory_pressure_signals_still_exist(self):
        """vlm_engine.apc_memory_pressure reads these to say a miss was APC
        holding back for memory rather than an ordinary miss."""
        from heylook_llm.providers.common.vlm_engine import make_apc_manager

        mgr = make_apc_manager()
        assert "memory_evictions" in mgr.stats_snapshot()
        assert isinstance(mgr._memory_headroom(), int)
        assert isinstance(mgr.memory_reserve_bytes, int)

    def test_apc_takes_our_overrides_and_no_disk(self):
        from heylook_llm.providers.common.vlm_engine import (
            APC_CHECKPOINT_ENTRIES, APC_CHECKPOINT_INTERVAL_TOKENS, make_apc_manager)

        mgr = make_apc_manager()
        assert mgr.disk is None
        # vlm_engine's "was the cache empty" check reads these two stores
        assert isinstance(mgr._exact_cache, dict) and isinstance(mgr.hash_table, dict)
        # refresh_snapshots re-derives a snapshot's key the way the store does
        from collections import OrderedDict
        from mlx_vlm import apc as _apc
        assert isinstance(mgr._exact_cache, OrderedDict) and hasattr(mgr.lock, "acquire")
        assert "key = _sequence_hash(token_tuple, extra_hash, self.block_size)" in \
            inspect.getsource(_apc.APCManager.store_exact_cache)
        assert mgr.checkpoint_interval_tokens == APC_CHECKPOINT_INTERVAL_TOKENS
        assert mgr._exact_cache_max == APC_CHECKPOINT_ENTRIES

    def test_our_capture_rule_is_upstreams_with_its_own_count(self):
        # install_capture_policy replaces the coordinator's checkpoint_lengths
        # per generator. Without boundaries, and with the capture count set
        # to the store size, it must reproduce mlx-vlm's own rule exactly --
        # so an upstream change to that rule fails here instead of drifting.
        import importlib
        from types import SimpleNamespace

        from mlx_vlm.apc_coordinator import APCCoordinator
        from heylook_llm.providers.common.vlm_engine import capture_lengths

        for final, entries, interval in [(1000, 3, 64), (130, 3, 64), (5000, 5, 100), (40, 3, 64)]:
            mgr = SimpleNamespace(checkpoint_interval_tokens=interval, _exact_cache_max=entries,
                                  disk=None, block_size=16, exact_cache_min_tokens=16)
            stub = SimpleNamespace(manager=mgr, checkpoint_len=lambda ids, media, f=final: f)
            upstream = APCCoordinator.checkpoint_lengths(stub, list(range(final + 1)), set())
            ours = capture_lengths(final, interval=interval, block_size=16, captures=entries,
                                   min_tokens=16)
            assert ours == upstream, (final, entries, interval)
        # ... and the generator builds its coordinator per instance and asks
        # it (not a module function) for the lengths, so the override binds.
        src = inspect.getsource(importlib.import_module("mlx_vlm.generate.ar"))
        assert "self.apc = (" in src and "APCCoordinator(apc_manager, model)" in src
        assert "coordinator.checkpoint_lengths(ids_list" in src
        for attr in ("enabled", "is_checkpoint", "checkpoint_len"):
            assert hasattr(APCCoordinator, attr), attr

    def test_the_salt_helpers_take_what_we_pass(self):
        from mlx_vlm import apc as _apc
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer  # noqa: F401

        salt = inspect.signature(_apc.semantic_extra_hash).parameters
        for name in ("image_hash", "media", "model", "processor"):
            assert name in salt, name
        assert "pixel_values" in inspect.signature(_apc.hash_image_payload).parameters
