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

    def test_the_end_thinking_fields_still_exist(self):
        """vlm_engine.generate arms a mid-stream end of thinking (the
        reasoning-end routes) through private fields, read with getattr
        defaults: a rename upstream makes it do nothing, silently, the way a
        moved restore site left the #2356 carry installed and idle. Pin each
        name, and the criteria behaviour the arming relies on: with the budget
        exceeded and the first forced token already injected, the next call
        returns the rest of the closing sequence."""
        import importlib
        from types import SimpleNamespace

        from mlx_vlm.utils import ThinkingBudgetCriteria

        src = inspect.getsource(importlib.import_module("mlx_vlm.generate.ar"))
        for field in ("self._generation_batch", "self._next_tokens",
                      "self.thinking_budget_criteria"):
            assert field in src, field

        ids = {"<think>": [7], "</think>": [8], "\n": [9]}
        tok = SimpleNamespace(encode=lambda s, add_special_tokens=False: ids.get(s, [1]))
        crit = ThinkingBudgetCriteria(tok, 0, thinking_end_token="</think>",
                                      thinking_start_token="<think>", enable_thinking=True,
                                      prompt_preopens_thinking=True)
        for name in ("in_thinking", "budget_exceeded", "thinking_budget",
                     "thinking_start_token_id", "thinking_end_token_id"):
            assert hasattr(crit, name), name
        assert crit.in_thinking and crit._forced_sequence == [9, 8]
        crit.thinking_budget, crit.budget_exceeded = 0, True
        crit._forced_index = 1          # vlm_engine injected the newline itself
        assert crit(5) == 8, "the criteria no longer finishes a closing sequence it was armed into"

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
        from heylook_llm.providers.common.vlm_engine import (
            capture_lengths,
            install_capture_policy,
        )

        for final, entries, interval in [(1000, 3, 64), (130, 3, 64), (5000, 5, 100), (40, 3, 64)]:
            mgr = SimpleNamespace(checkpoint_interval_tokens=interval, _exact_cache_max=entries,
                                  disk=None, block_size=16, exact_cache_min_tokens=16,
                                  checkpoint_prefix_length=lambda *_a, **_kw: 0)
            stub = SimpleNamespace(manager=mgr, checkpoint_len=lambda ids, media, f=final: f)
            upstream = APCCoordinator.checkpoint_lengths(stub, list(range(final + 1)), set())
            ours = capture_lengths(final, interval=interval, block_size=16, captures=entries,
                                   min_tokens=16)
            assert ours == upstream, (final, entries, interval)

        # Verify install_capture_policy's replacement accepts every parameter
        # APCCoordinator.checkpoint_lengths takes (including extra_hash and
        # stable_prefix_len) and includes observed/passed prefix boundaries.
        mgr = SimpleNamespace(checkpoint_interval_tokens=64, _exact_cache_max=16,
                              disk=None, block_size=16, exact_cache_min_tokens=16,
                              checkpoint_prefix_length=lambda *_a, **_kw: 48)
        coord = SimpleNamespace(enabled=True, is_checkpoint=True, manager=mgr,
                                checkpoint_len=lambda ids, media: 200)
        bg = SimpleNamespace(apc=coord)
        install_capture_policy(bg, [32])
        expected_params = [
            name for name in inspect.signature(APCCoordinator.checkpoint_lengths).parameters
            if name != "self"
        ]
        assert list(inspect.signature(coord.checkpoint_lengths).parameters) == expected_params
        got = coord.checkpoint_lengths(list(range(201)), set(), extra_hash=7, stable_prefix_len=24)
        assert 24 in got and 32 in got and 48 in got and 200 in got

        # ... and the generator builds its coordinator per instance and asks
        # it (not a module function) for the lengths, so the override binds.
        src = inspect.getsource(importlib.import_module("mlx_vlm.generate.ar"))
        assert "self.apc = (" in src and "APCCoordinator(apc_manager, model)" in src
        assert "coordinator.checkpoint_lengths(" in src
        for attr in ("enabled", "is_checkpoint", "checkpoint_len"):
            assert hasattr(APCCoordinator, attr), attr

    def test_install_media_token_ids_covers_nested_vlm_configs(self, caplog):
        import importlib
        import logging
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from mlx_vlm import apc as _apc
        from mlx_vlm.generate.ar import BatchGenerator
        from mlx_vlm.models.gemma4.config import ModelConfig as Gemma4ModelConfig, TextConfig as Gemma4TextConfig
        from heylook_llm.providers.common.vlm_engine import (
            install_media_token_ids,
            multimodal_token_ids,
        )

        # Upstream BatchGenerator._apc_media_token_ids takes only self and is called
        # by all four APC boundary/suffix checks.
        assert list(inspect.signature(BatchGenerator._apc_media_token_ids).parameters) == ["self"]
        ar_src = inspect.getsource(importlib.import_module("mlx_vlm.generate.ar"))
        for callsite in (
            "_apc_safe_prefix_lookup_min",
            "_apc_suffix_is_text_only",
            "_apc_exact_checkpoint_len",
            "_apc_exact_checkpoint_lengths",
        ):
            assert f"def {callsite}(" in ar_src
        assert ar_src.count("self._apc_media_token_ids()") >= 5

        cfg = Gemma4ModelConfig(
            text_config=Gemma4TextConfig(),
            image_token_id=258880,
            audio_token_id=258881,
        )
        assert _apc.multimodal_token_ids_from_config(cfg.text_config) == set()

        fake_vlm = SimpleNamespace(
            config=cfg,
            language_model=SimpleNamespace(config=cfg.text_config),
        )
        proc = SimpleNamespace(video_token_id=258884)
        assert multimodal_token_ids(fake_vlm, proc) == {258880, 258881, 258884}

        bg = SimpleNamespace(_apc_media_token_ids=lambda: {999})
        assert install_media_token_ids(bg, fake_vlm, proc) is True
        assert bg._apc_media_token_ids() == {999, 258880, 258881, 258884}

        # Verify APC checkpoint boundary advances past a 280-token image span instead
        # of splitting at token 256.
        toks = [1, 2, *([258880] * 280), 3, 4, 5]
        adjusted = _apc.adjust_prefix_to_text_suffix_boundary(
            toks,
            256,
            bg._apc_media_token_ids(),
            max_prefix_tokens=len(toks) - 1,
        )
        assert adjusted == 282

        # Mock / changed signature guards must stand down cleanly.
        assert install_media_token_ids(MagicMock(), fake_vlm, proc) is False
        assert multimodal_token_ids(MagicMock(), MagicMock()) == set()
        bad_bg = SimpleNamespace(_apc_media_token_ids=lambda extra: set())
        with caplog.at_level(logging.WARNING):
            assert install_media_token_ids(bad_bg, fake_vlm, proc) is False
        assert any("stood down" in r.message for r in caplog.records)

    def test_the_salt_helpers_take_what_we_pass(self):
        from mlx_vlm import apc as _apc
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer  # noqa: F401

        salt = inspect.signature(_apc.semantic_extra_hash).parameters
        for name in ("image_hash", "media", "model", "processor"):
            assert name in salt, name
        assert "pixel_values" in inspect.signature(_apc.hash_image_payload).parameters


class TestSpeculativeSurface:
    """What MLX speculative decoding consumes (vlm_engine, mlx_provider,
    loader_routing.mlx_drafter_kind)."""

    def test_the_drafter_surface_we_call(self):
        from mlx_vlm.generate.ar import BatchGenerator, SpeculativeGenerationBatch
        from mlx_vlm.speculative import drafters
        from mlx_vlm.speculative.drafters import mtp_split

        params = inspect.signature(BatchGenerator.__init__).parameters
        for name in ("draft_model", "draft_kind", "draft_block_size"):
            assert name in params, name
        for name in ("load_drafter", "validate_drafter_compatibility"):
            assert callable(getattr(drafters, name)), name
        assert isinstance(drafters.DRAFTER_KIND_BY_MODEL_TYPE, dict)
        for name in ("get_mtp_splitter", "detect_mtp_splitter"):
            assert callable(getattr(mtp_split, name)), name
        assert isinstance(mtp_split.MTP_SPLITTERS, dict)
        # vlm_engine.drafting_blocker keeps a request with logits processors
        # off the drafter because this batch takes none. When it does, the
        # blocker is dead weight. A thinking budget drafts because the batch
        # takes one (mlx-vlm#2641); if that goes, the blocker needs its rule
        # back or a budget is dropped without a word.
        spec = inspect.signature(SpeculativeGenerationBatch.__init__).parameters
        assert "logits_processors" not in spec, (
            "mlx-vlm's speculative batch now takes logits_processors: drop "
            "that rule from vlm_engine.drafting_blocker")
        assert "thinking_budget_criteria" in spec, (
            "mlx-vlm's speculative batch no longer takes thinking_budget_criteria: "
            "restore the thinking-budget rule in vlm_engine.drafting_blocker")

class TestCarriedPatches:
    """providers/common/mlx_vlm_patches.py applies heylook-owned runtime
    customizations and wrappers over mlx-vlm. Each contract test verifies
    safe binding, upstream call-site/signature compatibility, and clean
    stand-down rather than blocking package bumps for upstream PR retirement."""

    def test_2356_carry_binds_per_generator_under_the_cold_rule(self):
        from types import SimpleNamespace

        import mlx.core as mx
        from mlx_vlm import apc as _apc
        from mlx_vlm.apc_coordinator import APCCoordinator
        from mlx_vlm.models.cache import ArraysCache, KVCache
        from heylook_llm.providers.common import mlx_vlm_patches

        def filled():
            arrays, kv = ArraysCache(2), KVCache()
            arrays.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
            kv.keys = mx.ones((1, 1, 32, 4))
            kv.values, kv.offset = kv.keys * 2, 32
            return [arrays, kv]

        manager = _apc.APCManager(num_blocks=8, block_size=16, disk=None)
        tokens = list(range(32))
        assert manager.store_exact_cache(tokens, filled())
        model = SimpleNamespace(make_cache=lambda: [ArraysCache(2), KVCache()])
        name = mlx_vlm_patches.RESTORE_MERGE
        class_method = APCCoordinator.__dict__[name]

        def generator(**over):
            fields = dict(apc=manager.coordinator(model), model=model, kv_bits=None,
                          draft_model=None, draft_kind=None)
            return SimpleNamespace(**{**fields, **over})

        def hit(runner):
            return runner.lookup(tokens + [99], extra_hash=0, safe_lookup_min=0,
                                 suffix_is_text_only=lambda _: True,
                                 prefix_has_media=lambda _: False)

        # The rule leaves these generators on batch caches, as mlx-vlm does
        # cold: a drafter, KV quantization, a model without make_cache.
        for over in ({"draft_model": object(), "draft_kind": "mtp"}, {"kv_bits": 8},
                     {"model": SimpleNamespace()}, {"apc": None}):
            bg = generator(**over)
            assert not mlx_vlm_patches.apply_to(bg), over
            if bg.apc is not None:
                assert not hasattr(getattr(bg.apc, name), "_heylook_patch"), over

        bg = generator()
        assert mlx_vlm_patches.apply_to(bg) and mlx_vlm_patches.apply_to(bg)
        runner = bg.apc
        merge = getattr(runner, name)
        assert merge._heylook_patch == mlx_vlm_patches.PATCH_2356
        assert not hasattr(merge._heylook_upstream, "_heylook_patch")
        assert APCCoordinator.__dict__[name] is class_method, (
            "the carry must bind to one generator, never to the class")

        one = hit(runner)
        before = manager.stats_snapshot()["restored_tokens"]
        caches, prefix = merge([one], [one["prefix_len"]])
        assert caches is one["warm_cache"] and prefix == one["prefix_len"]
        assert manager.stats_snapshot()["restored_tokens"] == before + prefix
        runner.release_hit(one)

        two = hit(runner)
        caches, _ = merge([two, None], [two["prefix_len"], 0])
        assert caches is not two["warm_cache"]
        runner.release_hit(two)

    def test_2356_restore_site_calls_the_method_the_carry_binds(self):
        """The carry wraps one coordinator method. If the restore site stops
        calling it, the carry installs cleanly and never runs: upstream #2646
        moved the site from merge_rows to _merge_and_release_cache_rows and
        nothing noticed until this retirement test errored on the mock."""
        import inspect

        from mlx_vlm.generate.ar import BatchGenerator
        from heylook_llm.providers.common import mlx_vlm_patches

        site = inspect.getsource(BatchGenerator._build_mixed_prompt_batch)
        assert f"coordinator.{mlx_vlm_patches.RESTORE_MERGE}(" in site, (
            "mlx-vlm's restore site no longer calls APCCoordinator."
            f"{mlx_vlm_patches.RESTORE_MERGE}: move the #2356 carry "
            "(mlx_vlm_patches.RESTORE_MERGE) to the call it makes now")

    def test_gdn_packed_kernel_carry_and_retirement(self):
        """Contract & stand-down test for PATCH_GDN_PACKED:
        1. If upstream mlx-vlm already defines _gated_delta_kernel_packed,
           install_gated_delta_packed_kernel() stands down cleanly (returns False).
        2. Otherwise, verify install_gated_delta_packed_kernel() binds cleanly to
           mlx_vlm.models.gated_delta.gated_delta_kernel."""
        import mlx.core as mx
        from mlx_vlm.models import gated_delta as gd
        from heylook_llm.providers.common import mlx_vlm_patches

        if hasattr(gd, "_gated_delta_kernel_packed"):
            assert not mlx_vlm_patches.install_gated_delta_packed_kernel()
            return

        if not mx.metal.is_available():
            pytest.skip("Metal required to compile packed Gated DeltaNet kernel")

        orig = getattr(gd.gated_delta_kernel, "_heylook_upstream", gd.gated_delta_kernel)
        try:
            gd.gated_delta_kernel = orig
            assert mlx_vlm_patches.install_gated_delta_packed_kernel()
            assert getattr(gd.gated_delta_kernel, "_heylook_patch", None) == mlx_vlm_patches.PATCH_GDN_PACKED
            # Idempotent second call
            assert mlx_vlm_patches.install_gated_delta_packed_kernel()
        finally:
            gd.gated_delta_kernel = orig



class TestOwnedVisionPositionGather:
    """heylook binds its own row gather over Gemma 4's one-hot position lookup
    (providers/common/gemma4_vision.bind_position_gather, shared by
    mlx_provider and mlx_embeddings). It must equal the upstream method
    exactly on the CPU stream, and stand down silently on unrelated VLM
    architectures."""

    def test_gather_equals_upstream_one_hot_lookup(self):
        import mlx.core as mx
        from mlx_vlm.models.gemma4 import vision as gemma4_vision
        from mlx_vlm.models.gemma4.config import VisionConfig
        from heylook_llm.providers.common.gemma4_vision import bind_position_gather
        from heylook_llm.providers.mlx_embeddings import (
            bind_position_gather as embed_bind_position_gather,
        )
        assert embed_bind_position_gather is bind_position_gather
        params = list(inspect.signature(gemma4_vision.VisionPatchEmbedder._position_embeddings).parameters)
        assert params == ["self", "patch_positions", "padding_positions"]
        with mx.stream(mx.cpu):
            mx.random.seed(0)
            upstream = gemma4_vision.VisionPatchEmbedder(VisionConfig(hidden_size=8, position_embedding_size=16))
            upstream.position_embedding_table = mx.random.normal((2, 16, 8))
            positions = mx.random.randint(0, 16, (2, 6, 2))
            positions = mx.where(mx.arange(6)[None, :, None] >= mx.array([[[6]], [[4]]]), -1, positions)
            padding = mx.all(positions == -1, axis=-1)
            expected = upstream._position_embeddings(positions, padding)
            model = type("M", (), {})()
            model.vision_tower = type("V", (), {})()
            model.vision_tower.patch_embedder = upstream
            assert bind_position_gather(model)
            got = upstream._position_embeddings(positions, padding)
            assert mx.array_equal(expected, got).item()

    def test_bind_position_gather_ignores_non_gemma_patch_embedder(self, caplog):
        import logging
        import mlx.core as mx
        from mlx_vlm.models.muse_spark import vision as muse_vision
        from mlx_vlm.models.muse_spark.config import VisionConfig
        from heylook_llm.providers.common.gemma4_vision import bind_position_gather
        with mx.stream(mx.cpu), caplog.at_level(logging.WARNING):
            cfg = VisionConfig(
                hidden_size=16,
                num_attention_heads=4,
                intermediate_size=32,
                num_hidden_layers=1,
                pos_emb_height=4,
                pos_emb_width=4,
            )
            model = type("M", (), {})()
            model.config = type("C", (), {"model_type": "muse_spark"})()
            model.vision_tower = type("V", (), {})()
            model.vision_tower.patch_embedder = muse_vision.VisionPatchEmbedder(cfg)
            assert bind_position_gather(model) is False
            assert not caplog.records


class TestOwnedGemma4TextMasks:
    """heylook binds gemma4_make_masks onto loaded Gemma 4 text model instances
    to eliminate per-step GPU syncs and redundant 4D masks on unpadded single-row
    caches while keeping exact output equivalence with upstream _make_masks."""

    @staticmethod
    def _tiny_pair():
        import copy
        from mlx.utils import tree_flatten
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel
        from heylook_llm.providers.common.gemma4_vision import bind_text_masks

        cfg = TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_bidirectional_attention="vision",
            exact_speculative_verify=True,
        )
        upstream = LanguageModel(cfg)
        bound = copy.deepcopy(upstream)
        bound.load_weights(tree_flatten(upstream.parameters()))
        wrapper = type("M", (), {"language_model": bound})()
        assert bind_text_masks(wrapper) is True
        return upstream, bound

    @staticmethod
    def _batch_cache(mx):
        from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache
        return [BatchRotatingKVCache(8, [0]), BatchKVCache([0])]

    def test_single_row_batch_cache_matches_upstream_across_window_and_verify(self):
        import mlx.core as mx
        from mlx_vlm.models.gemma4 import language as gemma4_lang

        params = list(inspect.signature(gemma4_lang.Gemma4TextModel._make_masks).parameters)
        assert params == ["self", "h", "cache", "mm_token_type_ids"]

        with mx.stream(mx.cpu):
            mx.random.seed(7)
            upstream, bound = self._tiny_pair()
            c_up = self._batch_cache(mx)
            c_bd = self._batch_cache(mx)

            # 1. Cold prefill (< sliding_window=8): single-row unpadded batch caches
            # get fast-path ("causal", "causal") instead of 4D boolean masks.
            p1 = mx.array([[1, 2, 3, 4, 5]])
            h1 = mx.random.normal((1, 5, 16))
            assert bound.model._make_masks(h1, c_bd, None) == ["causal", "causal"]
            out_up = upstream(p1, cache=c_up).logits
            out_bd = bound(p1, cache=c_bd).logits
            assert mx.allclose(out_up, out_bd, atol=1e-5).item()

            # 2. Suffix prefill crossing sliding_window=8 (total 10 > 8): sliding gets
            # 4D mask while full_attention stays on the causal fast path.
            p2 = mx.array([[6, 7, 8, 9, 10]])
            masks_suf = bound.model._make_masks(mx.random.normal((1, 5, 16)), c_bd, None)
            assert isinstance(masks_suf[0], mx.array) and masks_suf[0].ndim == 4
            assert masks_suf[1] == "causal"
            out_up = upstream(p2, cache=c_up).logits
            out_bd = bound(p2, cache=c_bd).logits
            assert mx.allclose(out_up, out_bd, atol=1e-5).item()

            # 3. Decode steps past rotation (BatchRotatingKVCache decrements left_padding):
            # both layers skip mask allocation completely (`None`).
            h_dec = mx.random.normal((1, 1, 16))
            for tok in (11, 12, 13):
                assert bound.model._make_masks(h_dec, c_bd, None) == [None, None]
                step = mx.array([[tok]])
                assert mx.allclose(
                    upstream(step, cache=c_up).logits,
                    bound(step, cache=c_bd).logits,
                    atol=1e-5,
                ).item()

            # 4. Exact 4-token speculative verification both inside and past window:
            # past window keeps the 4D sliding mask that _attention slices by ndim >= 4,
            # while full_attention uses causal -> mask=None per query.
            masks_ver = bound.model._make_masks(mx.random.normal((1, 4, 16)), c_bd, None)
            assert isinstance(masks_ver[0], mx.array) and masks_ver[0].ndim >= 4
            assert masks_ver[1] == "causal"
            v_toks = mx.array([[14, 15, 16, 17]])
            assert mx.allclose(
                upstream(v_toks, cache=c_up, speculative_verify=True).logits,
                bound(v_toks, cache=c_bd, speculative_verify=True).logits,
                atol=1e-5,
            ).item()

            # Fresh inside-window verify check (offset=3, 3+4=7 <= 8)
            c_up_short = self._batch_cache(mx)
            c_bd_short = self._batch_cache(mx)
            upstream(mx.array([[1, 2, 3]]), cache=c_up_short)
            bound(mx.array([[1, 2, 3]]), cache=c_bd_short)
            assert mx.allclose(
                upstream(v_toks, cache=c_up_short, speculative_verify=True).logits,
                bound(v_toks, cache=c_bd_short, speculative_verify=True).logits,
                atol=1e-5,
            ).item()

    def test_vision_bidirectional_and_padded_or_batched_caches_match_upstream(self):
        import mlx.core as mx
        from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache

        with mx.stream(mx.cpu):
            mx.random.seed(11)
            upstream, bound = self._tiny_pair()
            h_vis = mx.random.normal((1, 6, 16))
            mm_types = mx.array([[0, 1, 1, 1, 0, 0]])
            masks_up = upstream.model._make_masks(h_vis, self._batch_cache(mx), mm_types)
            masks_bd = bound.model._make_masks(h_vis, self._batch_cache(mx), mm_types)
            for a, b in zip(masks_up, masks_bd):
                assert mx.array_equal(a, b).item()

            # Padded single row and B=2 fall back to exact upstream array masks.
            for pad in ([2], [0, 2]):
                b = len(pad)
                h = mx.random.normal((b, 4, 16))
                cu = [BatchRotatingKVCache(8, list(pad)), BatchKVCache(list(pad))]
                cb = [BatchRotatingKVCache(8, list(pad)), BatchKVCache(list(pad))]
                mu = upstream.model._make_masks(h, cu, None)
                mb = bound.model._make_masks(h, cb, None)
                for a, x in zip(mu, mb):
                    assert isinstance(x, mx.array)
                    assert mx.array_equal(a, x).item()

    def test_bind_text_masks_ignores_non_gemma4_vlm_models(self, caplog):
        import logging
        import mlx.core as mx
        from mlx_vlm.models.embedding_gemma2.config import TextConfig as EmbedTextConfig
        from mlx_vlm.models.embedding_gemma2.language import TextModel as EmbedTextModel
        from heylook_llm.providers.common.gemma4_vision import bind_text_masks

        with mx.stream(mx.cpu), caplog.at_level(logging.WARNING):
            inner = EmbedTextModel(
                EmbedTextConfig(
                    hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                    vocab_size=32, hidden_size_per_layer_input=0, embedding_dim=16,
                    layer_types=["full_attention"],
                )
            )
            model = type("M", (), {"language_model": type("L", (), {"model": inner})()})()
            assert bind_text_masks(model) is False
            assert not caplog.records


class TestVideoWindowHooks:
    """providers/video_windows.py calls two EmbeddingGemma 2 video processor internals:
    the decoder with a caller sampler, and the pre-decoded frames path."""

    def test_decoder_takes_a_sampler_and_preprocess_takes_decoded_frames(self):
        from mlx_vlm.models.embedding_gemma2.video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor as P
        assert list(inspect.signature(P._decode_video).parameters)[-2:] == ["path", "sampler"]
        assert "video_metadata" in inspect.signature(P.preprocess).parameters
        source = inspect.getsource(P.preprocess)
        assert "do_sample_frames" in source and "Set `do_sample_frames=False`" in source


class TestOwnedGemma4Runtime:
    """heylook binds affine QMV kernels, scalar RoPE offsets, Router norm
    caching, unpadded B=1 shared KV views, hoisted drafter masks, and single-
    shot B=1 MTP verification onto Gemma 4 instances."""

    @staticmethod
    def _tiny_config(**overrides):
        from mlx_vlm.models.gemma4.config import TextConfig

        defaults = dict(
            hidden_size=256,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=64,
            global_head_dim=64,
            vocab_size=64,
            hidden_size_per_layer_input=0,
            sliding_window=8,
            sliding_window_pattern=2,
            num_kv_shared_layers=0,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 10000.0},
            },
            final_logit_softcapping=30.0,
        )
        defaults.update(overrides)
        return TextConfig(**defaults)

    @pytest.mark.parametrize("bits", [8, 4])
    def test_qmv_multi_matches_t1_qmv_and_stands_down(self, bits):
        import mlx.core as mx
        import mlx.nn as nn
        from heylook_llm.providers.common.gemma4_runtime import gemma4_qmv_multi

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            pytest.skip("Metal GPU required for affine QMV kernel")

        mx.random.seed(0)
        k_dim = 256 if bits == 8 else 512
        box = nn.Module()
        box.lin = nn.Linear(k_dim, 64, bias=False)
        box.lin.update({"weight": mx.random.normal((64, k_dim)).astype(mx.bfloat16)})
        box.emb = nn.Embedding(64, k_dim)
        box.emb.update({"weight": mx.random.normal((64, k_dim)).astype(mx.bfloat16)})
        nn.quantize(box, group_size=64, bits=bits)
        lin, emb = box.lin, box.emb
        x = mx.random.normal((1, 5, k_dim)).astype(mx.bfloat16)

        ref_lin = mx.concatenate([lin(x[:, t : t + 1, :]) for t in range(5)], axis=1)
        got_lin = gemma4_qmv_multi(lin, x, min_out_rows=64)
        mx.eval(ref_lin, got_lin)
        assert got_lin is not None and mx.array_equal(got_lin, ref_lin).item()

        ref_emb = mx.concatenate(
            [emb.as_linear(x[:, t : t + 1, :]) for t in range(5)], axis=1
        )
        got_emb = gemma4_qmv_multi(emb, x, min_out_rows=64)
        mx.eval(ref_emb, got_emb)
        assert got_emb is not None and mx.array_equal(got_emb, ref_emb).item()

        # Default threshold stands down on narrow matrices, unaligned 4-bit K, and float32 scales.
        assert gemma4_qmv_multi(lin, x) is None
        if bits == 4:
            short_box = nn.Module()
            short_box.lin = nn.Linear(256, 64, bias=False)
            short_box.lin.update({"weight": mx.random.normal((64, 256)).astype(mx.bfloat16)})
            nn.quantize(short_box, group_size=64, bits=4)
            assert gemma4_qmv_multi(short_box.lin, x[:, :, :256], min_out_rows=64) is None
        f32_lin = nn.QuantizedLinear(k_dim, 64, group_size=64, bits=bits, bias=False)
        assert gemma4_qmv_multi(f32_lin, x, min_out_rows=64) is None

    def test_qmv_fused_and_mlp_dynamic_subclass(self):
        import copy
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.gemma4.language import MLP
        from heylook_llm.providers.common.gemma4_runtime import (
            _bind_mlp_subclass,
            gemma4_qmv_fused,
        )

        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            pytest.skip("Metal GPU required for fused affine QMV kernel")

        mx.random.seed(1)
        cfg = self._tiny_config(hidden_size=256, intermediate_size=64)
        mlp_up = MLP(cfg)
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp_up, name)
            proj.update(
                {"weight": mx.random.normal(proj.weight.shape).astype(mx.bfloat16)}
            )
        nn.quantize(mlp_up, group_size=64, bits=8)
        mlp_bd = copy.deepcopy(mlp_up)
        assert _bind_mlp_subclass(mlp_bd, min_out_rows=128) is True

        x = mx.random.normal((1, 5, 256)).astype(mx.bfloat16)
        fg, fu = gemma4_qmv_fused((mlp_up.gate_proj, mlp_up.up_proj), x, min_out_rows=128)
        rg = mx.concatenate([mlp_up.gate_proj(x[:, t : t + 1, :]) for t in range(5)], axis=1)
        ru = mx.concatenate([mlp_up.up_proj(x[:, t : t + 1, :]) for t in range(5)], axis=1)
        mx.eval(fg, fu, rg, ru)
        assert mx.array_equal(fg, rg).item() and mx.array_equal(fu, ru).item()

        # Verify nn.Module __call__ special-method dispatch hits the subclass on mlp(x).
        from mlx_vlm.models.gemma4.language import geglu

        out_bd = mlp_bd(x)
        ref_out = mlp_up.down_proj(geglu(rg, ru))
        mx.eval(out_bd, ref_out)
        assert mx.array_equal(out_bd, ref_out).item()

    def test_attention_scalar_rope_offset_and_cache_lifecycle(self):
        import copy
        import mlx.core as mx
        from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache
        from mlx_vlm.models.gemma4.language import Attention
        from mlx_vlm.speculative.cache_state import start_speculative_cache
        from heylook_llm.providers.common.gemma4_runtime import _bind_attention_subclass

        mx.random.seed(2)
        cfg = self._tiny_config(hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64)
        attn_up = Attention(cfg, layer_idx=0)
        attn_bd = copy.deepcopy(attn_up)
        assert _bind_attention_subclass(attn_bd) is True

        # 1. KVCache & BatchKVCache([0])
        for make_c in (lambda: KVCache(), lambda: BatchKVCache([0])):
            cu, cb = make_c(), make_c()
            x_pre = mx.random.normal((1, 4, 128))
            y_u, _, off_u = attn_up(x_pre, cache=cu)
            y_b, _, off_b = attn_bd(x_pre, cache=cb)
            assert isinstance(off_b, int) and off_b == 0
            x_dec = mx.random.normal((1, 1, 128))
            y_u2, _, _ = attn_up(x_dec, cache=cu)
            y_b2, _, off_b2 = attn_bd(x_dec, cache=cb)
            assert isinstance(off_b2, int) and off_b2 == 4
            mx.eval(y_u, y_b, y_u2, y_b2)
            assert mx.array_equal(y_u, y_b).item()
            assert mx.array_equal(y_u2, y_b2).item()

        # 2. BatchRotatingKVCache across rotation + _RotatingCacheTransaction rollback
        ru = BatchRotatingKVCache(8, [0])
        rb = BatchRotatingKVCache(8, [0])
        x_fill = mx.random.normal((1, 10, 128))
        attn_up(x_fill, cache=ru)
        attn_bd(x_fill, cache=rb)
        # Trigger partial speculative rollback (abort clones left_padding array id!)
        tu = start_speculative_cache([ru], 4)
        tb = start_speculative_cache([rb], 4)
        x_spec = mx.random.normal((1, 4, 128))
        attn_up(x_spec, cache=ru)
        attn_bd(x_spec, cache=rb)
        tu.commit([2])
        tb.commit([2])
        x_after = mx.random.normal((1, 1, 128))
        yu, _, offu = attn_up(x_after, cache=ru)
        yb, _, offb = attn_bd(x_after, cache=rb)
        mx.eval(yu, yb, offu)
        assert isinstance(offb, int) and offb == int(offu.item()) == 12
        assert mx.array_equal(yu, yb).item()

        # 3. Merged rotated BatchRotatingKVCache where offset (25) > max_size (8)
        rc = RotatingKVCache(8)
        rc.update_and_fetch(mx.ones((1, 1, 25, 64)), mx.ones((1, 1, 25, 64)))
        mu = BatchRotatingKVCache.merge([copy.deepcopy(rc)])
        mb = BatchRotatingKVCache.merge([copy.deepcopy(rc)])
        yu_m, _, offu_m = attn_up(x_after, cache=mu)
        yb_m, _, offb_m = attn_bd(x_after, cache=mb)
        mx.eval(yu_m, yb_m, offu_m)
        assert isinstance(offb_m, int) and offb_m == int(offu_m.item()) == 25
        assert mx.array_equal(yu_m, yb_m).item()
        _, _, offb_m2 = attn_bd(x_after, cache=mb)
        assert isinstance(offb_m2, int) and offb_m2 == 26

        # 4. Padded multi-row batch falls back to mx.array offset cleanly
        pu, pb = BatchKVCache([2, 0]), BatchKVCache([2, 0])
        x2 = mx.random.normal((2, 3, 128))
        yu2, _, _ = attn_up(x2, cache=pu)
        yb2, _, off2 = attn_bd(x2, cache=pb)
        mx.eval(yu2, yb2)
        assert isinstance(off2, mx.array) and mx.array_equal(yu2, yb2).item()

    def test_router_and_shared_kv_from_cache_match_upstream(self):
        import copy
        import mlx.core as mx
        from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache
        from mlx_vlm.models.gemma4.language import LanguageModel, Router
        from heylook_llm.providers.common.gemma4_runtime import (
            _bind_router_subclass,
            bind_gemma4_runtime,
        )

        mx.random.seed(3)
        cfg = self._tiny_config(
            hidden_size=64,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            vocab_size=64,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=32,
            enable_moe_block=True,
        )
        r_up = Router(cfg)
        r_bd = copy.deepcopy(r_up)
        assert _bind_router_subclass(r_bd) is True
        xr = mx.random.normal((1, 4, 64))
        iu, wu = r_up(xr)
        ib, wb = r_bd(xr)
        first_id = id(r_bd._heylook_scaled_norm_weight)
        ib2, wb2 = r_bd(xr)
        assert id(r_bd._heylook_scaled_norm_weight) == first_id
        mx.eval(iu, wu, ib, wb, ib2, wb2)
        assert mx.array_equal(iu, ib).item() and mx.array_equal(wu, wb).item()

        lm_up = LanguageModel(cfg)
        lm_bd = copy.deepcopy(lm_up)
        wrapper = type("M", (), {"language_model": lm_bd})()
        assert bind_gemma4_runtime(wrapper) is True

        # Both unrotated and rotated BatchRotatingKVCache + BatchKVCache
        for step_len in (4, 11):
            c0 = BatchRotatingKVCache(6, [0])
            c1 = BatchKVCache([0])
            c0.update_and_fetch(
                mx.random.normal((1, 1, step_len, 32)),
                mx.random.normal((1, 1, step_len, 32)),
            )
            c1.update_and_fetch(
                mx.random.normal((1, 1, step_len, 32)),
                mx.random.normal((1, 1, step_len, 32)),
            )
            su = lm_up.shared_kv_from_cache([c0, c1])
            sb = lm_bd.shared_kv_from_cache([c0, c1])
            assert set(su) == set(sb)
            for k in su:
                assert mx.array_equal(su[k][0], sb[k][0]).item()
                assert mx.array_equal(su[k][1], sb[k][1]).item()

        # Multi-row padded fallback matches upstream
        p0, p1 = BatchRotatingKVCache(6, [1, 0]), BatchKVCache([1, 0])
        p0.update_and_fetch(mx.random.normal((2, 1, 4, 32)), mx.random.normal((2, 1, 4, 32)))
        p1.update_and_fetch(mx.random.normal((2, 1, 4, 32)), mx.random.normal((2, 1, 4, 32)))
        su_p = lm_up.shared_kv_from_cache([p0, p1])
        sb_p = lm_bd.shared_kv_from_cache([p0, p1])
        for k in su_p:
            assert mx.array_equal(su_p[k][0], sb_p[k][0]).item()
            assert mx.array_equal(su_p[k][1], sb_p[k][1]).item()

    def test_bind_gemma4_drafter_survives_copy_and_hoists_masks(self, monkeypatch):
        import copy
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.speculative.drafters.gemma4_assistant.config import (
            Gemma4AssistantConfig,
        )
        from mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant import (
            Gemma4AssistantDraftModel,
        )
        from mlx_vlm.speculative.drafters.gemma4_assistant import masks as drafter_masks
        from heylook_llm.providers.common.gemma4_runtime import bind_gemma4_drafter

        mx.random.seed(4)
        text_cfg = self._tiny_config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            sliding_window=8,
            vocab_size=32,
            num_kv_shared_layers=2,
            tie_word_embeddings=True,
        )
        d_cfg = Gemma4AssistantConfig(
            backbone_hidden_size=16,
            text_config=text_cfg,
            tie_word_embeddings=True,
            use_ordered_embeddings=False,
        )
        target_embed = nn.Embedding(32, 16)
        target = type(
            "Target",
            (),
            {
                "language_model": type(
                    "LM",
                    (),
                    {
                        "config": text_cfg,
                        "model": type(
                            "M", (), {"embed_tokens": target_embed, "embed_scale": 1.0}
                        )(),
                    },
                )()
            },
        )()

        d_up = Gemma4AssistantDraftModel(d_cfg).bind(target)
        d_bd = copy.deepcopy(d_up)
        assert bind_gemma4_drafter(d_bd) is True

        d_copy = copy.copy(d_bd)
        d_copy.reset(target)
        shared_kv = {
            "sliding_attention": (mx.random.normal((1, 1, 6, 8)), mx.random.normal((1, 1, 6, 8))),
            "full_attention": (mx.random.normal((1, 1, 6, 8)), mx.random.normal((1, 1, 6, 8))),
        }
        d_up.set_shared_kv(shared_kv, kv_offset=6, position=5, kv_valid_len=6)
        d_copy.set_shared_kv(shared_kv, kv_offset=6, position=5, kv_valid_len=6)

        hidden = mx.random.normal((1, 1, 16))
        sampler = lambda logits: mx.argmax(logits, axis=-1)
        toks_up = d_up.draft_block(3, hidden, None, 5, sampler, greedy=False)

        calls = 0
        orig_make_masks = drafter_masks.make_drafter_masks

        def counting_make_masks(*args, **kwargs):
            nonlocal calls
            calls += 1
            return orig_make_masks(*args, **kwargs)

        monkeypatch.setattr(drafter_masks, "make_drafter_masks", counting_make_masks)
        toks_bd = d_copy.draft_block(3, hidden, None, 5, sampler, greedy=False)
        mx.eval(toks_up, toks_bd)
        assert calls == 1
        assert mx.array_equal(toks_up, toks_bd).item()

    def test_install_mtp_single_row_fast_paths_matches_upstream(self):
        import mlx.core as mx
        from mlx_vlm.models.cache import BatchKVCache
        from mlx_vlm.speculative import mtp
        from heylook_llm.providers.common import mlx_vlm_patches

        mx.random.seed(5)
        orig_walk = getattr(
            mtp._speculative_walk_batch_deferred_greedy,
            "_heylook_upstream",
            mtp._speculative_walk_batch_deferred_greedy,
        )
        orig_pad = getattr(
            mtp._batch_cache_left_padding,
            "_heylook_upstream",
            mtp._batch_cache_left_padding,
        )
        try:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk
            mtp._batch_cache_left_padding = orig_pad

            calls = []

            class FakeLM:
                _heylook_gemma4_runtime = True

                def speculative_logits_from_hidden(self, h):
                    calls.append(int(h.shape[1]))
                    # Encode deterministic preferred token per time step.
                    B, T, _ = h.shape
                    vocab = mx.zeros((B, T, 16))
                    # Position values from h[:, :, 0] pick argmax token index.
                    idx = h[0, :, 0].astype(mx.int32).tolist()
                    rows = []
                    for tok in idx:
                        row = [0.0] * 16
                        row[tok % 16] = 10.0
                        rows.append(row)
                    return mx.array(rows)[None, :, :]

            lm = FakeLM()
            draft_tokens = mx.array([[2, 4, 6, 8]])
            # Expect accept pos 0 (2), pos 1 (4), reject pos 2 (got 7 != 6) -> 2 accepted, [2, 4, 7]
            hidden = mx.array([[[2.0], [4.0], [7.0], [8.0], [1.0]]])
            greedy = lambda lp: mx.argmax(lp, axis=-1)

            up_acc, up_toks = orig_walk(lm, hidden, draft_tokens, greedy, [10])
            assert calls == [1, 1, 1]

            calls.clear()
            assert mlx_vlm_patches.install_mtp_single_row_fast_paths()
            assert mlx_vlm_patches.install_mtp_single_row_fast_paths()

            bd_acc, bd_toks = mtp._speculative_walk_batch_deferred_greedy(
                lm, hidden, draft_tokens, greedy, [10]
            )
            assert calls == [5]
            assert (bd_acc, bd_toks) == (up_acc, up_toks) == ([2], [[2, 4, 7]])

            # Positioned sampler parity (row_ids / positions recorded identically).
            class PositionedSampler:
                def __init__(self):
                    self.seen = []

                def sample_target(self, lp, *, row_ids, positions):
                    self.seen.append((list(row_ids), list(positions)))
                    return mx.argmax(lp, axis=-1)

            ps_up = PositionedSampler()
            up_p = orig_walk(lm, hidden, draft_tokens, ps_up, [2], row_ids=[3], base_positions=[10])
            ps_bd = PositionedSampler()
            bd_p = mtp._speculative_walk_batch_deferred_greedy(
                lm, hidden, draft_tokens, ps_bd, [2], row_ids=[3], base_positions=[10]
            )
            assert bd_p == up_p == ([2], [[2, 4]])
            assert ps_bd.seen == [([3, 3, 3, 3, 3], [10, 11, 12, 13, 14])]

            # Unpadded B=1 BatchKVCache skips left_padding array extraction; B=2 falls back.
            assert mtp._batch_cache_left_padding([BatchKVCache([0])]) is None
            pad_b2 = mtp._batch_cache_left_padding([BatchKVCache([2, 0])])
            assert pad_b2 is not None and pad_b2.tolist() == [2, 0]
        finally:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk
            mtp._batch_cache_left_padding = orig_pad
