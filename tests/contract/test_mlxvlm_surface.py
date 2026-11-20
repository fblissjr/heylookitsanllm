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
    def _is_mlx(name: str) -> bool:
        return name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx_vlm"))

    saved = {name: mod for name, mod in sys.modules.items() if _is_mlx(name)}
    sys.modules.update(_REAL_MLX_SYS_MODULES)
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _is_mlx(n)]:
            if name not in saved:
                sys.modules.pop(name, None)
        sys.modules.update(saved)


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

    def test_apc_takes_our_overrides_and_no_disk(self, monkeypatch):
        from heylook_llm.providers.common.vlm_engine import (
            APC_CHECKPOINT_ENTRIES, APC_CHECKPOINT_INTERVAL_TOKENS, APC_MEMORY_ENV,
            APC_MEMORY_SHARE, make_apc_manager)

        # The byte budget is ours (a share of the working set), not mlx-vlm's
        # 8 GiB ceiling: an override key mlx-vlm stops reading would put every
        # model back on that ceiling with nothing failing. An operator's
        # APC_MEMORY_MAX_GB still wins.
        monkeypatch.delenv(APC_MEMORY_ENV, raising=False)
        mgr = make_apc_manager()
        assert mgr._working_set_bytes, "mlx-vlm no longer reads the working set it sizes from"
        assert mgr.memory_max_bytes == int(mgr._working_set_bytes * APC_MEMORY_SHARE)
        monkeypatch.setenv(APC_MEMORY_ENV, "3")
        assert make_apc_manager().memory_max_bytes == 3 * 2**30
        monkeypatch.delenv(APC_MEMORY_ENV)

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
        # stable_prefix_len), retains real boundaries >= interval apart, skips
        # <64-token micro-deltas, and adds coarse interior checkpoints.
        stable_val = 220
        mgr = SimpleNamespace(checkpoint_interval_tokens=64, _exact_cache_max=32,
                              disk=None, block_size=16, exact_cache_min_tokens=16,
                              checkpoint_prefix_length=lambda *_a, **_kw: stable_val)
        coord = SimpleNamespace(enabled=True, is_checkpoint=True, manager=mgr,
                                checkpoint_len=lambda ids, media: 400)
        bg = SimpleNamespace(apc=coord)
        install_capture_policy(bg, [32])
        expected_params = [
            name for name in inspect.signature(APCCoordinator.checkpoint_lengths).parameters
            if name != "self"
        ]
        assert list(inspect.signature(coord.checkpoint_lengths).parameters) == expected_params
        got = coord.checkpoint_lengths(list(range(401)), set(), extra_hash=7, stable_prefix_len=128)
        assert 32 in got and 128 in got and 220 in got and 400 in got
        stable_val = 48  # 48 - 32 = 16 < 64: suppressed as micro-delta
        got_micro = coord.checkpoint_lengths(list(range(401)), set(), extra_hash=7)
        assert 32 in got_micro and 48 not in got_micro
        coord_long = SimpleNamespace(
            enabled=True,
            is_checkpoint=True,
            manager=SimpleNamespace(
                checkpoint_interval_tokens=64,
                _exact_cache_max=32,
                disk=None,
                block_size=16,
                exact_cache_min_tokens=16,
                checkpoint_prefix_length=lambda *_a, **_kw: 0,
            ),
            checkpoint_len=lambda ids, media: 2174,
        )
        bg_long = SimpleNamespace(apc=coord_long)
        install_capture_policy(bg_long, [])
        assert 1024 in coord_long.checkpoint_lengths(list(range(2175)), set())

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
        # SpeculativeGenerationBatch takes no logits_processors natively, so
        # vlm_engine.drafting_blocker keeps stateful/structured processors and
        # non-MTP drafters off the drafter while allowing MTP-safe stateless
        # penalties inside PATCH_MTP_B1_WALK. A thinking budget drafts because
        # the batch takes thinking_budget_criteria (mlx-vlm#2641).
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

        # The rule leaves non-#2356 generators returning False so callers know #2356
        # did not bind: drafted checkpoint generators receive PATCH_RESTORE_BATCH
        # instead, while KV quantization, missing make_cache, or disabled APC
        # leave _merge_and_release_cache_rows untouched.
        drafted_bg = generator(draft_model=object(), draft_kind="mtp")
        assert not mlx_vlm_patches.apply_to(drafted_bg)
        assert not mlx_vlm_patches.apply_to(drafted_bg)
        assert getattr(getattr(drafted_bg.apc, name), "_heylook_patch", None) == (
            mlx_vlm_patches.PATCH_RESTORE_BATCH
        )
        for over in ({"kv_bits": 8}, {"model": SimpleNamespace()}, {"apc": None}):
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

    def test_store_checkpoint_carry_trims_rotating_and_preserves_semantics(self, caplog):
        """PATCH_STORE_CHECKPOINT binds per checkpoint generator (drafted and
        undrafted), trims RotatingKVCache/BatchRotatingKVCache to max_size,
        avoids double copy, and falls back or stands down cleanly."""
        import logging
        from types import SimpleNamespace

        import mlx.core as mx
        from mlx_vlm import apc as _apc
        from mlx_vlm.apc_coordinator import APCCoordinator
        from mlx_vlm.generate.ar import PromptProcessingBatch
        from mlx_vlm.models.cache import (
            ArraysCache,
            BatchKVCache,
            BatchRotatingKVCache,
            KVCache,
            RotatingKVCache,
        )
        from heylook_llm.providers.common import mlx_vlm_patches

        # Pin upstream call sites that invoke APCCoordinator.store_checkpoint.
        assert f"coordinator.{mlx_vlm_patches.STORE_CHECKPOINT}(" in inspect.getsource(
            PromptProcessingBatch._store_apc_exact_checkpoints
        )
        assert f"self.{mlx_vlm_patches.STORE_CHECKPOINT}(" in inspect.getsource(
            APCCoordinator.commit
        )

        # Stand-down check on altered signature.
        bad_coord = SimpleNamespace(store_checkpoint=lambda wrong: False)
        with caplog.at_level(logging.WARNING):
            assert not mlx_vlm_patches._patch_store_checkpoint(bad_coord)
        assert any(mlx_vlm_patches.PATCH_STORE_CHECKPOINT in r.message for r in caplog.records)

        class_method = APCCoordinator.__dict__[mlx_vlm_patches.STORE_CHECKPOINT]
        model = SimpleNamespace(make_cache=lambda: [RotatingKVCache(max_size=8), KVCache()])

        def make_mgr():
            return _apc.APCManager(num_blocks=8, block_size=16, disk=None, overrides={"checkpoint_entries": 8})

        # Binds on BOTH undrafted and drafted checkpoint generators without mutating class.
        mgr_u = make_mgr()
        bg_u = SimpleNamespace(apc=mgr_u.coordinator(model), model=model, kv_bits=None,
                               draft_model=None, draft_kind=None)
        assert mlx_vlm_patches.apply_to(bg_u) is True
        assert getattr(bg_u.apc.store_checkpoint, "_heylook_patch", None) == mlx_vlm_patches.PATCH_STORE_CHECKPOINT

        mgr_d = make_mgr()
        bg_d = SimpleNamespace(apc=mgr_d.coordinator(model), model=model, kv_bits=None,
                               draft_model=object(), draft_kind="mtp")
        # apply_to still returns False when #2356 does not bind (drafting), while
        # still patching store_checkpoint on the checkpoint coordinator.
        assert mlx_vlm_patches.apply_to(bg_d) is False
        assert getattr(bg_d.apc.store_checkpoint, "_heylook_patch", None) == mlx_vlm_patches.PATCH_STORE_CHECKPOINT
        assert APCCoordinator.__dict__[mlx_vlm_patches.STORE_CHECKPOINT] is class_method

        # 1. RotatingKVCache (keep=0) & BatchRotatingKVCache (B=1, left_padding=[0]):
        # 14 keys with max_size=8, offset=20 -> trims to trailing 8, offset=20, _idx=8.
        base_k = mx.broadcast_to(mx.arange(14, dtype=mx.float32).reshape(1, 1, 14, 1), (1, 1, 14, 4))
        base_v = base_k + 100.0
        mx.eval(base_k, base_v)

        def make_rot(*, keep=0):
            r = RotatingKVCache(max_size=8, keep=keep)
            r.keys = mx.array(base_k)
            r.values = mx.array(base_v)
            r.offset = 20
            r._idx = 14
            return r

        rot = make_rot(keep=0)
        kv = KVCache()
        kv.keys = mx.ones((1, 1, 20, 4))
        kv.values = kv.keys * 2
        kv.offset = 20

        tokens_u = list(range(20))
        assert bg_u.apc.store_checkpoint(tokens_u, [rot, kv])
        stored_u = mgr_u._exact_cache[next(iter(mgr_u._exact_cache))].prompt_cache
        assert stored_u[0].keys.shape == (1, 1, 8, 4)
        assert mx.array_equal(stored_u[0].keys, base_k[..., -8:, :])
        assert mx.array_equal(stored_u[0].values, base_v[..., -8:, :])
        assert (stored_u[0].offset, stored_u[0]._idx) == (20, 8)
        # Detached from source mutation.
        rot.keys = rot.keys * 0.0
        assert mx.array_equal(stored_u[0].keys, base_k[..., -8:, :])

        # BatchRotatingKVCache + BatchKVCache single-row store.
        brot = BatchRotatingKVCache(max_size=8, left_padding=[0])
        brot.keys = mx.array(base_k)
        brot.values = mx.array(base_v)
        brot.offset = mx.array([20])
        brot._idx = 14
        brot._offset = 20
        bkv = BatchKVCache(left_padding=[0])
        bkv.keys = mx.ones((1, 1, 24, 4))  # step-padded beyond _idx=20
        bkv.values = bkv.keys * 3
        bkv.offset = mx.array([20])
        bkv._idx = 20
        assert bg_d.apc.store_checkpoint(tokens_u, [brot, bkv])
        stored_d = mgr_d._exact_cache[next(iter(mgr_d._exact_cache))].prompt_cache
        assert isinstance(stored_d[0], RotatingKVCache)
        assert isinstance(stored_d[1], KVCache)
        assert stored_d[0].keys.shape == (1, 1, 8, 4)
        assert mx.array_equal(stored_d[0].keys, base_k[..., -8:, :])
        assert mx.array_equal(stored_d[0].values, base_v[..., -8:, :])
        assert (stored_d[0].offset, stored_d[0]._idx) == (20, 8)
        assert stored_d[1].keys.shape == (1, 1, 20, 4)
        assert stored_d[1].offset == 20

        # 2. RotatingKVCache with keep=2 preserves 2 sink tokens + trailing 6 tokens.
        rot_keep = make_rot(keep=2)
        mgr_k = make_mgr()
        bg_k = SimpleNamespace(apc=mgr_k.coordinator(model), model=model, kv_bits=None,
                               draft_model=None, draft_kind=None)
        mlx_vlm_patches.apply_to(bg_k)
        assert bg_k.apc.store_checkpoint(tokens_u, [rot_keep])
        stored_k = mgr_k._exact_cache[next(iter(mgr_k._exact_cache))].prompt_cache[0]
        expected_keep = mx.concatenate([base_k[..., :2, :], base_k[..., -6:, :]], axis=2)
        assert stored_k.keys.shape == (1, 1, 8, 4)
        assert mx.array_equal(stored_k.keys, expected_keep)
        assert (stored_k.offset, stored_k._idx, stored_k.keep) == (20, 8, 2)

        # 3. Bit-exact equivalence after restore across BatchRotatingKVCache.merge,
        # direct S=1 _update_in_place, and direct S=3 _update_concat.
        untrimmed_ref = make_rot(keep=0)
        restored_hit = mgr_u.lookup_exact_cache(tokens_u + [999])[0][0]
        m_trim = BatchRotatingKVCache.merge([restored_hit])
        m_untrim = BatchRotatingKVCache.merge([untrimmed_ref])
        assert mx.array_equal(m_trim.keys, m_untrim.keys)
        assert mx.array_equal(m_trim.values, m_untrim.values)
        assert m_trim.offset.tolist() == m_untrim.offset.tolist()
        assert (m_trim._idx, m_trim._offset) == (m_untrim._idx, m_untrim._offset)

        k1 = mx.full((1, 1, 1, 4), 77.0, dtype=mx.float32)
        v1 = mx.full((1, 1, 1, 4), 88.0, dtype=mx.float32)
        r_trim_1 = mgr_u.lookup_exact_cache(tokens_u + [999])[0][0]
        r_untrim_1 = make_rot(keep=0)
        out_t1 = r_trim_1.update_and_fetch(k1, v1)
        out_u1 = r_untrim_1.update_and_fetch(k1, v1)
        assert mx.array_equal(out_t1[0], out_u1[0]) and mx.array_equal(out_t1[1], out_u1[1])
        assert (r_trim_1.offset, r_trim_1._idx) == (r_untrim_1.offset, r_untrim_1._idx)

        k3 = mx.full((1, 1, 3, 4), 55.0, dtype=mx.float32)
        v3 = mx.full((1, 1, 3, 4), 66.0, dtype=mx.float32)
        r_trim_3 = mgr_u.lookup_exact_cache(tokens_u + [999])[0][0]
        r_untrim_3 = make_rot(keep=0)
        out_t3 = r_trim_3.update_and_fetch(k3, v3)
        out_u3 = r_untrim_3.update_and_fetch(k3, v3)
        assert mx.array_equal(out_t3[0], out_u3[0]) and mx.array_equal(out_t3[1], out_u3[1])
        assert (r_trim_3.offset, r_trim_3._idx) == (r_untrim_3.offset, r_untrim_3._idx)

        # 4. ArraysCache + BatchKVCache single-row store detaches cleanly.
        arr = ArraysCache(2)
        arr.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
        assert bg_u.apc.store_checkpoint(tokens_u, [arr, BatchKVCache([0])])
        stored_arr = mgr_u._exact_cache[next(reversed(mgr_u._exact_cache))].prompt_cache[0]
        arr.cache[0] = arr.cache[0] * 0.0
        assert mx.array_equal(stored_arr.cache[0], mx.ones((1, 2, 4)))

        # 5. Fallback to upstream on batch_idx=1, B=2, and rotated=True.
        fallback_calls = []
        custom_coord = SimpleNamespace(
            enabled=True,
            is_checkpoint=True,
            manager=make_mgr(),
            store_checkpoint=lambda token_ids, prompt_cache, *, extra_hash=0, batch_idx=None: (
                fallback_calls.append(batch_idx) or True
            ),
        )
        assert mlx_vlm_patches._patch_store_checkpoint(custom_coord)
        assert custom_coord.store_checkpoint(tokens_u, [rot, kv], batch_idx=1) is True
        b2 = BatchKVCache([0, 0])
        assert custom_coord.store_checkpoint(tokens_u, [b2], batch_idx=0) is True
        brot_rot = BatchRotatingKVCache(max_size=8, left_padding=[0])
        brot_rot.keys, brot_rot.values = mx.array(base_k), mx.array(base_v)
        brot_rot.offset, brot_rot._idx, brot_rot.rotated = mx.array([20]), 8, True
        assert custom_coord.store_checkpoint(tokens_u, [brot_rot], batch_idx=0) is True
        assert fallback_calls == [1, 0, 0]

    def test_batch_restore_carry_wraps_single_row_without_double_copy(self, caplog):
        """PATCH_RESTORE_BATCH binds per drafted checkpoint generator, wraps
        single-row snapshots into Batch* caches with one batched eval, preserves
        pre-reserved KVCache capacity, matches upstream numerically, and falls
        back or stands down cleanly."""
        import logging
        from types import SimpleNamespace

        import mlx.core as mx
        from mlx_vlm import apc as _apc
        from mlx_vlm.apc_coordinator import APCCoordinator
        from mlx_vlm.models.cache import (
            ArraysCache,
            BatchKVCache,
            BatchRotatingKVCache,
            KVCache,
            RotatingKVCache,
        )
        from heylook_llm.providers.common import mlx_vlm_patches

        # Stand-down check on altered signature.
        bad_coord = SimpleNamespace(_merge_and_release_cache_rows=lambda wrong: (None, 0))
        with caplog.at_level(logging.WARNING):
            assert not mlx_vlm_patches._patch_batch_restore(bad_coord)
        assert any(mlx_vlm_patches.PATCH_RESTORE_BATCH in r.message for r in caplog.records)

        class_method = APCCoordinator.__dict__[mlx_vlm_patches.RESTORE_MERGE]
        model = SimpleNamespace(make_cache=lambda: [RotatingKVCache(max_size=8), KVCache()])
        mgr = _apc.APCManager(
            num_blocks=8, block_size=16, disk=None, overrides={"checkpoint_entries": 8}
        )
        bg = SimpleNamespace(
            apc=mgr.coordinator(model),
            model=model,
            kv_bits=None,
            draft_model=object(),
            draft_kind="mtp",
        )
        assert mlx_vlm_patches.apply_to(bg) is False
        assert getattr(bg.apc._merge_and_release_cache_rows, "_heylook_patch", None) == (
            mlx_vlm_patches.PATCH_RESTORE_BATCH
        )
        assert APCCoordinator.__dict__[mlx_vlm_patches.RESTORE_MERGE] is class_method

        # 1. Exact RotatingKVCache + pre-reserved KVCache parity & update equivalence.
        base_rk = mx.broadcast_to(
            mx.arange(8, dtype=mx.float32).reshape(1, 1, 8, 1), (1, 1, 8, 4)
        )
        base_rv = base_rk + 10.0
        base_kk = mx.broadcast_to(
            mx.arange(32, dtype=mx.float32).reshape(1, 1, 32, 1), (1, 1, 32, 4)
        )
        base_kv = base_kk + 50.0
        mx.eval(base_rk, base_rv, base_kk, base_kv)

        r_exact = RotatingKVCache(max_size=8)
        r_exact.keys, r_exact.values = mx.array(base_rk), mx.array(base_rv)
        r_exact.offset, r_exact._idx = 20, 8

        kv_init = KVCache()
        kv_init.keys, kv_init.values, kv_init.offset = mx.array(base_kk), mx.array(base_kv), 32

        tokens = list(range(32))
        assert mgr.store_exact_cache(tokens, [r_exact, kv_init])
        source_entry = mgr._exact_cache[next(iter(mgr._exact_cache))].prompt_cache

        hit_pat = bg.apc.lookup(
            tokens + [99],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _: True,
            prefix_has_media=lambda _: False,
        )
        hit_up = bg.apc.lookup(
            tokens + [99],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _: True,
            prefix_has_media=lambda _: False,
        )
        assert hit_pat["warm_cache"][1].keys.shape[2] >= 256  # reserved by lookup

        before_tok = mgr.stats_snapshot()["restored_tokens"]
        orig_warm = hit_pat["warm_cache"]
        b_pat, pref_pat = bg.apc._merge_and_release_cache_rows(
            [hit_pat], [hit_pat["prefix_len"]]
        )
        assert pref_pat == 32
        assert all(x is None for x in orig_warm)
        assert mgr.stats_snapshot()["restored_tokens"] == before_tok + 32

        upstream_fn = bg.apc._merge_and_release_cache_rows._heylook_upstream
        b_up, pref_up = upstream_fn([hit_up], [hit_up["prefix_len"]])
        assert pref_pat == pref_up

        assert type(b_pat[0]) is BatchRotatingKVCache
        assert mx.array_equal(b_pat[0].keys, b_up[0].keys)
        assert mx.array_equal(b_pat[0].values, b_up[0].values)
        assert b_pat[0].offset.tolist() == b_up[0].offset.tolist()
        assert b_pat[0].left_padding.tolist() == b_up[0].left_padding.tolist()
        assert (b_pat[0]._idx, b_pat[0]._offset, b_pat[0].rotated) == (
            b_up[0]._idx,
            b_up[0]._offset,
            b_up[0].rotated,
        )

        # KVCache preserves pre-reserved step capacity while keeping state/extract identical.
        assert type(b_pat[1]) is BatchKVCache
        assert b_pat[1].keys.shape[2] == 256 and b_up[1].keys.shape[2] == 32
        for arr_p, arr_u in zip(b_pat[1].state, b_up[1].state):
            assert mx.array_equal(arr_p, arr_u)
        ext_p, ext_u = b_pat[1].extract(0), b_up[1].extract(0)
        assert mx.array_equal(ext_p.keys, ext_u.keys) and ext_p.offset == ext_u.offset

        k1 = mx.full((1, 1, 1, 4), 7.0, dtype=mx.float32)
        v1 = mx.full((1, 1, 1, 4), 9.0, dtype=mx.float32)
        r_upd_p = b_pat[0].update_and_fetch(k1, v1)
        r_upd_u = b_up[0].update_and_fetch(k1, v1)
        assert mx.array_equal(r_upd_p[0], r_upd_u[0]) and mx.array_equal(r_upd_p[1], r_upd_u[1])
        k_upd_p = b_pat[1].update_and_fetch(k1, v1)
        k_upd_u = b_up[1].update_and_fetch(k1, v1)
        assert mx.array_equal(k_upd_p[0], k_upd_u[0]) and mx.array_equal(k_upd_p[1], k_upd_u[1])
        assert b_pat[1].trim(1) == b_up[1].trim(1) == 1

        # Source snapshot inside APCManager remains unmutated after Batch* writes.
        assert mx.array_equal(source_entry[0].keys, base_rk)
        assert mx.array_equal(source_entry[1].keys, base_kk)

        # S=3 concat update + rotated/partially-filled RotatingKVCache + ArraysCache parity.
        def make_extra_caches():
            rr = RotatingKVCache(max_size=8)
            rr.keys, rr.values = mx.array(base_rk), mx.array(base_rv)
            rr.offset, rr._idx = 20, 3  # rotated mid-buffer
            ap = ArraysCache(2)
            ap.cache = [mx.ones((1, 2, 4)), None]
            ae = ArraysCache(2)
            ke = KVCache()
            return [rr, ap, ae, ke]

        wrapped = mlx_vlm_patches._single_row_batch_caches(make_extra_caches())
        ref_merged, _ = _apc._merge_and_release_cache_rows([make_extra_caches()], [20])
        k3 = mx.full((1, 1, 3, 4), 12.0, dtype=mx.float32)
        v3 = mx.full((1, 1, 3, 4), 15.0, dtype=mx.float32)
        assert mx.array_equal(
            wrapped[0].update_and_fetch(k3, v3)[0],
            ref_merged[0].update_and_fetch(k3, v3)[0],
        )
        assert mx.array_equal(wrapped[1].cache[0], ref_merged[1].cache[0])
        assert wrapped[1].cache[1] is None and wrapped[1].left_padding is None
        assert wrapped[2].left_padding.tolist() == ref_merged[2].left_padding.tolist() == [0]
        assert type(wrapped[3]) is BatchKVCache and wrapped[3].keys is None

        # Empty RotatingKVCache wraps identically to BatchRotatingKVCache.merge directly.
        r_empty = RotatingKVCache(max_size=8)
        wrapped_empty = mlx_vlm_patches._single_row_batch_caches([r_empty])[0]
        ref_empty = BatchRotatingKVCache.merge([r_empty])
        assert type(wrapped_empty) is BatchRotatingKVCache and wrapped_empty.keys is None
        assert wrapped_empty.left_padding.tolist() == ref_empty.left_padding.tolist() == [0]
        assert wrapped_empty.offset.tolist() == ref_empty.offset.tolist() == [0]

        # 2. Fallbacks preserve warm_cache intact and delegate to upstream;
        # APCAdmissionError propagates when prepare_prefill denies admission.
        r_keep = RotatingKVCache(max_size=8, keep=2)
        r_keep.keys, r_keep.values, r_keep.offset, r_keep._idx = (
            mx.array(base_rk),
            mx.array(base_rv),
            8,
            8,
        )
        fb_calls = []
        custom_mgr = _apc.APCManager(num_blocks=8, block_size=16, disk=None)
        fb_coord = SimpleNamespace(
            enabled=True,
            is_checkpoint=True,
            manager=custom_mgr,
            prepare_prefill=lambda *a, **kw: True,
            _merge_and_release_cache_rows=lambda picks, prefix_lens, *, kv_quant_config=None: (
                fb_calls.append((len(picks), kv_quant_config)) or ([], prefix_lens[0])
            ),
        )
        assert mlx_vlm_patches._patch_batch_restore(fb_coord)

        keep_pick = {"warm_cache": [r_keep]}
        fb_coord._merge_and_release_cache_rows([keep_pick], [8])
        assert keep_pick["warm_cache"][0] is r_keep  # unmutated before fallback
        fb_coord._merge_and_release_cache_rows([{"warm_cache": [object()]}], [8])
        fb_coord._merge_and_release_cache_rows([{"warm_cache": [kv_init]}, None], [32, 0])
        fb_coord._merge_and_release_cache_rows(
            [{"warm_cache": [kv_init]}], [32], kv_quant_config={"bits": 4}
        )
        assert fb_calls == [(1, None), (1, None), (2, None), (1, {"bits": 4})]

        custom_mgr.memory_plan.lengths = [64]
        fb_coord.prepare_prefill = lambda *a, **kw: False
        with pytest.raises(_apc.APCAdmissionError, match="admission"):
            fb_coord._merge_and_release_cache_rows([{"warm_cache": [kv_init]}], [32])



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

        biased_box = nn.Module()
        biased_box.lin = nn.Linear(k_dim, 64, bias=True)
        biased_box.lin.update(
            {
                "weight": mx.random.normal((64, k_dim)).astype(mx.bfloat16),
                "bias": mx.random.normal((64,)).astype(mx.bfloat16),
            }
        )
        nn.quantize(biased_box, group_size=64, bits=bits)
        b_lin = biased_box.lin
        ref_biased = mx.concatenate([b_lin(x[:, t : t + 1, :]) for t in range(5)], axis=1)
        got_biased = gemma4_qmv_multi(b_lin, x, min_out_rows=64)
        mx.eval(ref_biased, got_biased)
        assert got_biased is not None and mx.array_equal(got_biased, ref_biased).item()

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

    def test_fast_b1_rotating_transaction_matches_upstream(self):
        import copy
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache
        from mlx_vlm.models.gemma4.language import LanguageModel
        from heylook_llm.providers.common.gemma4_runtime import (
            _PATCH_VERIFY_HIDDEN,
            bind_gemma4_runtime,
        )

        mx.random.seed(19)
        cfg = self._tiny_config(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            sliding_window=8,
            vocab_size=32,
        )
        lm_up = LanguageModel(cfg)
        lm_bd = copy.deepcopy(lm_up)
        lm_bd.load_weights(tree_flatten(lm_up.parameters()))
        assert bind_gemma4_runtime(type("M", (), {"language_model": lm_bd})()) is True
        assert (
            getattr(lm_bd.speculative_verify_hidden.__func__, "_heylook_patch", None)
            == _PATCH_VERIFY_HIDDEN
        )

        def make_caches(pad):
            return [BatchRotatingKVCache(8, list(pad)), BatchKVCache(list(pad))]

        verify_in = mx.array([[3, 7, 11, 15]])
        followup_in = mx.array([[19, 23]])
        for init_len in (0, 5, 7, 8, 13):
            for keep in (0, 1, 2, 3, 4):
                cu, cb = make_caches([0]), make_caches([0])
                if init_len > 0:
                    p = mx.arange(1, init_len + 1, dtype=mx.int32)[None, :] % 30
                    lm_up(p, cache=cu)
                    lm_bd(p, cache=cb)
                    if init_len >= 8:
                        step = mx.array([[5]])
                        lm_up(step, cache=cu)
                        lm_bd(step, cache=cb)
                hu, _, tu = lm_up.speculative_verify_hidden(verify_in, cu)
                hb, _, tb = lm_bd.speculative_verify_hidden(verify_in, cb)
                mx.eval(hu, hb)
                assert mx.array_equal(hu, hb).item()
                if keep == 0:
                    tu.abort()
                    tb.abort()
                else:
                    tu.commit([keep])
                    tb.commit([keep])
                lu = lm_up.logits_from_hidden(lm_up.model(followup_in, cache=cu))
                lb = lm_bd.logits_from_hidden(lm_bd.model(followup_in, cache=cb))
                mx.eval(lu, lb, cu[0].offset, cb[0].offset)
                assert int(cu[0].offset[0].item()) == int(cb[0].offset[0].item())
                assert mx.allclose(lu, lb, atol=1e-4).item()

        # Padded B=2 falls back cleanly to upstream transaction without failing.
        cu2, cb2 = make_caches([2, 0]), make_caches([2, 0])
        v2 = mx.array([[1, 2, 3], [4, 5, 6]])
        h2u, _, t2u = lm_up.speculative_verify_hidden(v2, cu2)
        h2b, _, t2b = lm_bd.speculative_verify_hidden(v2, cb2)
        mx.eval(h2u, h2b)
        assert mx.array_equal(h2u, h2b).item()
        t2u.commit([2, 1])
        t2b.commit([2, 1])

    def test_leviathan_rejection_sampling_and_non_greedy_walk(self):
        import mlx.core as mx
        from mlx_vlm.speculative import mtp
        from heylook_llm.providers.common import mlx_vlm_patches, samplers

        mx.random.seed(123)
        orig_walk = getattr(
            mtp._speculative_walk_batch_deferred_greedy,
            "_heylook_upstream",
            mtp._speculative_walk_batch_deferred_greedy,
        )
        try:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk
            assert mlx_vlm_patches.install_mtp_single_row_fast_paths()

            greedy_s, _ = samplers.build({"temperature": 0.0, "top_p": 1.0, "top_k": 0})
            assert getattr(greedy_s, "_heylook_greedy", None) is True
            assert not hasattr(greedy_s, "_heylook_spec_probs")

            sampler, _ = samplers.build(
                {"temperature": 1.0, "top_p": 0.9, "top_k": 5, "min_p": 0.0}
            )
            assert getattr(sampler, "_heylook_greedy", None) is False
            spec_probs = getattr(sampler, "_heylook_spec_probs", None)
            assert callable(spec_probs)

            target_logits = mx.array(
                [
                    [2.4, 2.0, 1.4, 0.8, 0.2, -0.5, -1.2, -2.0],
                    [1.5, 2.1, 1.2, 0.4, -0.1, -0.8, -1.5, -2.5],
                    [0.9, 1.8, 2.0, 1.1, 0.3, -0.4, -1.0, -2.0],
                    [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0],
                ],
                dtype=mx.float32,
            )
            draft_logits = mx.array(
                [
                    [2.1, 2.2, 1.3, 0.7, 0.1, -0.6, -1.1, -2.1],
                    [1.3, 2.0, 1.4, 0.5, 0.0, -0.7, -1.4, -2.4],
                    [1.0, 1.6, 1.9, 1.2, 0.2, -0.3, -1.1, -2.2],
                ],
                dtype=mx.float32,
            )
            lp_t = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
            p_true, _ = spec_probs(lp_t, already_logprobs=True)
            q_true, q_scaled = spec_probs(draft_logits, already_logprobs=False)
            mx.eval(p_true, q_true, q_scaled)
            p0 = p_true[0].tolist()
            expected_accept0 = sum(min(a, b) for a, b in zip(p0, q_true[0].tolist()))
            independent_match0 = sum(a * b for a, b in zip(p0, q_true[0].tolist()))
            assert expected_accept0 > independent_match0 + 0.35

            class FixedLM:
                _heylook_gemma4_runtime = True

                def speculative_logits_from_hidden(self, h):
                    del h
                    return target_logits[None, :, :]

            lm = FixedLM()
            n_rounds = 4000
            counts0 = [0] * 8
            step0_accepts = 0
            for _ in range(n_rounds):
                drafts = mx.random.categorical(q_scaled)[None, :].astype(mx.int32)
                mx.eval(drafts)
                sampler._heylook_pending_draft = (drafts, q_true)
                acc, toks = mtp._speculative_walk_batch_deferred_greedy(
                    lm, mx.zeros((1, 4, 4)), drafts, sampler, [4]
                )
                assert sampler._heylook_pending_draft is None
                counts0[toks[0][0]] += 1
                if acc[0] >= 1:
                    step0_accepts += 1

            empirical0 = [c / n_rounds for c in counts0]
            max_abs_err = max(abs(a - b) for a, b in zip(empirical0, p0))
            accept_rate0 = step0_accepts / n_rounds
            assert max_abs_err < 0.03, (empirical0, p0)
            assert abs(accept_rate0 - expected_accept0) < 0.035, (
                accept_rate0,
                expected_accept0,
            )

            # Fallback non-greedy sampler without _heylook_pending_draft must project logits once
            # and evaluate sampler sequentially per position until rejection.
            proj_calls = []

            class CountingLM:
                _heylook_gemma4_runtime = True

                def speculative_logits_from_hidden(self, h):
                    proj_calls.append(int(h.shape[1]))
                    return target_logits[None, : h.shape[1], :]

            sample_shapes = []

            def non_greedy_no_spec(lp):
                sample_shapes.append(tuple(lp.shape))
                return mx.array([0] if len(sample_shapes) == 1 else [7])

            non_greedy_no_spec._heylook_greedy = False
            acc_fb, toks_fb = mtp._speculative_walk_batch_deferred_greedy(
                CountingLM(),
                mx.zeros((1, 4, 4)),
                mx.array([[0, 1, 2]]),
                non_greedy_no_spec,
                [4],
            )
            assert proj_calls == [4]
            assert sample_shapes == [(1, 8), (1, 8)]
            assert (acc_fb, toks_fb) == ([1], [[0, 7]])
        finally:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk

    def test_bind_mtp_drafter_and_non_gemma_leviathan_walk(self):
        import copy
        from types import SimpleNamespace

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.speculative import mtp
        from mlx_vlm.speculative.drafters.mtp_base import AutoregressiveMTPDraftModel
        from heylook_llm.providers.common import mlx_vlm_patches, samplers

        mx.random.seed(7)
        target_logits = mx.array(
            [
                [2.3, 1.9, 1.3, 0.7, 0.1, -0.6, -1.2, -2.0],
                [1.4, 2.0, 1.1, 0.4, -0.2, -0.8, -1.5, -2.4],
                [0.9, 1.7, 2.1, 1.0, 0.2, -0.4, -1.0, -2.0],
                [1.8, 1.4, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0],
            ],
            dtype=mx.float32,
        )
        draft_logits = mx.array(
            [
                [2.0, 2.1, 1.2, 0.6, 0.0, -0.7, -1.1, -2.1],
                [1.2, 1.9, 1.3, 0.5, 0.1, -0.7, -1.4, -2.3],
                [1.0, 1.5, 1.8, 1.1, 0.2, -0.3, -1.1, -2.2],
            ],
            dtype=mx.float32,
        )

        class _FakeMTPCache:
            def __init__(self):
                self.offset = 0

            def trim(self, n):
                self.offset = max(0, self.offset - int(n))

        class _FakeAutoregressiveMTP(AutoregressiveMTPDraftModel):
            def make_cache(self):
                return [_FakeMTPCache()]

            def _forward_hidden(self, token_embed, hidden, tokens, caches):
                del token_embed, tokens
                steps = int(hidden.shape[1])
                step_offset = max(0, min(caches[0].offset, 2))
                caches[0].offset += steps
                return (
                    mx.zeros((1, steps, 4), dtype=mx.float32) + step_offset,
                    mx.zeros((1, steps, 4), dtype=mx.float32),
                )

            def _set_seed_from_hidden(self, hidden, sampler, greedy):
                logits = draft_logits[0:1][None, :, :]
                self._seed_token = (
                    mx.argmax(logits, axis=-1) if greedy else sampler(logits)
                )
                self._seed_hidden = hidden

            def _forward_tokens(self, tokens, hidden, token_dtype):
                del tokens
                caches = self._cache
                caches[0].offset += 1
                idx = min(max(self._round_appended + 1, 1), 2)
                return mx.array([[[idx]]], dtype=mx.float32), hidden

        drafter = _FakeAutoregressiveMTP(SimpleNamespace(block_size=4))
        drafter._lm_head_fn = (
            lambda h: draft_logits[int(h[0, 0, 0].item()) : int(h[0, 0, 0].item()) + 1][
                None, :, :
            ]
        )
        target = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=nn.Embedding(8, 4)),
                lm_head=lambda h: target_logits[: int(h.shape[1])][None, :, :],
            )
        )

        assert mlx_vlm_patches.bind_mtp_drafter(drafter)
        copied = copy.copy(drafter)
        copied.reset(target)
        assert getattr(copied, "_heylook_mtp_drafter", False)

        orig_walk = getattr(
            mtp._speculative_walk_batch_deferred_greedy,
            "_heylook_upstream",
            mtp._speculative_walk_batch_deferred_greedy,
        )
        try:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk
            assert mlx_vlm_patches.install_mtp_single_row_fast_paths()

            sampler, _ = samplers.build(
                {"temperature": 0.9, "top_p": 0.95, "top_k": 5, "min_p": 0.0}
            )
            spec_probs = sampler._heylook_spec_probs
            lp_t = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
            p_true, _ = spec_probs(lp_t, already_logprobs=True)
            mx.eval(p_true)
            p0 = p_true[0].tolist()

            # Deliberately plain LM without _heylook_gemma4_runtime
            plain_lm = SimpleNamespace(
                speculative_logits_from_hidden=lambda h: target_logits[: int(h.shape[1])][
                    None, :, :
                ]
            )

            # Prefill sets _seed_token + _heylook_seed_q (used as slot 0 of round 1)
            copied.prefill_from_target_hidden(
                mx.array([[1, 2]], dtype=mx.int32),
                mx.zeros((1, 2, 4), dtype=mx.float32),
                0,
                sampler,
                mx.int32,
                greedy=False,
            )
            assert copied._heylook_seed_q is not None
            assert getattr(sampler, "_heylook_captured_q", None) is None

            n_rounds = 3000
            counts0 = [0] * 8
            for _ in range(n_rounds):
                drafts = copied.draft_block(
                    0,
                    mx.zeros((1, 1, 4), dtype=mx.float32),
                    None,
                    4,
                    sampler,
                    mx.int32,
                    greedy=False,
                )
                assert getattr(sampler, "_heylook_captured_q", None) is None
                pending = getattr(sampler, "_heylook_pending_draft", None)
                assert pending is not None and pending[1].shape == (3, 8)
                acc, toks = mtp._speculative_walk_batch_deferred_greedy(
                    plain_lm, mx.zeros((1, 4, 4)), drafts, sampler, [4]
                )
                assert sampler._heylook_pending_draft is None
                counts0[toks[0][0]] += 1
                copied.accept_verified_tokens(
                    mx.zeros((1, 4, 4), dtype=mx.float32),
                    drafts,
                    acc[0],
                    toks[0],
                    sampler,
                    mx.int32,
                    greedy=False,
                )

            empirical0 = [c / n_rounds for c in counts0]
            assert max(abs(a - b) for a, b in zip(empirical0, p0)) < 0.035, (
                empirical0,
                p0,
            )
        finally:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk

    def test_mtp_walk_applies_penalties_per_position_greedy_and_stochastic(self):
        import mlx.core as mx
        from mlx_vlm.speculative import mtp
        from heylook_llm.providers.common import mlx_vlm_patches, samplers

        orig_walk = getattr(
            mtp._speculative_walk_batch_deferred_greedy,
            "_heylook_upstream",
            mtp._speculative_walk_batch_deferred_greedy,
        )
        try:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk
            assert mlx_vlm_patches.install_mtp_single_row_fast_paths()

            # Unpenalized argmax at pos 0, 1, 2, 3 would pick [1, 2, 3, 4].
            # gen_history=[1], draft=[2, 3, 0].
            # With strong penalties:
            # - pos 0 sees hist=[1] -> token 1 penalized -> argmax shifts to 2 (matches draft[0]=2)
            # - pos 1 sees hist=[1, 2] -> token 2 penalized -> argmax shifts to 3 (matches draft[1]=3)
            # - pos 2 sees hist=[1, 2, 3] -> token 3 penalized -> argmax shifts to 4 (rejects draft[2]=0)
            target_logits = mx.array(
                [
                    [-5.0, 5.0, 4.2, -5.0, -5.0, -5.0],
                    [-5.0, -5.0, 5.0, 4.2, -5.0, -5.0],
                    [-5.0, -5.0, -5.0, 5.0, 4.2, -5.0],
                    [-5.0, -5.0, -5.0, -5.0, 5.0, 1.0],
                ],
                dtype=mx.float32,
            )

            class PenLM:
                def speculative_logits_from_hidden(self, h):
                    del h
                    return target_logits[None, :, :]

            sampler, procs = samplers.build(
                {"temperature": 0.0, "presence_penalty": 2.0, "repetition_penalty": 1.5}
            )
            sampler._heylook_mtp_penalties = (procs, [1])
            acc, toks = mtp._speculative_walk_batch_deferred_greedy(
                PenLM(),
                mx.zeros((1, 4, 4)),
                mx.array([[2, 3, 0]], dtype=mx.int32),
                sampler,
                [8],
            )
            assert (acc, toks) == ([2], [[2, 3, 4]])

            # Stochastic path (Leviathan): same penalties block token 1 at pos 0.
            st_sampler, st_procs = samplers.build(
                {"temperature": 0.05, "presence_penalty": 2.0, "repetition_penalty": 1.5}
            )
            st_sampler._heylook_mtp_penalties = (st_procs, [1])
            q_probs = mx.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=mx.float32,
            )
            st_sampler._heylook_pending_draft = (
                mx.array([[2, 3, 0]], dtype=mx.int32),
                q_probs,
            )
            acc_s, toks_s = mtp._speculative_walk_batch_deferred_greedy(
                PenLM(),
                mx.zeros((1, 4, 4)),
                mx.array([[2, 3, 0]], dtype=mx.int32),
                st_sampler,
                [8],
            )
            assert (acc_s, toks_s) == ([2], [[2, 3, 4]])
        finally:
            mtp._speculative_walk_batch_deferred_greedy = orig_walk

    def test_native_window_size_sdpa_matches_upstream_4d_mask(self, monkeypatch):
        import copy
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.models.cache import (
            BatchKVCache,
            BatchRotatingKVCache,
            KVCache,
            RotatingKVCache,
        )
        from mlx_vlm.models.gemma4.language import LanguageModel
        from heylook_llm.providers.common import gemma4_runtime, gemma4_vision

        mx.random.seed(23)
        cfg = self._tiny_config(
            hidden_size=128,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=64,
            global_head_dim=64,
            sliding_window=8,
            vocab_size=64,
            use_bidirectional_attention="vision",
            exact_speculative_verify=False,
        )
        lm_up = LanguageModel(cfg)
        lm_bd = copy.deepcopy(lm_up)
        lm_bd.load_weights(tree_flatten(lm_up.parameters()))
        assert (
            gemma4_runtime.bind_gemma4_runtime(type("M", (), {"language_model": lm_bd})())
            is True
        )
        assert lm_bd.model.layers[0].self_attn._heylook_sdpa_window_size == 8
        assert lm_bd.model.layers[1].self_attn._heylook_sdpa_window_size == 0

        native_ws = gemma4_vision.mlx_sdpa_supports_window_size()

        for expected_ndim, make_caches in (
            (4, lambda: [BatchRotatingKVCache(8, [0]), BatchKVCache([0])]),
            (2, lambda: [RotatingKVCache(8, keep=0), KVCache()]),
        ):
            cu, cb = make_caches(), make_caches()

            # 1. Cold prefill inside sliding_window=8.
            p1 = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
            out_up = lm_up(p1, cache=cu).logits
            out_bd = lm_bd(p1, cache=cb).logits
            mx.eval(out_up, out_bd)
            assert mx.allclose(out_up, out_bd, atol=1e-5).item()

            # 2. Suffix prefill crossing window_size=8 (5 -> 11 tokens).
            p2 = mx.array([[6, 7, 8, 9, 10, 11]], dtype=mx.int32)
            h_p2 = mx.random.normal((1, 6, 128))
            masks_suf = lm_bd.model._make_masks(h_p2, cb, None)
            if native_ws:
                assert masks_suf == ["causal", "causal"]
            else:
                assert isinstance(masks_suf[0], mx.array) and masks_suf[0].ndim == expected_ndim
            out_up = lm_up(p2, cache=cu).logits
            out_bd = lm_bd(p2, cache=cb).logits
            mx.eval(out_up, out_bd)
            assert mx.allclose(out_up, out_bd, atol=1e-5).item()

            # 3. Single-token decodes rotate/wrap the sliding cache buffer.
            for tok in (12, 13, 14):
                step = mx.array([[tok]], dtype=mx.int32)
                out_up = lm_up(step, cache=cu).logits
                out_bd = lm_bd(step, cache=cb).logits
                mx.eval(out_up, out_bd)
                assert mx.allclose(out_up, out_bd, atol=1e-5).item()

            # 4. N=3 multi-token verification AFTER rotation matches 4D masks.
            h_ver = mx.random.normal((1, 3, 128))
            masks_ver = lm_bd.model._make_masks(h_ver, cb, None)
            if native_ws:
                assert masks_ver == ["causal", "causal"]
            else:
                assert isinstance(masks_ver[0], mx.array) and masks_ver[0].ndim == expected_ndim
            ver = mx.array([[15, 16, 17]], dtype=mx.int32)
            out_up = lm_up(ver, cache=cu).logits
            out_bd = lm_bd(ver, cache=cb).logits
            mx.eval(out_up, out_bd)
            assert mx.allclose(out_up, out_bd, atol=1e-5).item()

        # Vision bidirectional overlay and padded B=2 always fall back to 4D array masks.
        mm_vis = mx.array([[0, 1, 1, 0, 0]], dtype=mx.int32)
        h_vis = mx.random.normal((1, 5, 128))
        c_vis = [BatchRotatingKVCache(8, [0]), BatchKVCache([0])]
        for m in lm_bd.model._make_masks(h_vis, c_vis, mm_vis):
            assert isinstance(m, mx.array) and m.ndim == 4
        h_pad = mx.random.normal((2, 4, 128))
        c_pad = [BatchRotatingKVCache(8, [0, 2]), BatchKVCache([0, 2])]
        for m in lm_bd.model._make_masks(h_pad, c_pad, None):
            assert isinstance(m, mx.array) and m.ndim == 4

        # Verify capability gate forwarding directly (covers both 0.32.4 & 0.32.5+ logic).
        recorded_ws = []
        orig_sdpa = mx.fast.scaled_dot_product_attention

        def spy_sdpa(q, k, v, *, scale, mask=None, sinks=None, window_size=-1):
            recorded_ws.append(window_size)
            return mx.zeros_like(q)

        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy_sdpa)
        # Re-bind once against spy signature so closure resolves updated gate cleanly.
        assert gemma4_vision.mlx_sdpa_supports_window_size() is True

        attn_slide = lm_bd.model.layers[0].self_attn
        x_in = mx.random.normal((1, 3, 128))

        # Unquantized sliding cache with mask="causal" forwards window_size=8.
        attn_slide(x_in, mask="causal", cache=RotatingKVCache(8, keep=0))
        assert recorded_ws[-1] == 8

        # Quantized cache must NOT forward window_size to raw mx.fast SDPA.
        class _FakeQuantCache:
            offset = 0
            bits = 4
            group_size = 64

            def update_and_fetch(self, k, v):
                return k, v

        from mlx_vlm.models import base as mlx_vlm_base

        base_calls = []
        monkeypatch.setattr(
            mlx_vlm_base,
            "scaled_dot_product_attention",
            lambda q, k, v, *, cache=None, scale=1.0, mask=None: (
                base_calls.append(type(cache).__name__) or orig_sdpa(q, k, v, scale=scale)
            ),
        )
        attn_slide(x_in, mask="causal", cache=_FakeQuantCache())
        assert base_calls[-1] == "_FakeQuantCache"

        # Disabled capability gate stands down cleanly without passing window_size.
        monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda: False)
        monkeypatch.setattr(gemma4_runtime, "mlx_sdpa_supports_window_size", lambda: False)
        c_rot_past = RotatingKVCache(8, keep=0)
        c_rot_past.update_and_fetch(mx.zeros((1, 2, 10, 64)), mx.zeros((1, 2, 10, 64)))
        masks_off = lm_bd.model._make_masks(x_in, [c_rot_past, KVCache()], None)
        assert isinstance(masks_off[0], mx.array) and masks_off[0].ndim == 2
        attn_slide(x_in, mask="causal", cache=RotatingKVCache(8, keep=0))
        assert len(base_calls) == 2

    def test_adaptive_8bit_qmv_stands_down_when_native_probe_wins(self, monkeypatch):
        import copy
        import mlx.nn as nn
        from mlx_vlm.models.gemma4.language import LanguageModel
        from heylook_llm.providers.common import gemma4_runtime

        cfg = self._tiny_config(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            vocab_size=64,
        )
        base_lm = LanguageModel(cfg)
        nn.quantize(base_lm, group_size=64, bits=8)

        monkeypatch.setattr(gemma4_runtime, "MIN_QMV_OUT_ROWS", 64)

        # Native probe wins -> skip 8-bit MLP & LM head hoisted kernels while keeping
        # attention/shared_kv/verify_hidden/masks and _heylook_gemma4_runtime=True.
        lm_native = copy.deepcopy(base_lm)
        monkeypatch.setattr(gemma4_runtime, "_native_8bit_qmv_beats_hoisted", lambda embed=None: True)
        assert (
            gemma4_runtime.bind_gemma4_runtime(
                type("M", (), {"language_model": lm_native})(),
                min_out_rows=64,
            )
            is True
        )
        assert lm_native._heylook_gemma4_runtime is True
        assert getattr(lm_native.model.layers[0].self_attn, "_heylook_gemma4_attn", False) is True
        assert getattr(lm_native.model.layers[0].mlp, "_heylook_gemma4_mlp", False) is False
        assert (
            getattr(
                getattr(lm_native.logits_from_hidden, "__func__", None),
                "_heylook_patch",
                None,
            )
            is None
        )

        # Native probe slower -> bind hoisted 8-bit MLP & LM head kernels as usual.
        lm_hoisted = copy.deepcopy(base_lm)
        monkeypatch.setattr(gemma4_runtime, "_native_8bit_qmv_beats_hoisted", lambda embed=None: False)
        assert (
            gemma4_runtime.bind_gemma4_runtime(
                type("M", (), {"language_model": lm_hoisted})(),
                min_out_rows=64,
            )
            is True
        )
        assert lm_hoisted._heylook_gemma4_runtime is True
        assert getattr(lm_hoisted.model.layers[0].mlp, "_heylook_gemma4_mlp", False) is True
        assert (
            getattr(
                getattr(lm_hoisted.logits_from_hidden, "__func__", None),
                "_heylook_patch",
                None,
            )
            == gemma4_runtime._PATCH_LOGITS
        )

    def test_buffered_and_rotated_sliding_masks(self, monkeypatch):
        import copy
        import mlx.core as mx
        from mlx_vlm.models.cache import (
            BatchKVCache,
            BatchRotatingKVCache,
            BufferedRotatingKVCache,
            KVCache,
            RotatingKVCache,
        )
        from mlx_vlm.models.gemma4.language import LanguageModel
        from heylook_llm.providers.common import gemma4_runtime, gemma4_vision

        cfg = self._tiny_config(
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=2,
            head_dim=32,
            global_head_dim=32,
            sliding_window=8,
            vocab_size=64,
        )
        lm_up = LanguageModel(cfg)
        lm_bd = copy.deepcopy(lm_up)
        gemma4_runtime.bind_gemma4_runtime(type("M", (), {"language_model": lm_bd})())

        cu = [BufferedRotatingKVCache(8, keep=0, buffer_size=16), KVCache()]
        cb = [BufferedRotatingKVCache(8, keep=0, buffer_size=16), KVCache()]
        prefix = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32)
        out_u0 = lm_up(prefix, cache=cu).logits
        out_b0 = lm_bd(prefix, cache=cb).logits
        mx.eval(out_u0, out_b0)
        assert mx.allclose(out_u0, out_b0, atol=1e-5).item()

        # Check n=1 and n=5 past sliding_window=8 both with native window_size=False and real probe.
        for gate_val in (False, gemma4_vision.mlx_sdpa_supports_window_size()):
            monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda g=gate_val: g)
            monkeypatch.setattr(gemma4_runtime, "mlx_sdpa_supports_window_size", lambda g=gate_val: g)

            cu_s = copy.deepcopy(cu)
            cb_s = copy.deepcopy(cb)
            step1 = mx.array([[11]], dtype=mx.int32)
            m1 = lm_bd.model._make_masks(mx.zeros((1, 1, 128)), cb_s, None)
            if gate_val:
                assert m1 == ["causal", None]
            else:
                assert isinstance(m1[0], mx.array) and m1[0].ndim == 2
            lu1 = lm_up(step1, cache=cu_s).logits
            lb1 = lm_bd(step1, cache=cb_s).logits
            mx.eval(lu1, lb1)
            assert mx.allclose(lu1, lb1, atol=1e-5).item()

            step5 = mx.array([[12, 13, 14, 15, 16]], dtype=mx.int32)
            m5 = lm_bd.model._make_masks(mx.zeros((1, 5, 128)), cb_s, None)
            if gate_val:
                assert m5 == ["causal", "causal"]
            else:
                assert isinstance(m5[0], mx.array) and m5[0].ndim == 2
            lu5 = lm_up(step5, cache=cu_s).logits
            lb5 = lm_bd(step5, cache=cb_s).logits
            mx.eval(lu5, lb5)
            assert mx.allclose(lu5, lb5, atol=1e-5).item()

        # Verify rotated BatchRotatingKVCache / RotatingKVCache (n > 1):
        # without native window_size= they fall back to array masks; with native window_size= they use "causal".
        rot_batch = BatchRotatingKVCache(8, [0])
        rot_batch.update_and_fetch(mx.zeros((1, 2, 8, 32)), mx.zeros((1, 2, 8, 32)))
        rot_batch.update_and_fetch(mx.zeros((1, 2, 1, 32)), mx.zeros((1, 2, 1, 32)))
        assert rot_batch.rotated is True

        rot_plain = RotatingKVCache(8, keep=0)
        rot_plain.update_and_fetch(mx.zeros((1, 2, 8, 32)), mx.zeros((1, 2, 8, 32)))
        rot_plain.update_and_fetch(mx.zeros((1, 2, 1, 32)), mx.zeros((1, 2, 1, 32)))
        assert rot_plain.offset == 9 and rot_plain._idx < rot_plain.offset

        monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda: False)
        monkeypatch.setattr(gemma4_runtime, "mlx_sdpa_supports_window_size", lambda: False)
        masks_rot_off = lm_bd.model._make_masks(
            mx.zeros((1, 3, 128)), [rot_batch, BatchKVCache([0])], None
        )
        assert isinstance(masks_rot_off[0], mx.array) and masks_rot_off[0].ndim == 4
        masks_plain_off = lm_bd.model._make_masks(
            mx.zeros((1, 3, 128)), [rot_plain, KVCache()], None
        )
        assert isinstance(masks_plain_off[0], mx.array) and masks_plain_off[0].ndim == 2

        monkeypatch.setattr(gemma4_vision, "mlx_sdpa_supports_window_size", lambda: True)
        monkeypatch.setattr(gemma4_runtime, "mlx_sdpa_supports_window_size", lambda: True)
        assert lm_bd.model._make_masks(
            mx.zeros((1, 3, 128)), [rot_batch, BatchKVCache([0])], None
        ) == ["causal", "causal"]
        assert lm_bd.model._make_masks(
            mx.zeros((1, 3, 128)), [rot_plain, KVCache()], None
        ) == ["causal", "causal"]
