# tests/unit/test_vision_budget.py
"""Model-agnostic vision token budget -> per-family processor kwargs.

One wire knob (``vision_tokens``: target visual tokens per image) mapped by
duck-typing the loaded model's image processor: gemma-4 exposes discrete
``max_soft_tokens`` buckets (snap to nearest), qwen2/3-VL exposes a
continuous pixel budget (tokens x (patch x merge)^2 -> ``max_pixels``).
Unknown processors map to {} so the request degrades gracefully.
"""

from types import SimpleNamespace

import pytest

from heylook_llm.providers.common.vision_budget import vision_budget_kwargs


def _proc(**ip_attrs):
    return SimpleNamespace(image_processor=SimpleNamespace(**ip_attrs))


@pytest.mark.unit
class TestGemmaFamily:
    def _gemma_proc(self):
        return _proc(max_soft_tokens=280, patch_size=16, pooling_kernel_size=3)

    def test_exact_bucket_passes_through(self):
        assert vision_budget_kwargs(self._gemma_proc(), 1120) == {"max_soft_tokens": 1120}

    def test_snaps_to_nearest_bucket(self):
        assert vision_budget_kwargs(self._gemma_proc(), 400) == {"max_soft_tokens": 280}
        assert vision_budget_kwargs(self._gemma_proc(), 900) == {"max_soft_tokens": 1120}

    def test_clamps_to_extremes(self):
        assert vision_budget_kwargs(self._gemma_proc(), 16) == {"max_soft_tokens": 70}
        assert vision_budget_kwargs(self._gemma_proc(), 99999) == {"max_soft_tokens": 1120}


@pytest.mark.unit
class TestQwenFamily:
    def test_tokens_to_pixels(self):
        # patch 16, merge 2 -> one token per 32x32 px block -> 1024 px/token
        proc = _proc(patch_size=16, merge_size=2)
        assert vision_budget_kwargs(proc, 256) == {"max_pixels": 256 * 1024}

    def test_other_geometry(self):
        proc = _proc(patch_size=14, merge_size=2)
        assert vision_budget_kwargs(proc, 100) == {"max_pixels": 100 * 28 * 28}


@pytest.mark.unit
class TestDegradation:
    def test_none_or_zero_budget_is_empty(self):
        proc = _proc(max_soft_tokens=280)
        assert vision_budget_kwargs(proc, None) == {}
        assert vision_budget_kwargs(proc, 0) == {}

    def test_no_image_processor_is_empty(self):
        assert vision_budget_kwargs(SimpleNamespace(), 280) == {}
        assert vision_budget_kwargs(None, 280) == {}

    def test_unknown_processor_family_is_empty(self):
        assert vision_budget_kwargs(_proc(size={"shortest_edge": 336}), 280) == {}
