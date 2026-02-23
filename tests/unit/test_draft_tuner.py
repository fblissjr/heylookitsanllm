# tests/unit/test_draft_tuner.py
"""Tests for DraftTuner dynamic draft token adjustment.

Covers:
- Default returns configured value when model not tracked
- Insufficient samples returns configured default
- High acceptance increases draft tokens
- Low acceptance decreases draft tokens
- Bounds enforced (min 1, max 8)
- Per-model isolation
- Thread safety
"""

import threading


class TestDraftTunerDefaults:
    """Verify default/baseline behavior."""

    def test_unknown_model_returns_configured_default(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        assert tuner.get_num_draft_tokens("unknown-model", 3) == 3

    def test_ensure_baseline_sets_initial(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 5)
        assert tuner.get_num_draft_tokens("model-a", 3) == 5

    def test_ensure_baseline_idempotent(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 5)
        tuner._ensure_baseline("model-a", 7)  # Should not override
        assert tuner.get_num_draft_tokens("model-a", 3) == 5


class TestDraftTunerInsufficientSamples:
    """Verify no adjustment with too few samples."""

    def test_no_adjustment_below_min_samples(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 3)

        # Record 5 tokens (below MIN_SAMPLES=10)
        tuner.record("model-a", 5, 5)  # 100% acceptance, but too few samples
        assert tuner.get_num_draft_tokens("model-a", 3) == 3  # Unchanged


class TestDraftTunerHighAcceptance:
    """Verify upward adjustment on high acceptance."""

    def test_high_acceptance_increases(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 3)

        # Fill window with high acceptance (>80%)
        # First record sets the window but doesn't adjust (no baseline recorded yet via record)
        # Since _ensure_baseline already set current=3, subsequent records with enough data adjust
        tuner.record("model-a", 50, 50)  # 100% acceptance, 50 samples

        assert tuner.get_num_draft_tokens("model-a", 3) == 4  # 3 -> 4

    def test_incremental_increase(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 3)

        # First batch: 100% acceptance
        tuner.record("model-a", 50, 50)
        assert tuner.get_num_draft_tokens("model-a", 3) == 4

        # Second batch: still high acceptance
        tuner.record("model-a", 50, 50)
        assert tuner.get_num_draft_tokens("model-a", 3) == 5


class TestDraftTunerLowAcceptance:
    """Verify downward adjustment on low acceptance."""

    def test_low_acceptance_decreases(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 4)

        # 30% acceptance -- below 50% threshold
        tuner.record("model-a", 15, 50)

        assert tuner.get_num_draft_tokens("model-a", 3) == 3  # 4 -> 3


class TestDraftTunerBounds:
    """Verify min/max bounds are enforced."""

    def test_max_bound(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 8)  # Already at max

        tuner.record("model-a", 50, 50)  # 100% acceptance
        assert tuner.get_num_draft_tokens("model-a", 3) == 8  # Cannot exceed 8

    def test_min_bound(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 1)  # Already at min

        tuner.record("model-a", 5, 50)  # 10% acceptance
        assert tuner.get_num_draft_tokens("model-a", 3) == 1  # Cannot go below 1


class TestDraftTunerIsolation:
    """Verify per-model tracking is isolated."""

    def test_models_tracked_independently(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 3)
        tuner._ensure_baseline("model-b", 5)

        # model-a gets high acceptance
        tuner.record("model-a", 50, 50)
        # model-b gets low acceptance
        tuner.record("model-b", 10, 50)

        assert tuner.get_num_draft_tokens("model-a", 3) == 4  # Increased
        assert tuner.get_num_draft_tokens("model-b", 3) == 4  # Decreased from 5


class TestDraftTunerEdgeCases:
    """Edge cases."""

    def test_zero_total_ignored(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 3)
        tuner.record("model-a", 0, 0)  # Should be a no-op
        assert tuner.get_num_draft_tokens("model-a", 3) == 3

    def test_empty_model_id_ignored(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner.record("", 10, 10)  # Empty model_id should be ignored
        # Should not crash or create entries

    def test_middle_acceptance_no_change(self, mock_mlx):
        """Acceptance between 50-80% should not trigger adjustment."""
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 4)

        # 65% acceptance -- between thresholds
        tuner.record("model-a", 33, 50)
        assert tuner.get_num_draft_tokens("model-a", 3) == 4  # Unchanged


class TestDraftTunerThreadSafety:
    """Basic thread safety verification."""

    def test_concurrent_records(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import DraftTuner

        tuner = DraftTuner()
        tuner._ensure_baseline("model-a", 4)

        errors = []

        def record_loop():
            try:
                for _ in range(100):
                    tuner.record("model-a", 8, 10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Value should be valid (between 1 and 8)
        val = tuner.get_num_draft_tokens("model-a", 4)
        assert 1 <= val <= 8


class TestGetDraftTuner:
    """Verify singleton access."""

    def test_returns_same_instance(self, mock_mlx):
        from heylook_llm.providers.common.generation_core import get_draft_tuner

        tuner1 = get_draft_tuner()
        tuner2 = get_draft_tuner()
        assert tuner1 is tuner2
