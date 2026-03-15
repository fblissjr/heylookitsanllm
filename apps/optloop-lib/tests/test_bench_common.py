"""Unit tests for bench_common pure functions.

Covers scoring, variance checking, hard constraints, per-prompt constraints,
suspicion detection, fingerprinting, config extraction, and coderef helpers.
"""

import importlib

import pytest

import orjson

from bench_common import (
    REPOS_DIR,
    baseline_metrics_from_result,
    build_cycle_data,
    check_fingerprints,
    check_hard_constraints,
    check_per_prompt_constraints,
    check_suspicion,
    check_variance,
    compute_composite_score,
    fingerprint_output,
    get_bench_params,
    get_coderef_head,
    get_constraints,
    get_scoring_weights,
    load_baseline,
    snapshot_coderef,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline(gen_tps=50.0, ttft_ms=100.0, prefill_tps=200.0, memory_gb=8.0, avg_completion_tokens=256):
    return {
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "memory_gb": memory_gb,
        "avg_completion_tokens": avg_completion_tokens,
    }


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_baseline_identical_returns_1(self):
        b = _baseline()
        score = compute_composite_score(50.0, 100.0, 200.0, 8.0, b)
        assert score == pytest.approx(1.0)

    def test_higher_gen_tps_improves_score(self):
        b = _baseline()
        score = compute_composite_score(60.0, 100.0, 200.0, 8.0, b)
        assert score > 1.0

    def test_lower_ttft_improves_score(self):
        b = _baseline()
        score = compute_composite_score(50.0, 80.0, 200.0, 8.0, b)
        assert score > 1.0

    def test_weight_sensitivity(self):
        b = _baseline()
        # Double gen_tps weight, halve memory weight
        w1 = {"gen_tps": 0.55, "ttft": 0.25, "prefill_tps": 0.20, "memory": 0.00}
        w2 = {"gen_tps": 0.00, "ttft": 0.25, "prefill_tps": 0.20, "memory": 0.55}
        # Improve gen_tps, worsen memory
        score1 = compute_composite_score(60.0, 100.0, 200.0, 10.0, b, weights=w1)
        score2 = compute_composite_score(60.0, 100.0, 200.0, 10.0, b, weights=w2)
        # w1 should produce higher score (gen_tps improved, memory regressed)
        assert score1 > score2

    def test_zero_baseline_returns_1(self):
        b = _baseline(gen_tps=0.0)
        score = compute_composite_score(50.0, 100.0, 200.0, 8.0, b)
        assert score == 1.0

    def test_weights_not_summing_to_1_warns(self, capsys):
        b = _baseline()
        w = {"gen_tps": 0.50, "ttft": 0.50, "prefill_tps": 0.50, "memory": 0.50}
        compute_composite_score(50.0, 100.0, 200.0, 8.0, b, weights=w)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "sum to" in captured.err


# ---------------------------------------------------------------------------
# check_variance
# ---------------------------------------------------------------------------

class TestCheckVariance:
    def test_identical_runs_no_warnings(self):
        runs = [
            [{"name": "p1", "gen_tps": 50.0}],
            [{"name": "p1", "gen_tps": 50.0}],
        ]
        assert check_variance(runs) == []

    def test_sample_variance_n_minus_1(self):
        """Verify N-1 (sample) variance, not N (population).

        Values [10, 20, 30]: mean=20
          Sample variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / (3-1) = 200/2 = 100
          Sample stddev = 10, CV = 10/20 = 0.5
          Population variance would be 200/3 = 66.67, stddev 8.16, CV = 0.408
        With threshold 0.45, sample CV (0.5) triggers but population CV (0.408) would not.
        """
        runs = [
            [{"name": "p1", "gen_tps": 10.0}],
            [{"name": "p1", "gen_tps": 20.0}],
            [{"name": "p1", "gen_tps": 30.0}],
        ]
        # Threshold between population CV (0.408) and sample CV (0.5)
        warnings = check_variance(runs, constraints={"variance_max_cv": 0.45})
        assert len(warnings) == 1
        assert "CV=0.50" in warnings[0]

    def test_cv_above_threshold_warns(self):
        runs = [
            [{"name": "p1", "gen_tps": 40.0}],
            [{"name": "p1", "gen_tps": 60.0}],
        ]
        warnings = check_variance(runs, constraints={"variance_max_cv": 0.10})
        assert len(warnings) == 1

    def test_cv_below_threshold_no_warning(self):
        runs = [
            [{"name": "p1", "gen_tps": 49.0}],
            [{"name": "p1", "gen_tps": 51.0}],
        ]
        warnings = check_variance(runs, constraints={"variance_max_cv": 0.15})
        assert warnings == []

    def test_single_run_no_warning(self):
        runs = [[{"name": "p1", "gen_tps": 50.0}]]
        assert check_variance(runs) == []

    def test_empty_runs_no_warning(self):
        assert check_variance([]) == []


# ---------------------------------------------------------------------------
# check_hard_constraints
# ---------------------------------------------------------------------------

class TestHardConstraints:
    def test_no_regression(self):
        b = _baseline()
        violations = check_hard_constraints(50.0, 100.0, 200.0, 8.0, 256, b)
        assert violations == []

    def test_gen_tps_regression(self):
        b = _baseline()
        # 30% regression threshold, gen_tps drops from 50 to 30 (40% drop)
        violations = check_hard_constraints(30.0, 100.0, 200.0, 8.0, 256, b)
        assert len(violations) == 1
        assert "gen_tps" in violations[0]

    def test_ttft_regression(self):
        b = _baseline()
        # TTFT increases from 100 to 150 (50% increase, beyond 30% threshold)
        violations = check_hard_constraints(50.0, 150.0, 200.0, 8.0, 256, b)
        assert len(violations) == 1
        assert "ttft" in violations[0]

    def test_token_count_deviation(self):
        b = _baseline(avg_completion_tokens=256)
        # 25% deviation, beyond 20% threshold
        violations = check_hard_constraints(50.0, 100.0, 200.0, 8.0, 320, b)
        assert len(violations) == 1
        assert "token count" in violations[0]

    def test_multiple_violations(self):
        b = _baseline()
        # Both gen_tps and ttft regressed
        violations = check_hard_constraints(30.0, 150.0, 200.0, 8.0, 256, b)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# check_per_prompt_constraints
# ---------------------------------------------------------------------------

class TestPerPromptConstraints:
    def test_one_prompt_regressed(self):
        per_prompt = [
            {"name": "short", "gen_tps": 60.0, "ttft_ms": 90.0},
            {"name": "long", "gen_tps": 30.0, "ttft_ms": 100.0},  # regressed
        ]
        baseline_data = {
            "per_prompt": [
                {"name": "short", "gen_tps": 50.0, "ttft_ms": 100.0},
                {"name": "long", "gen_tps": 50.0, "ttft_ms": 100.0},
            ]
        }
        violations = check_per_prompt_constraints(per_prompt, baseline_data)
        assert len(violations) == 1
        assert "long" in violations[0]

    def test_all_improved(self):
        per_prompt = [
            {"name": "short", "gen_tps": 60.0, "ttft_ms": 90.0},
            {"name": "long", "gen_tps": 60.0, "ttft_ms": 90.0},
        ]
        baseline_data = {
            "per_prompt": [
                {"name": "short", "gen_tps": 50.0, "ttft_ms": 100.0},
                {"name": "long", "gen_tps": 50.0, "ttft_ms": 100.0},
            ]
        }
        assert check_per_prompt_constraints(per_prompt, baseline_data) == []

    def test_disabled_returns_empty(self):
        per_prompt = [{"name": "short", "gen_tps": 10.0, "ttft_ms": 500.0}]
        baseline_data = {
            "per_prompt": [{"name": "short", "gen_tps": 50.0, "ttft_ms": 100.0}],
        }
        constraints = {"per_prompt_regression_check": False}
        assert check_per_prompt_constraints(per_prompt, baseline_data, constraints=constraints) == []


# ---------------------------------------------------------------------------
# check_suspicion
# ---------------------------------------------------------------------------

class TestSuspicion:
    def test_small_gain_no_warning(self):
        assert check_suspicion(1.05) == []

    def test_large_gain_warns(self):
        warnings = check_suspicion(1.35)
        assert len(warnings) == 1
        assert "Suspiciously" in warnings[0]

    def test_regression_no_warning(self):
        assert check_suspicion(0.90) == []

    def test_custom_threshold(self):
        # 1.15 gain = 15%, threshold 10%
        warnings = check_suspicion(1.15, constraints={"max_suspicion_gain": 0.10})
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# fingerprint_output and check_fingerprints
# ---------------------------------------------------------------------------

class TestFingerprinting:
    def test_same_tokens_same_fingerprint(self):
        fp1 = fingerprint_output([1, 2, 3, 4, 5])
        fp2 = fingerprint_output([1, 2, 3, 4, 5])
        assert fp1 == fp2

    def test_different_tokens_different_fingerprint(self):
        fp1 = fingerprint_output([1, 2, 3])
        fp2 = fingerprint_output([4, 5, 6])
        assert fp1 != fp2

    def test_empty_token_list_returns_hash_of_empty(self):
        """Empty list hashes b'' -- callers guard with `if token_ids` before calling."""
        fp = fingerprint_output([])
        assert len(fp) == 16  # still a valid hash, just of empty input

    def test_fingerprint_length(self):
        fp = fingerprint_output([100, 200, 300])
        assert len(fp) == 16  # truncated sha256

    def test_check_fingerprints_match(self):
        fp = fingerprint_output([1, 2, 3])
        per_prompt = [{"name": "short", "fingerprint": fp}]
        baseline = {"per_prompt": [{"name": "short", "fingerprint": fp}]}
        assert check_fingerprints(per_prompt, baseline) == []

    def test_check_fingerprints_mismatch(self):
        per_prompt = [{"name": "short", "fingerprint": "aaaa"}]
        baseline = {"per_prompt": [{"name": "short", "fingerprint": "bbbb"}]}
        violations = check_fingerprints(per_prompt, baseline)
        assert len(violations) == 1
        assert "mismatch" in violations[0]

    def test_missing_baseline_fingerprint_no_violation(self):
        per_prompt = [{"name": "short", "fingerprint": "aaaa"}]
        baseline = {"per_prompt": [{"name": "short"}]}
        assert check_fingerprints(per_prompt, baseline) == []


# ---------------------------------------------------------------------------
# get_scoring_weights
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_with_full_config(self):
        config = {"scoring": {
            "gen_tps_weight": 0.50,
            "ttft_weight": 0.20,
            "prefill_tps_weight": 0.15,
            "memory_weight": 0.15,
        }}
        w = get_scoring_weights(config)
        assert w["gen_tps"] == 0.50
        assert w["ttft"] == 0.20
        assert w["prefill_tps"] == 0.15
        assert w["memory"] == 0.15

    def test_empty_config_returns_defaults(self):
        w = get_scoring_weights({})
        assert w == {"gen_tps": 0.40, "ttft": 0.25, "prefill_tps": 0.20, "memory": 0.15}

    def test_defaults_sum_to_1(self):
        w = get_scoring_weights({})
        assert sum(w.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# baseline_metrics_from_result
# ---------------------------------------------------------------------------

class TestBaselineMetrics:
    def test_standard_result(self):
        result = {
            "metrics": {
                "avg_gen_tps": 55.0,
                "avg_ttft_ms": 90.0,
                "avg_prefill_tps": 210.0,
                "peak_memory_gb": 7.5,
            },
            "per_prompt": [
                {"name": "short", "completion_tokens": 100},
                {"name": "long", "completion_tokens": 200},
            ],
        }
        m = baseline_metrics_from_result(result)
        assert m["gen_tps"] == 55.0
        assert m["ttft_ms"] == 90.0
        assert m["prefill_tps"] == 210.0
        assert m["memory_gb"] == 7.5
        assert m["avg_completion_tokens"] == 150.0

    def test_empty_per_prompt(self):
        result = {
            "metrics": {
                "avg_gen_tps": 50.0,
                "avg_ttft_ms": 100.0,
                "avg_prefill_tps": 200.0,
                "peak_memory_gb": 8.0,
            },
            "per_prompt": [],
        }
        m = baseline_metrics_from_result(result)
        assert m["avg_completion_tokens"] == 0

    def test_multiple_per_prompt_averages(self):
        result = {
            "metrics": {
                "avg_gen_tps": 50.0,
                "avg_ttft_ms": 100.0,
                "avg_prefill_tps": 200.0,
                "peak_memory_gb": 8.0,
            },
            "per_prompt": [
                {"name": "a", "completion_tokens": 100},
                {"name": "b", "completion_tokens": 200},
                {"name": "c", "completion_tokens": 300},
            ],
        }
        m = baseline_metrics_from_result(result)
        assert m["avg_completion_tokens"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# get_bench_params
# ---------------------------------------------------------------------------

class TestGetBenchParams:
    def test_empty_config_returns_defaults(self):
        p = get_bench_params({})
        assert p == {
            "runs": 3,
            "warmup": 1,
            "max_tokens": 256,
            "seed": 42,
            "timeout_seconds": 600,
        }

    def test_partial_config_overrides(self):
        config = {"bench": {"runs": 5, "seed": 99}}
        p = get_bench_params(config)
        assert p["runs"] == 5
        assert p["seed"] == 99
        assert p["warmup"] == 1  # default
        assert p["max_tokens"] == 256  # default
        assert p["timeout_seconds"] == 600  # default


# ---------------------------------------------------------------------------
# get_constraints
# ---------------------------------------------------------------------------

class TestGetConstraints:
    def test_empty_config_returns_defaults(self):
        c = get_constraints({})
        assert c["max_single_metric_regression"] == 0.30
        assert c["max_token_deviation"] == 0.20
        assert c["per_prompt_regression_check"] is True
        assert c["require_fingerprint_match"] is True
        assert c["require_tests_pass"] is True
        assert c["max_suspicion_gain"] == 0.30
        assert c["variance_max_cv"] == 0.15

    def test_partial_config_overrides(self):
        config = {"constraints": {"max_single_metric_regression": 0.50, "variance_max_cv": 0.25}}
        c = get_constraints(config)
        assert c["max_single_metric_regression"] == 0.50
        assert c["variance_max_cv"] == 0.25
        assert c["max_token_deviation"] == 0.20  # default


# ---------------------------------------------------------------------------
# check_per_prompt_constraints -- additional branches
# ---------------------------------------------------------------------------

class TestPerPromptConstraintsExtra:
    def test_ttft_regression_violation(self):
        """TTFT regression beyond threshold returns violation."""
        per_prompt = [{"name": "short", "gen_tps": 50.0, "ttft_ms": 200.0}]
        baseline_data = {
            "per_prompt": [{"name": "short", "gen_tps": 50.0, "ttft_ms": 100.0}],
        }
        violations = check_per_prompt_constraints(per_prompt, baseline_data)
        assert len(violations) == 1
        assert "ttft" in violations[0]

    def test_prompt_not_in_baseline_skipped(self):
        """Prompt not in baseline is skipped (no violation)."""
        per_prompt = [{"name": "new_prompt", "gen_tps": 10.0, "ttft_ms": 500.0}]
        baseline_data = {
            "per_prompt": [{"name": "short", "gen_tps": 50.0, "ttft_ms": 100.0}],
        }
        assert check_per_prompt_constraints(per_prompt, baseline_data) == []


# ---------------------------------------------------------------------------
# check_hard_constraints -- additional branches
# ---------------------------------------------------------------------------

class TestHardConstraintsExtra:
    def test_prefill_tps_regression(self):
        b = _baseline()
        # prefill_tps drops from 200 to 120 (40% drop, beyond 30% threshold)
        violations = check_hard_constraints(50.0, 100.0, 120.0, 8.0, 256, b)
        assert len(violations) == 1
        assert "prefill_tps" in violations[0]

    def test_memory_regression(self):
        b = _baseline()
        # memory increases from 8.0 to 12.0 (50% increase, beyond 30% threshold)
        violations = check_hard_constraints(50.0, 100.0, 200.0, 12.0, 256, b)
        assert len(violations) == 1
        assert "memory" in violations[0]


# ---------------------------------------------------------------------------
# check_fingerprints -- additional branches
# ---------------------------------------------------------------------------

class TestFingerprintingExtra:
    def test_multiple_prompts_one_mismatch(self):
        """Multiple prompts, only one has mismatch -- returns exactly 1 violation."""
        per_prompt = [
            {"name": "short", "fingerprint": "aaaa"},
            {"name": "long", "fingerprint": "cccc"},
        ]
        baseline = {
            "per_prompt": [
                {"name": "short", "fingerprint": "aaaa"},  # match
                {"name": "long", "fingerprint": "dddd"},   # mismatch
            ]
        }
        violations = check_fingerprints(per_prompt, baseline)
        assert len(violations) == 1
        assert "long" in violations[0]

    def test_current_no_fingerprint_baseline_has_one(self):
        """Current has no fingerprint, baseline does -- no violation (both must exist)."""
        per_prompt = [{"name": "short"}]
        baseline = {"per_prompt": [{"name": "short", "fingerprint": "aaaa"}]}
        assert check_fingerprints(per_prompt, baseline) == []

    def test_empty_baseline_per_prompt(self):
        """Empty baseline per_prompt list returns empty violations."""
        per_prompt = [{"name": "short", "fingerprint": "aaaa"}]
        baseline = {"per_prompt": []}
        assert check_fingerprints(per_prompt, baseline) == []


# ---------------------------------------------------------------------------
# check_variance -- additional branch
# ---------------------------------------------------------------------------

class TestCheckVarianceExtra:
    def test_all_zero_gen_tps_no_warnings(self):
        """All zero gen_tps: mean <= 0 guard prevents division by zero."""
        runs = [
            [{"name": "p1", "gen_tps": 0.0}],
            [{"name": "p1", "gen_tps": 0.0}],
        ]
        assert check_variance(runs) == []


# ---------------------------------------------------------------------------
# load_baseline -- JSON error handling
# ---------------------------------------------------------------------------

class TestLoadBaseline:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_baseline(tmp_path) is None

    def test_valid_file_returns_dict(self, tmp_path):
        data = {"gen_tps": 50.0, "ttft_ms": 100.0}
        (tmp_path / "baseline.json").write_bytes(orjson.dumps(data))
        result = load_baseline(tmp_path)
        assert result == data

    def test_corrupt_json_returns_none(self, tmp_path, capsys):
        (tmp_path / "baseline.json").write_bytes(b"{corrupt json!!")
        result = load_baseline(tmp_path)
        assert result is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "corrupt" in captured.err


# ---------------------------------------------------------------------------
# Coderef git helpers
# ---------------------------------------------------------------------------

class TestCoderefHelpers:
    def test_get_coderef_head_returns_dict(self):
        """get_coderef_head returns dict with branch and commit for existing repo."""
        result = get_coderef_head("mlx-lm")
        if REPOS_DIR.joinpath("mlx-lm").is_dir():
            assert result is not None
            assert "branch" in result
            assert "commit" in result
        else:
            assert result is None

    def test_get_coderef_head_nonexistent_returns_none(self):
        """get_coderef_head returns None for nonexistent repo."""
        result = get_coderef_head("nonexistent-repo-xyz")
        assert result is None

    def test_snapshot_coderef_returns_both_repos(self):
        """snapshot_coderef returns entries for repos that exist."""
        snapshot = snapshot_coderef()
        assert isinstance(snapshot, dict)
        # If repos exist, they should be in the snapshot
        if REPOS_DIR.joinpath("mlx-lm").is_dir():
            assert "mlx-lm" in snapshot
        if REPOS_DIR.joinpath("mlx-vlm").is_dir():
            assert "mlx-vlm" in snapshot

    def test_build_cycle_data_with_coderef_changes(self):
        """build_cycle_data includes coderef_changes in git section."""
        coderef = {
            "mlx-lm": {"branch": "optloop-lib/test", "commit": "abc1234", "diff_stat": "1 file changed"},
        }
        data = build_cycle_data(
            cycle_id=1,
            git_info={"branch": "main", "commit": "def5678"},
            optimizer_info={"description": "test"},
            config_snapshot={},
            text_results=None,
            vlm_results=None,
            verification={},
            cumulative={},
            decision="keep",
            decision_reason="test",
            coderef_changes=coderef,
        )
        assert "coderef_changes" in data["git"]
        assert data["git"]["coderef_changes"]["mlx-lm"]["commit"] == "abc1234"

    def test_build_cycle_data_without_coderef_changes_backward_compat(self):
        """build_cycle_data without coderef_changes preserves backward compat."""
        data = build_cycle_data(
            cycle_id=1,
            git_info={"branch": "main", "commit": "def5678"},
            optimizer_info={"description": "test"},
            config_snapshot={},
            text_results=None,
            vlm_results=None,
            verification={},
            cumulative={},
            decision="discard",
            decision_reason="test",
        )
        assert "coderef_changes" not in data["git"]
        assert data["decision"] == "discard"


# ---------------------------------------------------------------------------
# Smoke imports -- verify bench scripts are importable
# ---------------------------------------------------------------------------

class TestSmokeImports:
    def test_bench_text_importable(self):
        """bench_text.py can be imported (verifies bench_common path setup)."""
        pytest.importorskip("mlx")
        mod = importlib.import_module("bench_text")
        assert hasattr(mod, "run_benchmark")
        assert hasattr(mod, "PROMPTS")

    def test_bench_vlm_importable(self):
        """bench_vlm.py can be imported (verifies all mlx_vlm imports resolve)."""
        pytest.importorskip("mlx")
        mod = importlib.import_module("bench_vlm")
        assert hasattr(mod, "run_benchmark")
        assert hasattr(mod, "TEXT_PROMPTS")
        assert hasattr(mod, "VISION_PROMPTS")

    def test_bench_analysis_importable(self):
        """bench_analysis.py can be imported."""
        mod = importlib.import_module("bench_analysis")
        assert hasattr(mod, "load_results_tsv")
        assert hasattr(mod, "print_summary")
