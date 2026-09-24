# tests/unit/test_perf_ab_verdict.py
"""scripts/perf_ab.py's verdict rules: what may be called a change.

The claim: overlap, too few values, or a negligible difference is never a
change, and direction follows the metric. An instrument that calls noise a
win is worse than none (seen while building it: one cold load read as a
load-time 'win', a 0.3% footprint difference as 'worse')."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "scripts" / "perf_ab.py"
_spec = importlib.util.spec_from_file_location("perf_ab", _path)
perf_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf_ab)


@pytest.mark.unit
@pytest.mark.parametrize("a,b,higher,expected", [
    ([70, 72, 71], [78, 79, 78], True, "better"),       # decode rate up
    ([70, 72, 71], [78, 79, 78], False, "worse"),       # the same numbers as a latency
    ([70, 80, 75], [74, 79, 76], True, "noise"),        # ranges overlap
    ([29.70, 29.71, 29.70], [29.79, 29.80, 29.79], False, "noise"),  # separated, but 0.3%
    ([6.0, 3.5], [3.4, 3.5], False, "too few runs"),    # one value per round
    ([], [1, 2, 3], True, "n/a"),
])
def test_verdict(a, b, higher, expected):
    assert perf_ab.verdict(a, b, higher) == expected


@pytest.mark.unit
def test_overall():
    assert perf_ab.overall(["better", "noise"]) == "free lunch"
    assert perf_ab.overall(["better", "worse"]) == "tradeoff"
    assert perf_ab.overall(["worse", "noise"]) == "regression"
    assert perf_ab.overall(["noise", "too few runs"]) == "no change beyond noise"
