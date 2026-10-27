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


@pytest.mark.unit
def test_timed_on_gguf_shaped_stream_keys_ttft_on_first_output_not_token_count(monkeypatch):
    """GGUF emits text/thinking deltas with generation_tokens=0 and reports
    generation_tokens only on the final usage chunk. Keying `first` on
    `generation_tokens` made TTFT equal total wall time and `decode_tps = None`."""
    from types import SimpleNamespace

    # Clock ticks: t0=10.0, chunk1(thinking, tokens=0) at 10.2,
    # chunk2(text, tokens=0) at 10.5, chunk3(text, tokens=0) at 11.2,
    # chunk4(empty usage chunk, tokens=5) at 11.2, final wall clock at 11.25.
    clock = iter([10.0, 10.2, 10.5, 11.2, 11.2, 11.25])
    monkeypatch.setattr(perf_ab.time, "perf_counter", lambda: next(clock))

    chunks = [
        SimpleNamespace(
            text="",
            thinking="let me think",
            generation_tokens=0,
            prompt_tokens=None,
            cache=SimpleNamespace(cached_tokens=16),
        ),
        SimpleNamespace(
            text="Hello ",
            thinking=None,
            generation_tokens=0,
            prompt_tokens=None,
            cache=None,
        ),
        SimpleNamespace(
            text="world!",
            thinking=None,
            generation_tokens=0,
            prompt_tokens=None,
            cache=None,
        ),
        SimpleNamespace(
            text="",
            thinking=None,
            generation_tokens=5,
            prompt_tokens=32,
            cache=None,
        ),
    ]

    class _FakeGGUFProvider:
        def create_chat_completion(self, _req):
            yield from chunks

    res = perf_ab._timed(_FakeGGUFProvider(), object())
    # First output arrived at 10.2 (0.2s after t0=10.0), NOT at 11.2 when generation_tokens=5 arrived!
    assert res["ttft_s"] == pytest.approx(0.2)
    # 5 tokens over (11.2 - 10.2) = 1.0s -> (5 - 1) / 1.0 = 4.0 tok/s
    assert res["decode_tps"] == pytest.approx(4.0)
    assert res["gen_tokens"] == 5
    assert res["prompt_tokens"] == 32
    assert res["cached_tokens"] == 16
    assert res["reply"] == "Hello world!"

