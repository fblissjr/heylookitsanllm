"""The Messages `performance` object against the model that declares it.

`MessageStopEvent.performance` is typed `Optional[PerformanceInfo]` but the
payload is a raw dict written straight out by `_sse`, so nothing validates one
against the other. Two mismatches lived there undetected: the model declared
both rates REQUIRED while the stream never sent them, and the stream sent
three telemetry keys the model never declared -- so a client generated from
`/openapi.json` dropped them off every message_stop without an error.

WHAT CHANGED IN v1.79.58, and why this file was rewritten rather than
extended. Until then there were TWO builders, and drift entered at the CALL
SITE through a caller-supplied `timing` dict. A test could not close that: any
test supplies its own dict, so it asserted "given declared keys, the output is
declared" -- true by construction and green through the exact bug. There is no
caller-supplied dict any more. `perf_collector.build_performance` spells every
field itself, for both modes and both routes.

That single emit site is what makes the SECOND direction checkable for the
first time. v1.79.55 filtered emitted down to declared and was structurally
blind to declared-but-never-emitted -- which is precisely what
`peak_memory_gb` was until .50 and both rates were until .54. With one builder
the two sets can be compared in both directions, and
`test_every_declared_field_is_reachable` is that comparison.
"""

import json

import pytest

from heylook_llm.messages_api import StreamingEventTranslator
from heylook_llm.perf_collector import ChunkTelemetry, build_performance
from heylook_llm.providers.base import CacheReport, SpecReport
from heylook_llm.schema.responses import PerformanceInfo


def _fully_measured() -> ChunkTelemetry:
    """A run in which the engine reported everything it can report."""
    t = ChunkTelemetry()
    t.prompt_tps = 120.5
    t.generation_tps = 34.5
    t.peak_memory_gb = 12.25
    t.queue_wait_ms = 7.5
    t.cache = CacheReport(prompt_tokens=100, cached_tokens=60, outcome="reused")
    t.spec = SpecReport(accepted=5, drafted=10, emitted=8)
    return t


@pytest.mark.unit
class TestThePayloadAndItsModelAgreeBothWays:
    def test_every_emitted_key_is_declared(self):
        perf = build_performance(
            _fully_measured(),
            request_duration_ms=1000,
            generation_duration_ms=900,
            thinking_duration_ms=100,
            content_duration_ms=800,
        )
        undeclared = set(perf) - set(PerformanceInfo.model_fields)
        assert not undeclared, f"emitted but undeclared: {undeclared}"

    def test_every_declared_field_is_reachable(self):
        """The direction v1.79.55's filter could not see.

        A field declared here and emitted by NOTHING is a promise on
        `/openapi.json` that no run can keep -- `peak_memory_gb` was exactly
        that until .50, and both rates until .54, and in each case the filter
        was green because it only ever looked the other way. One emit site is
        what makes this askable at all.
        """
        perf = build_performance(
            _fully_measured(),
            request_duration_ms=1000,
            generation_duration_ms=900,
            thinking_duration_ms=100,
            content_duration_ms=800,
        )
        never_emitted = set(PerformanceInfo.model_fields) - set(perf)
        assert not never_emitted, (
            f"declared on PerformanceInfo but no run can produce them: "
            f"{never_emitted} -- either build_performance should emit them or "
            "they should not be declared"
        )

    # An undeclared key is dropped (the wire stays correct) and logged at
    # ERROR: degrading silently would be the failure this exists to end, and
    # the log is the only thing that tells anyone. It goes through ordinary
    # `logging` on purpose -- the JSONL spine is off by default and would
    # have swallowed it. A clean payload logs nothing: an error path that
    # fires on every normal generation is a log nobody reads.
    @pytest.mark.parametrize("undeclared", ["queue_wait_ms", None],
                             ids=["undeclared_key_dropped_and_logged", "clean_payload_logs_nothing"])
    def test_undeclared_keys_are_dropped_with_an_error_log(self, caplog, monkeypatch, undeclared):
        if undeclared:
            real = PerformanceInfo.model_fields
            monkeypatch.setattr(
                PerformanceInfo, "model_fields",
                {k: v for k, v in real.items() if k != undeclared},
            )
        with caplog.at_level("ERROR"):
            perf = build_performance(_fully_measured(), request_duration_ms=1)
        logs = [r.getMessage() for r in caplog.records if "performance" in r.getMessage()]
        if undeclared:
            assert undeclared not in perf, "an undeclared key reached the wire"
            assert any(undeclared in m for m in logs), "the drop was silent"
        else:
            assert not logs


@pytest.mark.unit
class TestAbsentMeansUnmeasurable:
    """The contract: present = measured exactly what the name says."""

    def test_the_generation_span_excludes_the_queue_wait(self):
        """Its own description promises "EXCLUDING queue wait and model load".

        Both callers time the span from before the generator is first
        advanced, and `create_chat_completion` is a GENERATOR FUNCTION, so the
        gate is acquired on that first `next()` -- inside the consume loop,
        after the clock started. The raw span therefore contains the wait.
        """
        t = ChunkTelemetry()
        t.queue_wait_ms = 30_000.0        # 30s behind another generation
        perf = build_performance(t, request_duration_ms=35_000,
                                 generation_duration_ms=35_000)
        assert perf["generation_duration_ms"] == 5_000, (
            "the throughput denominator still contains the queue wait; a "
            "client dividing by it reports a fraction of the true rate"
        )
        assert perf["request_duration_ms"] == 35_000, "the wide span keeps it"

    # Each performance key is present iff measured.
    # - unmeasured_queue_wait: v1.79.58 published this zero; v1.79.59 took it
    #   back, on measurement. The wait is an elapsed perf_counter difference,
    #   so an idle gate yields a tiny NONZERO float (live idle runs never
    #   reported 0.0). Exactly 0.0 is the unmeasured set and only it: gguf
    #   never assigns the field (it bypasses this gate), and an MLX run
    #   yielding no chunk loses the tag, which rides the first one. On gguf a
    #   published 0.0 would be every single request.
    # - real_queue_wait: what an IDLE gate actually reports survives.
    # - unreported_rate: prompt_tps shipped a raw 0.0 non-streaming,
    #   indistinguishable from a measured zero (an infinitely slow prefill).
    # - no_rate_synthesized: non-streaming ran generation_tps through
    #   headline_tps, a plausible figure the engine never measured while the
    #   stream omitted it. headline_tps belongs to the internal perf records,
    #   not to this wire.
    # - unmeasurable_span: a span the caller cannot measure is omitted.
    @pytest.mark.parametrize("telemetry, kwargs, present, absent", [
        (dict(), dict(request_duration_ms=10), {}, ("queue_wait_ms",)),
        (dict(queue_wait_ms=0.0044), dict(request_duration_ms=10), {"queue_wait_ms": 0.0044}, ()),
        (dict(), dict(request_duration_ms=10), {}, ("prompt_tps", "generation_tps")),
        (dict(completion_tokens=500), dict(request_duration_ms=1000, generation_duration_ms=1000),
         {}, ("generation_tps",)),
        (None, dict(request_duration_ms=100), {}, ("generation_duration_ms", "thinking_duration_ms")),
    ], ids=["unmeasured_queue_wait_absent", "real_queue_wait_survives", "unreported_rate_absent_not_zero",
            "no_rate_synthesized", "unmeasurable_span_omitted"])
    def test_present_iff_measured(self, telemetry, kwargs, present, absent):
        if telemetry is None:
            t = _fully_measured()
        else:
            t = ChunkTelemetry()  # never latched: the UNMEASURED state
            for k, v in telemetry.items():
                setattr(t, k, v)
        perf = build_performance(t, **kwargs)
        assert {k: perf.get(k) for k in present} == present
        assert not set(absent) & set(perf), f"unmeasured keys published: {set(absent) & set(perf)}"


@pytest.mark.unit
class TestBothRoutesOnTheGrammarUseTheBuilder:
    def _perf(self, **kw):
        t = StreamingEventTranslator("msg_x", "test-model")
        sse = t.message_stop_event(_fully_measured(), **kw)
        line = [l for l in sse.splitlines() if l.startswith("data:")][0]
        return json.loads(line[len("data:"):])["performance"]

    # The generation span is always reported. The request span only when the
    # caller gives a start time: the translator's clock starts when the stream
    # does, which is AFTER get_provider, so it can never be the request span.
    @pytest.mark.parametrize("started_ago_s", [None, 5], ids=["no_start_time", "given_start_time"])
    def test_message_stop_spans(self, started_ago_s):
        import time
        kw = {} if started_ago_s is None else {"request_start_time": time.time() - started_ago_s}
        perf = self._perf(**kw)
        assert "generation_duration_ms" in perf
        if started_ago_s is None:
            assert "request_duration_ms" not in perf
        else:
            assert perf["request_duration_ms"] >= started_ago_s * 1000 - 100
            assert perf["generation_duration_ms"] < perf["request_duration_ms"]
