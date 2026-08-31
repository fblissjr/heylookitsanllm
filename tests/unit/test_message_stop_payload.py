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
from heylook_llm.schema.responses import PerformanceInfo


def _fully_measured() -> ChunkTelemetry:
    """A run in which the engine reported everything it can report."""
    t = ChunkTelemetry()
    t.prompt_tps = 120.5
    t.generation_tps = 34.5
    t.peak_memory_gb = 12.25
    t.kv_cache_bytes = 4096
    t.queue_wait_ms = 7.5
    t.draft_tokens = 10
    t.draft_accepted = 5
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

    def test_an_undeclared_key_is_dropped_and_logged(self, caplog, monkeypatch):
        """Degrading silently would be the failure this exists to end.

        The drop keeps the wire correct; the log is the only thing that tells
        anyone. It goes through ordinary `logging` on purpose -- the JSONL
        spine is off by default and would have swallowed it.
        """
        real = PerformanceInfo.model_fields
        monkeypatch.setattr(
            PerformanceInfo, "model_fields",
            {k: v for k, v in real.items() if k != "queue_wait_ms"},
        )
        with caplog.at_level("ERROR"):
            perf = build_performance(_fully_measured(), request_duration_ms=1)
        assert "queue_wait_ms" not in perf, "an undeclared key reached the wire"
        assert any("queue_wait_ms" in r.getMessage() for r in caplog.records), (
            "the drop was silent"
        )

    def test_a_clean_payload_logs_nothing(self, caplog):
        """An error path that fires on every normal generation is a log nobody
        reads, which is the same as no log at all."""
        with caplog.at_level("ERROR"):
            build_performance(_fully_measured(), request_duration_ms=1)
        assert not [r for r in caplog.records if "performance" in r.getMessage()]


@pytest.mark.unit
class TestAbsentMeansUnmeasurable:
    """The contract: present = measured exactly what the name says."""

    def test_an_unmeasured_queue_wait_is_absent(self):
        """v1.79.58 published this zero; v1.79.59 took it back, on measurement.

        .58 reasoned that "a request that waited no time really did wait zero,
        so dropping the zero hides a measurement on an idle server". That
        premise was false. The wait is an elapsed `perf_counter` difference,
        so an idle gate yields a tiny NONZERO float -- live runs on an idle
        server reported 0.0044, 0.0037 and 0.0024 ms, never 0.0. The set that
        emits exactly 0.0 is the unmeasured set and only it: gguf never assigns
        the field (it bypasses this gate entirely), and an MLX run yielding no
        chunk loses the tag, which rides the first one.

        So this asserts the ORIGINAL behaviour, restored. The test that stood
        here asserted the defect, using a bare ChunkTelemetry -- the unmeasured
        state -- as if it were the idle-server state.
        """
        t = ChunkTelemetry()  # never latched: the UNMEASURED state
        perf = build_performance(t, request_duration_ms=10)
        assert "queue_wait_ms" not in perf, (
            "an unmeasured queue wait was published as a measured 0.0 -- on "
            "gguf that is every single request"
        )

    def test_a_real_queue_wait_survives(self):
        t = ChunkTelemetry()
        t.queue_wait_ms = 0.0044          # what an IDLE gate actually reports
        perf = build_performance(t, request_duration_ms=10)
        assert perf["queue_wait_ms"] == 0.0044

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

    def test_an_unreported_rate_is_absent_not_zero(self):
        """`prompt_tps` shipped a raw 0.0 non-streaming, indistinguishable
        from a measured zero -- which reads as an infinitely slow prefill. A
        rate of exactly zero is not a measurement of anything."""
        t = ChunkTelemetry()  # prompt_tps defaults to 0.0, never latched
        perf = build_performance(t, request_duration_ms=10)
        assert "prompt_tps" not in perf
        assert "generation_tps" not in perf

    def test_no_rate_is_ever_synthesized(self):
        """Non-streaming ran `generation_tps` through `headline_tps`, so it
        produced a plausible figure the engine never measured while the stream
        omitted it -- one name, two guarantees, indistinguishable on the wire.
        """
        t = ChunkTelemetry()
        t.completion_tokens = 500
        perf = build_performance(t, request_duration_ms=1000,
                                 generation_duration_ms=1000)
        assert "generation_tps" not in perf, (
            "a rate was synthesized from the duration -- headline_tps belongs "
            "to the internal perf records, not to this wire"
        )

    def test_a_span_the_caller_cannot_measure_is_omitted(self):
        perf = build_performance(_fully_measured(), request_duration_ms=100)
        assert "generation_duration_ms" not in perf
        assert "thinking_duration_ms" not in perf


@pytest.mark.unit
class TestBothRoutesOnTheGrammarUseTheBuilder:
    def _perf(self, **kw):
        t = StreamingEventTranslator("msg_x", "test-model")
        sse = t.message_stop_event(_fully_measured(), **kw)
        line = [l for l in sse.splitlines() if l.startswith("data:")][0]
        return json.loads(line[len("data:"):])["performance"]

    def test_message_stop_reports_the_generation_span(self):
        perf = self._perf()
        assert "generation_duration_ms" in perf
        # The translator's clock starts when the stream does, which is AFTER
        # get_provider -- so it can never be the request span.
        assert "request_duration_ms" not in perf

    def test_message_stop_reports_the_request_span_when_given_one(self):
        import time
        perf = self._perf(request_start_time=time.time() - 5)
        assert perf["request_duration_ms"] >= 4900
        assert perf["generation_duration_ms"] < perf["request_duration_ms"]

    def test_total_duration_ms_is_gone(self):
        """Retired rather than aliased (v1.79.58).

        Two spellings for one value is the defect class v1.79.48 cited when it
        MOVED the load route instead of aliasing it. "Total" was the
        ambiguous name -- it meant request arrival in one mode and stream
        start in the other.
        """
        assert "total_duration_ms" not in self._perf()
        assert "total_duration_ms" not in PerformanceInfo.model_fields
