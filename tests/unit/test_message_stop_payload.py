"""The streaming `message_stop` payload against the model that declares it.

`MessageStopEvent.performance` is typed `Optional[PerformanceInfo]`, but the
payload is assembled as a RAW DICT by `StreamingEventTranslator` and written
straight out by `_sse` -- nothing validates one against the other. That is why
two mismatches lived there undetected: the model declared `prompt_tps` and
`generation_tps` REQUIRED while the stream never sent them, and the stream
sent `kv_cache_bytes`, `queue_wait_ms` and `draft_acceptance` which the model
never declared, so a client generated from `/openapi.json` dropped three
telemetry values off every message_stop without an error.

v1.79.54 made the two agree. This pins that they STAY agreeing, which the
emit path cannot do on its own: it is the construction guarantee replacing a
hand-maintained claim, and the reason it is a subset check rather than a list
is that a list here would be the same census that let the drift happen.
"""

import pytest

from heylook_llm.schema.responses import PerformanceInfo
from heylook_llm.messages_api import StreamingEventTranslator


@pytest.mark.unit
class TestMessageStopMatchesItsModel:
    def _payload(self, timing):
        t = StreamingEventTranslator("msg_x", "test-model")
        sse = t.message_stop_event(timing=timing)
        import json
        # `event: message_stop\ndata: {...}\n\n`
        line = [l for l in sse.splitlines() if l.startswith("data:")][0]
        return json.loads(line[len("data:"):])["performance"]

    def test_every_emitted_key_is_declared_on_performance_info(self):
        """A key the stream sends and the model does not declare is INVISIBLE
        to a generated client -- it arrives and is dropped, with nothing
        reporting it. That is the more dangerous direction of the two."""
        payload = self._payload({
            "prompt_tps": 12.5, "generation_tps": 34.5, "peak_memory_gb": 1.0,
            "kv_cache_bytes": 4096, "queue_wait_ms": 7.5, "draft_acceptance": 0.5,
        })
        undeclared = set(payload) - set(PerformanceInfo.model_fields)
        assert not undeclared, f"message_stop sends undeclared fields: {undeclared}"

    def test_no_declared_field_is_required(self):
        """The mirror direction: a REQUIRED field this payload can omit makes
        the schema unsatisfiable for the mode. Both rates were required while
        the stream never sent them."""
        required = [n for n, f in PerformanceInfo.model_fields.items() if f.is_required()]
        assert not required, f"message_stop cannot guarantee: {required}"

    def test_absent_telemetry_is_omitted_not_null(self):
        """The streaming spelling, which differs from non-streaming on purpose
        and is documented as such -- pinned so the two do not silently
        converge without the doc moving."""
        payload = self._payload({"peak_memory_gb": None, "kv_cache_bytes": None})
        assert "peak_memory_gb" not in payload
        assert "kv_cache_bytes" not in payload
