# tests/unit/test_busy_response.py
#
# MODEL_BUSY has TWO causes and they are not the same situation. Three
# endpoints turn it into a 503 and each had its own hand-written copy of the
# body, all three of which replaced whatever the server raised with a fixed
# sentence about the queue.

import json

import pytest

from heylook_llm.busy_response import QUEUE_FULL_MESSAGE, model_busy_response


def _body(resp):
    return json.loads(bytes(resp.body).decode())["error"]


class _Provider:
    def __init__(self, capacity=None, raises=False):
        self._capacity = capacity
        self._raises = raises

    def generation_queue_stats(self):
        if self._raises:
            raise RuntimeError("provider is mid-teardown")
        return {"capacity": self._capacity} if self._capacity else None


class TestTheMessageIsTheServersOwn:
    @pytest.mark.parametrize("spellings, expected, must_contain", [
        # The case the fixed sentence was wrong about: every loaded model is
        # generating, so there is no slot to load into. The raise names the
        # models and the remedy; "the generation queue is full" names neither
        # and is not true.
        pytest.param(("MODEL_BUSY: cannot make room -- ['big-model'] is "
                      "generating. Stop the generation or wait for it to finish.",),
                     None, ("cannot make room", "Stop the generation"),
                     id="eviction_blocked_keeps_the_raised_detail"),
        # The gate's own refusal raises the bare marker; there is nothing more
        # to say, and the queue sentence is the right answer for it.
        pytest.param(("MODEL_BUSY", "MODEL_BUSY:", "MODEL_BUSY: "),
                     QUEUE_FULL_MESSAGE, (),
                     id="a_bare_marker_falls_back_to_the_queue_wording"),
    ])
    def test_the_message(self, spellings, expected, must_contain):
        for spelling in spellings:
            message = _body(model_busy_response(RuntimeError(spelling)))["message"]
            if expected is not None:
                assert message == expected, spelling
            for fragment in must_contain:
                assert fragment in message
            assert "MODEL_BUSY" not in message   # the marker is not for users


_OMIT = object()   # call without the provider argument at all


class TestTheWireShapeIsUnchanged:
    """v3's streamTypedSSE retries on exactly (503, code=model_overloaded),
    at most MAX_BUSY_RETRIES=3 times, using Retry-After. Changing the code
    would silently disable that retry; lengthening the wait would make the
    user sit longer for the same outcome. Only the message may move.

    Every row asserts the whole wire shape; the rows are what the call sites
    actually pass as ``provider``."""

    @pytest.mark.parametrize("provider, limit", [
        pytest.param(_OMIT, "1", id="status_code_and_retry_header"),
        pytest.param(_Provider(capacity=4), "4", id="capacity_rides_the_rate_limit_header"),
        # Not hypothetical: _evict_lru_model raises from inside get_provider,
        # BEFORE any provider is bound. api.py's comment used to assert this
        # could not happen.
        pytest.param(None, "1", id="no_provider"),
        # A 503 that turns into a 500 because the header lookup threw would
        # lose the retry contract entirely.
        pytest.param(_Provider(raises=True), "1",
                     id="a_provider_whose_stats_raise_still_yields_503"),
    ])
    def test_the_wire_shape(self, provider, limit):
        e = RuntimeError("MODEL_BUSY")
        resp = model_busy_response(e) if provider is _OMIT else model_busy_response(e, provider)
        assert resp.status_code == 503
        assert _body(resp)["code"] == "model_overloaded"
        assert _body(resp)["type"] == "server_error"
        assert resp.headers["Retry-After"] == "1"
        assert resp.headers["X-RateLimit-Limit"] == limit
        assert resp.headers["X-RateLimit-Remaining"] == "0"
