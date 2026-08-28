# tests/unit/test_busy_response.py
#
# MODEL_BUSY has TWO causes and they are not the same situation. Three
# endpoints turn it into a 503 and each had its own hand-written copy of the
# body, all three of which replaced whatever the server raised with a fixed
# sentence about the queue.

import json

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
    def test_eviction_blocked_keeps_the_raised_detail(self):
        # The case the fixed sentence was wrong about: every loaded model is
        # generating, so there is no slot to load into. The raise names the
        # models and the remedy; "the generation queue is full" names neither
        # and is not true.
        e = RuntimeError("MODEL_BUSY: cannot make room -- ['big-model'] is "
                         "generating. Stop the generation or wait for it to finish.")
        body = _body(model_busy_response(e))
        assert "cannot make room" in body["message"]
        assert "Stop the generation" in body["message"]
        assert "MODEL_BUSY" not in body["message"]   # the marker is not for users

    def test_a_bare_marker_falls_back_to_the_queue_wording(self):
        # The gate's own refusal raises the bare marker; there is nothing more
        # to say, and the queue sentence is the right answer for it.
        for spelling in ("MODEL_BUSY", "MODEL_BUSY:", "MODEL_BUSY: "):
            assert _body(model_busy_response(RuntimeError(spelling)))["message"] \
                == QUEUE_FULL_MESSAGE


class TestTheWireShapeIsUnchanged:
    """v3's streamTypedSSE retries on exactly (503, code=model_overloaded),
    at most MAX_BUSY_RETRIES=3 times, using Retry-After. Changing the code
    would silently disable that retry; lengthening the wait would make the
    user sit longer for the same outcome. Only the message may move."""

    def test_status_code_and_retry_header(self):
        resp = model_busy_response(RuntimeError("MODEL_BUSY"))
        assert resp.status_code == 503
        assert _body(resp)["code"] == "model_overloaded"
        assert _body(resp)["type"] == "server_error"
        assert resp.headers["Retry-After"] == "1"

    def test_capacity_rides_the_rate_limit_header(self):
        resp = model_busy_response(RuntimeError("MODEL_BUSY"), _Provider(capacity=4))
        assert resp.headers["X-RateLimit-Limit"] == "4"
        assert resp.headers["X-RateLimit-Remaining"] == "0"


class TestItSurvivesWhatTheCallSitesActuallyPass:
    def test_no_provider(self):
        # Not hypothetical: _evict_lru_model raises from inside get_provider,
        # BEFORE any provider is bound. api.py's comment used to assert this
        # could not happen.
        resp = model_busy_response(RuntimeError("MODEL_BUSY"), None)
        assert resp.status_code == 503
        assert resp.headers["X-RateLimit-Limit"] == "1"

    def test_a_provider_whose_stats_raise_still_yields_503(self):
        # A 503 that turns into a 500 because the header lookup threw would
        # lose the retry contract entirely.
        resp = model_busy_response(RuntimeError("MODEL_BUSY"), _Provider(raises=True))
        assert resp.status_code == 503
        assert resp.headers["X-RateLimit-Limit"] == "1"
