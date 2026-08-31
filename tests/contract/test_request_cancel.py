# tests/contract/test_request_cancel.py
"""DELETE /v1/requests/{id} on the wire (v1.79.43).

The registry's rules are unit-tested; this pins the HTTP contract a client
actually codes against, and in particular the two answers that carry meaning:
a 404 for an id that is not running (most often because it already finished),
and a `cancelled` count rather than a bare boolean, because client-supplied
ids are not assumed unique.
"""

import pathlib

import pytest

from heylook_llm.providers.abort import AbortEvent
from heylook_llm.request_registry import get_request_registry, track_request


class TestCancelEndpoint:
    def test_cancelling_a_live_request_reports_what_it_signalled(self, client):
        event = AbortEvent()
        with track_request("live-req", event):
            r = client.delete("/v1/requests/live-req")
        assert r.status_code == 200
        body = r.json()
        assert body["cancelled"] == 1
        assert body["request_id"] == "live-req"
        assert event.is_set()

    def test_an_unknown_id_is_404(self, client):
        """Not a silent 200. A client that hung up too late must learn it was
        too late rather than believe it stopped something."""
        r = client.delete("/v1/requests/never-ran")
        assert r.status_code == 404
        assert "already finished" in r.json()["detail"]

    def test_a_finished_request_is_404(self, client):
        with track_request("done-req", AbortEvent()):
            pass
        assert client.delete("/v1/requests/done-req").status_code == 404

    def test_duplicate_ids_report_the_count(self, client):
        """The response says 2 rather than "ok" because the id is the
        client's, and two in-flight requests can legitimately share one."""
        first, second = AbortEvent(), AbortEvent()
        with track_request("dup", first), track_request("dup", second):
            r = client.delete("/v1/requests/dup")
        assert r.json()["cancelled"] == 2
        assert first.is_set() and second.is_set()

    def test_the_route_is_published_in_the_schema(self, client):
        """A cancel endpoint nobody can discover is a cancel endpoint nobody
        uses -- and this repo's integration guide points clients at
        /openapi.json as authoritative."""
        schema = client.get("/openapi.json").json()
        assert "delete" in schema["paths"]["/v1/requests/{request_id}"]

    def test_the_registry_is_left_clean(self, client):
        before = set(get_request_registry().live_ids())
        with track_request("temp", AbortEvent()):
            client.delete("/v1/requests/temp")
        assert set(get_request_registry().live_ids()) == before


class TestMessagesHonoursTheClientId:
    def test_the_messages_route_echoes_the_client_id_on_a_stream(self, client):
        """/v1/messages generated its own id and ignored the header, so the
        value a client sends -- and which the docs tell it to send -- named
        nothing the server could find. Cancellation is by that exact value, so
        honouring it is the precondition for the endpoint working at all."""
        r = client.post(
            "/v1/messages",
            headers={"X-Request-ID": "client-chosen-id"},
            json={"model": "test-mlx-model", "max_tokens": 8, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.headers.get("X-Request-ID") == "client-chosen-id"

class TestMalformedIdIsNotAMiss:
    """A malformed id answers 422, not 404.

    The distinction is the point, not the status code: the POST end replaces
    an unusable X-Request-ID with a generated one, so an id of this shape was
    never tracked and never could be. Answering 404 made that indistinguishable
    from "the run already finished", and a client quietly generating bad ids
    would read permanent 404s as "cancellation does not work".
    """

    def test_malformed_id_is_422(self, client):
        resp = client.delete("/v1/requests/not%20a%20valid%20id")
        assert resp.status_code == 422

    def test_a_trailing_newline_is_malformed(self, client):
        """The `$`-vs-fullmatch trap, from the other end.

        v1.79.46 fixed `resolve_request_id` accepting a trailing newline
        (Python's `$` matches before one). The cancel route asks through the
        same predicate, so it inherits the fix -- this pins that it is asked
        through it rather than through a second, re-anchored regex.
        """
        resp = client.delete("/v1/requests/abc%0A")
        assert resp.status_code == 422

    def test_a_well_formed_but_unknown_id_is_still_404(self, client):
        """The 422 must not swallow the case the endpoint mostly answers."""
        resp = client.delete("/v1/requests/msg-not-running-abc123")
        assert resp.status_code == 404

    def test_the_charset_is_not_duplicated(self):
        """The route must ask the resolver's predicate, not its own copy."""
        import heylook_llm.requests_api as mod
        src = pathlib.Path(mod.__file__).read_text()
        assert "is_valid_request_id" in src
        assert "A-Za-z0-9" not in src, (
            "requests_api re-declares the id charset; it must come from "
            "request_registry so the two ends cannot disagree")

