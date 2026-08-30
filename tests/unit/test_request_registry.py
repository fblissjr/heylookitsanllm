# tests/unit/test_request_registry.py
"""Cancel-by-request-id: the registry's rules (v1.79.43).

Built because a NON-streaming request had no cancellation at all. A stream is
cancellable by hanging up -- the server is writing, so it notices the peer is
gone -- while a non-streaming call writes nothing until it finishes and so
never notices, so an abandoned run continues and blocks what is queued
behind it. (A consuming client's timings prompted this; they are not recorded
here -- uncontrolled measurements do not belong in tracked files.)

The rules worth pinning are the ones a plain ``dict[str, AbortEvent]`` would
get wrong, and they all follow from the id being CLIENT-supplied rather than
server-issued.
"""

import pytest

from heylook_llm.providers.abort import AbortEvent
from heylook_llm.request_registry import (
    RequestRegistry,
    resolve_request_id,
    track_request,
    tracked_stream,
)


@pytest.mark.unit
class TestCancellingById:
    def test_cancel_sets_the_registered_event(self):
        reg, ev = RequestRegistry(), AbortEvent()
        reg.register("req-1", ev)
        assert reg.cancel("req-1") == 1
        assert ev.is_set()

    def test_an_unknown_id_cancels_nothing(self):
        """Which is what the route turns into a 404. A registry that reported
        success for an id it never held would tell a client it stopped
        something that had already finished."""
        assert RequestRegistry().cancel("never-existed") == 0

    def test_a_finished_request_becomes_unknown(self):
        """Against the MODULE registry, which is the one `track_request` writes
        to. This asserted against a fresh local `RequestRegistry()` that
        `track_request` never touches, so it passed for the wrong reason --
        `unregister` could have been deleted outright and it would still be
        green. It names the module's central lifetime rule; it has to actually
        exercise it."""
        from heylook_llm.request_registry import get_request_registry

        reg = get_request_registry()
        ev = AbortEvent()
        with track_request("req-finished", ev):
            assert reg.cancel("req-finished") == 1   # live: cancellable
        assert reg.cancel("req-finished") == 0       # finished: unknown
        assert "req-finished" not in reg.live_ids()

    def test_duplicate_ids_both_cancel(self):
        """The id is CLIENT-supplied, so uniqueness is not ours to assume: a
        retry, a buggy client or a shared correlation id can put two live
        requests under one name. A single-slot map would let the second
        registration orphan the first -- a running generation nothing can
        name, which is the exact condition this module exists to remove."""
        reg = RequestRegistry()
        first, second = AbortEvent(), AbortEvent()
        reg.register("shared", first)
        reg.register("shared", second)
        assert reg.cancel("shared") == 2
        assert first.is_set() and second.is_set()

    def test_one_of_two_finishing_leaves_the_other_cancellable(self):
        reg = RequestRegistry()
        done, running = AbortEvent(), AbortEvent()
        reg.register("shared", done)
        reg.register("shared", running)
        reg.unregister("shared", done)
        assert reg.cancel("shared") == 1
        assert running.is_set()
        assert not done.is_set()

    def test_cancel_does_not_evict_a_still_running_request(self):
        """Each request removes its OWN entry when it unwinds. Evicting here
        would make a second cancel of a still-running generation answer 404
        while it was demonstrably still running."""
        reg, ev = RequestRegistry(), AbortEvent()
        reg.register("req-1", ev)
        assert reg.cancel("req-1") == 1
        assert reg.cancel("req-1") == 1


@pytest.mark.unit
class TestRegistrationLifetime:
    def test_the_map_is_empty_once_a_request_unwinds(self):
        """Bounded by liveness, not by a clock -- there is no TTL sweeper, so
        a leaked entry would be a permanent one."""
        reg_ev = AbortEvent()
        with track_request("req-1", reg_ev):
            pass
        from heylook_llm.request_registry import get_request_registry
        assert "req-1" not in get_request_registry().live_ids()

    def test_an_exception_still_unregisters(self):
        """Route bodies raise HTTPException as ordinary control flow, so the
        error path is the common path, not the rare one."""
        from heylook_llm.request_registry import get_request_registry

        with pytest.raises(ValueError):
            with track_request("req-boom", AbortEvent()):
                raise ValueError("boom")
        assert "req-boom" not in get_request_registry().live_ids()

    @pytest.mark.asyncio
    async def test_a_stream_is_registered_for_its_own_lifetime(self):
        """The wrapper exists because a streaming body OUTLIVES the route
        function: a `with` around the return would unregister before the first
        token. Asserts registration is live mid-stream and gone after."""
        from heylook_llm.request_registry import get_request_registry

        seen = []

        async def body():
            seen.append(get_request_registry().live_ids())
            yield "a"
            seen.append(get_request_registry().live_ids())
            yield "b"

        out = [x async for x in tracked_stream(body(), "stream-1", AbortEvent())]
        assert out == ["a", "b"]
        assert all("stream-1" in ids for ids in seen)
        assert "stream-1" not in get_request_registry().live_ids()

    @pytest.mark.asyncio
    async def test_abandoning_a_stream_unregisters_it(self):
        """A client hang-up closes the generator rather than exhausting it.
        Registration must not survive that -- a leaked id would answer a later
        cancel with a success it cannot deliver."""
        from heylook_llm.request_registry import get_request_registry

        async def body():
            yield "a"
            yield "b"

        agen = tracked_stream(body(), "stream-2", AbortEvent())
        assert await agen.__anext__() == "a"
        await agen.aclose()
        assert "stream-2" not in get_request_registry().live_ids()


@pytest.mark.unit
class TestRequestIdResolution:
    def test_a_usable_client_id_is_honoured_verbatim(self):
        """Verbatim is load-bearing: the client cancels by the value it sent,
        so any server rewrite makes the request uncancellable."""
        assert resolve_request_id("my-trace-01", prefix="msg") == "my-trace-01"

    def test_a_missing_header_gets_a_generated_id(self):
        got = resolve_request_id(None, prefix="msg")
        assert got.startswith("msg-") and len(got) > 10

    @pytest.mark.parametrize("bad", [
        "has space", "has\nnewline", "semi;colon", "x" * 129, "",
        # TRAILING newline specifically: Python's `$` also matches just before
        # one, so `re.match(r"^...$", "abc\n")` is TRUTHY and an id ending in a
        # newline sailed straight through the log-forging guard. The interior
        # case above passed throughout and hid it.
        "abc\n", "abc\r\n", "abc\r",
    ])
    def test_a_malformed_id_is_replaced_rather_than_trusted(self, bad):
        """These reach logs and the JSONL telemetry streams. A newline could
        forge a log line and an unbounded value bloats every event carrying
        it, so the id is bounded and charset-restricted before use."""
        got = resolve_request_id(bad, prefix="req")
        assert got != bad
        assert got.startswith("req-")
