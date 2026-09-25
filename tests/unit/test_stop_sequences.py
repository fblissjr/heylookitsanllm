# tests/unit/test_stop_sequences.py
"""Client stop sequences on /v1/messages (stop_sequences.py).

One property for the filter (any chunking of any text streams exactly what
one-shot truncation gives), and the two Messages paths driven end to end:
the reply is cut before the match, stop_reason/stop_sequence say so, the
generation is told to stop, and thinking never triggers it.
"""
import asyncio
import json
import random
from unittest.mock import patch

import pytest

from heylook_llm.perf_collector import PerfCollector
from heylook_llm.providers.abort import AbortEvent
from heylook_llm.schema.messages import MessageCreateRequest
from heylook_llm.stop_sequences import StopSequenceFilter, truncate


class _Chunk:
    def __init__(self, text="", thinking=None):
        self.text = text
        self.thinking = thinking
        self.finish_reason = None
        self.token = None


@pytest.mark.unit
def test_any_chunking_streams_what_truncation_gives():
    rng = random.Random(7)
    alphabet = "abEND \n"
    sequences_pool = [["END"], ["EN", "END"], ["\n\n"], ["abab", "ba"], ["zzz"]]
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        sequences = rng.choice(sequences_pool)
        cuts = sorted(rng.sample(range(len(text) + 1), k=min(len(text) + 1, rng.randint(0, 6))))
        pieces = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
        f = StopSequenceFilter(sequences)
        streamed = "".join(f.feed(p) for p in pieces) + f.flush()
        assert (streamed, f.matched) == truncate(text, sequences), (text, sequences, pieces)


def _request(**extra):
    return MessageCreateRequest.model_validate(
        {"model": "m", "max_tokens": 64,
         "messages": [{"role": "user", "content": "hi"}], **extra})


def _events(parts):
    out = []
    for part in parts:
        for line in part.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def _stream(chunks, req, abort):
    from heylook_llm.messages_api import _stream_messages

    async def drain():
        return [p async for p in _stream_messages(
            iter(chunks), req, "rid", http_request=None, abort_event=abort)]

    with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
        return _events(asyncio.run(drain()))


@pytest.mark.unit
def test_streaming_cuts_the_reply_and_stops_the_generation():
    abort = AbortEvent()
    events = _stream([_Chunk("Hello wor"), _Chunk("ld. EN"), _Chunk("D and more")],
                     _request(stop_sequences=["END"]), abort)
    text = "".join(e["delta"].get("text", "") for e in events
                   if e.get("type") == "content_block_delta")
    delta = next(e for e in events if e.get("type") == "message_delta")["delta"]
    assert text == "Hello world. "
    assert delta == {"stop_reason": "stop_sequence", "stop_sequence": "END"}
    assert abort.is_set()


@pytest.mark.unit
def test_thinking_never_ends_the_reply():
    abort = AbortEvent()
    events = _stream([_Chunk(thinking="plan: END here"), _Chunk("no stop in the reply")],
                     _request(stop_sequences=["END"]), abort)
    delta = next(e for e in events if e.get("type") == "message_delta")["delta"]
    assert delta == {"stop_reason": "end_turn"}
    assert not abort.is_set()


@pytest.mark.unit
def test_non_streaming_cuts_the_reply_and_names_the_sequence():
    from heylook_llm.messages_api import _non_stream_messages

    abort = AbortEvent()
    with patch("heylook_llm.messages_api.get_perf_collector", return_value=PerfCollector()):
        resp = asyncio.run(_non_stream_messages(
            (c for c in [_Chunk("one two "), _Chunk("STOP three")]),
            _request(stop_sequences=["STOP"]), "rid", 0.0, abort_event=abort))
    assert "".join(b.text for b in resp.content if b.type == "text") == "one two "
    assert (resp.stop_reason, resp.stop_sequence) == ("stop_sequence", "STOP")
    assert abort.is_set()


@pytest.mark.unit
@pytest.mark.parametrize("field", ["tools", "tool_choice", "response_format"])
def test_unbuilt_fields_are_refused_not_dropped(field):
    with pytest.raises(ValueError, match="not supported"):
        _request(**{field: {}})
