"""Readers for the Messages SSE grammar, shared by the suites that assert on it.

The grammar these parse IS the contract under test on three surfaces
(`/v1/messages`, `/v1/conversations/{id}/generate`, and the OpenAI wire's
translator), so hand-rolling a reader per test file means a delta-shape change
has to be caught in every copy -- and a copy that silently returns "" makes an
assertion like `SPECIAL not in text` pass while testing nothing.
"""

import json as _json


def sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse '(event, data)' pairs out of an SSE body."""
    events = []
    for block in text.split("\n\n"):
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = _json.loads(line[len("data: "):])
        if ev:
            events.append((ev, data))
    return events


def streamed_text(body: str) -> str:
    """The assistant TEXT a Messages-grammar stream delivered.

    Thinking deltas are deliberately excluded -- callers asserting on what the
    user sees as the answer must not have reasoning silently concatenated in.
    """
    return "".join(
        (data.get("delta") or {}).get("text", "")
        for ev, data in sse_events(body)
        if ev == "content_block_delta"
        and (data.get("delta") or {}).get("type") == "text_delta"
    )


def event_data(body: str, event_type: str) -> dict:
    """The data payload of the first `event_type` event; raises if absent."""
    for ev, data in sse_events(body):
        if ev == event_type:
            return data
    raise AssertionError(f"no {event_type} event in stream:\n{body}")
