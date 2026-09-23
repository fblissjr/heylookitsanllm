"""The gguf cache "why" (plan W5): llama_cache_witness plus the provider's
output pump. The log patterns are pinned against the build tree in
test_gguf_non_causal_images.py, beside the other build-pinned facts.
"""
import io

import pytest

from heylook_llm.providers import llama_server_provider as llama_mod
from heylook_llm.providers.base import CacheReport
from heylook_llm.providers.llama_cache_witness import CacheWitness, fingerprint

SKIP = "srv        alloc:  - prompt state size 9100.000 MiB exceeds cache size limit 8192.000 MiB, skipping"
EVICT = "srv        alloc:  - making room for prompt cache entry, removing oldest entry (size = 812.000 MiB)"
SLEEP = "srv  handle_slee: server is entering sleeping state"

A1, A2, A3 = ("sys", "a1"), ("sys", "a1", "r1", "a2"), ("sys", "a1", "r1", "a2", "r2", "a3")
B1 = ("sys", "b1")
C1 = ("other", "c1")


def _req(messages, whole, cached, cause):
    return ("req", messages, whole, cached, cause)


# Each scenario: requests in order (messages, whole prompt, cached, the cause
# explain() must give) and log lines between them.
SCENARIOS = {
    "first request is cold": [_req(A1, 100, 0, "cold")],
    "an extended turn that reused is left alone": [
        _req(A1, 100, 0, "cold"), _req(A2, 180, 98, None)],
    "an extended turn that stopped short": [
        _req(A1, 100, 0, "cold"), _req(A2, 180, 10, "probable_template_diverged")],
    "back to a conversation whose save was skipped": [
        _req(A1, 100, 0, "cold"), ("log", SKIP),
        _req(B1, 60, 0, None),          # shares only the system prompt: no claim
        _req(A2, 180, 0, "probable_budget_skipped")],
    "back to a conversation that was evicted": [
        _req(A1, 100, 0, "cold"), _req(B1, 60, 0, None), ("log", EVICT),
        _req(A2, 180, 0, "probable_evicted")],
    "an eviction BEFORE the conversation was seen does not explain it": [
        ("log", EVICT), _req(A1, 100, 0, "cold"), _req(B1, 60, 0, None),
        _req(A2, 180, 0, "probable_template_diverged")],
    "a new conversation shares nothing": [
        _req(A1, 100, 0, "cold"), _req(C1, 40, 0, "no_common_prefix")],
    "sleep drops the caches": [
        _req(A1, 100, 0, "cold"), ("log", SLEEP), _req(A2, 180, 0, "cold")],
    "the furthest-extended match is the one judged": [
        _req(A1, 100, 0, "cold"), _req(A2, 180, 98, None), _req(B1, 60, 0, None),
        _req(A3, 260, 170, None)],
}


@pytest.mark.unit
@pytest.mark.parametrize("steps", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_explain(steps):
    w = CacheWitness()
    for step in steps:
        if step[0] == "log":
            w.note_line(step[1])
            continue
        _, messages, whole, cached, cause = step
        before = CacheReport(prompt_tokens=whole, cached_tokens=cached,
                             outcome="reused" if cached else "miss")
        after = w.explain(messages, "h", before)
        assert after.cause == cause, (messages, after.reason)
        assert (after.reason is None) == (cause is None)
        assert (after.prompt_tokens, after.cached_tokens, after.outcome) == (
            before.prompt_tokens, before.cached_tokens, before.outcome)
        w.record(messages, "h", whole)


@pytest.mark.unit
def test_fingerprint_is_per_message_and_keeps_no_text():
    payload = {"messages": [{"role": "user", "content": "a secret"}],
               "chat_template_kwargs": {"enable_thinking": True}}
    messages, head = fingerprint(payload)
    longer, _ = fingerprint({**payload, "messages": payload["messages"] * 2})
    assert longer[:1] == messages and len(longer) == 2
    assert "secret" not in repr((messages, head))
    assert fingerprint({**payload, "chat_template_kwargs": {}})[1] != head


@pytest.mark.unit
def test_pump_drains_everything_even_after_the_tee_closes():
    """A pipe nobody reads blocks llama-server: the pump must read to EOF
    whatever happens to the log file."""
    lines = [b"plain line\n", (SKIP + "\n").encode(), b"\xff not utf-8\n", (EVICT + "\n").encode()]

    class ClosingTee(io.BytesIO):
        def write(self, b):
            if self.tell() > 0:
                raise ValueError("I/O operation on closed file")
            return super().write(b)

    stream, tee, w = io.BytesIO(b"".join(lines)), ClosingTee(), CacheWitness()
    llama_mod._pump_output(stream, tee, w)
    assert stream.closed
    assert tee.getvalue() == lines[0]
    assert [kind for _, kind in w._events] == ["budget_skipped", "evicted"]


@pytest.mark.unit
def test_the_stream_carries_the_cause_and_records_the_request():
    frame = ('data: {"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":3,'
             '"prompt_tokens_details":{"cached_tokens":%d}},"timings":{"cache_n":%d}}')
    p = llama_mod.LlamaServerProvider("m", {"model_path": "/fake/m.gguf"}, False)
    p._cache_witness = CacheWitness()
    payload = {"messages": [{"role": "user", "content": "x"}]}

    def run(whole, cached):
        body = io.BytesIO(((frame % (whole, cached, cached)) + "\ndata: [DONE]\n").encode())
        chunks = list(p._explained(p._stream_chunks(body, abort_event=None), payload))
        return [c.cache for c in chunks if c.cache][-1]

    assert run(50, 0).cause == "cold"
    assert run(50, 49).cause is None  # the same request again, reused
