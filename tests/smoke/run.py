#!/usr/bin/env python3
# Live smoke test: the v3 contract and the generation lifecycle, against a
# REAL server, once per ENGINE.
#
# Why this exists. `tests/e2e/render.mjs` drives the real /v3 page against a
# STUBBED /v1, so it can prove the client behaves -- and nothing about whether
# the server does. Everything the client's preset and lifecycle work rests on
# (a preset store that refuses a duplicate name, params that round-trip onto a
# conversation, a run that DETACHES and finishes after the reader walks away)
# is invisible to it. This is the other half.
#
# Why per ENGINE and not per provider. The provider Literal has three values
# but they are not three engines:
#
#   provider "mlx"  -> mlx-lm   (text)     ) two SEPARATE upstream repos, on
#                   -> mlx-vlm  (vision)   ) separate release trains
#   provider "gguf" -> llama-server subprocess (one engine, one local binary)
#
# Which of the two MLX libraries runs is `MLXProvider.effective_loader`, not
# the provider field, so "we covered mlx" is a claim about a config value
# rather than about code. A text arm and a vision arm are different engines
# and this harness treats them that way. (mlx_embedding is deliberately out of
# scope -- owner call 2026-08-28.)
#
# This tool NEVER spawns a server -- same rule as tests/eval/run.py. Point
# --server at a running `heylookllm`.
#
# Usage:
#   uv run python tests/smoke/run.py --server http://127.0.0.1:8000
#   uv run python tests/smoke/run.py --server ... --contract-only   # no model loads
#   uv run python tests/smoke/run.py --server ... --arm gguf        # one engine
#   uv run python tests/smoke/run.py --server ... --model mlx-lm=Qwen3.5-0.8B-MLX-8bit
from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import struct
import sys
import threading
import time
import zlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `tests/` on the path: this runs as a SCRIPT, not under pytest, so nothing has
# inserted the rootdir for us. helpers/ is where shared test code already lives
# (mlx_mock, sse).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.engines import ARMS, classify, format_coverage  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[2m", "\x1b[0m"

def _wav(freq_hz=440, seconds=0.5, rate=16000):
    """A real mono WAV, stdlib only (same rule as the PNG above).

    SHORT and synthetic on purpose: nothing here judges what the model heard --
    that is the eval bank's job. This exists so an audio part is WELL-FORMED,
    because a malformed one gets rejected by the schema and the 400 below would
    then be the schema's, not the provider's, while reading identically.
    """
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq_hz * i / rate)))
            for i in range(int(rate * seconds))))
    return buf.getvalue()


def _png(width=64, height=64):
    """A real RGB PNG, built with stdlib only (the eval bank's rule: no deps).

    NOT a 1x1 pixel. The first version of this harness used one and the vision
    arm failed before the model ever saw it: gemma's aspect-ratio-preserving
    resize hands PIL a degenerate (1,1,1) array and PIL refuses it. A
    degenerate fixture tests the preprocessor's edge handling, which is not
    what a smoke test is for -- it should fail only when the ENGINE is broken.
    """
    raw = b"".join(
        b"\x00" + bytes(
            v for x in range(width)
            for v in ((x * 4) % 256, (y * 4) % 256, 128)
        )
        for y in range(height)
    )

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


SMOKE_PNG = _png()
SMOKE_WAV = _wav()


# ---------------------------------------------------------------------------
# tiny http
# ---------------------------------------------------------------------------

def call(server, method, path, body=None, timeout=120, raw=False) -> tuple[int, Any]:
    """Returns (status, parsed_or_bytes). Never raises on an HTTP status --
    a 409 IS the assertion in places here."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{server}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload
            return resp.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return e.code, json.loads(payload)
        except Exception:
            return e.code, payload.decode(errors="replace")


def _dict(body):
    """A response body as a dict, whatever `call` actually returned.

    `call` yields a parsed dict on success, but None on an empty body and a raw
    STRING when an error body is not JSON. Every read of a field therefore has
    to survive both. It did not: one unguarded `got["params"]` turned a failed
    GET into a TypeError that unwound the whole contract half -- a harness that
    reports a server problem as a crash reports nothing at all.
    """
    return body if isinstance(body, dict) else {}


def _params_of(body):
    return _dict(_dict(body).get("params"))


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class Report:
    passed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name):
        self.passed += 1
        print(f"  {GREEN}✓{RESET} {name}")

    def fail(self, name, why):
        self.failed.append((name, why))
        print(f"  {RED}✗{RESET} {name}\n      {RED}{why}{RESET}")

    def skip(self, name, why):
        self.skipped.append((name, why))
        print(f"  {YELLOW}-{RESET} {name} {DIM}({why}){RESET}")

    def check(self, name, cond, why=""):
        self.ok(name) if cond else self.fail(name, why or "assertion failed")
        return bool(cond)


# ---------------------------------------------------------------------------
# discovery: which engine is each model?  (helpers/engines.py -- shared with
# tests/eval/run.py, because two copies of a taxonomy is one drifting copy)
# ---------------------------------------------------------------------------


def model_size_gb(server, model_id):
    """Measured weight size in GB, or None. NO LOAD.

    `POST /v1/admin/models/{id}/fit` sizes a model from FILE STATS (plus a
    vm_stat/sysctl read for the verdict, which is ignored here). It exists for
    the models page's "will this fit" panel; the number it already computes is
    the only real cost signal any endpoint serves, and it is exactly what
    choosing an arm's model needs.
    """
    st, body = call(server, "POST", f"/v1/admin/models/{urllib.parse.quote(model_id)}/fit",
                    {"headroom_gb": 0}, timeout=60)
    if st != 200 or not isinstance(body, dict):
        return None
    gb = body.get("weights_gb")
    return gb if isinstance(gb, (int, float)) else None


def pick_models(server, by_arm, overrides, wanted, resident=frozenset()):
    """One model per requested arm: resident if there is one, else the SMALLEST.

    Two heuristics, in that order, and the order is the point.

    RESIDENT first because it costs no load at all, and on a single-slot router
    loading something else evicts whatever the owner had running.

    Otherwise the smallest by MEASURED weight size. This used to sort by
    `len(id)`, which is not a cost signal and never was -- the docstring said as
    much ("a short name is not a small model") and it was still wrong in
    practice: on this machine an unnarrowed run picked gpt-oss-120b, a 27B and a
    27B for the three arms. That makes the Phase 4 release standard -- green on
    all three arms -- something nobody would run, which is the same failure the
    whole plan is about, one level up.

    Sizing is one cheap POST per candidate and only for the arms actually
    wanted, skipped entirely for an arm with an override. Where the endpoint
    cannot answer (older server, unreadable path) the model sorts LAST rather
    than first: an unknown size must not win a contest about smallness.
    `--model ARM=ID` remains the way to be sure.
    """
    chosen = {}
    for arm in wanted:
        if arm in overrides:
            chosen[arm] = overrides[arm]
            continue
        candidates = [m for m, a in by_arm.items() if a == arm]
        if not candidates:
            continue
        already = [m for m in candidates if m in resident]
        if already:
            chosen[arm] = sorted(already)[0]
            continue
        sizes = {m: model_size_gb(server, m) for m in candidates}
        chosen[arm] = sorted(
            candidates,
            key=lambda m: (sizes[m] is None, sizes[m] if sizes[m] is not None else 0.0, m),
        )[0]
    return chosen


# ---------------------------------------------------------------------------
# provider-independent contract
# ---------------------------------------------------------------------------

def contract_checks(server, r):
    """The store's own rules -- the ones the stubbed render suite fakes.
    No model, no load, seconds."""
    print(f"\n{DIM}-- contract (no model) ------------------------------------{RESET}")
    stamp = f"smoke-{int(time.time())}"
    created = []

    try:
        # -- presets: create, and the DUPLICATE-NAME REFUSAL that "Save as new"
        # now depends on. Before v1.79.26 a duplicate name overwrote silently.
        st, p1 = call(server, "POST", "/v1/presets",
                      {"name": stamp, "system_prompt": "SMOKE PROMPT", "params": {"temperature": 1.23}})
        if not r.check("preset create -> 201", st == 201, f"got {st}: {p1}"):
            return
        created.append(p1["id"])

        st, dup = call(server, "POST", "/v1/presets",
                       {"name": stamp, "system_prompt": "SHOULD NOT LAND", "params": {}})
        r.check("a duplicate preset name is REFUSED, not overwritten", st == 409, f"got {st}: {dup}")

        st, back = call(server, "GET", "/v1/presets")
        row = next((p for p in (back or {}).get("presets", []) if p["id"] == p1["id"]), None)
        r.check("the preset round-trips prompt and params",
                row is not None and row["system_prompt"] == "SMOKE PROMPT"
                and row["params"].get("temperature") == 1.23,
                f"read back {row}")

        # -- update is the ONLY overwrite path (v1.79.26)
        st, upd = call(server, "PUT", f"/v1/presets/{p1['id']}",
                       {"system_prompt": "UPDATED PROMPT"})
        r.check("update overwrites the stored prompt", st == 200 and upd.get("system_prompt") == "UPDATED PROMPT",
                f"got {st}: {upd}")

        # -- a preset carrying NO prompt is storable and stays null: this is the
        # "settings only" state the dropdown now labels.
        st, bare = call(server, "POST", "/v1/presets",
                        {"name": f"{stamp}-bare", "system_prompt": None, "params": {"top_p": 0.9}})
        if st == 201:
            created.append(bare["id"])
            r.check("a promptless preset stores as null, not empty string",
                    bare.get("system_prompt") is None, f"stored {bare.get('system_prompt')!r}")
        else:
            r.fail("a promptless preset is storable", f"got {st}: {bare}")

        # -- conversations: params and the applied-preset stamp round-trip.
        # The drawer's whole scope story rests on params living on the ROW.
        st, conv = call(server, "POST", "/v1/conversations",
                        {"title": stamp, "params": {"temperature": 0.5}, "applied_preset_id": p1["id"]})
        if not r.check("conversation create -> 201", st == 201, f"got {st}: {conv}"):
            return
        created.append(("conv", conv["id"]))

        st, got = call(server, "GET", f"/v1/conversations/{conv['id']}")
        params = _params_of(got) if st == 200 else {}
        r.check("conversation params + preset stamp round-trip",
                st == 200 and params.get("temperature") == 0.5
                and _dict(got).get("applied_preset_id") == p1["id"],
                f"got {st}: read back params={params} stamp={_dict(got).get('applied_preset_id')}")

        st, _ = call(server, "PUT", f"/v1/conversations/{conv['id']}",
                     {"params": {"temperature": 0.9, "top_k": 40}})
        st, got = call(server, "GET", f"/v1/conversations/{conv['id']}")
        params = _params_of(got) if st == 200 else {}
        r.check("a params PUT replaces the bag wholesale",
                st == 200 and params.get("temperature") == 0.9 and params.get("top_k") == 40,
                f"got {st}: read back {params}")

        # -- Phase 3, row 4: chat-template source. TWO mechanisms that must
        # stay on their own providers -- `chat_template_source` is MLX-only
        # (it force-installs a template on the tokenizer at load) and
        # `chat_template_path` is gguf-only (`--chat-template-file`, taken at
        # SPAWN, which is why it requires a reload). The failure mode is
        # silence: setting the wrong one for a provider does nothing at all,
        # and the publisher's embedded template goes on picking your prompt
        # format. Model-free, because the settable set is DERIVED from the
        # provider config classes and the endpoint is where that lands.
        st, opts = call(server, "GET", "/v1/admin/model-options")
        providers = ((opts or {}).get("providers") or {}) if st == 200 else {}
        fields = {prov: {f.get("name") for f in (spec.get("fields") or [])}
                  for prov, spec in providers.items()}
        if st != 200 or not fields:
            r.fail("the two chat-template mechanisms stay on their own providers",
                   f"GET /v1/admin/model-options returned {st}: {str(opts)[:200]}")
        else:
            r.check("chat_template_source is offered for mlx and NOT for gguf",
                    "chat_template_source" in fields.get("mlx", set())
                    and "chat_template_source" not in fields.get("gguf", set()),
                    f"mlx={sorted(fields.get('mlx', set()) & {'chat_template_source', 'chat_template_path'})} "
                    f"gguf={sorted(fields.get('gguf', set()) & {'chat_template_source', 'chat_template_path'})}")
            r.check("chat_template_path is offered for gguf and NOT for mlx",
                    "chat_template_path" in fields.get("gguf", set())
                    and "chat_template_path" not in fields.get("mlx", set()),
                    f"mlx={sorted(fields.get('mlx', set()) & {'chat_template_source', 'chat_template_path'})} "
                    f"gguf={sorted(fields.get('gguf', set()) & {'chat_template_source', 'chat_template_path'})}")

        # -- the list carries `generating`, which is what the composer's third
        # state (Stop for a run this page never subscribed to) is built on.
        st, lst = call(server, "GET", "/v1/conversations")
        row = next((c for c in (lst or {}).get("conversations", []) if c["id"] == conv["id"]), None)
        r.check("the conversation list reports `generating`",
                row is not None and "generating" in row, f"row keys: {sorted(row or {})}")
    finally:
        for item in created:
            if isinstance(item, tuple):
                call(server, "DELETE", f"/v1/conversations/{item[1]}", timeout=30)
            else:
                call(server, "DELETE", f"/v1/presets/{item}", timeout=30)


# ---------------------------------------------------------------------------
# per-engine lifecycle
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    saw_delta: bool = False       # any text arrived
    completed: bool = False       # the body ended on its own
    text: str = ""                # the deltas WE saw, exactly
    tail: str = ""                # last of the body, so a failure can say why
    http_error: str | None = None # a non-2xx, surfaced instead of raised

    saved_end_reason: str | None = None  # parsed from heylook_saved, not the tail

    @property
    def ended_complete(self) -> bool:
        """The server's own statement that the run finished rather than failed.
        `heylook_saved` is always last (spec §4) and the error path skips
        message_stop entirely, so a stream can END cleanly having FAILED --
        checking only 'did it finish' reports a dead engine as a passing arm.

        Read from the PARSED event, not by substring-matching `tail`. `tail`
        is the last 1500 bytes, and `end_reason` sits before the `messages`
        array in heylook_saved -- so a run whose answer was long enough to
        push the payload past the window failed this check while the server
        was reporting "complete" correctly. Flaky by ANSWER LENGTH, which is
        the worst kind: it reds the board at random and looks like an engine
        fault."""
        return self.saved_end_reason == "complete"


def stream_until(server, conv_id, body, stop_after_bytes=None, timeout=300) -> StreamResult:
    """POST the generate stream. If stop_after_bytes is set, DISCONNECT once
    that much has arrived -- the walk-away this harness exists to check.

    Accumulates the delta text WE saw, which is what makes the walk-away check
    able to fail: comparing the persisted answer against this prefix is how you
    tell "the run detached and finished" from "the server truncated at the
    disconnect". Asserting only that a non-empty assistant row exists is true
    of both.

    Never raises on an HTTP status -- a 409 (run already active) or 503
    (model_overloaded) is a result to report, not a traceback that kills every
    later arm."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{server}/v1/conversations/{conv_id}/generate", data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    res = StreamResult()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        res.http_error = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
        return res
    except OSError as e:
        res.http_error = f"{type(e).__name__}: {e}"
        return res

    read = 0
    buf = b""
    tail = b""
    saved_end_reason = None
    try:
        while True:
            chunk = resp.read(256)
            if not chunk:
                res.completed = True
                break
            read += len(chunk)
            tail = (tail + chunk)[-1500:]
            buf += chunk
            # Consume whole lines only; a split JSON payload waits for its rest.
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.startswith(b"data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except Exception:
                    continue
                # heylook_saved is one line and always last. Read it HERE:
                # the body arrives in 256-byte chunks, so the payload spans
                # many of them and any per-chunk parse sees partial JSON.
                # This loop already waits for whole lines.
                if evt.get("type") == "heylook_saved":
                    saved_end_reason = evt.get("end_reason")
                # CONTENT deltas only: a thinking_delta carries `text` too (v3
                # reads it there), and counting it here made a thinking-only
                # reply look truncated against a content-only persisted row.
                delta = evt.get("delta") or {}
                piece = delta.get("text") if delta.get("type") == "text_delta" else None
                if piece:
                    res.saw_delta = True
                    res.text += piece
            if stop_after_bytes is not None and res.saw_delta and read >= stop_after_bytes:
                break  # walk away mid-stream
    finally:
        resp.close()
        res.tail = tail.decode(errors="replace")
        res.saved_end_reason = saved_end_reason
    return res


def wait_for_idle(server, conv_id, timeout=300) -> Any:
    """Poll until the conversation stops reporting `generating`. Returns the
    conversation, or None on timeout / a run of failed GETs -- and callers must
    treat None as NO ANSWER, never as 'idle'. Reading `(x or {})` off this
    silently converts a stuck run into a weaker assertion that passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st, conv = call(server, "GET", f"/v1/conversations/{conv_id}", timeout=30)
        if st == 200 and not conv.get("generating"):
            return conv
        time.sleep(1.0)
    return None


def assistant_text(conv) -> str:
    rows = [m for m in (conv or {}).get("messages", []) if m.get("role") == "assistant"]
    return (rows[-1].get("content") or "") if rows else ""


# ---------------------------------------------------------------------------
# Phase 3: the same-feature-two-mechanisms invariants
#
# Each of these is ONE feature with TWO implementations, split by engine. That
# is the reason they belong here and not in a unit test: reasoning from the
# provider Literal hides the second implementation, and only a live run of both
# arms can show that they still agree.
#
# The rule for every row: where the precondition is unmet, report UNCOVERED.
# Do not weaken a check to make it runnable.
# ---------------------------------------------------------------------------

def _messages_probe(server, model_id, content, extra=None, timeout=180):
    """One non-streaming POST /v1/messages. Returns (status, body).

    NON-streaming deliberately. The provider's request guards fire at the first
    next(), which on the streaming path is AFTER the headers have flushed -- so
    the same refusal arrives as a 200 carrying an in-band `error` event
    (messages_api types it invalid_request_error, correctly). Only the
    non-streaming form turns it into the status code these checks read.
    """
    body = {"model": model_id, "max_tokens": 24, "stream": False,
            "messages": [{"role": "user", "content": content}]}
    body.update(extra or {})
    return call(server, "POST", "/v1/messages", body, timeout=timeout)


def conformance_checks(server, r, arm, model_id, caps):
    """Anthropic Messages conformance, against a REAL server (v1.79.39-40).

    Unit tests drive StreamingEventTranslator and the Pydantic models
    DIRECTLY, so they prove the pieces and not the route: every one of them
    stayed green while the two Messages-grammar routes disagreed about
    stop_reason. These go over the wire.

    The image row is the one that matters most. Before 1.79.39 the ONLY
    accepted media spelling was heylook's flat `source_type`, so a
    correctly-formed Anthropic request -- what an Anthropic SDK sends -- was
    a 422 from an endpoint advertising itself as Messages-shaped.
    """
    # 1. stop_reason speaks Anthropic's vocabulary, not the provider's
    #    OpenAI finish_reason. Every arm can run this one.
    st, body = _messages_probe(server, model_id, "Say the single word: ok")
    if r.check(f"{arm}: messages responds", st == 200, f"got {st}: {str(body)[:200]}"):
        stop = (body or {}).get("stop_reason")
        r.check(f"{arm}: stop_reason is Anthropic vocabulary",
                stop in ("end_turn", "max_tokens", "stop_sequence"),
                f"got {stop!r} -- 'stop'/'length' is the provider's OpenAI "
                f"finish_reason reaching the wire unrenamed")

    # 2. Anthropic's NESTED source object is accepted on a media block.
    if "vision" not in caps:
        r.skip(f"{arm}: nested image source accepted",
               "this arm's model does not advertise vision -- the conformance "
               "row that regressed hardest is UNCOVERED here")
    else:
        nested = [
            {"type": "text", "text": "Answer in one word: what colour is this?"},
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(SMOKE_PNG).decode()}},
        ]
        st, body = _messages_probe(server, model_id, nested)
        detail = str((body or {}).get("detail") if isinstance(body, dict) else body)
        if st == 400 and "text-only" in detail:
            # The row asks whether the SPELLING is accepted, and a 400 saying
            # the model cannot take images means it was never reached -- so
            # UNCOVERED rather than red either way. What this branch means
            # CHANGED at v1.79.43: the capability and the provider's guard now
            # come from one resolver, so an MLX model reaching here is no
            # longer the ordinary over-report it was written for. It is now
            # either an explicit `capabilities` override in models.toml
            # (honoured verbatim, so an operator can still assert something the
            # server will not serve) or a real regression in that shared
            # derivation. Kept, and kept loud, because the second reading is
            # the one worth catching -- a run that lands here should be
            # investigated, not filed as a known wart.
            r.skip(f"{arm}: nested image source accepted",
                   f"UNCOVERED: {model_id} advertises the vision capability and "
                   f"the provider refuses images as text-only, so the spelling "
                   f"was never reached. Since v1.79.43 those two answers share "
                   f"one resolver, so this is either a hand-written "
                   f"`capabilities` override or a regression -- not the "
                   f"derived-from-config.json over-report it used to be")
        else:
            r.check(f"{arm}: nested image source accepted", st == 200,
                    f"got {st} -- a 422 means the server rejected Anthropic's own "
                    f"image spelling. body: {str(body)[:300]}")

    # 3. A thinking block names the field Anthropic names it. Only meaningful
    #    where the model actually produces one.
    if "thinking" not in caps:
        r.skip(f"{arm}: thinking block carries `thinking`",
               "this arm's model does not advertise thinking")
        return
    st, body = _messages_probe(
        server, model_id, "Think briefly, then answer: what is 17 + 25?",
        extra={"thinking": True, "max_tokens": 256})
    blocks = (body or {}).get("content") or [] if st == 200 else []
    thinking_blocks = [b for b in blocks if b.get("type") == "thinking"]
    if not thinking_blocks:
        r.skip(f"{arm}: thinking block carries `thinking`",
               "the model returned no thinking block for this prompt -- "
               "UNCOVERED, not passing")
        return
    r.check(f"{arm}: thinking block carries `thinking`",
            bool(thinking_blocks[0].get("thinking")),
            f"block has no non-empty `thinking` field (Anthropic's name); "
            f"keys: {sorted(thinking_blocks[0])}")


def audio_checks(server, r, arm, model_id, caps):
    """Audio input: supported on gguf, must fail LOUDLY on MLX.

    MLX loads audio-capable checkpoints with skip_audio=True -- the tower is
    stripped -- so an audio part that reached generation would be silently
    dropped and answered text-only. That is a wrong answer the user cannot see,
    which is why the guard raises rather than skips, and why this is the row
    worth having first.

    Sent to /v1/messages, NOT to the conversation generate endpoint. The
    conversation path drops media the current model cannot take AT THE WIRE,
    with a per-message disclosure (the mid-conversation model-switch rule), so
    the provider guard is unreachable there BY DESIGN and a check aimed at it
    would pass without ever running the guard. (Near-unreachable rather than
    unreachable: the drop keys on the DERIVED capability, so a hand-written
    `capabilities = ["audio"]` override on an MLX entry would route the part
    through and hit the 400 after all.)
    """
    audio = [
        {"type": "text", "text": "Answer in one word: is this a tone or speech?"},
        {"type": "audio", "source_type": "base64", "media_type": "audio/wav",
         "data": base64.b64encode(SMOKE_WAV).decode()},
    ]
    if arm == "gguf":
        if "audio" not in caps:
            r.skip(f"{arm}: audio is accepted", "this gguf model declares no audio modality "
                                                "-- the supported half of the row is UNCOVERED")
            return
        st, body = _messages_probe(server, model_id, audio)
        r.check(f"{arm}: audio is accepted", st == 200, f"got {st}: {str(body)[:300]}")
        return

    # mlx-lm / mlx-vlm
    st, body = _messages_probe(server, model_id, audio, timeout=60)
    r.check(f"{arm}: audio is REFUSED, loudly", st == 400,
            f"got {st} -- a 200 means the audio part was dropped and the model "
            f"answered text-only, which is the failure this check exists for. "
            f"body: {str(body)[:300]}")
    if st == 400:
        # The message has to name the way out, or a loud failure is just a
        # failure. gguf is where audio works and the text says so.
        detail = str((body or {}).get("detail") if isinstance(body, dict) else body)
        r.check(f"{arm}: the refusal names gguf as the way to run audio",
                "gguf" in detail.lower(), f"detail: {detail[:200]}")


def thinking_checks(server, r, arm, model_id, caps):
    """Thinking, and thinking DEPTH -- one feature, two mechanisms each.

    Capability:  MLX probes the model's template FILE for `enable_thinking`;
                 gguf rides the entry's `supports_thinking`, because the
                 template lives inside GGUF metadata with nothing cheap to scan.
    Depth:       MLX passes `reasoning_effort` as an apply_chat_template kwarg;
                 gguf sends it in `chat_template_kwargs`.

    What is checked is that a model ADVERTISING the capability ACCEPTS the
    corresponding request. That pairing is the invariant worth holding: a
    capability nothing honours is a control the UI shows and the model ignores,
    and on gguf a value the template rejects is a raised jinja exception, which
    llama-server returns as a 500.
    """
    if "thinking" in caps:
        st, body = _messages_probe(server, model_id, "Name one colour.",
                                   {"chat_template_kwargs": {"enable_thinking": True}})
        r.check(f"{arm}: a thinking-capable model accepts enable_thinking",
                st == 200, f"got {st}: {str(body)[:300]}")
    else:
        r.skip(f"{arm}: thinking capability", "this arm's model does not advertise thinking "
                                              "-- that mechanism is UNCOVERED on this arm")

    if "reasoning_effort" in caps:
        # "medium" is the one value in BOTH published sets (Qwen3.8 takes
        # xhigh|medium|low, harmony low|medium|high) and the accepted set is
        # PER MODEL -- a wrong-for-this-model value reaches the template, where
        # llama-server turns a raised jinja exception into a 500. Picking the
        # intersection is what keeps this a check about the MECHANISM rather
        # than about one model's vocabulary.
        st, body = _messages_probe(server, model_id, "Name one colour.",
                                   {"reasoning_effort": "medium"})
        r.check(f"{arm}: a depth-capable model accepts reasoning_effort=medium",
                st == 200, f"got {st}: {str(body)[:300]}")
    else:
        r.skip(f"{arm}: thinking depth", "this arm's model does not advertise reasoning_effort "
                                         "-- that mechanism is UNCOVERED on this arm")


def arm_checks(server, r, arm, model_id, load_timeout):
    print(f"\n{DIM}-- {arm}: {model_id} ---------------------------------------{RESET}")

    st, loaded = call(server, "POST", f"/v1/models/{urllib.parse.quote(model_id)}/load?warm=true",
                      timeout=load_timeout)
    # `warmed` matters: the endpoint returns 200 with warmed=false + warm_error
    # when the first forward pass RAISED -- the single most informative signal
    # a smoke test can collect, and it used to be discarded into `_`.
    if not r.check(f"{arm}: load + warm",
                   st == 200 and (loaded or {}).get("warmed") is not False,
                   f"load returned {st}: {(loaded or {}).get('warm_error') or loaded}"):
        r.skip(f"{arm}: lifecycle", "model would not load or warm")
        return

    st, models = call(server, "GET", "/v1/models", timeout=30)
    caps = next((set(m.get("capabilities") or []) for m in models["data"] if m["id"] == model_id), set())
    if arm == "mlx-vlm":
        r.check(f"{arm}: reports the vision capability", "vision" in caps, f"caps: {sorted(caps)}")

    # Phase 3 rows, before the lifecycle: they are cheap (one short
    # non-streaming request each) and they run on a model that is now loaded
    # and warm. A failure here says something about the ENGINE rather than
    # about the store, so it is worth knowing before the long checks.
    audio_checks(server, r, arm, model_id, caps)
    thinking_checks(server, r, arm, model_id, caps)
    conformance_checks(server, r, arm, model_id, caps)

    conv_id = None
    try:
        # Thinking OFF for the lifecycle checks (v1.79.66): they assert on
        # persisted CONTENT, and since v1.79.62 a thinking-capable model thinks
        # by default, so a small model's whole budget went into the thinking
        # block and "the reply persisted" read an empty content field as a
        # lost reply. Thinking has its own checks above.
        st, conv = call(server, "POST", "/v1/conversations",
                        {"title": f"smoke-{arm}", "model_id": model_id,
                         "params": {"max_tokens": 400, "temperature": 0.7,
                                    "enable_thinking": False}})
        if not r.check(f"{arm}: conversation create", st == 201, f"got {st}: {conv}"):
            return
        conv_id = conv["id"]

        def say(text):
            """Append a user turn. CHECKED: _refuse_while_generating 409s an
            append while a run is active, and an unchecked 409 silently changes
            what the next generation is testing."""
            st, _ = call(server, "POST", f"/v1/conversations/{conv_id}/messages",
                         {"role": "user", "content": text})
            return st == 201

        # -- an ordinary generation completes and persists ------------------
        content = "Say hello in one short sentence."
        if arm == "mlx-vlm":
            content = [
                {"type": "text", "text": "Reply with one short sentence about this image."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(SMOKE_PNG).decode()}},
            ]
        if not r.check(f"{arm}: user message accepted", say(content), "append refused"):
            return

        res = stream_until(server, conv_id, {"mode": "append"})
        # ended_complete, not just `completed`: the error path yields an error
        # event, persists the partial and ends the body cleanly, so "it finished"
        # is true of a FAILED run too.
        r.check(f"{arm}: a generation streams and completes",
                res.saw_delta and res.completed and res.ended_complete,
                f"delta={res.saw_delta} completed={res.completed} "
                f"end_reason_complete={res.ended_complete} err={res.http_error} "
                f"tail={res.tail[-300:]!r}")
        conv = wait_for_idle(server, conv_id)
        r.check(f"{arm}: the reply persisted", bool(assistant_text(conv)),
                "no assistant content" if conv is not None else "the run never went idle")

        # -- THE walk-away check --------------------------------------------
        # Disconnect mid-stream. The run must DETACH and commit the WHOLE
        # answer. The discriminating comparison is against the prefix WE saw:
        # if the server truncated at the disconnect, the persisted answer is
        # that prefix. Asserting only "a non-empty assistant row exists" is
        # equally true of a truncation, which is the bug this arm exists for.
        if not r.check(f"{arm}: second user message accepted",
                       say("Count slowly from one to twenty, one number per line."),
                       "append refused (is a run still active?)"):
            return
        res = stream_until(server, conv_id, {"mode": "append"}, stop_after_bytes=400)
        if res.http_error or not res.saw_delta:
            r.skip(f"{arm}: a walked-away run finishes and commits",
                   res.http_error or "no delta arrived before the disconnect")
            # Wait it out regardless: a run left active 409s every later append
            # and every later check in this arm then measures THIS generation.
            wait_for_idle(server, conv_id, timeout=300)
        else:
            r.check(f"{arm}: disconnected mid-stream", not res.completed,
                    "the stream ended on its own before we could disconnect")
            conv = wait_for_idle(server, conv_id, timeout=300)
            final = assistant_text(conv)
            if conv is None:
                r.fail(f"{arm}: a walked-away run finishes and commits",
                       "the run never went idle -- no answer either way")
            else:
                r.check(f"{arm}: a walked-away run finishes and commits",
                        len(final) > len(res.text),
                        f"persisted {len(final)} chars vs the {len(res.text)} we saw before "
                        f"disconnecting -- the run was TRUNCATED at the disconnect, not detached")

        # -- Stop keeps the partial, and only the partial --------------------
        if not r.check(f"{arm}: third user message accepted",
                       say("Count slowly from one to two hundred, one number per line."),
                       "append refused"):
            return
        holder = {}

        def _run():
            # Only this thread writes holder; only the main thread reads it,
            # and only after join().
            try:
                holder["res"] = stream_until(server, conv_id, {"mode": "append"})
            except Exception as e:  # pragma: no cover -- transport oddities
                holder["err"] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        time.sleep(3.0)
        st, _ = call(server, "DELETE", f"/v1/conversations/{conv_id}/generate", timeout=30)
        th.join(timeout=120)
        finished_first = bool(holder.get("res") and holder["res"].ended_complete)
        # 200 ONLY -- a 404 means "no active generation", i.e. nothing was
        # stopped, and accepting it passed the Stop path without exercising it.
        # But a 404 because the run FINISHED inside our sleep window is an
        # unexercised check, not a failed one: a fast small model answers well
        # inside 3s. Say so rather than either passing or failing falsely.
        if st == 404 and finished_first:
            r.skip(f"{arm}: stop", "the run completed before Stop could be pressed "
                                   "(fast model) -- the Stop path was not exercised")
            return
        r.check(f"{arm}: stop accepted", st == 200,
                f"stop returned {st} (404 = nothing was running, so nothing was stopped)")
        r.check(f"{arm}: the stopped stream ended", not th.is_alive() and "err" not in holder,
                f"thread alive={th.is_alive()} err={holder.get('err')}")
        conv = wait_for_idle(server, conv_id, timeout=120)
        stopped_text = assistant_text(conv)
        r.check(f"{arm}: the conversation is idle after a stop", conv is not None,
                "still reported generating")
        # The endpoint's contract is that the partial IS persisted. Nothing used
        # to look at the content at all, so a regression that dropped it entirely
        # passed this arm unchanged.
        r.check(f"{arm}: the stop persisted its partial", bool(stopped_text),
                "the aborted run committed no content")
    finally:
        if conv_id:
            call(server, "DELETE", f"/v1/conversations/{conv_id}", timeout=60)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True, help="a RUNNING heylookllm (this never spawns one)")
    ap.add_argument("--arm", action="append", choices=ARMS,
                    help="engine arm to run; repeatable. Default: all that have a model.")
    ap.add_argument("--model", action="append", default=[], metavar="ARM=ID",
                    help="pin an arm's model, e.g. --model gguf=google_gemma-4-E4B-it-qat-q4_0-gguf")
    ap.add_argument("--contract-only", action="store_true",
                    help="skip every model load; run only the store/contract half (seconds)")
    ap.add_argument("--load-timeout", type=int, default=900)
    args = ap.parse_args()

    server = args.server.rstrip("/")
    host = urllib.parse.urlsplit(server)
    s = socket.socket(); s.settimeout(2)
    if s.connect_ex((host.hostname, host.port or 80)) != 0:
        raise SystemExit(f"{RED}nothing listening on {server}{RESET} -- start heylookllm first "
                         "(this harness never spawns one)")
    s.close()

    r = Report()
    contract_checks(server, r)

    if not args.contract_only:
        overrides = {}
        for kv in args.model:
            if "=" not in kv:
                raise SystemExit(f"--model wants ARM=ID, got {kv!r}")
            arm, mid = kv.split("=", 1)
            if arm not in ARMS:
                raise SystemExit(f"--model: unknown arm {arm!r} (choose from {', '.join(ARMS)})")
            overrides[arm] = mid
        cov = classify(server)
        wanted = args.arm or list(ARMS)
        chosen = pick_models(server, cov.by_engine, overrides, wanted, cov.resident)
        # Said BEFORE the arms run, not after: the point of the paragraph is to
        # tell you what this run is about to be evidence FOR, and a summary you
        # read after ten minutes of loads has already let you assume.
        print(f"\n{DIM}-- engine coverage ---------------------------------------{RESET}")
        print(format_coverage(cov, spanned=[a for a in ARMS if a in chosen],
                              narrowed=bool(args.arm or args.model)))
        for arm in wanted:
            if arm not in chosen:
                r.skip(f"{arm}: whole arm", "no model of this engine is served")
                continue
            # Say it ONCE per arm, not once per model: on a server too old to
            # serve `effective_loader` the engine is inferred from the vision
            # capability, and a harness that names engines should say when it
            # is guessing.
            if chosen[arm] in cov.unconfirmable:
                r.skip(f"{arm}: engine identity NOT confirmed", cov.unconfirmable[chosen[arm]])
            arm_checks(server, r, arm, chosen[arm], args.load_timeout)

    total = r.passed + len(r.failed)
    print(f"\n{'-' * 58}")
    if r.failed:
        print(f"{RED}FAIL{RESET}  {r.passed}/{total} checks passed")
        for name, why in r.failed:
            print(f"  {RED}✗{RESET} {name}: {why}")
    else:
        print(f"{GREEN}PASS{RESET}  {r.passed}/{total} checks passed")
    missing = [n for n, _ in r.skipped if n.endswith(": whole arm")]
    if r.skipped:
        print(f"{YELLOW}{len(r.skipped)} skipped{RESET} — a skipped ENGINE is uncovered, not green:")
        for name, why in r.skipped:
            print(f"  {YELLOW}-{RESET} {name}: {why}")
    # The machine-readable signal has to agree with the banner. `return 1 if
    # failed` printed PASS and exited 0 for a run where every arm was skipped,
    # which any wrapper reads as full engine coverage -- the exact opposite of
    # what this file promises. A run the caller NARROWED (--arm / --model /
    # --contract-only) is a decision and exits on its failures alone.
    narrowed = bool(args.arm or args.model or args.contract_only)
    if missing and not narrowed:
        print(f"{RED}UNCOVERED{RESET}  {len(missing)} engine arm(s) ran no model; "
              "pass --arm to say that was deliberate")
        return 2
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
