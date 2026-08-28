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
from typing import Any

GREEN, RED, YELLOW, DIM, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[2m", "\x1b[0m"

# The three engines. `loader` is what actually decodes; `provider` is only how
# models.toml spells it. Keep these apart -- conflating them is the mistake
# this file exists to prevent.
ARMS = ("mlx-lm", "mlx-vlm", "gguf")

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
# discovery: which engine is each model?
# ---------------------------------------------------------------------------

def classify(server):
    """model_id -> arm. The vision capability is what splits provider 'mlx'
    into its two upstream libraries -- see the header."""
    st, models = call(server, "GET", "/v1/models", timeout=30)
    if st != 200:
        raise SystemExit(f"GET /v1/models failed: {st} {models}")
    st, admin = call(server, "GET", "/v1/admin/models", timeout=30)
    provider_by_id = {}
    if st == 200:
        for m in (admin or {}).get("models", []):
            provider_by_id[m.get("id")] = m.get("provider")

    out = {}
    for entry in (models or {}).get("data", []):
        mid = entry["id"]
        caps = set(entry.get("capabilities") or [])
        prov = provider_by_id.get(mid)
        if prov == "gguf":
            out[mid] = "gguf"
        elif prov == "mlx":
            out[mid] = "mlx-vlm" if "vision" in caps else "mlx-lm"
        # anything else (embeddings, unknown) is deliberately unclassified
    return out


def pick_models(by_arm, overrides, wanted):
    """One model per requested arm. Prefers the smallest-looking id, because a
    smoke test that costs a 122B load will not get run."""
    chosen = {}
    for arm in wanted:
        if arm in overrides:
            chosen[arm] = overrides[arm]
            continue
        candidates = sorted([m for m, a in by_arm.items() if a == arm], key=lambda s: (len(s), s))
        if candidates:
            chosen[arm] = candidates[0]
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
        r.check("conversation params + preset stamp round-trip",
                st == 200 and got["params"].get("temperature") == 0.5
                and got.get("applied_preset_id") == p1["id"],
                f"read back params={got.get('params')} stamp={got.get('applied_preset_id')}")

        st, _ = call(server, "PUT", f"/v1/conversations/{conv['id']}",
                     {"params": {"temperature": 0.9, "top_k": 40}})
        st, got = call(server, "GET", f"/v1/conversations/{conv['id']}")
        r.check("a params PUT replaces the bag wholesale",
                got["params"].get("temperature") == 0.9 and got["params"].get("top_k") == 40,
                f"read back {got.get('params')}")

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

def stream_until(server, conv_id, body, stop_after_bytes=None, timeout=300) -> tuple[bool, bool, str]:
    """POST the generate stream. If stop_after_bytes is set, DISCONNECT once
    that much has arrived -- that is the walk-away this whole harness exists
    to check.

    Returns (saw_any_delta, completed_normally, tail). `tail` is the last of
    the body so a no-delta ending can SAY WHY -- RETURNED, not stashed in a
    module global: the stop check below runs this on a worker thread, and a
    global cleared-then-appended from a worker is a race waiting for the first
    person to drive two streams at once."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{server}/v1/conversations/{conv_id}/generate", data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    read = 0
    saw = False
    tail = b""
    resp = urllib.request.urlopen(req, timeout=timeout)
    try:
        while True:
            chunk = resp.read(256)
            if not chunk:
                return saw, True, tail.decode(errors="replace")
            read += len(chunk)
            # Keep the last of the body so a no-delta ending can SAY WHY. The
            # first run of this harness reported only "saw_delta=False" for a
            # bad image fixture, and the cause was reachable only by digging
            # through the server log.
            tail = (tail + chunk)[-1200:]
            if b"text_delta" in chunk:
                saw = True
            if stop_after_bytes is not None and saw and read >= stop_after_bytes:
                return saw, False, tail.decode(errors="replace")  # walk away mid-stream
    finally:
        resp.close()


def wait_for_idle(server, conv_id, timeout=300) -> Any:
    """Poll until the conversation stops reporting `generating`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st, conv = call(server, "GET", f"/v1/conversations/{conv_id}", timeout=30)
        if st == 200 and not conv.get("generating"):
            return conv
        time.sleep(1.0)
    return None


def arm_checks(server, r, arm, model_id, load_timeout):
    print(f"\n{DIM}-- {arm}: {model_id} ---------------------------------------{RESET}")

    st, _ = call(server, "POST", f"/v1/admin/models/{model_id}/load?warm=true", timeout=load_timeout)
    if not r.check(f"{arm}: load + warm", st == 200, f"load returned {st}"):
        r.skip(f"{arm}: lifecycle", "model would not load")
        return

    st, models = call(server, "GET", "/v1/models", timeout=30)
    caps = next((set(m.get("capabilities") or []) for m in models["data"] if m["id"] == model_id), set())
    if arm == "mlx-vlm":
        r.check(f"{arm}: reports the vision capability", "vision" in caps, f"caps: {sorted(caps)}")

    conv_id = None
    try:
        st, conv = call(server, "POST", "/v1/conversations",
                        {"title": f"smoke-{arm}", "model_id": model_id,
                         "params": {"max_tokens": 220, "temperature": 0.7}})
        if not r.check(f"{arm}: conversation create", st == 201, f"got {st}: {conv}"):
            return
        conv_id = conv["id"]

        # -- an ordinary generation completes and persists ------------------
        content = "Say hello in one short sentence."
        if arm == "mlx-vlm":
            content = [
                {"type": "text", "text": "Reply with one short sentence about this image."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(SMOKE_PNG).decode()}},
            ]
        st, _ = call(server, "POST", f"/v1/conversations/{conv_id}/messages",
                     {"role": "user", "content": content})
        if not r.check(f"{arm}: user message accepted", st == 201, f"got {st}"):
            return

        saw, done, tail = stream_until(server, conv_id, {"mode": "append"})
        r.check(f"{arm}: a generation streams and completes", saw and done,
                f"saw_delta={saw} completed={done}; stream tail: {tail[-400:]!r}")
        conv = wait_for_idle(server, conv_id)
        replies = [m for m in (conv or {}).get("messages", []) if m["role"] == "assistant"]
        r.check(f"{arm}: the reply persisted", bool(replies) and bool(replies[-1].get("content")),
                f"{len(replies)} assistant rows")

        # -- THE walk-away check: drop the connection mid-stream. The run must
        # DETACH and commit the whole answer, not truncate. This is exactly
        # what the client now discloses and what the stubbed suite cannot see.
        st, _ = call(server, "POST", f"/v1/conversations/{conv_id}/messages",
                     {"role": "user", "content": "Count slowly from one to twenty, one number per line."})
        before = len([m for m in (wait_for_idle(server, conv_id) or {}).get("messages", [])])
        saw, done, tail = stream_until(server, conv_id, {"mode": "append"}, stop_after_bytes=400)
        if not saw:
            r.skip(f"{arm}: a walked-away run finishes on the server", "no delta arrived before the disconnect")
        else:
            r.check(f"{arm}: disconnected mid-stream", not done, "the stream ended on its own")
            conv = wait_for_idle(server, conv_id, timeout=300)
            rows = (conv or {}).get("messages", [])
            last = rows[-1] if rows else {}
            r.check(f"{arm}: a walked-away run finishes and commits",
                    conv is not None and len(rows) > before
                    and last.get("role") == "assistant" and bool(last.get("content")),
                    f"rows {before} -> {len(rows)}, last={last.get('role')!r} "
                    f"len={len(last.get('content') or '')}")

        # -- Stop is the OTHER thing, and must keep only the partial --------
        st, _ = call(server, "POST", f"/v1/conversations/{conv_id}/messages",
                     {"role": "user", "content": "Count slowly from one to fifty, one number per line."})
        holder = {}

        def _run():
            # Results go in `holder`, which only this thread writes and only
            # the main thread reads AFTER join() -- no shared mutable state
            # while both are running.
            try:
                holder["res"] = stream_until(server, conv_id, {"mode": "append"})
            except Exception as e:  # the abort surfaces here on some transports
                holder["err"] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        time.sleep(3.0)
        st, _ = call(server, "DELETE", f"/v1/conversations/{conv_id}/generate", timeout=30)
        r.check(f"{arm}: stop accepted", st in (200, 404), f"stop returned {st}")
        th.join(timeout=120)
        conv = wait_for_idle(server, conv_id, timeout=120)
        r.check(f"{arm}: the conversation is idle after a stop", conv is not None,
                "still reported generating")
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
        overrides = dict(kv.split("=", 1) for kv in args.model)
        by_arm = classify(server)
        wanted = args.arm or list(ARMS)
        chosen = pick_models(by_arm, overrides, wanted)
        for arm in wanted:
            if arm not in chosen:
                r.skip(f"{arm}: whole arm", "no model of this engine is served")
                continue
            arm_checks(server, r, arm, chosen[arm], args.load_timeout)

    total = r.passed + len(r.failed)
    print(f"\n{'-' * 58}")
    if r.failed:
        print(f"{RED}FAIL{RESET}  {r.passed}/{total} checks passed")
        for name, why in r.failed:
            print(f"  {RED}✗{RESET} {name}: {why}")
    else:
        print(f"{GREEN}PASS{RESET}  {r.passed}/{total} checks passed")
    if r.skipped:
        print(f"{YELLOW}{len(r.skipped)} skipped{RESET} — a skipped ENGINE is uncovered, not green:")
        for name, why in r.skipped:
            print(f"  {YELLOW}-{RESET} {name}: {why}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
