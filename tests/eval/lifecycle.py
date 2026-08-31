#!/usr/bin/env python3
"""Generation-lifecycle checks that need a REAL model actually decoding.

Sibling of run.py, same cost profile -- opt-in, points at an already-running
server, never spawns one, not part of /test-suite. Different KIND though:
run.py judges model behavior, this asserts the plumbing around it.

    uv run python tests/eval/lifecycle.py --server http://127.0.0.1:8991 \
        --models Qwen3.5-0.8B-MLX-8bit,google_gemma-4-E4B-it-qat-q4_0-gguf

WHY THIS IS NOT A UNIT TEST. It was one, briefly, and it certified a guard
that did not work. The teardown guard reads a provider's in-flight generation
count; with MagicMock providers the count is whatever the test says it is, so
the suite was green while the guard returned False for every gguf model --
LlamaServerProvider did not implement the signal at all. Mocks cannot answer
"does the real provider report this", which is the only question that matters
here. RUN IT AGAINST BOTH PROVIDER KINDS: the two fail differently (MLX frees
weights under a live Metal command buffer; llama-server gets SIGTERMed out
from under an open HTTP stream) and gguf is the one that had no protection.

What it pins, per model:

  1. A model that is generating cannot be torn down. The admin unload must
     409, naming the model. Without the guard it returns 200 and kills the
     generation mid-decode -- verified by removing the counter and watching
     this check go from 409 to 200.
  2. Stop actually stops, and the run clears from the active set.
  3. A stopped generation persists EXACTLY what the wire delivered, byte for
     byte. This is what makes a stopped reply editable afterwards: the abort
     has to land on the token it stopped at, with nothing dropped in the
     reasoning parser's rolling holdback and nothing duplicated.
  4. Never more than one llama-server subprocess.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

GREEN, RED, DIM, RESET = "\x1b[32m", "\x1b[31m", "\x1b[2m", "\x1b[0m"


def call(base, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def llama_server_count() -> int:
    try:
        out = subprocess.run(["pgrep", "-f", "llama-server"],
                             capture_output=True, text=True, timeout=10)
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        return -1  # unknown, not zero -- do not assert on a failed probe


def check_model(base: str, model_id: str) -> list[str]:
    """Returns a list of failure strings; empty means everything held."""
    failures: list[str] = []

    def bad(msg):
        failures.append(f"[{model_id}] {msg}")

    status, _ = call(base, "POST", f"/v1/models/{model_id}/load?warm=true", timeout=900)
    if status != 200:
        bad(f"could not load: {status}")
        return failures

    _, conv = call(base, "POST", "/v1/conversations",
                   {"title": "lifecycle probe", "model_id": model_id})
    cid = conv["id"]
    call(base, "POST", f"/v1/conversations/{cid}/messages",
         {"role": "user",
          "content": "Count slowly from 1 to 400, one number per line, no other words."})

    delivered: list[str] = []
    state: dict = {}

    def stream():
        req = urllib.request.Request(
            f"{base}/v1/conversations/{cid}/generate",
            data=json.dumps({"mode": "append", "overrides": {"max_tokens": 4096}}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace")
                    if not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[6:])
                    except Exception:
                        continue
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta" and d.get("text"):
                        delivered.append(d["text"])
        except Exception as e:
            state["err"] = repr(e)

    t = threading.Thread(target=stream, daemon=True)
    t.start()
    for _ in range(400):
        time.sleep(0.05)
        if len(delivered) > 20:
            break
    if len(delivered) <= 20:
        bad(f"never started generating (err={state.get('err')})")
        return failures

    # 1. teardown must be refused while generating
    status, body = call(base, "POST", f"/v1/admin/models/{model_id}/unload", timeout=60)
    if status != 409:
        bad(f"unload during generation returned {status}, expected 409 "
            f"-- the model can be torn down mid-decode ({str(body)[:120]})")
    elif model_id not in str(body):
        bad(f"the 409 does not name the model: {str(body)[:120]}")

    # 2. stop
    status, _ = call(base, "DELETE", f"/v1/conversations/{cid}/generate")
    if status != 200:
        bad(f"stop returned {status}, expected 200")
    t.join(timeout=120)
    wire = "".join(delivered)

    # 3. persisted == delivered, exactly
    time.sleep(1.5)
    _, stored = call(base, "GET", f"/v1/conversations/{cid}")
    rows = [m for m in stored.get("messages", []) if m["role"] == "assistant"]
    saved = rows[-1]["content"] if rows else ""
    if saved != wire:
        if wire.startswith(saved):
            bad(f"stored is SHORT by {len(wire) - len(saved)} chars -- a stopped "
                f"reply is missing its tail: {wire[len(saved):][:80]!r}")
        elif saved.startswith(wire):
            bad(f"stored has {len(saved) - len(wire)} chars MORE than the wire "
                f"delivered: {saved[len(wire):][:80]!r}")
        else:
            bad(f"stored and delivered DIVERGE\n   wire tail: {wire[-80:]!r}\n"
                f" stored tail: {saved[-80:]!r}")
    if stored.get("generating"):
        bad("the conversation still reports generating after a stop")

    call(base, "DELETE", f"/v1/conversations/{cid}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="an ALREADY-RUNNING heylookllm")
    ap.add_argument("--models", required=True,
                    help="comma-separated ids; include an MLX AND a gguf one")
    args = ap.parse_args()

    base = args.server.rstrip("/")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_failures: list[str] = []

    for model_id in models:
        print(f"{DIM}-- {model_id}{RESET}")
        fails = check_model(base, model_id)
        for f in fails:
            print(f"  {RED}x{RESET} {f}")
        if not fails:
            print(f"  {GREEN}ok{RESET} teardown refused, stop clean, stored == delivered")
        all_failures.extend(fails)

    # 4. one subprocess, whatever else happened
    n = llama_server_count()
    if n > 1:
        all_failures.append(f"{n} llama-server processes alive; the invariant is at most one")
    elif n >= 0:
        print(f"{DIM}-- llama-server processes: {n}{RESET}")

    print()
    if all_failures:
        print(f"{RED}FAIL{RESET}  {len(all_failures)} problem(s)")
        return 1
    print(f"{GREEN}PASS{RESET}  generation lifecycle holds for {len(models)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
