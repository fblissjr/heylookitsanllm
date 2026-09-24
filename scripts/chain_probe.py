"""Does a restored MLX prompt cache generate what a fresh one does?

Opt-in instrument, never part of a suite: it talks to a RUNNING server
(scripts/dev_server.sh) and loads a real model. Run it before and after any
change to the prompt-cache slot, trim, restore or position state, on each
model class the change touches -- unit tests on fakes stayed green through
every live cache failure in this repo, and the eval bank never hits a restore.

For each hop, the same request runs twice at temperature 0:
  fresh    -- after an unrelated request, so nothing of this chain is fresh
             in the model's cache to restore from the last hop
  restored -- right after the previous hop, so its prefix is cached
The texts must match.
Hops: extends (multi-turn), then one edit (diverges mid-history).
Only a CHAIN discriminates: single-hop restores have passed on models whose
chained restores were broken. Greedy is right here and nowhere else: this is
a text EQUALITY check, not a throughput measurement. A restored run that
does not report `reused` proves nothing about the restore: every extend
hop is required to restore, an edit may re-prefill (it can diverge before the
earliest kept checkpoint), and the restored hops are printed. Exits 1 on a
mismatch or a required restore that did not happen.

  uv run python scripts/chain_probe.py --server http://127.0.0.1:8991 --model ID

Results (texts, cache reports, commit, conditions) go to --out as JSON.
"""
import argparse, json, subprocess, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--out", default="internal/claude/chain_probe",
                help="directory for the JSON record (gitignored by default)")
a = ap.parse_args()

# Longer than the MLX prefix cache's checkpoint interval
# (vlm_engine.APC_CHECKPOINT_INTERVAL_TOKENS), so a checkpoint model has a
# boundary to restore from; a shorter prompt proves nothing either way.
SYSTEM = ("You are a concise assistant. Answer in one or two sentences. "
          "Prefer plain words, name the thing you mean, and do not hedge. ") * 8

def ask(messages):
    body = {"model": a.model, "max_tokens": a.max_tokens, "stream": False,
            "temperature": 0.0, "system": SYSTEM, "messages": messages,
            "thinking": False}
    r = urllib.request.Request(a.server + "/v1/messages", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    b = json.loads(urllib.request.urlopen(r, timeout=900).read())
    text = "".join(x.get("text") or "" for x in b["content"] if x.get("type") == "text")
    return text, (b.get("performance") or {}).get("cache")

def evict():
    ask([{"role": "user", "content": "Unrelated: name a prime number."}])

questions = ["Name one planet.", "How far is it from the sun, roughly?",
             "What is its largest moon?", "Is that moon larger than ours?"]
hops, convo, prev = [], [], None
for i, q in enumerate(questions):
    msgs = convo + [{"role": "user", "content": q}]
    evict()
    fresh, fcache = ask(msgs)
    if prev is not None:
        ask(prev)                       # prime the slot with the previous hop
    restored, rcache = ask(msgs)
    hops.append({"hop": f"extend-{i}", "match": fresh == restored,
                 "fresh_cache": fcache, "restored_cache": rcache,
                 "fresh": fresh, "restored": restored})
    prev = msgs
    convo = msgs + [{"role": "assistant", "content": fresh}]

# edit hop: rewrite the second user turn, keep the first exchange
edited = convo[:2] + [{"role": "user", "content": "Name its closest neighbour planet."}]
evict()
fresh, fcache = ask(edited)
ask(prev)
restored, rcache = ask(edited)
hops.append({"hop": "edit", "match": fresh == restored, "fresh_cache": fcache,
             "restored_cache": rcache, "fresh": fresh, "restored": restored})

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
out = {"model": a.model, "commit": commit, "src_dirty": dirty, "temperature": 0.0,
       "max_tokens": a.max_tokens, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "hops": hops}
Path(a.out).mkdir(parents=True, exist_ok=True)
path = Path(a.out) / f"chain_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(out, indent=2))
for h in hops:
    rc = h["restored_cache"] or {}
    print(f"{h['hop']:9} match={h['match']} restored={rc.get('outcome')} cause={rc.get('cause')} "
          f"cached={rc.get('cached_tokens')}/{rc.get('prompt_tokens')}")
print("wrote", path)
# Per-hop expectation. Every extend hop (the multi-turn follow-up, and the
# exact repeat that opens the chain) MUST restore: if reuse breaks entirely,
# every hop re-prefills, restored == fresh trivially, and only this catches
# it. An edit MAY re-prefill: a checkpoint model keeps a bounded set of
# checkpoints near the end of the previous prompt, so an edit that diverges
# before the earliest one legitimately starts over.
restored = [h["hop"] for h in hops if (h["restored_cache"] or {}).get("outcome") == "reused"]
print("restored:", ", ".join(restored) or "none")
missed = [h["hop"] for h in hops
          if h["hop"].startswith("extend") and h["hop"] not in restored]
mismatch = [h["hop"] for h in hops if not h["match"]]
if missed:
    print("REQUIRED RESTORE MISSED:", ", ".join(missed))
if mismatch:
    print("MISMATCH:", ", ".join(mismatch))
if missed or mismatch:
    raise SystemExit(1)
