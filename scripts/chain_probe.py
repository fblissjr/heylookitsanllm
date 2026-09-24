"""Does a restored MLX prompt cache generate what a fresh one does?

Opt-in instrument, never part of a suite: it loads a real model. Run it
before and after any change to the prompt-cache slot, trim, restore or
position state, on each model class the change touches -- unit tests on fakes
stayed green through every live cache failure in this repo, and the eval bank
never hits a restore.

For each hop, the same request runs twice at temperature 0:
  fresh    -- after the model's prefix cache is cleared, so nothing of this
             chain is in it (the store keeps many conversations, so an
             unrelated request no longer evicts it); a fresh run that
             reports reuse fails the probe
  restored -- right after the previous hop, so its prefix is cached
The texts must match.
Hops: extends (multi-turn), then one edit (diverges mid-history).
Only a CHAIN discriminates: single-hop restores have passed on models whose
chained restores were broken. Greedy is right here and nowhere else: this is
a text EQUALITY check, not a throughput measurement. A restored run that
does not report `reused` proves nothing about the restore: every extend
hop is required to restore, an edit may re-prefill (it can diverge before the
earliest kept checkpoint), and the restored hops are printed.

NEAR-TIES. A restored cache and a fresh prefill are different float paths
(different chunking, a copied state), so where the model's top two tokens are
one bf16 quantum apart either can win, and the texts part with nothing wrong.
In-process (the default) the probe judges each divergence the way vLLM's
model tests do: at the FIRST differing token it runs one fresh forward over
the prompt plus the shared reply prefix and reads the top two log-probs.
It is a NEAR-TIE only if the two picked tokens ARE those two and their gap is
at most one quantum (_inproc.NEAR_TIE_MARGIN); the hop is not compared past
it. Anything else is a MISMATCH. Over HTTP (--server) there are no token ids
or log-probs, so every divergence is reported as a mismatch.

Exits 1 on a mismatch or a required restore that did not happen.

  uv run python scripts/chain_probe.py --model ID                # in-process
  uv run python scripts/chain_probe.py --server URL --model ID   # a running server

Results (texts, cache reports, verdicts, commit, conditions) go to --out as
JSON.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--server", help="probe a running server over HTTP (no tie verdicts)")
ap.add_argument("--model", required=True)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--config", default="{}",
                help="in-process only: JSON merged into the model's config (a variant, or a moved model_path)")
ap.add_argument("--out", default="internal/claude/chain_probe",
                help="directory for the JSON record (gitignored by default)")
a = ap.parse_args()

# Longer than the MLX prefix cache's checkpoint interval
# (vlm_engine.APC_CHECKPOINT_INTERVAL_TOKENS), so a checkpoint model has a
# boundary to restore from; a shorter prompt proves nothing either way.
SYSTEM = ("You are a concise assistant. Answer in one or two sentences. "
          "Prefer plain words, name the thing you mean, and do not hedge. ") * 8


def http_asker():
    def ask(messages):
        body = {"model": a.model, "max_tokens": a.max_tokens, "stream": False,
                "temperature": 0.0, "system": SYSTEM, "messages": messages,
                "thinking": False}
        r = urllib.request.Request(a.server + "/v1/messages", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        b = json.loads(urllib.request.urlopen(r, timeout=900).read())
        text = "".join(x.get("text") or "" for x in b["content"] if x.get("type") == "text")
        return text, (b.get("performance") or {}).get("cache"), None

    def clear():
        r = urllib.request.Request(a.server + "/v1/cache/clear", method="POST",
                                   data=json.dumps({"model": a.model}).encode(),
                                   headers={"Content-Type": "application/json"})
        if json.loads(urllib.request.urlopen(r, timeout=60).read()).get("deleted_count") != 1:
            sys.exit(f"could not clear {a.model}'s prefix cache; a fresh run would not be fresh")
    return ask, None, clear


def inproc_asker():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _inproc import load_provider, near_tie_verdict

    from heylook_llm.config import ChatRequest

    provider, name, on_worker = load_provider(a.model, json.loads(a.config))
    if name != "mlx":
        sys.exit("the chain probe checks the MLX prefix cache; this model is gguf")
    on_worker(provider.load_model)

    def request(messages):
        return ChatRequest.model_validate({
            "model": a.model, "max_tokens": a.max_tokens, "temperature": 0.0,
            "enable_thinking": False,
            "messages": [{"role": "system", "content": SYSTEM}, *messages]})

    def ask(messages):
        def go():
            text, cache, tokens = "", None, []
            for c in provider.create_chat_completion(request(messages)):
                text += c.text or ""
                if c.token is not None:
                    tokens.append(int(c.token))
                if getattr(c, "cache", None) is not None:
                    cache = asdict(c.cache)
            return text, cache, tokens
        return on_worker(go)

    def judge(messages, fresh_tokens, restored_tokens):
        return near_tie_verdict(provider, on_worker, request(messages),
                                fresh_tokens, restored_tokens)

    def clear():
        if not on_worker(provider.clear_cache):
            sys.exit(f"could not clear {a.model}'s prefix cache; a fresh run would not be fresh")

    return ask, judge, clear


ask, judge, evict = http_asker() if a.server else inproc_asker()


def run_hop(label, msgs, prime):
    evict()
    fresh, fcache, ftoks = ask(msgs)
    if prime is not None:
        evict()                         # or the restore is of this same prompt
        ask(prime)                      # prime the cache with the previous hop
    restored, rcache, rtoks = ask(msgs)
    hop = {"hop": label, "match": fresh == restored, "fresh_cache": fcache,
           "restored_cache": rcache, "fresh": fresh, "restored": restored}
    if not hop["match"]:
        hop.update(judge(msgs, ftoks, rtoks) if judge else
                   {"verdict": "MISMATCH", "why": "over HTTP: rerun in-process for a tie verdict"})
    return hop, fresh


questions = ["Name one planet.", "How far is it from the sun, roughly?",
             "What is its largest moon?", "Is that moon larger than ours?"]
hops, convo, prev = [], [], None
for i, q in enumerate(questions):
    msgs = convo + [{"role": "user", "content": q}]
    hop, fresh = run_hop(f"extend-{i}", msgs, prev)
    hops.append(hop)
    prev = msgs
    convo = msgs + [{"role": "assistant", "content": fresh}]

# edit hop: rewrite the second user turn, keep the first exchange
edited = convo[:2] + [{"role": "user", "content": "Name its closest neighbour planet."}]
hop, _ = run_hop("edit", edited, prev)
hops.append(hop)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip())
out = {"model": a.model, "mode": "http" if a.server else "inproc", "commit": commit,
       "src_dirty": dirty, "temperature": 0.0, "max_tokens": a.max_tokens,
       "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "hops": hops}
Path(a.out).mkdir(parents=True, exist_ok=True)
path = Path(a.out) / f"chain_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(out, indent=2))
for h in hops:
    rc = h["restored_cache"] or {}
    tail = "" if h["match"] else f" {h['verdict']}" + (
        f" at token {h['first_divergence']} (margin {h['margin']})" if "margin" in h else "")
    print(f"{h['hop']:9} match={h['match']} restored={rc.get('outcome')} cause={rc.get('cause')} "
          f"cached={rc.get('cached_tokens')}/{rc.get('prompt_tokens')}{tail}")
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
ties = [h["hop"] for h in hops if h.get("verdict") == "NEAR-TIE"]
mismatch = [h["hop"] for h in hops if h.get("verdict") == "MISMATCH"]
not_fresh = [h["hop"] for h in hops if (h["fresh_cache"] or {}).get("outcome") == "reused"]
if not_fresh:
    print("FRESH RUN REUSED A PREFIX:", ", ".join(not_fresh))
if missed:
    print("REQUIRED RESTORE MISSED:", ", ".join(missed))
if ties:
    print("NEAR-TIE (not a failure; not compared past the tie):", ", ".join(ties))
if mismatch:
    print("MISMATCH:", ", ".join(mismatch))
if missed or mismatch or not_fresh:
    raise SystemExit(1)
