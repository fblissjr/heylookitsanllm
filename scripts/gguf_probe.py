# scripts/gguf_probe.py
#
# Direct llama-server diagnostics for ONE gguf model -- the layer BELOW
# scripts/dev_server.sh (no FastAPI, no DB, no router): spawn the provider,
# read /props modalities, diff /apply-template with thinking on/off, run one
# short generation, scrape draft acceptance from the subprocess log, tear
# down. Wrapped by the gguf-probe skill; run UNSANDBOXED (Metal + localhost).
#
# Usage:
#   uv run python scripts/gguf_probe.py <model-dir-or-gguf-file>
#       [--spec-type draft-mtp] [--spec-draft-n-max N] [--draft PATH]
#       [--ctx N] [--prompt TEXT] [--max-tokens N] [--no-gen]
#       [--seed N] [--temp F]
#       [--lora PATH ...] [--lora-scale F] [--lora-ab]
#
# Comparing runs (what this script is mostly for): --seed is PINNED by
# default (1234) and --draft overrides sidecar pairing, so two builds of one
# drafter can be A/B'd. Seeding is necessary but NOT sufficient for identical
# output: speculative decoding changes the verify batch composition per eval,
# which perturbs floating-point reductions and can still diverge the text.
# For a tight drafter A/B add `--temp 0`; otherwise treat single runs as
# samples and repeat across seeds (`--seed -1` = random, for variance checks).
#
# LoRA: `--lora PATH` rides `extra_args`, the SAME raw-passthrough field a
# models.toml gguf entry uses -- so a flag proven here transfers to a server
# config verbatim. `--lora-ab` then answers the three questions an adapter
# actually raises, in ONE process and one model load, by toggling scale over
# llama-server's POST /lora-adapters between two otherwise identical runs:
# is it applied at all (identical output = it is not; the tok/s drop is the
# other tell), what does it cost, and does it wreck speculative decoding
# (draft acceptance off vs on -- an embedded MTP head is NOT adapted, so a
# collapse here is the expected failure and the reason to measure). Pair it
# with `--temp 0`, same as the drafter A/B. Note the adapter switch CLEARS
# the prompt cache server-side, which the printed `cached=` makes visible.
#
# Given a DIRECTORY, sidecar pairing (mmproj / mtp- drafter) reuses the
# importer's own pickers -- the single source of pairing truth. Drift note:
# everything here rides llama-server's stable surfaces (/props,
# /apply-template, OpenAI-compat chat) via LlamaServerProvider, which is the
# one place that tracks llama.cpp; the only fragile bit is the
# "draft acceptance" log-line grep at the bottom.

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from heylook_llm.config import ChatRequest
from heylook_llm.model_importer import ModelImporter
from heylook_llm.providers.llama_server_provider import LlamaServerProvider


def build_config(target: Path, args) -> dict:
    cfg: dict  # values are heterogeneous (str/int/list), so don't infer per-branch
    if target.is_dir():
        imp = ModelImporter()
        entry = imp._create_gguf_entry(target)
        if entry is None:
            sys.exit(f"no primary .gguf found in {target}")
        cfg = {k: v for k, v in entry["config"].items() if v is not None}
    else:
        cfg = {"model_path": str(target)}
    if args.draft:
        # Explicit override of the paired drafter. Needed to A/B two builds of
        # the SAME speculative module (e.g. DeepSeek-V4's dspark Q8_0 vs BF16,
        # which HF ships in a `dspark/` subdir the root-level picker cannot
        # see) -- drafter fidelity drives acceptance rate, so comparing them
        # is a real measurement, not a config convenience.
        cfg["draft_model_path"] = str(args.draft)
    if args.spec_type:
        cfg["spec_type"] = args.spec_type
    if args.spec_draft_n_max:
        cfg["spec_draft_n_max"] = args.spec_draft_n_max
    if args.ctx:
        cfg["ctx_size"] = args.ctx
    if args.lora:
        # extra_args is raw passthrough -- the same field a models.toml gguf
        # entry carries -- so these flags transfer to a server config as-is.
        # llama.cpp defaults an adapter to scale 1.0; --lora-scaled states it.
        # APPEND: the importer may already have put flags in there.
        flags: list[str] = []
        for path in args.lora:
            if args.lora_scale is None:
                flags += ["--lora", str(path)]
            else:
                flags += ["--lora-scaled", f"{path}:{args.lora_scale}"]
        cfg["extra_args"] = list(cfg.get("extra_args") or []) + flags
    cfg.pop("modalities", None)  # description only; /props is the live truth
    return cfg


def get_json(url: str, body: dict | list | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def set_lora_scales(base: str, n_adapters: int, scale: float) -> None:
    """Set EVERY loaded adapter's scale on a running server.

    Always sends the full list: llama.cpp's construct_lora_list zeroes any
    adapter the payload omits, so a partial post silently disables the rest.
    """
    get_json(base + "/lora-adapters",
             [{"id": i, "scale": scale} for i in range(n_adapters)])


def run_gen(p, args, label: str, extra: dict) -> tuple:
    """One generation through the PROVIDER (not raw HTTP), so the reported
    telemetry is the same GenerationChunk surface the server serves from."""
    t = time.time()
    chunks = list(p.create_chat_completion(ChatRequest.model_validate({
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens, "seed": args.seed,
        **({} if args.temp is None else {"temperature": args.temp}),
        **extra})))
    final = chunks[-1]
    thinking = sum(len(c.thinking or "") for c in chunks)
    print(f"[probe] {label}: {time.time()-t:.1f}s | gen={final.generation_tokens} tok "
          f"@ {final.generation_tps:.1f} tok/s | thinking={thinking} ch | "
          f"cached={final.cached_tokens} | draft={final.draft_accepted}/{final.draft_tokens}")
    return final, "".join(c.text or "" for c in chunks)


def apply_template(base: str, kwargs: dict | None) -> str:
    body: dict = {"messages": [{"role": "user", "content": "hi"}]}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    return get_json(base + "/apply-template", body)["prompt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--draft", type=Path,
                    help="drafter .gguf, overriding whatever sidecar pairing found")
    ap.add_argument("--spec-type")
    ap.add_argument("--spec-draft-n-max", type=int)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--prompt", default="Write a 150-word story about a lighthouse keeper.")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--no-gen", action="store_true")
    # Pinned by DEFAULT, because this script's whole job is comparing runs.
    # Unseeded, each run generates different text and draft acceptance tracks
    # content: two nominally-identical DSpark runs came out 11.7 acceptance
    # points apart, wider than the Q8_0-vs-BF16 effect being measured. Noise
    # that large silently turns a null result into an apparent one.
    # `--seed -1` restores llama-server's random behaviour when you actually
    # want sampling variety rather than a comparison.
    ap.add_argument("--seed", type=int, default=1234,
                    help="sampling seed; pinned by default so runs are comparable "
                         "(-1 = random, for variance checks)")
    # Seeding alone does NOT make two configs produce identical text: spec
    # decode changes the verify batch composition per eval, FP reductions
    # shift, and the streams diverge. `--temp 0` removes sampling as a
    # variable entirely, which is the right control when the thing being
    # compared is a drafter rather than the model's prose.
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature; 0 for a deterministic A/B "
                         "(default: leave to the sampler cascade)")
    ap.add_argument("--lora", type=Path, action="append", metavar="PATH",
                    help="LoRA adapter .gguf, repeatable; must be converted for "
                         "THIS base (llama.cpp rejects an arch mismatch)")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="scale for every --lora given (default: llama.cpp's 1.0)")
    ap.add_argument("--lora-ab", action="store_true",
                    help="generate twice in one process, adapter off then on, and "
                         "report the tok/s, draft-acceptance and output deltas")
    args = ap.parse_args()
    if args.lora_ab and not args.lora:
        ap.error("--lora-ab needs at least one --lora to toggle")

    cfg = build_config(args.target, args)
    print(f"[probe] config: { {k: (str(v)[-60:] if isinstance(v, str) else v) for k, v in cfg.items()} }")

    p = LlamaServerProvider("gguf-probe", cfg, False)
    # Which BINARY produced a number is part of the number: this machine can
    # carry more than one llama-server build (a vendored one and whatever
    # $HEYLOOK_LLAMA_SERVER points at), and the probe silently inherits the
    # env one unless `server_binary` says otherwise. Print it so an A/B is
    # attributable to a build rather than assumed to share one.
    print(f"[probe] binary: {p._resolve_binary()}")
    t0 = time.time()
    p.load_model()
    base = p._base_url
    assert base is not None  # set by load_model
    print(f"[probe] loaded in {time.time()-t0:.1f}s at {base}")
    # Every run reuses this filename and the provider APPENDS, so remember where
    # THIS run's output starts -- otherwise the acceptance grep at the bottom
    # reports the previous probe's model as if it were this one's.
    log = Path("logs") / "llama_server_gguf-probe.log"
    log_start = log.stat().st_size if log.exists() else 0
    try:
        props = get_json(base + "/props")
        print(f"[probe] /props modalities: {props.get('modalities')}")

        on = apply_template(base, {"enable_thinking": True})
        off = apply_template(base, {"enable_thinking": False})
        unset = apply_template(base, None)
        print(f"[probe] template: on==off {on == off} | unset==off {unset == off} "
              f"(unset!=off means thinking-off must be sent EXPLICITLY)")
        if on != off:
            # Fall back to the end of the common prefix: when one template is a
            # strict prefix of the other (Qwen3.6 appends rather than edits),
            # zip() yields no differing pair and a bare next() raises.
            i = next((i for i, (a, b) in enumerate(zip(on, off)) if a != b),
                     min(len(on), len(off)))
            print(f"[probe] first template diff @{i}: ON={on[max(0,i-15):i+35]!r}")

        if args.lora_ab:
            loaded = get_json(base + "/lora-adapters")
            n = len(loaded)
            print(f"[probe] /lora-adapters: {n} loaded")
            if n == 0:
                sys.exit("server loaded 0 adapters -- --lora path wrong, or the "
                         "adapter's arch does not match this base")
            runs = {}
            for label, scale in (("lora-off", 0.0), ("lora-on", 1.0)):
                set_lora_scales(base, n, scale)
                runs[label] = run_gen(p, args, label, {})
            off_f, off_text = runs["lora-off"]
            on_f, on_text = runs["lora-on"]

            def acc(f) -> float | None:
                return f.draft_accepted / f.draft_tokens * 100 if f.draft_tokens else None

            def fmt(v) -> str:
                return "n/a" if v is None else f"{v:.1f}%"

            delta = ("n/a" if not off_f.generation_tps else
                     f"{(on_f.generation_tps - off_f.generation_tps) / off_f.generation_tps * 100:+.1f}%")
            print(f"[probe] A/B tok/s: off={off_f.generation_tps:.1f} "
                  f"on={on_f.generation_tps:.1f} ({delta})")
            a_off, a_on = acc(off_f), acc(on_f)
            if a_off is not None or a_on is not None:
                print(f"[probe] A/B draft acceptance: off={fmt(a_off)} on={fmt(a_on)}"
                      "  (a collapse means the draft path is UNADAPTED)")
            # Identical text is proof the adapter did nothing; differing text is
            # only consistent with it working, so run --temp 0 before believing
            # either direction.
            if off_text == on_text:
                print("[probe] A/B output: IDENTICAL -- adapter had no effect on generation")
            else:
                print("[probe] A/B output: differs (expected when the adapter applies)")
        elif not args.no_gen:
            for label, extra in (("gen", {}), ("gen-think-off", {"enable_thinking": False})):
                run_gen(p, args, label, extra)

        if log.exists():
            with log.open("rb") as fh:
                fh.seek(log_start)
                tail = fh.read().decode(errors="replace")
            hits = re.findall(r"draft acceptance = [\d.]+ \([^)]*\), mean len = *[\d.]+", tail)
            for h in hits[-2:]:
                print(f"[probe] log: {h}")
    finally:
        p.unload()
        print("[probe] unloaded")


if __name__ == "__main__":
    main()
