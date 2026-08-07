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
#
# Comparing runs (what this script is mostly for): --seed is PINNED by
# default (1234) and --draft overrides sidecar pairing, so two builds of one
# drafter can be A/B'd. Seeding is necessary but NOT sufficient for identical
# output: speculative decoding changes the verify batch composition per eval,
# which perturbs floating-point reductions and can still diverge the text.
# For a tight drafter A/B add `--temp 0`; otherwise treat single runs as
# samples and repeat across seeds (`--seed -1` = random, for variance checks).
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

from heylook_llm.config import ChatRequest
from heylook_llm.model_importer import ModelImporter
from heylook_llm.providers.llama_server_provider import LlamaServerProvider


def build_config(target: Path, args) -> dict:
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
    cfg.pop("modalities", None)  # description only; /props is the live truth
    return cfg


def get_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


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
    args = ap.parse_args()

    cfg = build_config(args.target, args)
    print(f"[probe] config: { {k: (str(v)[-60:] if isinstance(v, str) else v) for k, v in cfg.items()} }")

    p = LlamaServerProvider("gguf-probe", cfg, False)
    t0 = time.time()
    p.load_model()
    base = p._base_url
    assert base is not None  # set by load_model
    print(f"[probe] loaded in {time.time()-t0:.1f}s at {base}")
    try:
        props = get_json(base + "/props")
        print(f"[probe] /props modalities: {props.get('modalities')}")

        on = apply_template(base, {"enable_thinking": True})
        off = apply_template(base, {"enable_thinking": False})
        unset = apply_template(base, None)
        print(f"[probe] template: on==off {on == off} | unset==off {unset == off} "
              f"(unset!=off means thinking-off must be sent EXPLICITLY)")
        if on != off:
            i = next(i for i, (a, b) in enumerate(zip(on, off)) if a != b)
            print(f"[probe] first template diff @{i}: ON={on[max(0,i-15):i+35]!r}")

        if not args.no_gen:
            for label, extra in (("gen", {}), ("gen-think-off", {"enable_thinking": False})):
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

        log = Path("logs") / f"llama_server_gguf-probe.log"
        if log.exists():
            hits = re.findall(r"draft acceptance = [\d.]+ \([^)]*\), mean len = *[\d.]+",
                              log.read_text(errors="replace"))
            for h in hits[-2:]:
                print(f"[probe] log: {h}")
    finally:
        p.unload()
        print("[probe] unloaded")


if __name__ == "__main__":
    main()
