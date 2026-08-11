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
# Do not draw PERFORMANCE conclusions at `--temp 0`: greedy is a DIFFERENT
# REGIME, not a quieter one. Speculative acceptance under greedy is exact argmax
# matching; at temp>0 it is the rejection-sampling criterion, so n_max/p_min
# conclusions can differ in KIND and name the wrong setting. It stays available
# (reproducing an exact output is a real need) and the probe warns rather than
# refusing -- but it can never be a perf number or a shipped default.
# Use the model's VENDOR-RECOMMENDED sampling (unsloth.ai/docs/models/<model>:
# gemma-4 = temp 1.0/top_p 0.95/top_k 64; Qwen3.6 thinking = temp 1.0/top_p 0.95/
# top_k 20/min_p 0). Comparability comes from the PINNED SEED plus repeats, not
# from flattening the sampler (`--seed -1` = random, for variance checks).
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
# at vendor-recommended sampling, same as the drafter A/B. Note the switch CLEARS
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
import shlex
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
    if args.spec_draft_p_min is not None:
        # `is not None`, not truthiness: 0.0 is a real setting AND llama.cpp's
        # default, so truthiness would silently drop an explicit 0.0 and make
        # "set to default" indistinguishable from "unset".
        cfg["spec_draft_p_min"] = args.spec_draft_p_min
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
    if args.arg:
        # shlex per token: argparse refuses a value starting with "-", so a
        # flag+value must arrive as ONE quoted string ("--spec-draft-p-min 0.3").
        cfg["extra_args"] = list(cfg.get("extra_args") or []) + [
            part for chunk in args.arg for part in shlex.split(chunk)]
    cfg.pop("modalities", None)  # description only; /props is the live truth
    return cfg


def reject_banned_flags(cfg: dict, fail) -> None:
    """Enforce the temp-0 ban on RAW PASSTHROUGH too, not just --temp.

    A control with a bypass is not a control. `--arg '--temp 0'` put
    `--temp 0` in extra_args and reached llama-server as a server-side
    sampling default, defeating the --temp guard completely -- and the bypass
    shipped in the same file, an hour earlier. Found by attacking the control
    rather than reading it, which is the only way this class shows up.
    """
    extra = list(cfg.get("extra_args") or [])
    for i, tok in enumerate(extra):
        val = None
        if tok in ("--temp", "--temperature") and i + 1 < len(extra):
            val = extra[i + 1]
        elif tok.startswith(("--temp=", "--temperature=")):
            val = tok.split("=", 1)[1]
        if val is None:
            continue
        try:
            if float(val) == 0:
                fail(f"passthrough sets {tok} to {val}: temp 0 is banned as a "
                     "measurement setting here, and routing it through --arg or "
                     "models.toml extra_args does not make it valid. See the "
                     "module header.")
        except ValueError:
            pass


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
        **({} if args.top_p is None else {"top_p": args.top_p}),
        **({} if args.top_k is None else {"top_k": args.top_k}),
        **({} if args.min_p is None else {"min_p": args.min_p}),
        **extra})))
    final = chunks[-1]
    thinking = sum(len(c.thinking or "") for c in chunks)
    # prompt= is load-bearing, not decoration: without it a SILENTLY TRUNCATED
    # prompt (ctx smaller than the input) looks like a legitimate fast run, and
    # two arms that processed different amounts of context compare as if equal.
    print(f"[probe] {label}: {time.time()-t:.1f}s | prompt={final.prompt_tokens} "
          f"| gen={final.generation_tokens} tok "
          f"@ {final.generation_tps:.1f} tok/s | thinking={thinking} ch | "
          f"cached={final.cached_tokens} | draft={final.draft_accepted}/{final.draft_tokens}")
    # THINKING COUNTS AS OUTPUT for the A/B comparison. Comparing `text` alone
    # silently reports "no effect" for any model that spends the whole budget
    # reasoning: both arms come back empty and two empty strings are equal.
    # That is a false negative on the one check the A/B exists to make.
    return final, "".join((c.thinking or "") + (c.text or "") for c in chunks)


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
    # Drives the CONFIG FIELD, not --arg passthrough. Once a flag earns a
    # GGUFModelConfig field the probe must exercise that path, or it keeps
    # certifying a route production does not take. --arg stays for the flags
    # that are still unmodelled.
    ap.add_argument("--spec-draft-p-min", type=float)
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
    # shift, and the streams diverge. The control for that is REPEATS at a
    # pinned seed -- NOT flattening the sampler, which would move the run into
    # a decoding regime nobody serves and can invert which setting looks best.
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature; use the model's vendor-recommended "
                         "value for anything you will act on. 0 warns: valid for "
                         "reproducibility, never for perf (default: leave to the "
                         "sampler cascade)")
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--min-p", type=float)
    ap.add_argument("--lora", type=Path, action="append", metavar="PATH",
                    help="LoRA adapter .gguf, repeatable; must be converted for "
                         "THIS base (llama.cpp rejects an arch mismatch)")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="scale for every --lora given (default: llama.cpp's 1.0)")
    ap.add_argument("--lora-ab", action="store_true",
                    help="generate twice in one process, adapter off then on, and "
                         "report the tok/s, draft-acceptance and output deltas")
    # Reaching flags heylook does NOT model. Several spec-decode knobs
    # (--spec-draft-p-min, --spec-draft-n-min, --spec-draft-p-split) have no
    # GGUFModelConfig field, so without this the probe cannot measure the very
    # levers that would justify adding one.
    ap.add_argument("--arg", action="append", metavar="FLAG", default=[],
                    help="raw llama-server flag+value as ONE quoted string, "
                         "repeatable, e.g. --arg '--spec-draft-p-min 0.5'")
    args = ap.parse_args()
    if args.temp is not None and args.temp == 0:
        # WARN, do not refuse. temp 0 is legitimate for reproducing something
        # exactly -- it is only invalid as a PERFORMANCE number or a default,
        # because greedy is a different decoding regime (speculative acceptance
        # is exact argmax matching rather than the rejection-sampling criterion)
        # and no real workload runs there.
        print("[probe] WARNING: --temp 0 is greedy. Fine for reproducing a "
              "specific output; NOT valid as a perf measurement or a basis for "
              "a default -- acceptance dynamics differ in kind, so a tuning "
              "conclusion drawn here can name the wrong setting. Use the "
              "model's vendor-recommended sampling for anything you will act on.",
              file=sys.stderr)
    if args.lora_ab and not args.lora:
        ap.error("--lora-ab needs at least one --lora to toggle")

    cfg = build_config(args.target, args)
    reject_banned_flags(cfg, ap.error)
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
            # only consistent with it working, so repeat at a pinned seed before
            # believing either direction.
            if not off_text and not on_text:
                # Distinct from "no effect": there is nothing to compare, so the
                # check did not run. Reporting no-effect here would be a pass
                # earned by vacuity.
                print("[probe] A/B output: NO OUTPUT in either arm -- comparison "
                      "did not run (raise --max-tokens, or the model emitted nothing)")
            elif off_text == on_text:
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
