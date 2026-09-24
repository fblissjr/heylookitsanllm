"""Does heylook's MLX vision path generate what mlx-vlm's own loop generates?

Opt-in instrument, never part of a suite: it loads a real model. One model per
process, no server. Run it BEFORE and AFTER any change to the vision prefill,
the pre-filled cache, or VLM position state -- unit tests on fakes cannot see
that class of bug, and the eval bank is blind to it.

    uv run python scripts/vlm_parity_probe.py --model Qwen3.5-0.8B-MLX-8bit
    uv run python scripts/vlm_parity_probe.py --model <id> --tokens 64 --out result.json

HOW IT COMPARES. heylook's path is driven through MLXProvider exactly as a
request would drive it, while a recorder around the provider module's
`vlm_prepare_inputs` captures the tensors it built. Those SAME tensors are then
replayed through `mlx_vlm.generate.ar.generate_step`. Identical inputs on both
sides means a difference can only come from how the prompt was prefilled and
the first tokens were produced -- never from the chat template, the vision
token budget, or the stop set.

Greedy on both sides. That is correct HERE and nowhere else: this is a token
EQUALITY check, and temp 0 is the only regime where two implementations can be
expected to agree token for token. It says nothing about throughput.

WHAT CAN FOOL IT, and what the report carries so you can tell:
- A short comparison is decided by the template (every reply starts the same
  way). --tokens defaults high enough to get past that, and the NEGATIVE
  CONTROL (a second, different image) must change the output or the run is
  vacuous and says so.
- Chunked-vs-unchunked prefill, MoE kernels and 2-D vs 3-D rope are different
  float paths. A divergence where upstream's top-1/top-2 logprob margin is one
  bf16 quantum (NEAR_TIE_MARGIN) is a tie broken by rounding, reported as
  NEAR-TIE rather than as a failure. Once the two sides prefill the same way
  there is no such excuse left, so expect exact MATCH and read a NEAR-TIE as a
  sign the paths have drifted apart again.
- heylook stops at its resolved stop set and generate_step does not stop at
  all. The shared prefix is compared, and when heylook ended early, upstream's
  NEXT token must be a stop token (heylook's engine stops without yielding
  it) -- ending early anywhere else means a token was lost, reported as SHORT.
- Position state on the language model survives between requests. The probe
  runs vision after TEXT and vision after VISION in the same process; both must
  agree with upstream.

Everything runs on ONE worker thread inside the provider's generation stream:
MLX streams are thread-local and a forward on the main thread is not the
condition the server runs under.
"""
import argparse
import base64
import io
import json
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# One quantum. Logprobs come out of bf16 logits whose magnitude puts the
# representable step at 0.125, so a top-1/top-2 margin of 0.125 is the SMALLEST
# nonzero gap the dtype can express -- a tie broken by rounding, not a
# preference. Measured on the 27B and the PE model against the pre-v2.0.54
# path, whose all-N prefill was a different float path from upstream's
# N-1-then-step: every divergence sat at exactly this value.
NEAR_TIE_MARGIN = 0.125


def _image(variant: int) -> str:
    """A small synthetic picture as a data URI. Two variants that differ enough
    that any model reading the pixels must describe them differently."""
    from PIL import Image, ImageDraw

    if variant == 0:
        im = Image.new("RGB", (448, 448), (30, 120, 200))
        d = ImageDraw.Draw(im)
        d.ellipse((120, 120, 330, 330), fill=(240, 60, 40))
        d.rectangle((20, 380, 430, 430), fill=(40, 180, 70))
    else:
        im = Image.new("RGB", (448, 448), (250, 220, 40))
        d = ImageDraw.Draw(im)
        d.polygon([(224, 60), (400, 380), (48, 380)], fill=(20, 20, 20))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_config(model_id: str) -> dict:
    """The provider config the ROUTER would hand MLXProvider: the merged entry
    validated through ModelConfig and dumped. A raw dict skips the validator
    that derives `modalities`, and a vision model then routes to mlx-lm."""
    from heylook_llm.config import ModelConfig
    from heylook_llm.model_registry import discover, merge_discovered

    data = tomllib.loads(Path("models.toml").read_text())
    for m in merge_discovered(data, discover(data))["models"]:
        if m["id"] == model_id:
            if m.get("provider") != "mlx":
                sys.exit(f"{model_id} is provider={m.get('provider')}; this probe is MLX-only")
            return ModelConfig.model_validate(m).config.model_dump()
    sys.exit(f"{model_id}: not served (checked models.toml + discovery)")


def _request(model_id: str, content, max_tokens: int):
    from heylook_llm.config import ChatRequest

    return ChatRequest.model_validate({
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        # Every knob that could make the two sides sample differently, off.
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "enable_thinking": False, "seed": 0,
    })


def _vision_content(variant: int):
    return [{"type": "image_url", "image_url": {"url": _image(variant)}},
            {"type": "text", "text": "Describe this picture in detail: every shape, its colour, its position, and the background. Write at least six sentences."}]


def _run(args) -> dict:
    import mlx.core as mx
    from mlx_vlm.generate.ar import generate_step

    from heylook_llm.providers import mlx_provider as mp
    from mlx_vlm.generate.common import generation_stream

    cfg = _resolve_config(args.model)
    if args.step:
        cfg["prefill_step_size"] = args.step
    provider = mp.MLXProvider(args.model, cfg, False)
    provider.load_model()
    assert provider.model is not None
    if not provider.is_vlm:
        sys.exit(f"{args.model} is served as text; nothing to compare")

    captured: list[dict] = []
    real_prepare = mp.vlm_prepare_inputs

    def recording_prepare(*a, **kw):
        out = real_prepare(*a, **kw)
        captured.append(dict(out))
        return out

    mp.vlm_prepare_inputs = recording_prepare
    tokenizer = provider.get_tokenizer()
    stop_ids = set(provider._stop_tokens)

    def heylook(content) -> list[int]:
        captured.clear()
        return [c.token for c in provider.create_chat_completion(
            _request(args.model, content, args.tokens)) if c.token is not None]

    def upstream() -> tuple[list[int], list]:
        inputs = captured[-1]
        extras = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "pixel_values", "attention_mask")}
        kw = {} if not args.step else {"prefill_step_size": args.step}
        toks, lps = [], []
        for tok, lp in generate_step(
                inputs["input_ids"], provider.model, inputs.get("pixel_values"),
                inputs.get("attention_mask"), max_tokens=args.tokens,
                temperature=0.0, **kw, **extras):
            toks.append(int(tok))
            lps.append(lp)
        return toks, lps

    def compare(label: str, ours: list[int], theirs: list[int], lps) -> dict:
        n = min(len(ours), len(theirs))
        first = next((i for i in range(n) if ours[i] != theirs[i]), None)
        row = {"case": label, "heylook_tokens": len(ours), "upstream_tokens": len(theirs),
               "compared": n, "first_divergence": first,
               "heylook_text": tokenizer.decode(ours), "upstream_text": tokenizer.decode(theirs[:n])}
        if first is not None:
            top2 = mx.sort(lps[first].reshape(-1))[-2:]
            margin = float((top2[1] - top2[0]).tolist())
            row["upstream_top1_top2_margin"] = margin
            row["verdict"] = "NEAR-TIE" if margin <= NEAR_TIE_MARGIN else "DIVERGED"
        elif len(ours) < len(theirs):
            # heylook ended first: legitimate only if upstream's NEXT token is
            # a stop token (heylook's engine stops without yielding it).
            row["heylook_stopped_at_upstream_stop"] = theirs[len(ours)] in stop_ids
            row["verdict"] = "MATCH" if theirs[len(ours)] in stop_ids else "SHORT"
        else:
            row["verdict"] = "MATCH" if n else "EMPTY"
        return row

    rows = []
    with mx.stream(generation_stream):
        # Text first, so the first vision run inherits whatever position state
        # a text generation leaves on the language model.
        heylook("Say hello in five words.")
        ours_a = heylook(_vision_content(0))
        theirs_a, lps_a = upstream()
        rows.append(compare("vision after text", ours_a, theirs_a, lps_a))

        ours_b = heylook(_vision_content(1))
        theirs_b, lps_b = upstream()
        rows.append(compare("vision after vision (different image)", ours_b, theirs_b, lps_b))

    control_ok = ours_a != ours_b
    return {"model": args.model, "tokens": args.tokens, "prefill_step_size": args.step,
            "negative_control_images_differ": control_ok, "cases": rows,
            "ok": control_ok and all(r["verdict"] in ("MATCH", "NEAR-TIE") for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True, help="exact served MLX model id")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--step", type=int, default=0,
                    help="prefill_step_size for BOTH sides (0 = each side's default)")
    ap.add_argument("--out", type=Path, help="also write the report as JSON here")
    args = ap.parse_args()

    with ThreadPoolExecutor(max_workers=1) as pool:
        report = pool.submit(_run, args).result()

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    if not report["negative_control_images_differ"]:
        print("VACUOUS: two different images produced the same tokens", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
