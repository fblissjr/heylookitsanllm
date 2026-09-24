"""Shared by the in-process instruments (perf_ab, chain_probe,
vlm_parity_probe): resolve a served model's config the way the router does,
and run MLX work where the server runs it.

One copy on purpose: three scripts each carried their own resolver, and a
resolver that skips validation routes a vision model to text (validation is
what derives `modalities`).
"""
from __future__ import annotations

import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# One bf16 quantum at the logit magnitudes these models produce: the
# smallest nonzero top-1/top-2 gap the dtype can express. A divergence at a
# gap this small is a tie broken by float rounding (a different prefill
# chunking, a restored cache), not a preference. Measured on the 27B and the
# PE model against the pre-v2.0.54 vision path, whose every divergence sat at
# exactly this value.
NEAR_TIE_MARGIN = 0.125


def resolve_config(model_id: str, overrides: dict | None = None) -> tuple[str, dict]:
    """(provider, config) for a served model: heylook.toml merged with
    discovery and VALIDATED through ModelConfig, as the router builds it."""
    from heylook_llm.config import ModelConfig
    from heylook_llm.model_registry import discover, merge_discovered

    from heylook_llm.router import CONFIG_FILENAME

    data = tomllib.loads((REPO / CONFIG_FILENAME).read_text())
    for m in merge_discovered(data, discover(data))["models"]:
        if m["id"] == model_id:
            mc = ModelConfig.model_validate(m)
            cfg = mc.config.model_dump()
            cfg.update(overrides or {})
            return mc.provider, cfg
    sys.exit(f"{model_id}: not served (checked heylook.toml + discovery)")


def load_provider(model_id: str, overrides: dict | None = None):
    """(provider, provider_name, on_worker). ``on_worker(fn)`` runs ``fn``
    where the server would: MLX on one pinned thread inside mlx-vlm's
    generation stream (streams are thread-local); gguf inline."""
    provider_name, cfg = resolve_config(model_id, overrides)
    if provider_name == "mlx":
        import mlx.core as mx
        from mlx_vlm.generate.common import generation_stream

        from heylook_llm.providers.mlx_provider import MLXProvider as Provider
        pool = ThreadPoolExecutor(1)

        def on_worker(fn):
            def run():
                with mx.stream(generation_stream):
                    return fn()
            return pool.submit(run).result()
    else:
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider as Provider

        def on_worker(fn):
            return fn()

    provider = Provider(model_id, cfg, False)
    return provider, provider_name, on_worker


def near_tie_verdict(provider, on_worker, request, fresh_tokens: list, restored_tokens: list) -> dict:
    """Judge a greedy divergence between two runs of ``request`` on an MLX
    provider, the way vLLM's model tests do: at the FIRST differing token,
    one fresh forward over the rendered prompt plus the shared reply prefix
    gives the top two log-probs there. NEAR-TIE only if the two picked
    tokens ARE those two and their gap is at most NEAR_TIE_MARGIN; the runs
    are not compared past it. Anything else, a failed forward included, is a
    MISMATCH."""
    import mlx.core as mx

    from heylook_llm.providers import mlx_provider as mp

    k = next((i for i in range(min(len(fresh_tokens), len(restored_tokens)))
              if fresh_tokens[i] != restored_tokens[i]), None)
    if k is None:
        return {"verdict": "MISMATCH", "why": "same tokens, different text (detokenizer)"}

    def forward():
        eff = provider._apply_model_defaults(request)
        prompt = provider._strategies["text"].build_prompt(
            request, eff, provider.model, provider.processor)
        ids = (mp.vlm_prepare_inputs(provider.processor, prompts=prompt)["input_ids"]
               if isinstance(prompt, str) else mx.array([list(prompt)]))
        if ids.ndim == 1:
            ids = ids[None, :]
        if k:
            ids = mx.concatenate([ids, mx.array([fresh_tokens[:k]], dtype=ids.dtype)], axis=1)
        out = provider.model.language_model(ids)
        logits = (out.logits if hasattr(out, "logits") else out)[0, -1].astype(mx.float32)
        lps = logits - mx.logsumexp(logits)
        top = mx.argsort(lps)[-2:].tolist()
        return top, float((lps[top[1]] - lps[top[0]]).item())

    try:
        top, margin = on_worker(forward)
    except Exception as e:  # noqa: BLE001
        return {"verdict": "MISMATCH", "first_divergence": k,
                "why": f"fresh forward failed, cannot judge: {e}"}
    picked = {fresh_tokens[k], restored_tokens[k]}
    tie = picked == set(top) and margin <= NEAR_TIE_MARGIN
    return {"verdict": "NEAR-TIE" if tie else "MISMATCH", "first_divergence": k,
            "top2": top, "picked": sorted(picked), "margin": round(margin, 4)}
