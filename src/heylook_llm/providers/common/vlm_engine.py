# src/heylook_llm/providers/common/vlm_engine.py
"""One MLX request through mlx-vlm's own engine (plan W10, outcome A2).

heylook drives every MLX model with mlx-vlm's ``BatchGenerator``, one request
per generator, with mlx-vlm's automatic prefix cache (APC) held per model.
That is how mlx-vlm's own server serves requests, and it is what gives
heylook, with no upstream hooks:

- prefill progress: the prompt batch's processed-column count, read after
  every ``next()`` (private fields, the ones mlx-vlm's server reads; pinned
  by ``TestVlmEngineSurface``);
- mid-prefill cancel: ``remove(uid)`` between prefill chunks;
- cross-request reuse on every model class, restored at checkpoints taken
  during prefill (hybrid and sliding-window models) or from hashed blocks
  (plain KV models).

Everything else heylook owns stays heylook's: the sampler and the
generated-only processors (``samplers.build`` + ``generated_only``), the stop
set (checked in this loop; nothing is added to the shared tokenizer's stop
list), detokenization, timing and the per-request reports.

Rules the W10 spike established (internal/claude/w10/spike_results.md):
- The APC salt is computed the way mlx-vlm's server computes it, from the
  request's media only. Left to the generator, the salt folds in the whole
  prompt's embeddings and no two different prompts can share anything.
- Checkpoint models need a short checkpoint interval and at least three
  entries, or a follow-up finds nothing to restore
  (``APC_CHECKPOINT_INTERVAL_TOKENS`` / ``APC_CHECKPOINT_ENTRIES``, measured
  in the spike).
- The APC disk tier stays off: it would write prompt- and image-derived cache
  state to disk.
- A generator is closed on the thread that created it (its stream is
  thread-local), in this generator's ``finally``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator, Iterable, Optional

import mlx.core as mx

from ..base import CacheReport, GenerationChunk, InvalidGenerationRequest

# The spike's settings (2026-09-23, measured): at mlx-vlm's defaults a
# checkpoint model keeps only its prompt end and one 2048-aligned boundary,
# so a follow-up that diverges at the previous reply finds nothing.
APC_CHECKPOINT_INTERVAL_TOKENS = 64
APC_CHECKPOINT_ENTRIES = 3


def make_apc_manager():
    """A per-model, in-memory-only APC store."""
    from mlx_vlm import apc as _apc

    return _apc.APCManager(disk=None, overrides={
        "checkpoint_interval_tokens": APC_CHECKPOINT_INTERVAL_TOKENS,
        "checkpoint_entries": APC_CHECKPOINT_ENTRIES,
    })


def semantic_hash(raw_inputs: dict, model, processor) -> int:
    """The APC salt, exactly as mlx-vlm's server computes it: the request's
    image, audio and video payloads plus the model/processor identity --
    never the token embeddings."""
    from mlx_vlm import apc as _apc

    pixel_values = raw_inputs.get("pixel_values")
    image_hash = _apc.hash_image_payload(pixel_values=pixel_values) if pixel_values is not None else 0
    return _apc.semantic_extra_hash(
        image_hash=image_hash,
        media={"audio": raw_inputs.get("input_features"),
               "video": raw_inputs.get("pixel_values_videos")},
        model=model.language_model, processor=processor)


def cache_report(prompt_tokens: int, cached_tokens: Optional[int]) -> Optional[CacheReport]:
    """The request's CacheReport from APC's cached count (whole-prompt
    normalized like every engine). None when APC reported nothing."""
    if cached_tokens is None:
        return None
    cached = max(0, min(int(cached_tokens), prompt_tokens))
    if cached:
        return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=cached, outcome="reused")
    return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=0, outcome="miss",
                       reason="no stored prefix matched this prompt")


def prefill_progress(prompt_batch, prompt_tokens: int, cached: int) -> Optional[tuple[int, int]]:
    """(done, total) of THIS request's prefill work, cached prefix excluded
    (the cross-engine meaning). ``done`` is the prompt batch's processed-column
    count (a private field; None when absent); ``total`` is the prompt's own
    length minus the cached prefix -- never the batch's embeddings, which it
    trims as it consumes them, so a total read there shrinks between reports."""
    if prompt_batch is None:
        return None
    processed = getattr(prompt_batch, "_processed_prompt_columns", None)
    if processed is None:
        return None
    total = max(0, prompt_tokens - cached)
    return min(int(processed), total), total


def _token_int(t) -> int:
    return int(t.item()) if hasattr(t, "item") else int(t)


def generate(
    *,
    model,
    processor,
    apc_manager,
    input_ids: mx.array,
    raw_inputs: dict,
    sampler,
    processors: list,
    stop_tokens: Iterable[int],
    max_tokens: int,
    greedy: bool = False,
    abort_event=None,
    continuing: bool = False,
    context_length: Optional[int] = None,
    prefill_step_size: Optional[int] = None,
    model_id: Optional[str] = None,
) -> Generator[GenerationChunk, None, None]:
    """One request, start to finish, yielding GenerationChunks.

    ``input_ids`` (shape (1, n)) is the rendered prompt's tokens; for a
    vision request ``raw_inputs`` is ``mlx_vlm.utils.prepare_inputs``'s
    output (pixel_values and the model's extra tensors), else just the ids.
    Runs entirely on the calling thread (the pinned MLX executor).
    """
    from mlx_vlm.generate.ar import BatchGenerator

    from .generation_core import generated_only

    prompt_list = input_ids.squeeze(0).tolist()
    n = len(prompt_list)
    if context_length and n > context_length:
        raise InvalidGenerationRequest(
            f"Prompt is {n} tokens; {model_id or 'this model'} has a "
            f"context of {context_length} tokens. Shorten the conversation or the "
            f"system prompt.")

    stop = set(int(t) for t in stop_tokens)
    mx.reset_peak_memory()
    started = time.perf_counter()

    data = {k: v for k, v in raw_inputs.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")}
    bg_kwargs: dict[str, Any] = {
        "sampler": sampler, "compute_logprobs": False,
        "apc_manager": apc_manager, "greedy_sampling": greedy,
        "max_tokens": max_tokens,
    }
    if prefill_step_size:
        bg_kwargs["prefill_step_size"] = prefill_step_size
    bg = BatchGenerator(model.language_model, processor, **bg_kwargs)
    detok = None
    try:
        if bg.apc is not None:
            bg.apc.prepare_prefill(n, prefill_step_size=bg.prefill_step_size)
        embed = model.get_input_embeddings(
            input_ids, raw_inputs.get("pixel_values"),
            mask=raw_inputs.get("attention_mask"), **data)
        gen_kwargs = {**data, **{k: v for k, v in embed.to_dict().items() if v is not None}}
        if apc_manager is not None:
            gen_kwargs["_apc_semantic_hash"] = semantic_hash(raw_inputs, model, processor)
        (uid,) = bg.insert([prompt_list], max_tokens=max_tokens, prompt_kwargs=[gen_kwargs],
                           logits_processors=[generated_only(processors) or []])

        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer
        detok = make_streaming_detokenizer(processor)
        report_progress = getattr(abort_event, "set_prefill_progress", None)
        cached = 0
        cache_rep = None
        prefill_done_at = None
        generated = 0
        first = True
        last_progress = None
        while True:
            if abort_event is not None and abort_event.is_set():
                bg.remove(uid)
                logging.info("Generation aborted")
                return
            prompt_responses, responses = bg.next()
            for pr in prompt_responses:
                if pr.uid == uid:
                    cached = int(getattr(pr, "cached_tokens", 0) or 0)
                    cache_rep = cache_report(n, cached)
                    prefill_done_at = time.perf_counter()
            if prefill_done_at is None and report_progress is not None:
                progress = prefill_progress(getattr(bg, "_prompt_batch", None), n, cached)
                if progress is not None and progress != last_progress:
                    last_progress = progress
                    report_progress(*progress)
            finish = None
            text = ""
            token = None
            for r in responses:
                if r.uid != uid:
                    continue
                if r.token is not None:
                    tok = _token_int(r.token)
                    if tok in stop:
                        finish = "stop"
                    else:
                        token = tok
                        generated += 1
                        detok.add_token(tok)
                        text += detok.last_segment
                if r.finish_reason is not None and finish is None:
                    finish = "stop" if r.finish_reason == "stop" else "length"
            if token is None and finish is None:
                continue
            if finish is not None:
                detok.finalize()
                text += detok.last_segment
                if finish == "stop" and generated < max_tokens:
                    bg.remove(uid)
            now = time.perf_counter()
            prefill_end = prefill_done_at or now
            chunk = GenerationChunk(
                text=text, token=token, finish_reason=finish,
                prompt_tokens=n, generation_tokens=generated,
                prompt_tps=(n - cached) / max(prefill_end - started, 1e-9),
                generation_tps=generated / max(now - prefill_end, 1e-9) if generated else 0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
            if first:
                # A fresh turn's first token carries the artifact space after
                # the role marker; a continuation's first token completes
                # prefilled text and its space is real.
                if not continuing and chunk.text.startswith(" "):
                    chunk.text = chunk.text.lstrip()
                chunk.cache = cache_rep
                first = False
            yield chunk
            if finish is not None:
                return
    finally:
        try:
            bg.close()
        except Exception as e:  # noqa: BLE001 - closed off its thread (GC)
            logging.debug(f"BatchGenerator close skipped: {e}")
