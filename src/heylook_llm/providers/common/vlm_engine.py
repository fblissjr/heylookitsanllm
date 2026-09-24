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
  captures near the prompt end, or a follow-up finds nothing to restore
  (``APC_CHECKPOINT_INTERVAL_TOKENS`` / ``APC_CHECKPOINT_CAPTURES``, measured
  in the spike). The store size is a separate constant
  (``APC_CHECKPOINT_ENTRIES``; see ``install_capture_policy``).
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
# so a follow-up that diverges at the previous reply finds nothing. CAPTURES
# is how many of those near-end states one request snapshots (the prompt end
# plus the aligned boundaries before it).
APC_CHECKPOINT_INTERVAL_TOKENS = 64
APC_CHECKPOINT_CAPTURES = 3
# How many snapshots the store keeps across requests. mlx-vlm spends one
# number on both (its checkpoint_entries), so at the spike's setting every
# request evicted the last conversation's states and a new chat never found
# its system prompt. The store's byte budget (APCManager.memory_max_bytes)
# also bounds it; that it binds before this count on long contexts is
# reasoned from snapshot sizes, not measured. Measured 2026-09-24 (sweep in
# internal/claude/improve/runs/2026-09-24): mlx-vlm re-counts every retained
# snapshot's bytes several times per request, so each entry costs every
# request a little CPU; this keeps a few conversations' snapshots.
APC_CHECKPOINT_ENTRIES = 16


def make_apc_manager():
    """A per-model, in-memory-only APC store."""
    from mlx_vlm import apc as _apc

    return _apc.APCManager(disk=None, overrides={
        "checkpoint_interval_tokens": APC_CHECKPOINT_INTERVAL_TOKENS,
        "checkpoint_entries": APC_CHECKPOINT_ENTRIES,
    })


def capture_lengths(final: int, *, interval: int, block_size: int, captures: int,
                    min_tokens: int, boundaries: Iterable[int] = (),
                    adjust=lambda n: n) -> list[int]:
    """The prompt lengths one request snapshots on a checkpoint model.

    mlx-vlm's own rule (``APCCoordinator.checkpoint_lengths``) with its
    capture count taken from ``captures`` instead of the store size: the
    prompt end (``final``) and the ``captures - 1`` interval-aligned lengths
    before it, which a follow-up restores when it diverges inside the
    previous reply. Plus ``boundaries``: prefix lengths other requests will
    share (the end of the system prompt), so a new conversation restores
    them. ``adjust`` moves a length to where the rest of the prompt is text
    only (mlx-vlm's media rule); lengths outside ``[min_tokens, final)`` are
    dropped. Pure: no MLX."""
    lengths = {final}
    candidates = list(boundaries)
    if interval > 0 and captures > 1:
        interval = ((interval + block_size - 1) // block_size) * block_size
        last = ((final - 1) // interval) * interval
        first = max(interval, last - (captures - 2) * interval)
        candidates.extend(range(first, last + 1, interval))
    for n in candidates:
        n = adjust(n)
        if min_tokens <= n < final:
            lengths.add(n)
    return sorted(lengths)


def shared_prefix_len(prompt: list[int], prefixes: Iterable[list[int]]) -> list[int]:
    """How far each of ``prefixes`` agrees with ``prompt``, token for token
    (the caller's renders can tokenize differently at their cut)."""
    out = []
    for prefix in prefixes:
        n = 0
        for a, b in zip(prefix, prompt):
            if a != b:
                break
            n += 1
        out.append(n)
    return out


def install_capture_policy(bg, boundaries: Iterable[int]) -> None:
    """Replace this generator's checkpoint capture rule with heylook's
    (``capture_lengths``). The coordinator is built per generator
    (``BatchGenerator.apc``), so this binds one request only. Plain KV models
    (block mode) have no checkpoints and are left alone."""
    coord = getattr(bg, "apc", None)
    if coord is None or not coord.enabled or not coord.is_checkpoint:
        return
    from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

    bounds = tuple(boundaries)

    def lengths(token_ids, media_token_ids):
        final = coord.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        mgr = coord.manager
        return capture_lengths(
            final, interval=mgr.checkpoint_interval_tokens, block_size=mgr.block_size,
            captures=APC_CHECKPOINT_CAPTURES, min_tokens=mgr.exact_cache_min_tokens,
            boundaries=bounds,
            adjust=lambda n: adjust_prefix_to_text_suffix_boundary(
                token_ids, n, media_token_ids, max_prefix_tokens=final))

    coord.checkpoint_lengths = lengths


def apc_is_empty(apc_manager) -> bool:
    """Did the model's prefix cache hold nothing (the ``cold`` miss cause)?
    No snapshot and no hashed block, asked of the two stores directly:
    ``stats_snapshot()`` re-counts every retained array's bytes, a cost every
    request paid for one bool. A store that is missing reads as not empty,
    so a renamed attribute never mislabels a miss as cold (the names are
    pinned in TestVlmEngineSurface)."""
    if apc_manager is None:
        return False
    return not (getattr(apc_manager, "_exact_cache", True) or getattr(apc_manager, "hash_table", True))


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


def cache_report(prompt_tokens: int, cached_tokens: Optional[int], *,
                 cold: bool = False, has_media: bool = False) -> Optional[CacheReport]:
    """The request's CacheReport from APC's cached count (whole-prompt
    normalized like every engine). None when APC reported nothing.

    A miss says why when the engine can know: ``cold`` (the model's prefix
    cache held nothing when the request started), or ``new_image_set`` (the
    request carries images and no stored prefix has its image set -- APC
    keys a request's images as ONE hash, so a turn that adds an image starts
    over; an accepted gap, internal/claude/w10/apc_new_image_turns.md)."""
    if cached_tokens is None:
        return None
    cached = max(0, min(int(cached_tokens), prompt_tokens))
    if cached:
        return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=cached, outcome="reused")
    if cold:
        return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=0, outcome="miss",
                           cause="cold", reason="this model's prefix cache was empty")
    if has_media:
        return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=0, outcome="miss",
                           cause="new_image_set",
                           reason=("no stored prefix has this request's image set: the "
                                   "prefix cache keys a request's images as one hash, so "
                                   "a turn that adds an image starts over"))
    return CacheReport(prompt_tokens=prompt_tokens, cached_tokens=0, outcome="miss",
                       reason="no stored prefix matched this prompt")


def prefill_progress(prompt_batch, prompt_tokens: int) -> Optional[tuple[int, int]]:
    """(done, total) of THIS request's prefill work, cached prefix excluded
    (the cross-engine meaning). ``done`` is the prompt batch's processed-column
    count, which counts the uncached tail only; ``total`` is the prompt's own
    length minus the batch's cached prefix (both private fields; None when
    absent). The cached count is read off the batch because the engine's
    prompt response, which also carries it, arrives only once prefill is
    over: a total without it left a mostly-cached follow-up stuck at a few
    percent. Never the batch's embeddings for the total -- it trims them as
    it consumes them, so a total read there shrinks between reports."""
    if prompt_batch is None:
        return None
    processed = getattr(prompt_batch, "_processed_prompt_columns", None)
    if processed is None:
        return None
    cached = max(getattr(prompt_batch, "_cached_tokens_per_row", None) or [0])
    total = max(0, prompt_tokens - int(cached))
    return min(int(processed), total), total


def release_prefill(bg) -> None:
    """Give back the prefix-cache blocks a request's prefill still holds.

    A prefix hit acquires the matched APC blocks (a reference count), and
    mlx-vlm releases them when the prompt batch finishes. ``remove()`` during
    prefill -- a cancel -- and an exception out of ``next()`` drop the batch
    without releasing, and a block with references is never evictable: every
    cancelled follow-up would pin its matched prefix in the model's cache
    until unload. Idempotent: the batch's metadata is emptied after release.
    """
    batch = getattr(bg, "_prompt_batch", None)
    if batch is None:
        return
    try:
        batch._release_apc_meta_blocks()
        batch._apc_meta = []
    except Exception as e:  # noqa: BLE001 - cleanup must never mask the cause
        logging.debug(f"APC block release skipped: {e}")


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
    embed_extras: Optional[dict] = None,
    detokenizer=None,
    processors: list,
    stop_tokens: Iterable[int],
    max_tokens: int,
    greedy: bool = False,
    abort_event=None,
    continuing: bool = False,
    context_length: Optional[int] = None,
    prefill_step_size: Optional[int] = None,
    model_id: Optional[str] = None,
    reset_peak: bool = True,
    thinking_budget=None,
    shared_prefixes: Iterable[list[int]] = (),
) -> Generator[GenerationChunk, None, None]:
    """One request, start to finish, yielding GenerationChunks.

    ``input_ids`` (shape (1, n)) is the rendered prompt's tokens; for a
    vision request ``raw_inputs`` is ``mlx_vlm.utils.prepare_inputs``'s
    output (pixel_values and the model's extra tensors), else just the ids.
    ``detokenizer``: a reset streaming detokenizer to use (the caller's,
    so the continuation seam and per-token streaming are the caller's
    choice); None takes mlx-vlm's. mlx-vlm's BPE detokenizer holds ALL text
    until finalize when no token starts with a space (a count, code, CJK) --
    the answer then arrives in one lump at the end.
    ``thinking_budget``: mlx-vlm's ``ThinkingBudgetCriteria`` for this
    request (plan W7), or None. The engine forces the thinking block shut
    once it is passed; the forced tokens stream like any other.
    ``shared_prefixes``: token lists other requests will start with (the
    system prompt as the template renders it); on a checkpoint model the
    engine snapshots where each agrees with this prompt, so the next
    conversation restores it (``install_capture_policy``).
    ``embed_extras`` reach ``get_input_embeddings`` only, never the
    generator's prompt kwargs (the vision cache as ``vision_cache`` and
    ``_image_key``; mlx-vlm's server strips the same kwargs the same way).
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
    if reset_peak:
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
        install_capture_policy(bg, shared_prefix_len(prompt_list, shared_prefixes))
        cold = apc_is_empty(apc_manager)
        has_media = raw_inputs.get("pixel_values") is not None
        if bg.apc is not None:
            bg.apc.prepare_prefill(n, prefill_step_size=bg.prefill_step_size)
        embed = model.get_input_embeddings(
            input_ids, raw_inputs.get("pixel_values"),
            mask=raw_inputs.get("attention_mask"), **data, **(embed_extras or {}))
        gen_kwargs = {**data, **{k: v for k, v in embed.to_dict().items() if v is not None}}
        if apc_manager is not None:
            gen_kwargs["_apc_semantic_hash"] = semantic_hash(raw_inputs, model, processor)
        (uid,) = bg.insert([prompt_list], max_tokens=max_tokens, prompt_kwargs=[gen_kwargs],
                           logits_processors=[generated_only(processors) or []],
                           thinking_budget_criteria=[thinking_budget])

        if detokenizer is not None:
            detok = detokenizer
        else:
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
                release_prefill(bg)
                bg.remove(uid)
                logging.info("Generation aborted")
                return
            prompt_responses, responses = bg.next()
            for pr in prompt_responses:
                if pr.uid == uid:
                    cached = int(getattr(pr, "cached_tokens", 0) or 0)
                    cache_rep = cache_report(n, cached, cold=cold, has_media=has_media)
                    prefill_done_at = time.perf_counter()
            if prefill_done_at is None and report_progress is not None:
                progress = prefill_progress(getattr(bg, "_prompt_batch", None), n)
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
                    # Either stop set ends the reply, and neither's token is
                    # text. mlx-vlm stops on its own list (config.json eos,
                    # the tokenizer's, the processor's extras), which can
                    # hold an id heylook's set lacks; treating that token as
                    # content streamed its text ("<end_of_utterance>") and
                    # counted it.
                    if tok in stop or r.finish_reason == "stop":
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
        release_prefill(bg)
        try:
            bg.close()
        except Exception as e:  # noqa: BLE001 - closed off its thread (GC)
            logging.debug(f"BatchGenerator close skipped: {e}")
