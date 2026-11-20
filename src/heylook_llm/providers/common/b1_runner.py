"""Thin single-row (B=1) MLX execution path backed by Heylook's CoW prefix cache.

Bypasses ``BatchGenerator`` / ``APCCoordinator`` multi-row batching, deep
cache clones, capacity pre-allocation, and mid-prefill 64-token checkpointing
for standard B=1 requests while keeping ``BatchGenerator`` intact as the
fallback for unsupported cache types, non-MTP drafters, KV quantization,
pipeline-parallel workers, and ``HEYLOOK_B1_RUNNER=0``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Generator, Iterable, Optional

import mlx.core as mx

from ..base import CacheReport, GenerationChunk, InvalidGenerationRequest, SpecReport
from . import mlx_vlm_patches
from .prefix_cache import (
    PrefixCacheStore,
    cache_has_recurrent,
    cache_layout_supported,
    capture_rotating_fallbacks_before_evict,
    snapshot_rotating_fallbacks,
)

_DEFAULT_PREFILL_STEP_SIZE = 1024
_DEFAULT_CHECKPOINT_ENTRIES = 32
_INTERIOR_STRIDE_TOKENS = 1024
_INTERIOR_MAX_CAPTURES = 4


def _is_mock_or_none(obj: Any) -> bool:
    return obj is None or "unittest.mock" in type(obj).__module__


def can_use_b1_runner(
    *,
    model: Any,
    apc_manager: Any = None,
    draft: Any = None,
    kv_bits: Optional[int] = None,
) -> bool:
    """True when this request can execute safely on the B=1 runner."""
    if os.environ.get("HEYLOOK_B1_RUNNER", "1") == "0":
        return False
    if kv_bits is not None:
        return False
    if getattr(apc_manager, "_heylook_apc_sync_wrapped", False):
        return False
    if draft is not None and (
        not isinstance(draft, tuple) or len(draft) < 2 or draft[1] != "mtp"
    ):
        return False
    if _is_mock_or_none(model):
        return False
    lm = getattr(model, "language_model", None)
    if _is_mock_or_none(lm) or not callable(getattr(lm, "make_cache", None)):
        return False

    cached_ok = getattr(lm, "_heylook_b1_layout_ok", None)
    if cached_ok is None:
        try:
            sample = lm.make_cache()
            cached_ok = bool(sample) and cache_layout_supported(sample)
        except Exception:  # noqa: BLE001 - conservative fallback on unusual model caches
            cached_ok = False
        try:
            lm._heylook_b1_layout_ok = cached_ok
        except Exception:  # noqa: BLE001
            pass
    return bool(cached_ok)


def get_or_create_b1_cache(apc_manager: Any) -> Optional[PrefixCacheStore]:
    """Attach and return the model-scoped CoW prefix cache on ``apc_manager``."""
    if apc_manager is None:
        return None
    store = getattr(apc_manager, "_heylook_b1_cache", None)
    if store is None:
        max_entries = int(
            getattr(apc_manager, "checkpoint_entries", 0) or _DEFAULT_CHECKPOINT_ENTRIES
        )
        store = PrefixCacheStore(
            memory_max_bytes=int(getattr(apc_manager, "memory_max_bytes", 0) or 0),
            max_entries=max_entries,
            memory_reserve_bytes=int(
                getattr(apc_manager, "memory_reserve_bytes", 0) or 0
            ),
        )
        apc_manager._heylook_b1_cache = store
    return store


def _generation_stream():
    try:
        from mlx_vlm._generation_stream import generation_stream

        return generation_stream
    except ImportError:
        return mx.default_stream(mx.default_device())


def _buffer_rotating_size(draft_model: Any, draft_block_size: Optional[int]) -> int:
    if draft_model is None:
        return 256
    cfg = getattr(draft_model, "config", None)
    configured = int(getattr(cfg, "block_size", draft_block_size or 1) or 1)
    requested = int(draft_block_size or configured)
    raw = max(256, max(configured, requested) * 8)
    aligned = ((raw + 255) // 256) * 256
    return min(512, aligned)


def _should_use_buffered_rotating(lm: Any, *, drafting: bool) -> bool:
    if drafting:
        return True
    from . import gemma4_vision as _gv

    return (
        _gv.mlx_sdpa_supports_window_size()
        and getattr(
            getattr(getattr(lm, "model", None), "_make_masks", None),
            "__func__",
            None,
        )
        is _gv.gemma4_make_masks
    )


def _ensure_buffered_rotating_caches(prompt_cache: list[Any], buffer_size: int) -> None:
    from mlx_vlm.models.cache import BufferedRotatingKVCache, CacheList, RotatingKVCache

    def _convert(entry: Any) -> Any:
        t = type(entry)
        if t is CacheList:
            entry.caches = tuple(_convert(ch) for ch in entry.caches)
            return entry
        if t is BufferedRotatingKVCache:
            entry.buffer_size = max(int(entry.buffer_size), buffer_size)
            return entry
        if t is RotatingKVCache and int(getattr(entry, "keep", 0) or 0) == 0:
            if entry.keys is None:
                return BufferedRotatingKVCache(
                    entry.max_size, keep=0, buffer_size=buffer_size
                )
            return BufferedRotatingKVCache.from_cache(entry, buffer_size=buffer_size)
        return entry

    for idx, entry in enumerate(prompt_cache):
        prompt_cache[idx] = _convert(entry)


def _slice_prompt_kwargs(
    kwargs: dict[str, Any],
    *,
    full_len: int,
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        from mlx_vlm.generate.ar import (
            _is_sequence_aligned_prompt_kwarg,
            _slice_sequence_aligned_prompt_kwarg,
        )
    except ImportError:
        return dict(kwargs)

    out = {}
    for k, v in kwargs.items():
        if isinstance(v, mx.array) and _is_sequence_aligned_prompt_kwarg(
            k, v, full_len
        ):
            out[k] = _slice_sequence_aligned_prompt_kwarg(k, v, start=start, stop=stop)
        else:
            out[k] = v
    return out


def _sample_next_token(
    logits: mx.array,
    sampler: Any,
    *,
    greedy: bool,
    active_processors: list,
    context_tokens: list[int],
    position: int,
) -> mx.array:
    if logits.ndim == 1:
        logits = logits[None, :]
    if active_processors:
        ctx_arr = mx.array(context_tokens)
        for proc in active_processors:
            logits = proc(ctx_arr, logits)

    if (
        greedy
        and not active_processors
        and (sampler is None or getattr(sampler, "_heylook_greedy", False))
    ):
        return mx.argmax(logits, axis=-1).reshape(1)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if sampler is None:
        return mx.argmax(logprobs, axis=-1).reshape(1)
    sample_target = getattr(sampler, "sample_target", None)
    if callable(sample_target):
        tok = sample_target(logprobs, row_ids=[0], positions=[position])
        return tok.reshape(1)
    return sampler(logprobs).reshape(1)


def _extract_rope_deltas(lm: Any, prompt_kwargs: dict[str, Any]) -> Optional[mx.array]:
    rd = (prompt_kwargs or {}).get("rope_deltas")
    if isinstance(rd, mx.array):
        if rd.ndim == 0:
            return rd.reshape(1, 1)
        if rd.ndim == 1:
            return rd[:, None]
        return rd
    lm_rd = getattr(lm, "_rope_deltas", None)
    if isinstance(lm_rd, mx.array):
        if lm_rd.ndim == 0:
            return lm_rd.reshape(1, 1)
        if lm_rd.ndim == 1:
            return lm_rd[:, None]
        return lm_rd
    return None


def _plan_chunk_stops(
    *,
    cached: int,
    total_len: int,
    step_size: Optional[int],
    boundaries: Iterable[int],
    adjust_fn,
    has_recurrent: bool = False,
    apc_manager: Any = None,
) -> tuple[list[int], set[int]]:
    """Return sorted chunk end positions strictly in ``(cached, total_len)`` and snapshot stops."""
    if has_recurrent:
        if total_len <= cached + 1:
            return [], set()
        from . import vlm_engine as _ve

        interval = int(
            getattr(
                apc_manager,
                "checkpoint_interval_tokens",
                _ve.APC_CHECKPOINT_INTERVAL_TOKENS,
            )
            or _ve.APC_CHECKPOINT_INTERVAL_TOKENS
        )
        block_size = int(getattr(apc_manager, "block_size", 16) or 16)
        min_tokens = int(getattr(apc_manager, "exact_cache_min_tokens", 16) or 16)
        planned = _ve.capture_lengths(
            total_len - 1,
            interval=interval,
            block_size=block_size,
            captures=_ve.APC_CHECKPOINT_CAPTURES,
            min_tokens=min_tokens,
            boundaries=boundaries,
            adjust=adjust_fn or (lambda m: m),
            interior_stride=_ve.APC_INTERIOR_STRIDE_TOKENS,
            interior_max=_ve.APC_INTERIOR_MAX_CAPTURES,
        )
        snap_stops = {p for p in planned if cached < p < total_len - 1}
        stops = set(snap_stops)
        if step_size is not None and step_size > 0 and (total_len - cached) > step_size:
            pos = cached + step_size
            while pos < total_len:
                stops.add(pos)
                if pos == total_len - 1 and (adjust_fn is None or adjust_fn(pos) == pos):
                    snap_stops.add(pos)
                pos += step_size
        return sorted(stops), snap_stops

    snap_stops: set[int] = set()
    for b in boundaries:
        if cached < b < total_len:
            adj = adjust_fn(int(b)) if adjust_fn is not None else int(b)
            if cached < adj < total_len:
                snap_stops.add(adj)

    if step_size is None or step_size <= 0 or (total_len - cached) <= step_size:
        return sorted(snap_stops), snap_stops

    stops = set(snap_stops)
    pos = cached + step_size
    while pos < total_len:
        stops.add(pos)
        pos += step_size

    span = total_len - cached
    stride = _INTERIOR_STRIDE_TOKENS
    while stride > 0 and (span - 1) // stride > _INTERIOR_MAX_CAPTURES:
        stride *= 2
    if stride > 0:
        p = ((cached // stride) + 1) * stride
        while p < total_len:
            adj = adjust_fn(p) if adjust_fn is not None else p
            if cached < adj < total_len:
                stops.add(adj)
                snap_stops.add(adj)
            p += stride

    return sorted(stops), snap_stops


def _uniform_cache_offset(prompt_cache: list[Any]) -> Optional[int]:
    from mlx_vlm.models.cache import CacheList

    offsets: list[int] = []

    def _visit(entry: Any) -> bool:
        if type(entry) is CacheList:
            return all(_visit(ch) for ch in entry.caches)
        off = getattr(entry, "offset", None)
        if isinstance(off, int):
            offsets.append(off)
        return True

    if not all(_visit(c) for c in prompt_cache) or not offsets:
        return None
    first = offsets[0]
    return first if all(o == first for o in offsets) else None


def _apply_forced_budget_check(criteria: Any, tok: int) -> Optional[int]:
    if criteria is None:
        return None
    criteria(tok)
    pop = getattr(criteria, "pop_forced_token_id", None)
    if callable(pop):
        return pop()
    return None


def _track_end_thinking_criteria(end_crit: Any, tok: int) -> None:
    if end_crit is None:
        return
    if (
        not getattr(end_crit, "in_thinking", False)
        and getattr(end_crit, "thinking_start_token_id", None) is not None
        and tok == end_crit.thinking_start_token_id
    ):
        end_crit.in_thinking = True
    elif (
        getattr(end_crit, "in_thinking", False)
        and tok == getattr(end_crit, "thinking_end_token_id", None)
    ):
        end_crit.in_thinking = False


def generate_b1(
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
    end_thinking_criteria=None,
    shared_prefixes: Iterable[list[int]] = (),
    draft=None,
    draft_block_size: Optional[int] = None,
    draft_max_context: Optional[int] = None,
) -> Generator[GenerationChunk, None, None]:
    """Single-row MLX generation path using zero-copy CoW prefix reuse."""
    from . import vlm_engine as _ve
    from .generation_core import generated_only

    prompt_list = [int(t) for t in input_ids.squeeze(0).tolist()]
    n = len(prompt_list)
    if context_length and n > context_length:
        raise InvalidGenerationRequest(
            f"Prompt is {n} tokens; {model_id or 'this model'} has a "
            f"context of {context_length} tokens. Shorten the conversation or the "
            f"system prompt."
        )

    stop = set(int(t) for t in stop_tokens)
    if reset_peak:
        mx.reset_peak_memory()
    started = time.perf_counter()

    mlx_vlm_patches.install_gated_delta_packed_kernel()
    mlx_vlm_patches.install_mtp_single_row_fast_paths()

    data = {
        k: v
        for k, v in raw_inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }
    active = generated_only(processors) or []
    drafting = False
    if draft is not None:
        blocker = _ve.drafting_blocker(
            active,
            draft_kind=draft[1],
            prompt_tokens=n,
            draft_max_context=draft_max_context,
        )
        if blocker is None:
            drafting = True
        else:
            logging.debug(f"{model_id}: not drafting this request: {blocker}")

    mtp_gen_tokens: Optional[list[int]] = None
    eff_greedy = greedy
    if drafting and active:
        mtp_gen_tokens = []
        raw_procs = [getattr(p, "_heylook_raw_processor", p) for p in active]
        sampler._heylook_mtp_penalties = (raw_procs, mtp_gen_tokens)
        eff_greedy = False

    lm = model.language_model
    b1_cache = get_or_create_b1_cache(apc_manager)
    detok = detokenizer
    if detok is None:
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer

        detok = make_streaming_detokenizer(processor)

    boundaries = _ve.shared_prefix_len(prompt_list, shared_prefixes)
    salt = None
    aborted = False
    finish = None
    generated = 0
    accepted = 0
    prompt_cache: list[Any] = []
    cache_tokens = list(prompt_list)
    stopped_mid_mtp_round = False
    has_media = any(
        raw_inputs.get(k) is not None
        for k in ("pixel_values", "pixel_values_videos", "audio_features", "input_features")
    )
    rope_deltas: Optional[mx.array] = None
    pregen_fallbacks: dict[int, Any] = {}
    has_recurrent = False
    has_uncaptured_rotating = False

    try:
        use_buf_rot = _should_use_buffered_rotating(lm, drafting=drafting)
        buf_size = (
            _buffer_rotating_size(draft[0] if drafting else None, draft_block_size)
            if use_buf_rot
            else 64
        )
        media_ids = _ve.multimodal_token_ids(model, processor)
        suffix_is_text = (
            (lambda m: not any(t in media_ids for t in prompt_list[m:]))
            if media_ids
            else None
        )
        cold = _ve.apc_is_empty(apc_manager) if b1_cache is not None else False

        with mx.stream(_generation_stream()):
            cached = 0
            restored_rd = None
            if b1_cache is not None:
                salt = _ve.semantic_hash(raw_inputs, model, processor)
                restored_cache, cached, restored_rd = b1_cache.lookup(
                    prompt_list,
                    extra_hash=salt,
                    suffix_is_text_only=suffix_is_text,
                    use_buffered_rotating=use_buf_rot,
                    buffer_size=buf_size,
                )
                if restored_cache is not None and cached > 0:
                    prompt_cache = restored_cache
                else:
                    cached = 0
            if not prompt_cache:
                prompt_cache = list(lm.make_cache())
                if use_buf_rot:
                    _ensure_buffered_rotating_caches(prompt_cache, buf_size)
                if hasattr(lm, "_rope_deltas"):
                    lm._rope_deltas = None
            elif hasattr(lm, "_rope_deltas"):
                lm._rope_deltas = restored_rd

            has_recurrent = cache_has_recurrent(prompt_cache)
            has_uncaptured_rotating = b1_cache is not None and not has_recurrent
            rem_len = n - cached
            suffix_ids = input_ids[:, cached:]

            gen_kwargs: dict[str, Any] = {}
            rem_kwargs: dict[str, Any] = {}
            suffix_embeds = None
            can_suffix_embed = (
                cached > 0
                and not has_media
                and not data
                and not embed_extras
                and not hasattr(lm, "get_rope_index")
            )
            if can_suffix_embed:
                embed = model.get_input_embeddings(suffix_ids, None, mask=None)
                emb_dict = {
                    k: v for k, v in embed.to_dict().items() if v is not None
                }
                if "inputs_embeds" in emb_dict and all(
                    k in ("inputs_embeds", "per_layer_inputs") for k in emb_dict
                ):
                    suffix_embeds = emb_dict.pop("inputs_embeds")
                    rem_kwargs = dict(emb_dict)
                    gen_kwargs = dict(emb_dict)

            if suffix_embeds is None:
                embed = model.get_input_embeddings(
                    input_ids,
                    raw_inputs.get("pixel_values"),
                    mask=raw_inputs.get("attention_mask"),
                    **data,
                    **(embed_extras or {}),
                )
                gen_kwargs = {
                    **data,
                    **{k: v for k, v in embed.to_dict().items() if v is not None},
                }
                full_embeds = gen_kwargs.pop("inputs_embeds", None)
                if full_embeds is None:
                    raise ValueError(
                        "get_input_embeddings did not produce inputs_embeds"
                    )
                suffix_embeds = full_embeds[:, cached:, :]
                rem_kwargs = _slice_prompt_kwargs(
                    gen_kwargs, full_len=n, start=cached
                )

            if restored_rd is not None:
                if "rope_deltas" not in gen_kwargs:
                    gen_kwargs["rope_deltas"] = restored_rd
                if "rope_deltas" not in rem_kwargs:
                    rem_kwargs["rope_deltas"] = restored_rd

            chunk_policy = getattr(lm, "chunked_prefill_policy", None)
            allow_chunk = True
            if callable(chunk_policy):
                allow_chunk = bool(
                    chunk_policy(
                        input_ids=suffix_ids,
                        inputs_embeds=suffix_embeds,
                        prompt_cache=prompt_cache,
                        draft_model=draft[0] if drafting else None,
                        draft_kind=draft[1] if drafting else None,
                        prefill_kwargs=rem_kwargs,
                    )
                )
            step_size = (
                (prefill_step_size or _DEFAULT_PREFILL_STEP_SIZE)
                if allow_chunk
                else None
            )

            adjust_fn = None
            if media_ids:
                from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

                adjust_fn = lambda m: adjust_prefix_to_text_suffix_boundary(
                    prompt_list, m, media_ids, max_prefix_tokens=n - 1
                )

            chunk_stops, snap_stops = _plan_chunk_stops(
                cached=cached,
                total_len=n,
                step_size=step_size,
                boundaries=boundaries,
                adjust_fn=adjust_fn,
                has_recurrent=has_recurrent,
                apc_manager=apc_manager,
            )

            report_progress = getattr(abort_event, "set_prefill_progress", None)
            pos = cached
            spec_prefill_kw: dict[str, Any] = {}
            if drafting:
                from mlx_vlm.speculative.utils import (
                    speculative_hidden_state,
                    speculative_prefill_kwargs,
                )

                spec_prefill_kw = speculative_prefill_kwargs(draft[1], draft[0])

            for stop_pos in chunk_stops:
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    return
                k = stop_pos - pos
                rel_start = pos - cached
                rel_end = stop_pos - cached
                step_kw = _slice_prompt_kwargs(
                    rem_kwargs, full_len=rem_len, start=rel_start, stop=rel_end
                )
                lm(
                    suffix_ids[:, rel_start:rel_end],
                    cache=prompt_cache,
                    inputs_embeds=suffix_embeds[:, rel_start:rel_end],
                    logits_to_keep=1,
                    **step_kw,
                )
                if stop_pos in snap_stops and b1_cache is not None:
                    mx.eval([c.state for c in prompt_cache])
                    b1_cache.store(
                        prompt_list[:stop_pos],
                        prompt_cache,
                        extra_hash=salt or 0,
                        trim_tail=0,
                        rope_deltas=_extract_rope_deltas(lm, gen_kwargs),
                    )
                else:
                    mx.async_eval([c.state for c in prompt_cache])
                pos = stop_pos
                if report_progress is not None:
                    report_progress(pos - cached, rem_len)

            if abort_event is not None and abort_event.is_set():
                aborted = True
                return

            final_rel = pos - cached
            tail_len = n - pos
            final_kw = _slice_prompt_kwargs(
                rem_kwargs, full_len=rem_len, start=final_rel
            )
            final_ids = suffix_ids[:, final_rel:]
            final_embeds = suffix_embeds[:, final_rel:]

            if has_recurrent and b1_cache is not None and tail_len >= 2:
                # ArraysCache cannot trim back after running token N; snapshot N-1 first.
                pre_ids = final_ids[:, :-1]
                pre_embeds = final_embeds[:, :-1]
                pre_kw = _slice_prompt_kwargs(
                    final_kw, full_len=tail_len, start=0, stop=tail_len - 1
                )
                lm(
                    pre_ids,
                    cache=prompt_cache,
                    inputs_embeds=pre_embeds,
                    logits_to_keep=1,
                    **pre_kw,
                )
                mx.eval([c.state for c in prompt_cache])
                b1_cache.store(
                    prompt_list[: n - 1],
                    prompt_cache,
                    extra_hash=salt or 0,
                    trim_tail=0,
                    rope_deltas=_extract_rope_deltas(lm, gen_kwargs),
                )
                last_kw = _slice_prompt_kwargs(
                    final_kw, full_len=tail_len, start=tail_len - 1, stop=tail_len
                )
                prefill_out = lm(
                    final_ids[:, -1:],
                    cache=prompt_cache,
                    inputs_embeds=final_embeds[:, -1:],
                    logits_to_keep=1,
                    **last_kw,
                    **spec_prefill_kw,
                )
            else:
                prefill_out = lm(
                    final_ids,
                    cache=prompt_cache,
                    inputs_embeds=final_embeds,
                    logits_to_keep=1,
                    **final_kw,
                    **spec_prefill_kw,
                )

            raw_logits = (
                prefill_out.logits if hasattr(prefill_out, "logits") else prefill_out
            )
            logits = raw_logits[:, -1, :]
            first_y = _sample_next_token(
                logits,
                sampler,
                greedy=eff_greedy,
                active_processors=[] if drafting else active,
                context_tokens=prompt_list,
                position=0,
            )
            rope_deltas = _extract_rope_deltas(lm, gen_kwargs)
            mx.eval(first_y, [c.state for c in prompt_cache])
            first_tok = int(first_y.item())

            if b1_cache is not None and has_recurrent:
                b1_cache.store(
                    prompt_list,
                    prompt_cache,
                    extra_hash=salt or 0,
                    trim_tail=0,
                    rope_deltas=rope_deltas,
                )

            prefill_done_at = time.perf_counter()
            if report_progress is not None:
                report_progress(rem_len, rem_len)
            memory = (
                _ve.apc_memory_pressure(apc_manager)
                if apc_manager is not None
                else None
            )
            cache_rep = _ve.cache_report(
                n, cached, cold=cold, has_media=has_media, memory=memory
            )

            def _emit_chunk(tok: Optional[int], *, is_last: bool, is_first: bool, fin: Optional[str]) -> GenerationChunk:
                nonlocal generated
                text = ""
                if tok is not None:
                    generated += 1
                    detok.add_token(tok)
                    text = detok.last_segment
                if is_last and fin is not None:
                    detok.finalize()
                    text += detok.last_segment
                now = time.perf_counter()
                chunk = GenerationChunk(
                    text=text,
                    token=tok,
                    finish_reason=fin if is_last else None,
                    prompt_tokens=n,
                    generation_tokens=generated,
                    prompt_tps=(n - cached) / max(prefill_done_at - started, 1e-9),
                    generation_tps=(
                        generated / max(now - prefill_done_at, 1e-9) if generated else 0.0
                    ),
                    peak_memory=mx.get_peak_memory() / 1e9,
                )
                if drafting and is_last and fin is not None:
                    chunk.spec = SpecReport(accepted=accepted, emitted=generated)
                if is_first:
                    if not continuing and chunk.text.startswith(" "):
                        chunk.text = chunk.text.lstrip()
                    chunk.cache = cache_rep
                return chunk

            # Handle first token (never emit a stop token as text).
            if first_tok in stop:
                finish = "stop"
                yield _emit_chunk(None, is_last=True, is_first=True, fin="stop")
                return

            if max_tokens <= 1:
                finish = "length"
                if mtp_gen_tokens is not None:
                    mtp_gen_tokens.append(first_tok)
                yield _emit_chunk(first_tok, is_last=True, is_first=True, fin="length")
                return

            if mtp_gen_tokens is not None:
                mtp_gen_tokens.append(first_tok)
            if thinking_budget is not None:
                if drafting:
                    thinking_budget(first_tok)
                    pending_forced = None
                else:
                    pending_forced = _apply_forced_budget_check(thinking_budget, first_tok)
            else:
                pending_forced = None
                if end_thinking_criteria is not None:
                    _track_end_thinking_criteria(end_thinking_criteria, first_tok)
            yield _emit_chunk(first_tok, is_last=False, is_first=True, fin=None)

            pop_end_reasoning = getattr(abort_event, "pop_end_reasoning", None)

            if not drafting:
                next_input_tok = first_tok
                fwd_kw = {"rope_deltas": rope_deltas} if rope_deltas is not None else {}
                while generated < max_tokens:
                    if abort_event is not None and abort_event.is_set():
                        aborted = True
                        return
                    if pop_end_reasoning is not None and pop_end_reasoning():
                        crit = thinking_budget or end_thinking_criteria
                        if (
                            crit is not None
                            and getattr(crit, "in_thinking", False)
                            and not getattr(crit, "budget_exceeded", False)
                        ):
                            crit.thinking_budget = 0
                            crit.budget_exceeded = True
                            seq = getattr(crit, "_forced_sequence", None) or []
                            idx = int(getattr(crit, "_forced_index", 0) or 0)
                            if pending_forced is None and idx < len(seq):
                                pending_forced = int(seq[idx])
                                crit._forced_index = idx + 1

                    if has_uncaptured_rotating:
                        has_uncaptured_rotating = capture_rotating_fallbacks_before_evict(
                            prompt_cache,
                            target_len=n,
                            incoming=1,
                            captured=pregen_fallbacks,
                        )
                    step_in = mx.array([[next_input_tok]], dtype=input_ids.dtype)
                    out = lm(step_in, cache=prompt_cache, **fwd_kw)
                    cache_tokens.append(next_input_tok)

                    if pending_forced is not None:
                        tok = int(pending_forced)
                        pending_forced = None
                    else:
                        step_logits = (out.logits if hasattr(out, "logits") else out)[
                            :, -1, :
                        ]
                        y = _sample_next_token(
                            step_logits,
                            sampler,
                            greedy=eff_greedy,
                            active_processors=active,
                            context_tokens=cache_tokens,
                            position=generated,
                        )
                        mx.eval(y)
                        tok = int(y.item())

                    if tok in stop:
                        finish = "stop"
                        yield _emit_chunk(None, is_last=True, is_first=False, fin="stop")
                        return

                    if thinking_budget is not None:
                        forced_after = _apply_forced_budget_check(thinking_budget, tok)
                        if forced_after is not None and pending_forced is None:
                            pending_forced = forced_after
                    elif end_thinking_criteria is not None:
                        _track_end_thinking_criteria(end_thinking_criteria, tok)

                    is_last = generated + 1 >= max_tokens
                    if is_last:
                        finish = "length"
                    yield _emit_chunk(
                        tok,
                        is_last=is_last,
                        is_first=False,
                        fin="length" if is_last else None,
                    )
                    if is_last:
                        return
                    if generated % 256 == 0:
                        mx.eval([c.state for c in prompt_cache])
                    next_input_tok = tok
            else:
                from mlx_vlm.speculative.mtp import MTPBatch

                block_size = draft_block_size or int(
                    getattr(getattr(draft[0], "config", None), "block_size", 5) or 5
                )
                mtp_batch = MTPBatch(
                    lm,
                    draft[0],
                    prompt_cache,
                    speculative_hidden_state(draft[1], prefill_out),
                    mx.array([first_tok], dtype=input_ids.dtype),
                    block_size,
                    sampler,
                    input_ids.dtype,
                    eff_greedy,
                    shared_kv_states=getattr(prefill_out, "shared_kv_states", None),
                    prompt_tokens=input_ids,
                    rope_deltas=rope_deltas,
                )
                active_crit = thinking_budget or end_thinking_criteria
                crit_arg = [active_crit] if active_crit is not None else None

                while generated < max_tokens:
                    if abort_event is not None and abort_event.is_set():
                        aborted = True
                        return
                    if pop_end_reasoning is not None and pop_end_reasoning():
                        crit = thinking_budget or end_thinking_criteria
                        if (
                            crit is not None
                            and getattr(crit, "in_thinking", False)
                            and not getattr(crit, "budget_exceeded", False)
                        ):
                            crit.thinking_budget = 0
                            crit.budget_exceeded = True

                    if has_uncaptured_rotating:
                        has_uncaptured_rotating = capture_rotating_fallbacks_before_evict(
                            prompt_cache,
                            target_len=n,
                            incoming=block_size + 1,
                            captured=pregen_fallbacks,
                        )
                    prev_bonus = int(mtp_batch.rows[0].bonus)
                    round_tokens = mtp_batch.step(
                        [max_tokens - generated],
                        [generated],
                        thinking_budget_criteria=crit_arg,
                    )
                    raw_toks = [int(t) for t in round_tokens[0]]

                    toks: list[int] = []
                    round_finish: Optional[str] = None
                    for tok in raw_toks:
                        if tok in stop:
                            round_finish = "stop"
                            break
                        toks.append(tok)
                    delivered = len(toks) + (1 if round_finish == "stop" else 0)
                    accepted += max(0, delivered - 1)
                    if round_finish is None and generated + len(toks) >= max_tokens:
                        round_finish = "length"

                    if round_finish != "stop" and len(raw_toks) >= 1:
                        committed_in_round = [prev_bonus, *raw_toks[:-1]]
                        if len(cache_tokens) + len(committed_in_round) == mtp_batch.rows[0].position:
                            cache_tokens.extend(committed_in_round)
                        else:
                            stopped_mid_mtp_round = True
                    else:
                        stopped_mid_mtp_round = True

                    if mtp_gen_tokens is not None and toks:
                        mtp_gen_tokens.extend(toks)

                    steps: list[Optional[int]] = toks if toks else [None]
                    for idx, step_tok in enumerate(steps):
                        is_last_in_round = idx == len(steps) - 1
                        is_final = is_last_in_round and round_finish is not None
                        if is_final:
                            finish = round_finish
                        yield _emit_chunk(
                            step_tok,
                            is_last=is_final,
                            is_first=False,
                            fin=round_finish if is_final else None,
                        )
                        if is_final:
                            return
    finally:
        if hasattr(sampler, "_heylook_mtp_penalties"):
            sampler._heylook_mtp_penalties = None
        if b1_cache is not None:
            if salt is not None and boundaries:
                b1_cache.touch(prompt_list, boundaries, extra_hash=salt)
            with mx.stream(_generation_stream()):
                if not has_recurrent and prompt_cache:
                    cur_off = _uniform_cache_offset(prompt_cache)
                    if cur_off is not None and cur_off >= n:
                        mx.eval([c.state for c in prompt_cache])
                        b1_cache.store(
                            prompt_list,
                            prompt_cache,
                            extra_hash=salt or 0,
                            trim_tail=0,
                            cache_trim_tail=cur_off - n,
                            rope_deltas=rope_deltas,
                            layer_fallbacks=pregen_fallbacks,
                        )
                        if (
                            not aborted
                            and generated > 0
                            and finish in ("stop", "length")
                            and not stopped_mid_mtp_round
                            and len(cache_tokens) > n
                            and cur_off == len(cache_tokens)
                        ):
                            b1_cache.store(
                                cache_tokens,
                                prompt_cache,
                                extra_hash=salt or 0,
                                trim_tail=0,
                                rope_deltas=rope_deltas,
                            )
                    pregen_fallbacks.clear()
                elif (
                    not aborted
                    and generated > 0
                    and finish in ("stop", "length")
                    and not stopped_mid_mtp_round
                    and len(cache_tokens) > n
                    and _uniform_cache_offset(prompt_cache) == len(cache_tokens)
                ):
                    mx.eval([c.state for c in prompt_cache])
                    b1_cache.store(
                        cache_tokens,
                        prompt_cache,
                        extra_hash=salt or 0,
                        trim_tail=0,
                        rope_deltas=rope_deltas,
                    )
                b1_cache.compact_pending()
