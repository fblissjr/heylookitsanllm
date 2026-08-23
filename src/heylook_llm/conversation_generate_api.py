# src/heylook_llm/conversation_generate_api.py
"""Conversation-scoped generation: the server-side saga.

POST /v1/conversations/{conv_id}/generate replaces the client-orchestrated
truncate -> stream -> persist sequence (plan_chat_orchestration.md Phase 1).
The server builds the provider request FROM THE STORE (system prompt, sampler
params, model, message rows), anchors truncation by message id, owns
persistence (completion, abort, and client disconnect all persist), and emits
the authoritative saved rows as the stream's final event -- the client's
post-stream state is assignment, not arithmetic.

Wire shape: the Messages SSE grammar (message_start / content_block_* /
message_delta / message_stop via StreamingEventTranslator -- this endpoint is
Phase 3b's first consumer) plus one namespaced extension event:

    event: heylook_saved
    data: {"type": "heylook_saved", "conversation_id", "mode",
           "end_reason": "complete"|"aborted"|"error",
           "messages": [<full stored rows, position order>],
           "dropped_media": {"images": n, "audio": m},
           "timing": {...ChunkTelemetry fields...}}

Modes:
- append:     optionally persist ``user_content`` as a new user turn first,
              then generate a fresh assistant turn.
- regenerate: drop the anchor message and everything after it (prompt-side),
              generate a replacement. The truncation COMMITS only together
              with the replacement row (db.replace_tail_with_message), so a
              failed/empty generation leaves the thread untouched.
- continue:   prefill with the anchor (any role; user-role is MLX-only and
              the provider 400s it as today), drop everything after it, and
              merge the continuation back onto the anchor row
              (db.replace_tail_with_update) -- same commit-together rule.

Arbitration: one active generation per conversation. A second POST gets 409;
DELETE /{conv_id}/generate aborts the active one (abort persists the partial,
same contract as the client Stop button).

Media gating: history blocks the model cannot take are dropped AT THE WIRE
(never from the store), counted in ``dropped_media`` -- the server-side twin
of v3's toWireContent, keyed off the same effective_capabilities.
"""

import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from heylook_llm import db
from heylook_llm.capabilities import effective_capabilities
from heylook_llm.config import ChatMessage, ChatRequest
from heylook_llm.db import get_db as _get_db
from heylook_llm.messages_api import StreamingEventTranslator
from heylook_llm.samplers import REQUEST_SAMPLER_FIELDS
from heylook_llm.optimizations import fast_json as json
from heylook_llm.perf_collector import (
    ChunkTelemetry,
    RequestEvent,
    get_perf_collector,
    headline_tps,
    net_ttft_ms,
)
from heylook_llm.providers.abort import AbortEvent
from heylook_llm.providers.base import GenerationFailed, InvalidGenerationRequest
from heylook_llm.reasoning_parser import (
    merge_presplit_thinking,
    parse_reasoning,
    select_reasoning_parser,
    specials_stripper,
)
from heylook_llm.router import ModelNotFound

logger = logging.getLogger(__name__)

generate_router = APIRouter(
    prefix="/v1/conversations",
    tags=["Conversations"],
)

# conv_id -> AbortEvent for the generation currently streaming into it.
# Single event loop, claimed before the first await after the 409 check --
# two concurrent POSTs cannot both pass. NOT persisted: a server restart has
# no active generations by construction.
_ACTIVE: dict[str, AbortEvent] = {}


# Sampler-bag keys that may reach ChatRequest from conv.params / overrides.
# An allowlist, not passthrough: params is the sampler bag by contract, but a
# stray key must never become a ChatRequest field by accident.
# DERIVED, not hand-listed. This was a copy of samplers.REQUEST_SAMPLER_FIELDS
# that drifted the moment a field was added to one and not the other:
# reasoning_effort landed in the cascade and NOT here, so v3 chat -- the only
# surface that generates server-side -- accepted the setting, stored it on the
# conversation, sent it, and silently dropped it (2026-08-17).
_SAMPLER_KEYS = REQUEST_SAMPLER_FIELDS + ("sampler",)
# Cap-gated keys (the server-side twin of v3's PARAM_META requiresCap).
_CAP_GATED = {"enable_thinking": "thinking", "vision_tokens": "vision",
              "reasoning_effort": "reasoning_effort"}


class GenerateRequest(BaseModel):
    mode: Literal["append", "regenerate", "continue"] = "append"
    # Anchor message for regenerate/continue.
    message_id: str | None = None
    # append mode: a new user turn to persist before generating (string or
    # Messages-style content-block list). Omit to generate from the current
    # tail (e.g. after an edit).
    user_content: str | list[dict] | None = None
    # One-shot sampler overrides layered over the conversation's params
    # (same allowlist + cap gates). May carry "model" to generate with a
    # model other than the conversation's stamped one.
    overrides: dict = Field(default_factory=dict)
    # v3's "Show special tokens" display pref (DESIGN.md §6). NOT a sampler and
    # deliberately NOT carried in `overrides`/params -- it never reaches the
    # model, and params is the sampler bag. It is per-REQUEST rather than
    # per-conversation because it decides what this reply RECORDS: the text is
    # persisted exactly as parsed, so a reply generated with it on keeps its
    # specials forever and one generated with it off never had them to keep.
    show_special_tokens: bool = False


# The ONE media-type -> capability mapping. Both the prefetch filter
# (_media_ids_for_wire) and the wire build (_wire_content) read it -- a
# hand-copied second spelling drifted once already in review: a type added to
# one and not the other means a blob is never prefetched and blob_b64 raises
# "store corruption" for a blob that exists.
_MEDIA_CAPS = {"image": "vision", "audio": "audio"}


def _media_ids_for_wire(rows: list[dict], caps: list[str]) -> list[str]:
    """Blob ids the wire build will need bytes for (schema v7 url sources).
    Cap-excluded media is dropped at the wire, so its bytes are never
    fetched."""
    ids = []
    for row in rows:
        for b in row.get("content_blocks") or []:
            cap = _MEDIA_CAPS.get(b.get("type") or "")
            src = b.get("source") or {}
            if cap in caps and src.get("media_id"):
                ids.append(src["media_id"])
    return list(dict.fromkeys(ids))


def _wire_content(blocks: list[dict], caps: list[str], dropped: dict,
                  media: dict[str, tuple[str, bytes]]) -> str | list[dict]:
    """Stored content blocks -> OpenAI wire content, dropping cap-excluded
    media (counted in ``dropped``). Text-only messages travel as a plain
    string -- the shape MLX templates see today.

    Blob-backed sources (schema v7: ``media_id`` on a url source) resolve to
    inline bytes from ``media`` -- providers cannot fetch our own relative
    URLs. A referenced blob missing from ``media`` is store corruption and
    raises (ValueError -> 500 before any stream starts), never a silent
    drop."""
    def blob_b64(src: dict) -> tuple[str, str]:
        entry = media.get(src["media_id"])
        if entry is None:
            raise ValueError(f"media blob {src['media_id']} referenced by a message is missing")
        media_type, raw = entry
        import base64
        return media_type, base64.b64encode(raw).decode()

    parts: list[dict] = []
    text_parts: list[str] = []
    for b in blocks:
        btype = b.get("type")
        if btype == "text":
            text_parts.append(b.get("text") or "")
            parts.append({"type": "text", "text": b.get("text") or ""})
        elif btype == "image":
            if _MEDIA_CAPS["image"] not in caps:
                dropped["images"] += 1
                continue
            src = b.get("source") or {}
            if src.get("type") == "url" and src.get("media_id"):
                media_type, b64 = blob_b64(src)
                url = f"data:{media_type};base64,{b64}"
            elif src.get("type") == "url":
                url = src.get("url")
            else:
                url = f"data:{src.get('media_type')};base64,{src.get('data')}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif btype == "audio":
            if _MEDIA_CAPS["audio"] not in caps:
                dropped["audio"] += 1
                continue
            src = b.get("source") or {}
            if src.get("type") == "url" and src.get("media_id"):
                media_type, b64 = blob_b64(src)
                input_audio: dict = {"data": b64}
                if "/" in media_type:
                    input_audio["format"] = media_type.split("/", 1)[1]
            elif src.get("type") == "url":
                input_audio = {"url": src.get("url")}
            else:
                input_audio = {"data": src.get("data")}
                media_type = src.get("media_type") or ""
                if "/" in media_type:
                    input_audio["format"] = media_type.split("/", 1)[1]
            parts.append({"type": "input_audio", "input_audio": input_audio})
        # unknown block types are not wire-able; skipped
    has_media = any(p.get("type") != "text" for p in parts)
    if not has_media:
        return "\n".join(text_parts)
    return parts if parts else "\n".join(text_parts)


def _build_chat_request(conv: dict, rows: list[dict], caps: list[str],
                        model_id: str, overrides: dict, *, continuing: bool,
                        dropped: dict, media: dict[str, tuple[str, bytes]]) -> ChatRequest:
    """The store IS the request: system prompt + sampler bag + rows."""
    messages: list[ChatMessage] = []
    if conv.get("system_prompt"):
        messages.append(ChatMessage(role="system", content=conv["system_prompt"]))
    for row in rows:
        # cast: pydantic validates the dict parts into ContentPart at
        # construction (same shape converters.to_chat_request builds).
        messages.append(ChatMessage(
            role=row["role"],
            content=cast("str | list", _wire_content(row.get("content_blocks") or [], caps, dropped, media)),
        ))

    params = {**(conv.get("params") or {}), **overrides}
    kwargs = {k: params[k] for k in _SAMPLER_KEYS if params.get(k) is not None}
    for key, cap in _CAP_GATED.items():
        if cap not in caps:
            kwargs.pop(key, None)

    return ChatRequest(
        model=model_id,
        messages=messages,
        stream=True,
        continue_final_message=True if continuing else None,
        **kwargs,
    )


def _strip_history_specials(request: ChatRequest, provider) -> None:
    """Remove the model's DECLARED specials from replayed ASSISTANT text.

    `show_special_tokens` keeps those specials in the row we PERSIST -- that is
    the point of the pref, they are what the model wrote. But the store IS the
    request: the next turn replays those rows verbatim, and a fast tokenizer
    matches a declared special's STRING and encodes the real control-token id.
    Left alone, an end-of-turn marker inside prior assistant content becomes a
    real turn boundary mid-prompt -- and in `continue` mode the prefill would
    END with one. A display pref that reached the model that way would be a
    generation-settings path wearing a display label (DESIGN.md §6), so the
    strip on REPLAY is what keeps "display-only" true rather than aspirational.

    ASSISTANT rows only: those are the ones this server chose to record
    unstripped. User-authored text keeps the semantics every other API surface
    gives it -- untouched, whatever the user typed or pasted.

    Uses the parser module's own one-shot stripper, so the strip SET and the
    compiled pattern are shared with the streaming filter by construction --
    `reasoning_parser` owns what "declared special" means, and asking it is what
    keeps this path from drifting onto a stale spelling of that set.
    """
    strip = specials_stripper(provider.template_info())
    if strip is None:   # model declares none (gguf routes here too)
        return
    for msg in request.messages:
        if msg.role != "assistant":
            continue
        if isinstance(msg.content, str):
            msg.content = strip(msg.content)
            continue
        for part in msg.content or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                part.text = strip(text)


@generate_router.post(
    "/{conv_id}/generate",
    summary="Generate Into Conversation",
    description="Server-side generation saga: builds the request from the "
                "stored conversation, streams Messages-grammar SSE, persists "
                "the result (completion, abort, and disconnect all persist), "
                "and emits the saved rows as a final heylook_saved event. "
                "One active generation per conversation (409 otherwise).",
)
async def generate_in_conversation(conv_id: str, request: Request, body: GenerateRequest):
    conn = _get_db(request)
    router = request.app.state.router_instance
    request_start_time = time.time()

    if conv_id in _ACTIVE:
        return JSONResponse(
            status_code=409,
            content={"error": {
                "message": "A generation is already streaming into this conversation. "
                           "Stop it (DELETE .../generate) or wait for it to finish.",
                "type": "invalid_request_error",
                "code": "generation_in_progress",
            }},
        )
    # Claim BEFORE the row snapshot, not merely before the provider awaits
    # (second review, 2026-08-13): with the claim first, any message write
    # that passed the CRUD gate earlier has already committed before our
    # snapshot read -- the store's single FIFO writer serializes it -- and
    # any later write 409s. Snapshot-before-claim left an interleaving where
    # a row landed between the two and was destroyed by the positional
    # commit: the exact phone+desktop hole the gate exists to close.
    abort_event = AbortEvent()
    _ACTIVE[conv_id] = abort_event

    try:
        conv = await db.get_conversation(conn, conv_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        model_id = body.overrides.get("model") or conv.get("model_id") \
            or getattr(router.app_config, "default_model", None)
        if not model_id:
            raise HTTPException(status_code=400, detail=(
                "Conversation has no model and the server has no default_model"))
        model_config = router.app_config.get_model_config(model_id)
        if model_config is None:
            raise HTTPException(status_code=400, detail=f"Unknown or disabled model: {model_id}")
        caps = effective_capabilities(model_config)

        # -- resolve mode against the stored rows --------------------------
        msgs: list[dict] = conv.get("messages") or []
        saved_user_row: dict | None = None
        continue_row: dict | None = None

        if body.mode == "append":
            user_content = body.user_content
            if user_content is not None and user_content != "":
                try:
                    saved_user_row = await db.append_message(
                        conn, conv_id, role="user", content=user_content)
                except ValueError as e:  # malformed content blocks
                    raise HTTPException(status_code=400, detail=str(e))
                if saved_user_row is None:  # conversation deleted mid-flight
                    raise HTTPException(status_code=404, detail="Conversation not found")
                msgs = msgs + [saved_user_row]
            if not msgs:
                raise HTTPException(status_code=400, detail=(
                    "Nothing to generate from: the conversation is empty and "
                    "no user_content was provided"))
            prompt_rows = msgs
            commit_after = msgs[-1]["position"]
        else:
            if not body.message_id:
                raise HTTPException(status_code=400, detail=f"mode={body.mode} requires message_id")
            anchor = next((m for m in msgs if m["id"] == body.message_id), None)
            if anchor is None:
                raise HTTPException(status_code=404, detail="Anchor message not found")
            if body.mode == "regenerate":
                prompt_rows = [m for m in msgs if m["position"] < anchor["position"]]
                if not prompt_rows:
                    raise HTTPException(status_code=400, detail=(
                        "Cannot regenerate the first message: nothing precedes it"))
                commit_after = anchor["position"] - 1
            else:  # continue
                prompt_rows = [m for m in msgs if m["position"] <= anchor["position"]]
                commit_after = anchor["position"]
                continue_row = anchor

        dropped = {"images": 0, "audio": 0}
        # Blob bytes for the wire (schema v7): resolved HERE, before any
        # stream starts, so a missing blob is a clean 500 rather than an
        # in-band error mid-stream. Cap-excluded media never fetches.
        media = await db.get_media_blobs(
            conn, conv_id, _media_ids_for_wire(prompt_rows, caps))
        try:
            chat_request = _build_chat_request(
                conv, prompt_rows, caps, model_id, body.overrides,
                continuing=continue_row is not None, dropped=dropped, media=media)
        except ValueError as e:  # referenced blob missing = store corruption
            raise HTTPException(status_code=500, detail=str(e))

        from heylook_llm.api import validate_request_sampler
        validate_request_sampler(chat_request.sampler)

        # -- provider + generator (same error mapping as /v1/messages) -----
        provider = None
        try:
            provider_get_start = time.time()
            provider = await asyncio.to_thread(router.get_provider, chat_request.model)
            provider_get_ms = (time.time() - provider_get_start) * 1000
            provider.check_capacity()
            # AFTER the provider resolves (it owns template_info) and BEFORE
            # the request is handed over -- see _strip_history_specials.
            _strip_history_specials(chat_request, provider)
            generator = await asyncio.to_thread(
                provider.create_chat_completion, chat_request, abort_event)
        except RuntimeError as e:
            if "MODEL_BUSY" in str(e):
                capacity = (provider.generation_queue_stats() or {}).get("capacity") if provider else None
                # A RETURN skips the except-BaseException release below and no
                # stream generator exists yet -- pop here or the conversation
                # is 409-locked forever (review finding 2026-08-13).
                _ACTIVE.pop(conv_id, None)
                return JSONResponse(
                    status_code=503,
                    content={"error": {
                        "message": "The generation queue is full. Retry shortly.",
                        "type": "server_error",
                        "code": "model_overloaded",
                    }},
                    headers={"Retry-After": "1",
                             "X-RateLimit-Limit": str(capacity) if capacity else "1",
                             "X-RateLimit-Remaining": "0"},
                )
            raise HTTPException(status_code=500, detail=str(e))
        except ModelNotFound as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[CONV-GEN] Provider error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        thinking_enabled = provider.effective_thinking(chat_request)
        perf_ctx = {"request_start_time": request_start_time,
                    "provider_get_ms": provider_get_ms,
                    "had_images": any(
                        b.get("type") == "image"
                        for r in prompt_rows for b in (r.get("content_blocks") or []))}

        # The stream generator's finally owns the _ACTIVE release -- but a
        # generator that is CANCELLED before its first step runs no code at
        # all (client aborts the fetch in the dispatch window), which would
        # leak the claim -- and since v1.67.0 a held claim also 409s every
        # message write, so a leak is a frozen conversation, not just a
        # blocked Stop. TWO releases close it (second review 2026-08-13):
        # the response's BackgroundTask runs on Starlette's cleanup path
        # even when the client disconnected (deterministic, prompt), and
        # the 60s watchdog stays as the belt for any path that skips both.
        # Every release is identity-guarded so none can pop a NEWER
        # generation's claim.
        started = {"flag": False}

        def _release_claim(source: str):
            if _ACTIVE.get(conv_id) is abort_event:
                logger.warning(f"[CONV-GEN {conv_id[:8]}] claim released by {source}")
                _ACTIVE.pop(conv_id, None)

        def _watchdog():
            if not started["flag"]:
                _release_claim("watchdog (stream never started)")
        asyncio.get_running_loop().call_later(60, _watchdog)

        return StreamingResponse(
            _stream_generate(
                conn, conv_id, generator, request,
                provider=provider, abort_event=abort_event,
                model_id=model_id, mode=body.mode,
                saved_user_row=saved_user_row, continue_row=continue_row,
                commit_after=commit_after, dropped=dropped,
                thinking_enabled=thinking_enabled,
                strip_specials=not body.show_special_tokens,
                perf_ctx=perf_ctx, started=started,
            ),
            media_type="text/event-stream",
            background=BackgroundTask(_release_claim, "response cleanup"),
        )
    except BaseException:
        # Any failure before the stream starts releases the claim; the
        # stream generator's finally owns it from here on.
        _ACTIVE.pop(conv_id, None)
        raise


@generate_router.delete(
    "/{conv_id}/generate",
    summary="Stop Active Generation",
    description="Abort the generation currently streaming into this "
                "conversation. The partial output is persisted (same "
                "contract as the Stop button). 404 if none is active.",
)
async def stop_generation(conv_id: str):
    ev = _ACTIVE.get(conv_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="No active generation for this conversation")
    ev.set()
    return {"status": "stopping", "conversation_id": conv_id}


# ---------------------------------------------------------------------------
# Streaming + persistence
# ---------------------------------------------------------------------------

async def _persist_result(conn, conv_id: str, *,
                          continue_row: dict | None, commit_after: int,
                          content: str, thinking: str | None,
                          model_id: str | None) -> dict | None:
    """Commit the generation outcome atomically (truncate + write together).

    ``model_id`` stamps a FRESH assistant row only. A continuation keeps the
    anchor's original stamp: the merged row was co-written, and any single
    stamp would misattribute half of it (see replace_tail_with_update)."""
    if continue_row is not None:
        combined = (continue_row.get("content") or "") + content
        combined_thinking = None
        new_thinking = thinking or ""
        if continue_row.get("thinking") or new_thinking:
            combined_thinking = (continue_row.get("thinking") or "") + new_thinking
        return await db.replace_tail_with_update(
            conn, conv_id, commit_after, continue_row["id"],
            content=combined, thinking=combined_thinking)
    return await db.replace_tail_with_message(
        conn, conv_id, commit_after,
        role="assistant", content=content, thinking=thinking, model_id=model_id)


async def _stream_generate(conn, conv_id, generator, http_request, *,
                           provider, abort_event, model_id, mode,
                           saved_user_row, continue_row, commit_after,
                           dropped, thinking_enabled, perf_ctx,
                           strip_specials=True,
                           started=None) -> AsyncGenerator[str, None]:
    if started is not None:
        started["flag"] = True  # the finally below owns _ACTIVE from here on
    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    # ONE dict, both parsers: the streamed split and the persisted split must
    # be built the same way or the row would disagree with what was rendered.
    parser_args = dict(thinking_enabled=thinking_enabled,
                       continuing=continue_row is not None,
                       strip_specials=strip_specials)
    translator = StreamingEventTranslator(
        message_id, model_id,
        thinking_parser=select_reasoning_parser(provider.template_info(), **parser_args))

    from heylook_llm.streaming_utils import async_generator_with_abort, keepalive_sse

    yield translator.message_start_event()

    telemetry = ChunkTelemetry()
    full_text = ""
    pre_thinking_parts: list[str] = []
    end_reason = "complete"
    error_message: str | None = None
    persisted = False

    def split_final():
        # A FRESH parser over the accumulated text: parser output is
        # chunk-invariant (TestParserInvariants), so this equals what the
        # translator streamed.
        content, thinking = parse_reasoning(
            full_text,
            select_reasoning_parser(provider.template_info(), **parser_args))
        return content, merge_presplit_thinking(pre_thinking_parts, thinking)

    async def persist():
        content, thinking = split_final()
        if not content and not thinking:
            return None  # nothing generated: the thread stays untouched
        return await _persist_result(
            conn, conv_id, continue_row=continue_row,
            commit_after=commit_after, content=content, thinking=thinking,
            model_id=model_id)

    try:
        try:
            async for chunk in async_generator_with_abort(
                    generator, http_request, abort_event,
                    log_prefix=f"[CONV-GEN {conv_id[:8]}] "):
                ka = keepalive_sse(chunk)  # sentinel guard FIRST (shared spelling)
                if ka:
                    yield ka
                    continue
                chunk_finish = getattr(chunk, "finish_reason", None)
                if chunk_finish:
                    translator.stop_reason = chunk_finish
                telemetry.absorb(chunk)
                translator.prompt_tokens = telemetry.prompt_tokens
                translator.completion_tokens = telemetry.completion_tokens
                chunk_thinking = getattr(chunk, "thinking", None)
                if chunk_thinking:
                    pre_thinking_parts.append(chunk_thinking)
                    for event_str in translator.process_presplit_thinking(chunk_thinking):
                        yield event_str
                if not chunk.text:
                    continue
                full_text += chunk.text
                for event_str in translator.process_chunk(
                        chunk.text, token_id=getattr(chunk, "token", None)):
                    yield event_str
        except InvalidGenerationRequest as e:
            # Request-shape guards fire on first advance, after headers --
            # typed in-band, nothing generated, nothing to persist.
            yield translator._sse("error", {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e)},
            })
            return
        except GenerationFailed as e:
            end_reason = "error"
            error_message = str(e)
            yield translator._sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            })
            # fall through: a partial that exists is persisted, not vanished

        if abort_event.is_set() and end_reason == "complete":
            end_reason = "aborted"

        for event_str in translator.flush():
            yield event_str
        if end_reason != "error":
            yield translator.message_delta_event()
            yield translator.message_stop_event()

        saved_row = await persist()
        persisted = True

        saved_rows = [r for r in (saved_user_row, saved_row) if r]
        payload = {
            "type": "heylook_saved",
            "conversation_id": conv_id,
            "mode": mode,
            "end_reason": end_reason,
            "messages": saved_rows,
            "dropped_media": dropped,
            "timing": {
                "peak_memory_gb": telemetry.peak_memory_gb or None,
                "kv_cache_bytes": telemetry.kv_cache_bytes or None,
                "queue_wait_ms": telemetry.queue_wait_ms or None,
                "draft_acceptance": (telemetry.draft_accepted / telemetry.draft_tokens)
                if telemetry.draft_tokens else None,
            },
        }
        if error_message:
            payload["error"] = error_message
        yield f"event: heylook_saved\ndata: {json.dumps(payload)}\n\n"

        _record_perf(perf_ctx, translator, telemetry, model_id)
    finally:
        _ACTIVE.pop(conv_id, None)
        if not persisted:
            # Client disconnected (or the response task was cancelled) before
            # the inline persist ran. The abort has already stopped the
            # provider; persistence must still happen -- that is the
            # server-owned half of "the phone locking mid-stream loses
            # nothing". A detached task survives this generator's close.
            asyncio.ensure_future(_persist_after_disconnect(persist, conv_id))


async def _persist_after_disconnect(persist, conv_id: str):
    try:
        row = await persist()
        if row is not None:
            logger.info(f"[CONV-GEN {conv_id[:8]}] partial persisted after disconnect")
    except Exception:
        logger.warning(f"[CONV-GEN {conv_id[:8]}] disconnect persistence failed", exc_info=True)


def _record_perf(perf_ctx, translator, telemetry, model_id):
    now = time.time()
    total_ms = (now - perf_ctx["request_start_time"]) * 1000
    gen_tokens = translator.completion_tokens or (translator.thinking_tokens + translator.content_tokens)
    gen_time_s = now - translator.start_time
    first_output = translator.thinking_start or translator.content_start
    raw_ttft_ms = (first_output - translator.start_time) * 1000 if first_output else 0.0
    get_perf_collector().record_request(RequestEvent(
        timestamp=now,
        model=model_id,
        success=True,
        total_ms=total_ms,
        queue_ms=perf_ctx["provider_get_ms"],
        model_load_ms=perf_ctx["provider_get_ms"] if perf_ctx["provider_get_ms"] >= 100 else 0.0,
        image_processing_ms=0.0,
        token_generation_ms=gen_time_s * 1000,
        first_token_ms=net_ttft_ms(raw_ttft_ms, telemetry.queue_wait_ms),
        prompt_tokens=translator.prompt_tokens,
        completion_tokens=gen_tokens,
        tokens_per_second=headline_tps(telemetry.generation_tps, gen_tokens, gen_time_s, telemetry.queue_wait_ms),
        had_images=perf_ctx.get("had_images", False),
        was_streaming=True,
        queue_wait_ms=round(telemetry.queue_wait_ms, 1),
        prompt_tps=telemetry.prompt_tps,
        draft_tokens=telemetry.draft_tokens,
        draft_accepted=telemetry.draft_accepted,
    ))
