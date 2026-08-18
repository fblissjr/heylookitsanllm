# src/heylook_llm/conversation_api.py
"""Conversation storage API endpoints.

CRUD for conversations and their messages, backed by the DuckDB store
(db.py). Message content persists as content-block lists; see
docs/frontend_v3_spec.md §4 for the wire contract.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from heylook_llm import db
from heylook_llm.db import get_db as _get_db

logger = logging.getLogger(__name__)


def _refuse_while_generating(conv_id: str) -> None:
    """409 message mutations while a generation streams into this conversation.

    The generate endpoint's commit deletes `position > commit_after` in the
    same transaction that writes its row -- rows another client appends
    mid-generation would be silently destroyed at that commit (review
    finding 2026-08-13; the phone+desktop case). The server is the
    arbiter now: mutate after the stream ends, or stop it first.
    Metadata PUTs (title/prompt/params) stay open -- they are not
    position-destructive and don't touch the pending commit.
    """
    from heylook_llm.conversation_generate_api import _ACTIVE
    if conv_id in _ACTIVE:
        raise HTTPException(status_code=409, detail=(
            "A generation is streaming into this conversation -- wait for it "
            "or stop it (DELETE .../generate) before mutating messages"))

conversation_router = APIRouter(
    prefix="/v1/conversations",
    tags=["Conversations"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: str = "New Conversation"
    model_id: str | None = None
    system_prompt: str | None = None
    params: dict = {}  # per-conversation sampler settings (temperature, top_p, ...)
    # A new document can START as a preset (v3's new-conversation inheritance)
    # -- that is an explicit apply at birth, so it stamps. Same contract as
    # ConversationUpdate.applied_preset_id otherwise.
    applied_preset_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    params: dict | None = None
    # Which preset this conversation is running. EXPLICIT stamps only (an
    # Apply or a Save writes it); a document whose state merely happens to
    # match a preset is labelled by live client-side inference and never
    # stamped -- storing an inference could bind stale state to the wrong
    # document (the v1.39.7 lesson). null clears the association.
    applied_preset_id: str | None = None


# content accepts a plain string OR a content-block list (Messages-style,
# e.g. [{"type":"image","source":{...}}, {"type":"text","text":"..."}]).
# Responses always carry both: `content` (flattened text, back-compatible)
# and `content_blocks` (the full list). Spec: docs/frontend_v3_spec.md §4.
class MessageCreate(BaseModel):
    role: str
    content: str | list[dict] = ""
    thinking: str | None = None


class MessageUpdate(BaseModel):
    content: str | list[dict] | None = None
    thinking: str | None = None


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

@conversation_router.get(
    "",
    summary="List Conversations",
    description="List all conversations, ordered by most recently updated.",
)
async def list_conversations(request: Request):
    conn = _get_db(request)
    convs = await db.list_conversations(conn)
    return {"conversations": convs, "total": len(convs)}


@conversation_router.post(
    "",
    summary="Create Conversation",
    description="Create a new conversation.",
    status_code=201,
)
async def create_conversation(request: Request, body: ConversationCreate):
    conn = _get_db(request)
    conv = await db.create_conversation(
        conn, title=body.title, model_id=body.model_id,
        system_prompt=body.system_prompt, params=body.params,
        applied_preset_id=body.applied_preset_id,
    )
    return conv


@conversation_router.get(
    "/{conv_id}",
    summary="Get Conversation",
    description="Get a conversation with all its messages.",
)
async def get_conversation(conv_id: str, request: Request):
    conn = _get_db(request)
    conv = await db.get_conversation(conn, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@conversation_router.put(
    "/{conv_id}",
    summary="Update Conversation",
    description="Update conversation metadata (title, system prompt, model).",
)
async def update_conversation(conv_id: str, request: Request, body: ConversationUpdate):
    conn = _get_db(request)
    kwargs = {k: getattr(body, k) for k in body.model_fields_set if k in {"title", "model_id", "system_prompt", "params", "applied_preset_id"}}
    if not kwargs:
        raise HTTPException(status_code=400, detail="No fields to update")
    conv = await db.update_conversation(conn, conv_id, **kwargs)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@conversation_router.delete(
    "/{conv_id}",
    summary="Delete Conversation",
    description="Delete a conversation and all its messages.",
)
async def delete_conversation(conv_id: str, request: Request):
    _refuse_while_generating(conv_id)
    conn = _get_db(request)
    deleted = await db.delete_conversation(conn, conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conv_id}


# ---------------------------------------------------------------------------
# Message endpoints
# ---------------------------------------------------------------------------

@conversation_router.post(
    "/{conv_id}/messages",
    summary="Append Message",
    description="Append a message to a conversation.",
    status_code=201,
)
async def append_message(conv_id: str, request: Request, body: MessageCreate):
    _refuse_while_generating(conv_id)
    conn = _get_db(request)
    try:
        msg = await db.append_message(
            conn, conv_id, role=body.role, content=body.content, thinking=body.thinking
        )
    except ValueError as e:  # malformed content blocks
        raise HTTPException(status_code=400, detail=str(e))
    if msg is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return msg


@conversation_router.put(
    "/{conv_id}/messages/{msg_id}",
    summary="Update Message",
    description="Update a message's content or thinking.",
)
async def update_message(conv_id: str, msg_id: str, request: Request, body: MessageUpdate):
    _refuse_while_generating(conv_id)
    conn = _get_db(request)
    kwargs = {k: getattr(body, k) for k in body.model_fields_set if k in {"content", "thinking"}}
    try:
        msg = await db.update_message(conn, conv_id, msg_id, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return msg


@conversation_router.delete(
    "/{conv_id}/messages",
    summary="Truncate Messages",
    description="Delete all messages after the given position.",
)
async def truncate_messages(conv_id: str, request: Request, after: int):
    _refuse_while_generating(conv_id)
    conn = _get_db(request)
    count = await db.truncate_messages_after(conn, conv_id, after)
    return {"deleted": count, "after_position": after}


@conversation_router.delete(
    "/{conv_id}/messages/{msg_id}",
    summary="Delete Message",
    description="Delete exactly one message. Later messages keep their "
                "positions (gaps are fine -- ordering and appends are "
                "position-based, nothing assumes density).",
)
async def delete_message(conv_id: str, msg_id: str, request: Request):
    _refuse_while_generating(conv_id)
    conn = _get_db(request)
    if not await db.delete_message(conn, conv_id, msg_id):
        raise HTTPException(status_code=404, detail="Message not found")
    return {"deleted": 1, "id": msg_id}


@conversation_router.get(
    "/{conv_id}/media/{media_id}",
    summary="Get Media Blob",
    description="Serve one media blob (schema v7: media lives by reference; "
                "message rows carry url sources pointing here). "
                "Content-addressed, so the response is immutable and "
                "cacheable forever.",
)
async def get_media(conv_id: str, media_id: str, request: Request):
    conn = _get_db(request)
    blob = await db.get_media_blob(conn, conv_id, media_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="Media not found")
    media_type, data = blob
    # private (conversation media must not park in shared caches), nosniff
    # (the type is caller-derived; never let the browser second-guess it into
    # something scriptable), immutable (content-addressed -- safe forever).
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )
