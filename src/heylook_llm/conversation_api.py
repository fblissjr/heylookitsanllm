# src/heylook_llm/conversation_api.py
"""Conversation storage API endpoints.

CRUD for conversations and their messages, backed by SQLite.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from heylook_llm import db
from heylook_llm.db import get_db as _get_db

logger = logging.getLogger(__name__)

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


class ConversationUpdate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None


class MessageCreate(BaseModel):
    role: str
    content: str = ""
    thinking: str | None = None


class MessageUpdate(BaseModel):
    content: str | None = None
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
        conn, title=body.title, model_id=body.model_id, system_prompt=body.system_prompt
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
    kwargs = {k: getattr(body, k) for k in body.model_fields_set if k in {"title", "model_id", "system_prompt"}}
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
    conn = _get_db(request)
    msg = await db.append_message(
        conn, conv_id, role=body.role, content=body.content, thinking=body.thinking
    )
    if msg is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return msg


@conversation_router.put(
    "/{conv_id}/messages/{msg_id}",
    summary="Update Message",
    description="Update a message's content or thinking.",
)
async def update_message(conv_id: str, msg_id: str, request: Request, body: MessageUpdate):
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
    conn = _get_db(request)
    count = await db.truncate_messages_after(conn, conv_id, after)
    return {"deleted": count, "after_position": after}
