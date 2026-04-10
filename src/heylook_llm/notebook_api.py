# src/heylook_llm/notebook_api.py
"""Notebook storage API endpoints.

CRUD for notebooks (simple text documents with optional LLM generation).
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from heylook_llm import db
from heylook_llm.db import get_db as _get_db

logger = logging.getLogger(__name__)

notebook_router = APIRouter(
    prefix="/v1/notebooks",
    tags=["Notebooks"],
)


class NotebookCreate(BaseModel):
    title: str = "Untitled"
    content: str = ""
    system_prompt: str | None = None
    model_id: str | None = None


class NotebookUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    system_prompt: str | None = None
    model_id: str | None = None


@notebook_router.get(
    "",
    summary="List Notebooks",
    description="List all notebooks, ordered by most recently updated.",
)
async def list_notebooks(request: Request):
    conn = _get_db(request)
    notebooks = await db.list_notebooks(conn)
    return {"notebooks": notebooks, "total": len(notebooks)}


@notebook_router.post(
    "",
    summary="Create Notebook",
    description="Create a new notebook.",
    status_code=201,
)
async def create_notebook(request: Request, body: NotebookCreate):
    conn = _get_db(request)
    nb = await db.create_notebook(
        conn, title=body.title, content=body.content,
        system_prompt=body.system_prompt, model_id=body.model_id,
    )
    return nb


@notebook_router.get(
    "/{notebook_id}",
    summary="Get Notebook",
    description="Get a notebook by ID.",
)
async def get_notebook(notebook_id: str, request: Request):
    conn = _get_db(request)
    nb = await db.get_notebook(conn, notebook_id)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return nb


@notebook_router.put(
    "/{notebook_id}",
    summary="Update Notebook",
    description="Update notebook fields (title, content, system_prompt, model_id).",
)
async def update_notebook(notebook_id: str, request: Request, body: NotebookUpdate):
    conn = _get_db(request)
    kwargs = {k: getattr(body, k) for k in body.model_fields_set if k in {"title", "content", "system_prompt", "model_id"}}
    if not kwargs:
        raise HTTPException(status_code=400, detail="No fields to update")
    nb = await db.update_notebook(conn, notebook_id, **kwargs)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return nb


@notebook_router.delete(
    "/{notebook_id}",
    summary="Delete Notebook",
    description="Delete a notebook.",
)
async def delete_notebook(notebook_id: str, request: Request):
    conn = _get_db(request)
    deleted = await db.delete_notebook(conn, notebook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"status": "deleted", "id": notebook_id}
