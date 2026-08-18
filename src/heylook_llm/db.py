# src/heylook_llm/db.py
"""DuckDB store for conversations, notebooks, and presets (Q5 migration).

Messages persist as CONTENT BLOCK lists (Messages-style JSON) so image
conversations round-trip; reads expose both ``content`` (flattened text of
the text blocks -- the back-compatible wire shape) and ``content_blocks``
(the full list). String input normalizes to a single text block.

Concurrency by construction: DuckDB's Python API is synchronous, so every
operation runs in a worker thread (``asyncio.to_thread``) holding a store-wide
lock, on the store's single connection. Each logical operation is atomic --
there is no shared implicit-transaction state to bleed across interleaved
handlers (the aiosqlite defect class this migration retires).

DB path defaults to ``data/conversations.duckdb`` relative to the working
directory, overridable with the ``HEYLOOK_DB_PATH`` environment variable.
No migration from the retired SQLite store: this is a fresh start by design.
"""

import asyncio
import base64
import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import orjson

logger = logging.getLogger(__name__)

# v2 = DuckDB + content blocks; v3 = FK removed (DuckDB's FK check rejects
# deleting a parent even when its children are deleted in the same
# transaction -- a documented DuckDB limitation. Referential integrity is
# enforced in code: single writer, explicit cascade in delete_conversation).
# v7: media by reference (media_blobs table; base64 media in content_blocks is
# relocated to it at the write boundary, rows carry url sources) + per-message
# model_id (which model produced an assistant row -- metadata only, no content).
_SCHEMA_VERSION = 7

_UPDATABLE_MESSAGE_FIELDS: frozenset[str] = frozenset({"content", "thinking"})
_UPDATABLE_NOTEBOOK_FIELDS: frozenset[str] = frozenset(
    {"title", "content", "system_prompt", "model_id", "params", "applied_preset_id"}
)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Conversation',
    model_id    TEXT,
    system_prompt TEXT,
    params        TEXT NOT NULL DEFAULT '{}',
    applied_preset_id TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content_blocks  TEXT NOT NULL DEFAULT '[]',
    thinking        TEXT,
    model_id        TEXT,
    position        INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(conversation_id, position)
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_pos
    ON messages(conversation_id, position);

CREATE TABLE IF NOT EXISTS media_blobs (
    conversation_id TEXT NOT NULL,
    id              TEXT NOT NULL,
    media_type      TEXT NOT NULL,
    data            BLOB NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, id)
);

CREATE TABLE IF NOT EXISTS notebooks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT 'Untitled',
    content       TEXT NOT NULL DEFAULT '',
    system_prompt TEXT,
    model_id      TEXT,
    params        TEXT NOT NULL DEFAULT '{}',
    applied_preset_id TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    system_prompt TEXT,
    params        TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def normalize_blocks(content) -> list[dict]:
    """Normalize message content to a block list.

    Strings become a single text block; block lists pass through
    (shallow-copied). ``None`` behaves like the empty string.

    Raises ``ValueError`` for malformed blocks -- validation lives at this
    boundary so garbage can never persist and then crash every later read
    or render of the conversation.
    """
    if content is None:
        content = ""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for b in content:
        if not isinstance(b, dict) or not isinstance(b.get("type"), str):
            raise ValueError("Each content block must be an object with a string 'type'")
        b = dict(b)
        if b["type"] == "text":
            if b.get("text") is not None and not isinstance(b["text"], str):
                raise ValueError("Text block 'text' must be a string")
            b["text"] = b.get("text") or ""
        elif b["type"] == "image":
            src = b.get("source")
            if not isinstance(src, dict):
                raise ValueError("Image block requires a 'source' object")
            if src.get("type") == "base64":
                if not (isinstance(src.get("media_type"), str) and isinstance(src.get("data"), str) and src["data"]):
                    raise ValueError("base64 image source requires 'media_type' and non-empty 'data'")
            elif src.get("type") == "url":
                if not (isinstance(src.get("url"), str) and src["url"]):
                    raise ValueError("url image source requires a non-empty 'url'")
            else:
                raise ValueError("Image source 'type' must be 'base64' or 'url'")
        # unknown block types pass through untouched -- forward-compatible with
        # future Messages block types; flatten treats them as non-text.
        blocks.append(b)
    return blocks


def flatten_blocks(blocks: list[dict]) -> str:
    """Back-compatible text view: the text blocks joined by newlines."""
    return "\n".join((b.get("text") or "") for b in blocks if b.get("type") == "text")


# ---------------------------------------------------------------------------
# Media by reference (schema v7)
# ---------------------------------------------------------------------------
# Base64 media never persists inside content_blocks: every message WRITE
# relocates it into media_blobs (content-addressed by sha256, per
# conversation) and stores a url source pointing at the serve endpoint, with
# ``media_id`` as the internal marker. This is a relocation, not new storage
# -- the same user-attached bytes the messages table held before, moved so a
# conversation read is text-sized instead of shipping every image inline.
# Consumers: the browser fetches the url (immutable, cacheable); the generate
# saga resolves media_id blocks back to bytes at wire-build time.

def _externalize_media(conn, conv_id: str, blocks: list[dict], now: str) -> list[dict]:
    """Rewrite base64 media blocks to blob-backed url sources. Runs inside a
    write op (single writer thread, same transaction as the message row).

    A caller-supplied url source carrying ``media_id`` (a stored row round-
    tripped back through a write) is honored only if that blob exists in THIS
    conversation -- otherwise the marker is stripped and the block is treated
    as a plain external url, so a cross-conversation or dangling reference
    can never persist.
    """
    out = []
    for b in blocks:
        src = b.get("source")
        if b.get("type") not in ("image", "audio") or not isinstance(src, dict):
            out.append(b)
            continue
        if src.get("type") == "base64" and src.get("data"):
            try:
                raw = base64.b64decode(src["data"])
            except Exception:
                raise ValueError(f"{b['type']} block carries invalid base64 data")
            media_id = hashlib.sha256(raw).hexdigest()
            # The stored media_type is later SERVED verbatim as a same-origin
            # Content-Type -- constrain it to the block type's own family, or
            # a block claiming text/html becomes stored XSS at the serve
            # endpoint (review finding). Off-family claims degrade to
            # octet-stream (served with nosniff), never to trust.
            media_type = src.get("media_type") or ""
            if not media_type.startswith(f"{b['type']}/"):
                media_type = "application/octet-stream"
            conn.execute(
                "INSERT OR IGNORE INTO media_blobs (conversation_id, id, media_type, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, media_id, media_type, raw, now),
            )
            b = dict(b)
            b["source"] = {
                "type": "url",
                "url": f"/v1/conversations/{conv_id}/media/{media_id}",
                "media_type": src.get("media_type"),
                "media_id": media_id,
            }
        elif src.get("type") == "url" and src.get("media_id"):
            exists = conn.execute(
                "SELECT 1 FROM media_blobs WHERE conversation_id = ? AND id = ?",
                (conv_id, src["media_id"]),
            ).fetchone()
            if not exists:
                b = dict(b)
                stripped = dict(src)
                stripped.pop("media_id")
                b["source"] = stripped
        out.append(b)
    return out


def _gc_media(conn, conv_id: str) -> None:
    """Drop blobs no message in this conversation references any more.

    The reference check is a substring LIKE on the JSON text. Its only
    false-positive direction is RETENTION (a 64-hex blob id pasted into a
    text block keeps the blob alive); it can never delete a blob a media
    block still references. Do not "fix" this into a JSON parse -- the safe
    direction is the point.
    """
    conn.execute(
        "DELETE FROM media_blobs WHERE conversation_id = ? AND NOT EXISTS ("
        "  SELECT 1 FROM messages m WHERE m.conversation_id = media_blobs.conversation_id"
        "  AND m.content_blocks LIKE '%' || media_blobs.id || '%')",
        (conv_id,),
    )


async def get_media_blob(db: Store, conv_id: str, media_id: str) -> tuple[str, bytes] | None:
    """Return (media_type, bytes) for one blob, or None."""
    def op(conn):
        row = conn.execute(
            "SELECT media_type, data FROM media_blobs WHERE conversation_id = ? AND id = ?",
            (conv_id, media_id),
        ).fetchone()
        return (row[0], bytes(row[1])) if row else None
    return await db.run(op)


async def get_media_blobs(db: Store, conv_id: str, media_ids: list[str]) -> dict[str, tuple[str, bytes]]:
    """Batch fetch for the generate saga's wire build: id -> (media_type, bytes)."""
    if not media_ids:
        return {}
    def op(conn):
        placeholders = ", ".join("?" for _ in media_ids)
        rows = conn.execute(
            f"SELECT id, media_type, data FROM media_blobs WHERE conversation_id = ? AND id IN ({placeholders})",
            (conv_id, *media_ids),
        ).fetchall()
        return {r[0]: (r[1], bytes(r[2])) for r in rows}
    return await db.run(op)


def _message_row_to_dict(names: list[str], row) -> dict:
    d = dict(zip(names, row))
    blocks = orjson.loads(d.pop("content_blocks"))
    d["content"] = flatten_blocks(blocks)
    d["content_blocks"] = blocks
    return d


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    env = os.environ.get("HEYLOOK_DB_PATH")
    if env:
        return Path(env)
    return Path("data") / "conversations.duckdb"


class Store:
    """Single-connection DuckDB store on its own dedicated worker thread.

    A max_workers=1 executor gives strict serialization (stronger than a
    lock: queued ops don't pile up blocking pooled threads) and keeps DB ops
    off asyncio's shared default executor, where long-running model loads and
    generation consumption would otherwise contend with trivial reads.

    Every operation runs inside an explicit transaction with rollback on
    exception -- DuckDB autocommits per statement, so without this a crash
    between the statements of one logical op leaves partial state, and an
    error mid-transaction would wedge the long-lived connection until
    ROLLBACK.
    """

    _CONNECT_RETRY_S = 10.0  # old aiosqlite had timeout=10; retry the file lock

    def __init__(self, resolved: str):
        deadline = time.monotonic() + self._CONNECT_RETRY_S
        while True:
            try:
                self._conn = duckdb.connect(resolved)
                break
            except duckdb.IOException:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duckdb-store")
        self._create_schema()
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        elif row[0] != str(_SCHEMA_VERSION):
            # Pre-release DuckDB schema (same-day churn only; the store is a
            # fresh start by design). Recreate rather than migrate. `presets`
            # is deliberately NOT in the drop list: it's versionless config
            # (additive CREATE TABLE, no FK to versioned tables) promised to
            # survive destructive operations -- a presets schema change needs
            # its own explicit migration, not this hammer.
            logger.warning(
                "DuckDB schema v%s != v%d -- recreating (fresh-start store)",
                row[0], _SCHEMA_VERSION,
            )
            for table in ("messages", "media_blobs", "conversations", "notebooks", "schema_meta"):
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._create_schema()
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(_SCHEMA_VERSION),),
            )

    def _create_schema(self):
        for stmt in _SCHEMA_SQL.split(";\n\n"):
            if stmt.strip():
                self._conn.execute(stmt)

    async def run(self, fn, *args):
        """Run a sync store operation, transactionally, on the store thread."""
        def op():
            self._conn.execute("BEGIN TRANSACTION")
            try:
                result = fn(self._conn, *args)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")
            return result
        return await asyncio.get_running_loop().run_in_executor(self._executor, op)

    async def close(self):
        await asyncio.get_running_loop().run_in_executor(self._executor, self._conn.close)
        self._executor.shutdown(wait=True)


async def get_connection(path: Path | str | None = None) -> Store:
    """Open (or create) the database and return the store.

    Caller is responsible for closing the store. ``:memory:`` is supported.
    """
    if isinstance(path, str) and path == ":memory:":
        resolved = ":memory:"
    else:
        resolved_path = Path(path) if isinstance(path, str) else (path or _default_db_path())
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(resolved_path)

    store = await asyncio.to_thread(Store, resolved)
    logger.info("Database ready at %s (schema v%d, DuckDB)", resolved, _SCHEMA_VERSION)
    return store


def get_db(request):
    """Get the shared store from app state. For use in FastAPI route handlers."""
    from fastapi import HTTPException
    conn = getattr(request.app.state, "db", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


async def clear_all_data(db: Store) -> dict:
    """Delete all conversations, messages, and notebooks. Returns counts.

    Presets deliberately survive -- they're configuration, not data.
    """
    def op(conn):
        conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        nb_count = conn.execute("SELECT COUNT(*) FROM notebooks").fetchone()[0]
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM media_blobs")
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM notebooks")
        return {"conversations_deleted": conv_count, "notebooks_deleted": nb_count}
    return await db.run(op)


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

# Single source of truth per table: the SELECT string derives from the names
# list, so the two can never drift (zip would silently mispair otherwise).
_CONV_NAMES = ["id", "title", "model_id", "system_prompt", "params", "applied_preset_id", "created_at", "updated_at"]
_MSG_NAMES = ["id", "role", "content_blocks", "thinking", "model_id", "position", "created_at", "updated_at"]
_CONV_COLS = ", ".join(_CONV_NAMES)
_MSG_COLS = ", ".join(_MSG_NAMES)


# Per-document sampler settings (`params`) are stored as a JSON TEXT blob, like
# `presets.params`. ONE decode/encode pair is shared by conversations AND
# notebooks so the two never branch into separate copies of the same logic.

def _decode_params(d: dict) -> dict:
    """JSON-decode the ``params`` blob on a conversation/notebook row dict."""
    try:
        d["params"] = orjson.loads(d["params"]) if d.get("params") else {}
    except (orjson.JSONDecodeError, TypeError):
        d["params"] = {}
    return d


def _encode_params(params) -> str:
    """Validate + JSON-encode a document's ``params`` at the write boundary."""
    if not isinstance(params, dict):
        raise ValueError("'params' must be an object")
    try:
        return orjson.dumps(params).decode()
    except TypeError as e:
        raise ValueError(f"'params' is not JSON-serializable: {e}")


def _conv_row_to_dict(row) -> dict:
    return _decode_params(dict(zip(_CONV_NAMES, row)))


def _touch_conversation(conn, conv_id: str, now: str) -> None:
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))


async def list_conversations(db: Store) -> list[dict]:
    """Return all conversations ordered by updated_at desc."""
    def op(conn):
        rows = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [_conv_row_to_dict(r) for r in rows]
    return await db.run(op)


async def get_conversation(db: Store, conv_id: str) -> dict | None:
    """Return a conversation with its messages, or None."""
    def op(conn):
        row = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row is None:
            return None
        conv = _conv_row_to_dict(row)
        msgs = conn.execute(
            f"SELECT {_MSG_COLS} FROM messages WHERE conversation_id = ? ORDER BY position",
            (conv_id,),
        ).fetchall()
        conv["messages"] = [_message_row_to_dict(_MSG_NAMES, m) for m in msgs]
        return conv
    return await db.run(op)


async def create_conversation(
    db: Store,
    *,
    title: str = "New Conversation",
    model_id: str | None = None,
    system_prompt: str | None = None,
    params: dict | None = None,
    applied_preset_id: str | None = None,
) -> dict:
    """Create a new conversation and return it.

    ``applied_preset_id`` at creation is the new-document preset inheritance:
    the document explicitly STARTS as that preset, which is an apply, so the
    stamp is legitimate under the explicit-stamps-only rule. A dangling id
    stays harmless -- readers resolve stamps against the live preset list.
    """
    conv_id = new_id()
    now = _now_iso()
    params = params or {}
    params_json = _encode_params(params)
    def op(conn):
        conn.execute(
            "INSERT INTO conversations (id, title, model_id, system_prompt, params, applied_preset_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conv_id, title, model_id, system_prompt, params_json, applied_preset_id, now, now),
        )
    await db.run(op)
    return {
        "id": conv_id,
        "title": title,
        "model_id": model_id,
        "system_prompt": system_prompt,
        "params": params,
        "applied_preset_id": applied_preset_id,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


async def update_conversation(
    db: Store,
    conv_id: str,
    **fields: str | None,
) -> dict | None:
    """Update mutable conversation fields. Returns updated conversation or None.

    Pass only the fields to change. Supports explicit ``None`` to clear nullable
    columns (model_id, system_prompt). Allowed fields: title, model_id,
    system_prompt, params (a JSON object of per-conversation sampler settings),
    applied_preset_id (which preset this conversation is running; explicit
    stamp only -- see the preset-bar provenance note in the v3 spec).
    """
    allowed = {"title", "model_id", "system_prompt", "params", "applied_preset_id"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return None

    # params is stored as a JSON blob -- validate/encode for SQL, keep the decoded
    # dict for the returned object.
    sql_updates = dict(updates)
    if "params" in sql_updates:
        sql_updates["params"] = _encode_params(sql_updates["params"])

    now = _now_iso()
    def op(conn):
        row = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row is None:
            return None
        existing = _conv_row_to_dict(row)
        set_clause = ", ".join(f"{k}=?" for k in sql_updates)
        values = list(sql_updates.values()) + [now, conv_id]
        conn.execute(
            f"UPDATE conversations SET {set_clause}, updated_at=? WHERE id=?", values
        )
        existing.update(**updates, updated_at=now)  # `updates` has the DECODED params
        return existing
    return await db.run(op)


async def delete_conversation(db: Store, conv_id: str) -> bool:
    """Delete a conversation and its messages. Returns True if it existed.

    DuckDB has no ON DELETE CASCADE -- messages are deleted explicitly first.
    """
    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone() is not None
        if not exists:
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM media_blobs WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        return True
    return await db.run(op)


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

async def append_message(
    db: Store,
    conv_id: str,
    *,
    role: str,
    content: str | list[dict] = "",
    thinking: str | None = None,
    model_id: str | None = None,
) -> dict | None:
    """Append a message to a conversation. Returns the message or None if conv not found.

    ``content`` may be a plain string (stored as one text block) or a
    content-block list. Base64 media blocks are relocated to media_blobs and
    stored as url sources (_externalize_media). ``model_id`` records which
    model produced the row (assistant rows only by convention; None for
    user turns).
    """
    blocks = normalize_blocks(content)
    msg_id = new_id()
    now = _now_iso()

    def op(conn):
        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone() is None:
            return None
        stored = _externalize_media(conn, conv_id, blocks, now)
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content_blocks, thinking, model_id, position, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, orjson.dumps(stored).decode(), thinking, model_id, position, now, now),
        )
        _touch_conversation(conn, conv_id, now)
        return position, stored

    result = await db.run(op)
    if result is None:
        return None
    position, stored = result
    return {
        "id": msg_id,
        "role": role,
        "content": flatten_blocks(stored),
        "content_blocks": stored,
        "thinking": thinking,
        "model_id": model_id,
        "position": position,
        "created_at": now,
        "updated_at": now,
    }


async def update_message(
    db: Store,
    conv_id: str,
    msg_id: str,
    **fields,
) -> dict | None:
    """Update a message's content and/or thinking. Returns updated message or None.

    ``content`` accepts a string or a content-block list. Raises ``ValueError``
    if no recognized fields are provided.
    """
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_MESSAGE_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_MESSAGE_FIELDS)})")

    now = _now_iso()
    col_updates = dict(updates)
    # Validation (normalize_blocks raises on garbage) happens BEFORE the op;
    # media externalization needs the connection, so it happens inside it.
    new_blocks = normalize_blocks(col_updates.pop("content")) if "content" in col_updates else None

    def op(conn):
        row = conn.execute(
            f"SELECT {_MSG_COLS} FROM messages WHERE id = ? AND conversation_id = ?",
            (msg_id, conv_id),
        ).fetchone()
        if row is None:
            return None
        if new_blocks is not None:
            stored = _externalize_media(conn, conv_id, new_blocks, now)
            col_updates["content_blocks"] = orjson.dumps(stored).decode()
        set_clause = ", ".join(f"{k}=?" for k in col_updates)
        values = list(col_updates.values()) + [now, msg_id]
        conn.execute(f"UPDATE messages SET {set_clause}, updated_at=? WHERE id=?", values)
        _touch_conversation(conn, conv_id, now)
        if new_blocks is not None:
            _gc_media(conn, conv_id)  # an edit can drop the last reference to a blob
        # Merge locally instead of re-SELECTing -- content_blocks can carry
        # multi-MB base64 images; one fetch is enough.
        raw = dict(zip(_MSG_NAMES, row))
        raw.update(col_updates, updated_at=now)
        return _message_row_to_dict(_MSG_NAMES, [raw[k] for k in _MSG_NAMES])
    return await db.run(op)


async def truncate_messages_after(
    db: Store,
    conv_id: str,
    after_position: int,
) -> int:
    """Delete all messages with position > after_position. Returns count deleted."""
    now = _now_iso()
    def op(conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND position > ?",
            (conv_id, after_position),
        ).fetchone()[0]
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND position > ?",
            (conv_id, after_position),
        )
        _touch_conversation(conn, conv_id, now)
        _gc_media(conn, conv_id)
        return count
    return await db.run(op)


async def delete_message(db: Store, conv_id: str, msg_id: str) -> bool:
    """Delete exactly one message. Returns True if it existed.

    Later rows keep their positions (gaps are fine: ordering and MAX+1
    appends are position-based, nothing assumes density). Blobs the deleted
    row was the last reference to are collected.
    """
    now = _now_iso()
    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM messages WHERE id = ? AND conversation_id = ?",
            (msg_id, conv_id),
        ).fetchone() is not None
        if not exists:
            return False
        conn.execute(
            "DELETE FROM messages WHERE id = ? AND conversation_id = ?",
            (msg_id, conv_id),
        )
        _touch_conversation(conn, conv_id, now)
        _gc_media(conn, conv_id)
        return True
    return await db.run(op)


# ---------------------------------------------------------------------------
# Generation commits (conversation-scoped generate endpoint)
# ---------------------------------------------------------------------------
# The server-side generation saga commits its DESTRUCTIVE truncation only
# together with the row it produced, in one transaction -- a failed or empty
# generation leaves the conversation untouched. This is the invariant the
# client-orchestrated flow could never offer (its DELETE ?after and POST were
# separate HTTP calls, and a failure between them stranded a truncated
# thread). Holds no prompt/response beyond what the messages table already
# stores by design.


async def replace_tail_with_message(
    db: Store,
    conv_id: str,
    after_position: int,
    *,
    role: str,
    content: str | list[dict] = "",
    thinking: str | None = None,
    model_id: str | None = None,
) -> dict | None:
    """Atomically delete position > after_position and append one message.

    Returns the appended message, or None if the conversation is gone.
    With after_position = the current last position this is a plain atomic
    append (nothing to truncate). ``model_id`` stamps which model produced
    the row -- unambiguous here (fresh row, one producer).
    """
    blocks = normalize_blocks(content)
    msg_id = new_id()
    now = _now_iso()

    def op(conn):
        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone() is None:
            return None
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND position > ?",
            (conv_id, after_position),
        )
        stored = _externalize_media(conn, conv_id, blocks, now)
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content_blocks, thinking, model_id, position, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, orjson.dumps(stored).decode(), thinking, model_id, position, now, now),
        )
        _touch_conversation(conn, conv_id, now)
        _gc_media(conn, conv_id)  # the truncated tail may have held the last references
        return position, stored

    result = await db.run(op)
    if result is None:
        return None
    position, stored = result
    return {
        "id": msg_id,
        "role": role,
        "content": flatten_blocks(stored),
        "content_blocks": stored,
        "thinking": thinking,
        "model_id": model_id,
        "position": position,
        "created_at": now,
        "updated_at": now,
    }


async def replace_tail_with_update(
    db: Store,
    conv_id: str,
    after_position: int,
    msg_id: str,
    *,
    content: str | list[dict],
    thinking: str | None = None,
) -> dict | None:
    """Atomically delete position > after_position and update the anchor row.

    The continuation commit: the anchor (at after_position) absorbs the
    combined text; everything after it goes in the same transaction.
    ``thinking`` is written only when not None (a continuation never clears
    prior thinking). Returns the updated message, or None if it is gone.
    The row's model_id is deliberately NOT restamped: a continuation merges
    onto a row possibly co-written by another model, and any single stamp
    would misattribute half of it -- the original stamp stands.
    """
    blocks = normalize_blocks(content)
    now = _now_iso()

    def op(conn):
        row = conn.execute(
            f"SELECT {_MSG_COLS} FROM messages WHERE id = ? AND conversation_id = ?",
            (msg_id, conv_id),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND position > ?",
            (conv_id, after_position),
        )
        stored = _externalize_media(conn, conv_id, blocks, now)
        col_updates: dict = {"content_blocks": orjson.dumps(stored).decode()}
        if thinking is not None:
            col_updates["thinking"] = thinking
        set_clause = ", ".join(f"{k}=?" for k in col_updates)
        conn.execute(
            f"UPDATE messages SET {set_clause}, updated_at=? WHERE id=?",
            list(col_updates.values()) + [now, msg_id],
        )
        _touch_conversation(conn, conv_id, now)
        _gc_media(conn, conv_id)
        raw = dict(zip(_MSG_NAMES, row))
        raw.update(col_updates, updated_at=now)
        return _message_row_to_dict(_MSG_NAMES, [raw[k] for k in _MSG_NAMES])

    return await db.run(op)


# ---------------------------------------------------------------------------
# Notebook CRUD
# ---------------------------------------------------------------------------

_NB_NAMES = ["id", "title", "content", "system_prompt", "model_id", "params", "applied_preset_id", "created_at", "updated_at"]
_NB_LIST_NAMES = ["id", "title", "system_prompt", "model_id", "applied_preset_id", "created_at", "updated_at"]
_NB_COLS = ", ".join(_NB_NAMES)
_NB_LIST_COLS = ", ".join(_NB_LIST_NAMES)


async def list_notebooks(db: Store) -> list[dict]:
    """Return all notebooks ordered by updated_at desc. Excludes content for efficiency."""
    def op(conn):
        rows = conn.execute(
            f"SELECT {_NB_LIST_COLS} FROM notebooks ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(zip(_NB_LIST_NAMES, r)) for r in rows]
    return await db.run(op)


async def get_notebook(db: Store, notebook_id: str) -> dict | None:
    """Return a notebook or None."""
    def op(conn):
        row = conn.execute(
            f"SELECT {_NB_COLS} FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        return _decode_params(dict(zip(_NB_NAMES, row))) if row else None
    return await db.run(op)


async def create_notebook(
    db: Store,
    *,
    title: str = "Untitled",
    content: str = "",
    system_prompt: str | None = None,
    model_id: str | None = None,
    params: dict | None = None,
    applied_preset_id: str | None = None,
) -> dict:
    """Create a new notebook and return it.

    ``applied_preset_id`` at creation: same new-document preset inheritance
    contract as ``create_conversation``."""
    nb_id = new_id()
    now = _now_iso()
    params = params or {}
    params_json = _encode_params(params)
    def op(conn):
        conn.execute(
            "INSERT INTO notebooks (id, title, content, system_prompt, model_id, params, applied_preset_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (nb_id, title, content, system_prompt, model_id, params_json, applied_preset_id, now, now),
        )
    await db.run(op)
    return {
        "id": nb_id, "title": title, "content": content,
        "system_prompt": system_prompt, "model_id": model_id, "params": params,
        "applied_preset_id": applied_preset_id, "created_at": now, "updated_at": now,
    }


async def update_notebook(
    db: Store,
    notebook_id: str,
    **fields: str | None,
) -> dict | None:
    """Update notebook fields. Returns updated notebook or None if not found.

    Allowed fields include ``params`` (per-notebook sampler settings, a JSON
    object -- same shape + shared encode/decode as conversations)."""
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_NOTEBOOK_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_NOTEBOOK_FIELDS)})")

    sql_updates = dict(updates)
    if "params" in sql_updates:
        sql_updates["params"] = _encode_params(sql_updates["params"])

    now = _now_iso()
    def op(conn):
        row = conn.execute(
            f"SELECT {_NB_COLS} FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        if row is None:
            return None
        existing = _decode_params(dict(zip(_NB_NAMES, row)))
        set_clause = ", ".join(f"{k}=?" for k in sql_updates)
        values = list(sql_updates.values()) + [now, notebook_id]
        conn.execute(f"UPDATE notebooks SET {set_clause}, updated_at=? WHERE id=?", values)
        existing.update(**updates, updated_at=now)  # `updates` has the DECODED params
        return existing
    return await db.run(op)


async def delete_notebook(db: Store, notebook_id: str) -> bool:
    """Delete a notebook. Returns True if it existed."""
    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone() is not None
        if exists:
            conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        return exists
    return await db.run(op)


# ---------------------------------------------------------------------------
# Preset CRUD
# ---------------------------------------------------------------------------
# User presets: named system_prompt + sampler-params bundles authored from the
# UI. Distinct from the bundled TOML sampler registry (presets.py), which is
# server-side and request-scoped via ``ChatRequest.preset`` -- these are
# expanded client-side into explicit request fields. Name uniqueness is
# enforced in code, not a constraint: the single serialized writer makes the
# check race-free (same rationale as the dropped messages FK). Deliberately
# NOT touched by clear_all_data -- presets are configuration, not data.

class PresetNameTaken(ValueError):
    """Raised when a preset name is already in use."""


_PRESET_NAMES = ["id", "name", "system_prompt", "params", "created_at", "updated_at"]
_PRESET_COLS = ", ".join(_PRESET_NAMES)
_UPDATABLE_PRESET_FIELDS: frozenset[str] = frozenset({"name", "system_prompt", "params"})


def _validate_preset_fields(fields: dict) -> dict:
    """Normalize name/params in place-of-write; ValueError at the boundary so
    garbage never persists (same policy as normalize_blocks)."""
    out = dict(fields)
    if "name" in out:
        if not isinstance(out["name"], str) or not out["name"].strip():
            raise ValueError("Preset 'name' must be a non-empty string")
        out["name"] = out["name"].strip()
    if "params" in out:
        if not isinstance(out["params"], dict):
            raise ValueError("Preset 'params' must be an object")
        try:
            orjson.dumps(out["params"])
        except TypeError as e:  # e.g. int beyond orjson's 64-bit range
            raise ValueError(f"Preset 'params' is not JSON-serializable: {e}")
    return out


def _preset_row_to_dict(row) -> dict:
    d = dict(zip(_PRESET_NAMES, row))
    d["params"] = orjson.loads(d["params"])
    return d


def _preset_name_taken(conn, name: str, *, exclude_id: str | None = None) -> bool:
    row = conn.execute(
        "SELECT id FROM presets WHERE name = ?", (name,)
    ).fetchone()
    return row is not None and row[0] != exclude_id


async def list_presets(db: Store) -> list[dict]:
    """Return all presets ordered by name."""
    def op(conn):
        rows = conn.execute(
            f"SELECT {_PRESET_COLS} FROM presets ORDER BY name"
        ).fetchall()
        return [_preset_row_to_dict(r) for r in rows]
    return await db.run(op)


async def create_preset(
    db: Store,
    *,
    name: str,
    system_prompt: str | None = None,
    params: dict | None = None,
) -> dict:
    """Create a preset. Raises PresetNameTaken if the name is in use."""
    fields = _validate_preset_fields({"name": name, "params": params or {}})
    preset_id = new_id()
    now = _now_iso()

    def op(conn):
        if _preset_name_taken(conn, fields["name"]):
            raise PresetNameTaken(f"Preset name already exists: {fields['name']}")
        conn.execute(
            "INSERT INTO presets (id, name, system_prompt, params, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (preset_id, fields["name"], system_prompt,
             orjson.dumps(fields["params"]).decode(), now, now),
        )
    await db.run(op)
    return {
        "id": preset_id,
        "name": fields["name"],
        "system_prompt": system_prompt,
        "params": fields["params"],
        "created_at": now,
        "updated_at": now,
    }


async def update_preset(db: Store, preset_id: str, **fields) -> dict | None:
    """Update preset fields. Returns the updated preset, or None if not found.

    Raises ValueError on no/invalid fields, PresetNameTaken on a name collision.
    """
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_PRESET_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_PRESET_FIELDS)})")
    updates = _validate_preset_fields(updates)
    if "params" in updates:
        updates["params"] = orjson.dumps(updates["params"]).decode()

    now = _now_iso()
    def op(conn):
        row = conn.execute(
            f"SELECT {_PRESET_COLS} FROM presets WHERE id = ?", (preset_id,)
        ).fetchone()
        if row is None:
            return None
        if "name" in updates and _preset_name_taken(conn, updates["name"], exclude_id=preset_id):
            raise PresetNameTaken(f"Preset name already exists: {updates['name']}")
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [now, preset_id]
        conn.execute(f"UPDATE presets SET {set_clause}, updated_at=? WHERE id=?", values)
        existing = dict(zip(_PRESET_NAMES, row))
        existing.update(**updates, updated_at=now)
        return _preset_row_to_dict([existing[k] for k in _PRESET_NAMES])
    return await db.run(op)


async def delete_preset(db: Store, preset_id: str) -> bool:
    """Delete a preset. Returns True if it existed."""
    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM presets WHERE id = ?", (preset_id,)
        ).fetchone() is not None
        if exists:
            conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        return exists
    return await db.run(op)


# ---------------------------------------------------------------------------
# Settings key->value store
# ---------------------------------------------------------------------------
# Operational settings (obs level/retention, etc.) edited via /v1/admin/config,
# persisted alongside presets. key -> JSON value. Schema-stable: a new setting is
# a new ROW, never a DDL change, so the table survives the drop/recreate schema
# policy without a drop-list carve-out (same posture as presets -- config, not
# data; NOT touched by clear_all_data). The single serialized writer makes the
# check-then-write upsert race-free (same rationale as presets' name check).

async def get_setting(db: Store, key: str) -> Any | None:
    """Return the JSON-decoded value for ``key``, or None if unset."""
    def op(conn):
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return orjson.loads(row[0]) if row is not None else None
    return await db.run(op)


async def get_all_settings(db: Store) -> dict:
    """Return all stored settings as a ``{key: decoded_value}`` map."""
    def op(conn):
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {k: orjson.loads(v) for k, v in rows}
    return await db.run(op)


async def set_setting(db: Store, key: str, value: Any) -> None:
    """Upsert one setting. ValueError at the boundary so garbage never persists."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Setting 'key' must be a non-empty string")
    try:
        encoded = orjson.dumps(value).decode()
    except TypeError as e:  # orjson.JSONEncodeError subclasses TypeError
        raise ValueError(f"Setting value is not JSON-serializable: {e}")
    now = _now_iso()

    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM settings WHERE key = ?", (key,)
        ).fetchone() is not None
        if exists:
            conn.execute(
                "UPDATE settings SET value = ?, updated_at = ? WHERE key = ?",
                (encoded, now, key),
            )
        else:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, encoded, now),
            )
    await db.run(op)


async def delete_setting(db: Store, key: str) -> bool:
    """Delete a setting (resets it to its schema default). True if it existed."""
    def op(conn):
        exists = conn.execute(
            "SELECT 1 FROM settings WHERE key = ?", (key,)
        ).fetchone() is not None
        if exists:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        return exists
    return await db.run(op)
