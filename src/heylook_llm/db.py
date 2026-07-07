# src/heylook_llm/db.py
"""DuckDB store for conversations and notebooks (Q5 migration).

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
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import orjson

logger = logging.getLogger(__name__)

# v2 = DuckDB + content blocks; v3 = FK removed (DuckDB's FK check rejects
# deleting a parent even when its children are deleted in the same
# transaction -- a documented DuckDB limitation. Referential integrity is
# enforced in code: single writer, explicit cascade in delete_conversation).
_SCHEMA_VERSION = 3

_UPDATABLE_MESSAGE_FIELDS: frozenset[str] = frozenset({"content", "thinking"})
_UPDATABLE_NOTEBOOK_FIELDS: frozenset[str] = frozenset({"title", "content", "system_prompt", "model_id"})

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Conversation',
    model_id    TEXT,
    system_prompt TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content_blocks  TEXT NOT NULL DEFAULT '[]',
    thinking        TEXT,
    position        INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(conversation_id, position)
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_pos
    ON messages(conversation_id, position);

CREATE TABLE IF NOT EXISTS notebooks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT 'Untitled',
    content       TEXT NOT NULL DEFAULT '',
    system_prompt TEXT,
    model_id      TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
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
            # fresh start by design). Recreate rather than migrate.
            logger.warning(
                "DuckDB schema v%s != v%d -- recreating (fresh-start store)",
                row[0], _SCHEMA_VERSION,
            )
            for table in ("messages", "conversations", "notebooks", "schema_meta"):
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
    """Delete all conversations, messages, and notebooks. Returns counts."""
    def op(conn):
        conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        nb_count = conn.execute("SELECT COUNT(*) FROM notebooks").fetchone()[0]
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM notebooks")
        return {"conversations_deleted": conv_count, "notebooks_deleted": nb_count}
    return await db.run(op)


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

# Single source of truth per table: the SELECT string derives from the names
# list, so the two can never drift (zip would silently mispair otherwise).
_CONV_NAMES = ["id", "title", "model_id", "system_prompt", "created_at", "updated_at"]
_MSG_NAMES = ["id", "role", "content_blocks", "thinking", "position", "created_at", "updated_at"]
_CONV_COLS = ", ".join(_CONV_NAMES)
_MSG_COLS = ", ".join(_MSG_NAMES)


def _touch_conversation(conn, conv_id: str, now: str) -> None:
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))


async def list_conversations(db: Store) -> list[dict]:
    """Return all conversations ordered by updated_at desc."""
    def op(conn):
        rows = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(zip(_CONV_NAMES, r)) for r in rows]
    return await db.run(op)


async def get_conversation(db: Store, conv_id: str) -> dict | None:
    """Return a conversation with its messages, or None."""
    def op(conn):
        row = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row is None:
            return None
        conv = dict(zip(_CONV_NAMES, row))
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
) -> dict:
    """Create a new conversation and return it."""
    conv_id = new_id()
    now = _now_iso()
    def op(conn):
        conn.execute(
            "INSERT INTO conversations (id, title, model_id, system_prompt, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title, model_id, system_prompt, now, now),
        )
    await db.run(op)
    return {
        "id": conv_id,
        "title": title,
        "model_id": model_id,
        "system_prompt": system_prompt,
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
    columns (model_id, system_prompt). Allowed fields: title, model_id, system_prompt.
    """
    allowed = {"title", "model_id", "system_prompt"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return None

    now = _now_iso()
    def op(conn):
        row = conn.execute(
            f"SELECT {_CONV_COLS} FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row is None:
            return None
        existing = dict(zip(_CONV_NAMES, row))
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [now, conv_id]
        conn.execute(
            f"UPDATE conversations SET {set_clause}, updated_at=? WHERE id=?", values
        )
        existing.update(**updates, updated_at=now)
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
) -> dict | None:
    """Append a message to a conversation. Returns the message or None if conv not found.

    ``content`` may be a plain string (stored as one text block) or a
    content-block list (stored verbatim).
    """
    blocks = normalize_blocks(content)
    msg_id = new_id()
    now = _now_iso()

    def op(conn):
        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone() is None:
            return None
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content_blocks, thinking, position, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, orjson.dumps(blocks).decode(), thinking, position, now, now),
        )
        _touch_conversation(conn, conv_id, now)
        return position

    position = await db.run(op)
    if position is None:
        return None
    return {
        "id": msg_id,
        "role": role,
        "content": flatten_blocks(blocks),
        "content_blocks": blocks,
        "thinking": thinking,
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
    if "content" in col_updates:
        col_updates["content_blocks"] = orjson.dumps(normalize_blocks(col_updates.pop("content"))).decode()

    def op(conn):
        row = conn.execute(
            f"SELECT {_MSG_COLS} FROM messages WHERE id = ? AND conversation_id = ?",
            (msg_id, conv_id),
        ).fetchone()
        if row is None:
            return None
        set_clause = ", ".join(f"{k}=?" for k in col_updates)
        values = list(col_updates.values()) + [now, msg_id]
        conn.execute(f"UPDATE messages SET {set_clause}, updated_at=? WHERE id=?", values)
        _touch_conversation(conn, conv_id, now)
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
        return count
    return await db.run(op)


# ---------------------------------------------------------------------------
# Notebook CRUD
# ---------------------------------------------------------------------------

_NB_NAMES = ["id", "title", "content", "system_prompt", "model_id", "created_at", "updated_at"]
_NB_LIST_NAMES = ["id", "title", "system_prompt", "model_id", "created_at", "updated_at"]
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
        return dict(zip(_NB_NAMES, row)) if row else None
    return await db.run(op)


async def create_notebook(
    db: Store,
    *,
    title: str = "Untitled",
    content: str = "",
    system_prompt: str | None = None,
    model_id: str | None = None,
) -> dict:
    """Create a new notebook and return it."""
    nb_id = new_id()
    now = _now_iso()
    def op(conn):
        conn.execute(
            "INSERT INTO notebooks (id, title, content, system_prompt, model_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (nb_id, title, content, system_prompt, model_id, now, now),
        )
    await db.run(op)
    return {
        "id": nb_id, "title": title, "content": content,
        "system_prompt": system_prompt, "model_id": model_id,
        "created_at": now, "updated_at": now,
    }


async def update_notebook(
    db: Store,
    notebook_id: str,
    **fields: str | None,
) -> dict | None:
    """Update notebook fields. Returns updated notebook or None if not found."""
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_NOTEBOOK_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_NOTEBOOK_FIELDS)})")

    now = _now_iso()
    def op(conn):
        row = conn.execute(
            f"SELECT {_NB_COLS} FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        if row is None:
            return None
        existing = dict(zip(_NB_NAMES, row))
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [now, notebook_id]
        conn.execute(f"UPDATE notebooks SET {set_clause}, updated_at=? WHERE id=?", values)
        existing.update(**updates, updated_at=now)
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
