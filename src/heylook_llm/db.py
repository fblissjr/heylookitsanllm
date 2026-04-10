# src/heylook_llm/db.py
"""SQLite database for conversation storage.

Provides async access via aiosqlite. Schema auto-creates on first connection.
DB path defaults to ``data/conversations.db`` relative to the working directory,
overridable with the ``HEYLOOK_DB_PATH`` environment variable.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

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
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
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


def _default_db_path() -> Path:
    env = os.environ.get("HEYLOOK_DB_PATH")
    if env:
        return Path(env)
    return Path("data") / "conversations.db"


async def get_connection(path: Path | str | None = None) -> aiosqlite.Connection:
    """Open (or create) the database and return a connection.

    Caller is responsible for closing the connection.
    """
    if isinstance(path, str) and path == ":memory:":
        resolved = ":memory:"
    else:
        resolved_path = Path(path) if isinstance(path, str) else (path or _default_db_path())
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(resolved_path)

    db = await aiosqlite.connect(resolved, timeout=10)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.executescript(_SCHEMA_SQL)

    # Stamp version if missing
    async with db.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        await db.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        await db.commit()

    logger.info("Database ready at %s (schema v%d)", resolved, _SCHEMA_VERSION)
    return db


def get_db(request):
    """Get the shared database connection from app state. For use in FastAPI route handlers."""
    from fastapi import HTTPException
    conn = getattr(request.app.state, "db", None)
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

async def list_conversations(db: aiosqlite.Connection) -> list[dict]:
    """Return all conversations ordered by updated_at desc."""
    async with db.execute(
        "SELECT id, title, model_id, system_prompt, created_at, updated_at "
        "FROM conversations ORDER BY updated_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_conversation(db: aiosqlite.Connection, conv_id: str) -> dict | None:
    """Return a conversation with its messages, or None."""
    async with db.execute(
        "SELECT id, title, model_id, system_prompt, created_at, updated_at "
        "FROM conversations WHERE id = ?",
        (conv_id,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    conv = dict(row)

    async with db.execute(
        "SELECT id, role, content, thinking, position, created_at, updated_at "
        "FROM messages WHERE conversation_id = ? ORDER BY position",
        (conv_id,),
    ) as cur:
        msgs = await cur.fetchall()
    conv["messages"] = [dict(m) for m in msgs]
    return conv


async def create_conversation(
    db: aiosqlite.Connection,
    *,
    title: str = "New Conversation",
    model_id: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Create a new conversation and return it."""
    conv_id = new_id()
    now = _now_iso()
    await db.execute(
        "INSERT INTO conversations (id, title, model_id, system_prompt, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, title, model_id, system_prompt, now, now),
    )
    await db.commit()
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
    db: aiosqlite.Connection,
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

    async with db.execute(
        "SELECT id, title, model_id, system_prompt, created_at, updated_at "
        "FROM conversations WHERE id = ?",
        (conv_id,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    existing = dict(row)
    now = _now_iso()

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [now, conv_id]
    await db.execute(
        f"UPDATE conversations SET {set_clause}, updated_at=? WHERE id=?",
        values,
    )
    await db.commit()
    existing.update(**updates, updated_at=now)
    return existing


async def delete_conversation(db: aiosqlite.Connection, conv_id: str) -> bool:
    """Delete a conversation and its messages. Returns True if it existed."""
    cursor = await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    await db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

async def _next_position(db: aiosqlite.Connection, conv_id: str) -> int:
    async with db.execute(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE conversation_id = ?",
        (conv_id,),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None  # COALESCE always returns a row
    return row[0]


async def append_message(
    db: aiosqlite.Connection,
    conv_id: str,
    *,
    role: str,
    content: str = "",
    thinking: str | None = None,
) -> dict | None:
    """Append a message to a conversation. Returns the message or None if conv not found."""
    # Verify conversation exists
    async with db.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)) as cur:
        if await cur.fetchone() is None:
            return None

    msg_id = new_id()
    now = _now_iso()
    position = await _next_position(db, conv_id)

    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, thinking, position, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conv_id, role, content, thinking, position, now, now),
    )
    # Touch conversation updated_at
    await db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
    )
    await db.commit()
    return {
        "id": msg_id,
        "role": role,
        "content": content,
        "thinking": thinking,
        "position": position,
        "created_at": now,
        "updated_at": now,
    }


async def update_message(
    db: aiosqlite.Connection,
    conv_id: str,
    msg_id: str,
    **fields: str | None,
) -> dict | None:
    """Update a message's content and/or thinking. Returns updated message or None.

    Pass only the fields you want to change, e.g. ``update_message(db, c, m, content="new")``.
    Raises ``ValueError`` if no recognized fields are provided.
    """
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_MESSAGE_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_MESSAGE_FIELDS)})")

    async with db.execute(
        "SELECT id, role, content, thinking, position, created_at, updated_at "
        "FROM messages WHERE id = ? AND conversation_id = ?",
        (msg_id, conv_id),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    msg = dict(row)
    now = _now_iso()

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [now, msg_id]
    await db.execute(
        f"UPDATE messages SET {set_clause}, updated_at=? WHERE id=?",
        values,
    )
    await db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
    )
    await db.commit()
    msg.update(**updates, updated_at=now)
    return msg


async def truncate_messages_after(
    db: aiosqlite.Connection,
    conv_id: str,
    after_position: int,
) -> int:
    """Delete all messages with position > after_position. Returns count deleted."""
    cursor = await db.execute(
        "DELETE FROM messages WHERE conversation_id = ? AND position > ?",
        (conv_id, after_position),
    )
    now = _now_iso()
    await db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
    )
    await db.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Notebook CRUD
# ---------------------------------------------------------------------------

async def list_notebooks(db: aiosqlite.Connection) -> list[dict]:
    """Return all notebooks ordered by updated_at desc. Excludes content for efficiency."""
    async with db.execute(
        "SELECT id, title, system_prompt, model_id, created_at, updated_at "
        "FROM notebooks ORDER BY updated_at DESC"
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_notebook(db: aiosqlite.Connection, notebook_id: str) -> dict | None:
    """Return a notebook or None."""
    async with db.execute(
        "SELECT id, title, content, system_prompt, model_id, created_at, updated_at "
        "FROM notebooks WHERE id = ?",
        (notebook_id,),
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def create_notebook(
    db: aiosqlite.Connection,
    *,
    title: str = "Untitled",
    content: str = "",
    system_prompt: str | None = None,
    model_id: str | None = None,
) -> dict:
    """Create a new notebook and return it."""
    nb_id = new_id()
    now = _now_iso()
    await db.execute(
        "INSERT INTO notebooks (id, title, content, system_prompt, model_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (nb_id, title, content, system_prompt, model_id, now, now),
    )
    await db.commit()
    return {
        "id": nb_id, "title": title, "content": content,
        "system_prompt": system_prompt, "model_id": model_id,
        "created_at": now, "updated_at": now,
    }


async def update_notebook(
    db: aiosqlite.Connection,
    notebook_id: str,
    **fields: str | None,
) -> dict | None:
    """Update notebook fields. Returns updated notebook or None if not found."""
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_NOTEBOOK_FIELDS}
    if not updates:
        raise ValueError(f"No updatable fields provided (allowed: {sorted(_UPDATABLE_NOTEBOOK_FIELDS)})")

    async with db.execute(
        "SELECT id, title, content, system_prompt, model_id, created_at, updated_at "
        "FROM notebooks WHERE id = ?",
        (notebook_id,),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None

    existing = dict(row)
    now = _now_iso()

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [now, notebook_id]
    await db.execute(
        f"UPDATE notebooks SET {set_clause}, updated_at=? WHERE id=?",
        values,
    )
    await db.commit()
    existing.update(**updates, updated_at=now)
    return existing


async def delete_notebook(db: aiosqlite.Connection, notebook_id: str) -> bool:
    """Delete a notebook. Returns True if it existed."""
    cursor = await db.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    await db.commit()
    return cursor.rowcount > 0
