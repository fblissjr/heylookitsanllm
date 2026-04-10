# tests/unit/test_notebook_api.py
"""Unit tests for notebook storage (db.py notebook CRUD)."""

import pytest
import pytest_asyncio

from heylook_llm import db


@pytest_asyncio.fixture
async def conn():
    connection = await db.get_connection(path=":memory:")
    yield connection
    await connection.close()


@pytest.mark.unit
class TestNotebookCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, conn):
        nb = await db.create_notebook(conn, title="Test", content="Hello world")
        assert nb["title"] == "Test"
        assert nb["content"] == "Hello world"
        assert nb["id"]

        fetched = await db.get_notebook(conn, nb["id"])
        assert fetched is not None
        assert fetched["title"] == "Test"
        assert fetched["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_create_defaults(self, conn):
        nb = await db.create_notebook(conn)
        assert nb["title"] == "Untitled"
        assert nb["content"] == ""
        assert nb["system_prompt"] is None
        assert nb["model_id"] is None

    @pytest.mark.asyncio
    async def test_list_ordered_by_updated(self, conn):
        n1 = await db.create_notebook(conn, title="First")
        n2 = await db.create_notebook(conn, title="Second")
        notebooks = await db.list_notebooks(conn)
        assert len(notebooks) == 2
        assert notebooks[0]["id"] == n2["id"]

    @pytest.mark.asyncio
    async def test_update(self, conn):
        nb = await db.create_notebook(conn, title="Original")
        updated = await db.update_notebook(conn, nb["id"], title="Renamed", content="New content")
        assert updated is not None
        assert updated["title"] == "Renamed"
        assert updated["content"] == "New content"

    @pytest.mark.asyncio
    async def test_update_partial(self, conn):
        nb = await db.create_notebook(conn, title="Test", content="Keep this", model_id="llama")
        updated = await db.update_notebook(conn, nb["id"], title="Changed")
        assert updated is not None
        assert updated["title"] == "Changed"
        assert updated["content"] == "Keep this"
        assert updated["model_id"] == "llama"

    @pytest.mark.asyncio
    async def test_update_clear_nullable(self, conn):
        nb = await db.create_notebook(conn, model_id="llama", system_prompt="Be helpful")
        updated = await db.update_notebook(conn, nb["id"], model_id=None)
        assert updated is not None
        assert updated["model_id"] is None
        assert updated["system_prompt"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_update_no_fields_raises(self, conn):
        nb = await db.create_notebook(conn)
        with pytest.raises(ValueError, match="No updatable fields"):
            await db.update_notebook(conn, nb["id"])

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, conn):
        result = await db.update_notebook(conn, "ghost", title="Nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, conn):
        nb = await db.create_notebook(conn, title="Doomed")
        assert await db.delete_notebook(conn, nb["id"]) is True
        assert await db.get_notebook(conn, nb["id"]) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, conn):
        assert await db.delete_notebook(conn, "ghost") is False

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, conn):
        assert await db.get_notebook(conn, "nope") is None
