# tests/unit/test_conversation_api.py
"""Unit tests for the conversation storage layer (db.py + conversation_api.py).

Tests run against an in-memory SQLite database -- no server required.
"""

import pytest
import pytest_asyncio

from heylook_llm import db


@pytest_asyncio.fixture
async def conn():
    """In-memory database connection, fresh per test."""
    connection = await db.get_connection(path=":memory:")
    yield connection
    await connection.close()


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConversationCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, conn):
        conv = await db.create_conversation(conn, title="Test Chat", model_id="llama-3")
        assert conv["title"] == "Test Chat"
        assert conv["model_id"] == "llama-3"
        assert conv["messages"] == []
        assert conv["id"]

        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert fetched["title"] == "Test Chat"
        assert fetched["messages"] == []

    @pytest.mark.asyncio
    async def test_list_ordered_by_updated(self, conn):
        c1 = await db.create_conversation(conn, title="First")
        c2 = await db.create_conversation(conn, title="Second")
        # c2 created after c1, so it should come first
        convs = await db.list_conversations(conn)
        assert len(convs) == 2
        assert convs[0]["id"] == c2["id"]
        assert convs[1]["id"] == c1["id"]

    @pytest.mark.asyncio
    async def test_update(self, conn):
        conv = await db.create_conversation(conn, title="Original")
        updated = await db.update_conversation(
            conn, conv["id"], title="Renamed", system_prompt="Be helpful."
        )
        assert updated is not None
        assert updated["title"] == "Renamed"
        assert updated["system_prompt"] == "Be helpful."
        # model_id unchanged
        assert updated["model_id"] == conv["model_id"]

    @pytest.mark.asyncio
    async def test_clear_model_id(self, conn):
        conv = await db.create_conversation(conn, title="Test", model_id="llama-3")
        updated = await db.update_conversation(conn, conv["id"], model_id=None)
        assert updated is not None
        assert updated["model_id"] is None
        assert updated["title"] == "Test"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, conn):
        result = await db.update_conversation(conn, "nonexistent", title="Nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, conn):
        conv = await db.create_conversation(conn, title="Doomed")
        assert await db.delete_conversation(conn, conv["id"]) is True
        assert await db.get_conversation(conn, conv["id"]) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, conn):
        assert await db.delete_conversation(conn, "ghost") is False

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, conn):
        assert await db.get_conversation(conn, "nope") is None


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMessageCRUD:
    @pytest.mark.asyncio
    async def test_append_and_retrieve(self, conn):
        conv = await db.create_conversation(conn)
        m1 = await db.append_message(conn, conv["id"], role="user", content="Hello")
        m2 = await db.append_message(conn, conv["id"], role="assistant", content="Hi there")

        assert m1 is not None
        assert m1["position"] == 0
        assert m1["role"] == "user"
        assert m1["content"] == "Hello"

        assert m2 is not None
        assert m2["position"] == 1

        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert len(fetched["messages"]) == 2
        assert fetched["messages"][0]["content"] == "Hello"
        assert fetched["messages"][1]["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_conversation(self, conn):
        result = await db.append_message(conn, "ghost", role="user", content="Hello?")
        assert result is None

    @pytest.mark.asyncio
    async def test_append_with_thinking(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(
            conn, conv["id"], role="assistant", content="Answer", thinking="Let me think..."
        )
        assert msg is not None
        assert msg["thinking"] == "Let me think..."

    @pytest.mark.asyncio
    async def test_update_content(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(conn, conv["id"], role="user", content="Original")
        assert msg is not None

        updated = await db.update_message(conn, conv["id"], msg["id"], content="Edited")
        assert updated is not None
        assert updated["content"] == "Edited"

    @pytest.mark.asyncio
    async def test_update_thinking_only(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(
            conn, conv["id"], role="assistant", content="Answer", thinking="Old thinking"
        )
        assert msg is not None

        updated = await db.update_message(conn, conv["id"], msg["id"], thinking="New thinking")
        assert updated is not None
        assert updated["thinking"] == "New thinking"
        assert updated["content"] == "Answer"  # Unchanged

    @pytest.mark.asyncio
    async def test_clear_thinking(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(
            conn, conv["id"], role="assistant", content="Answer", thinking="Some thinking"
        )
        assert msg is not None
        updated = await db.update_message(conn, conv["id"], msg["id"], thinking=None)
        assert updated is not None
        assert updated["thinking"] is None
        assert updated["content"] == "Answer"

    @pytest.mark.asyncio
    async def test_update_no_fields_raises(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(conn, conv["id"], role="user", content="Hello")
        assert msg is not None
        with pytest.raises(ValueError, match="No updatable fields"):
            await db.update_message(conn, conv["id"], msg["id"])

    @pytest.mark.asyncio
    async def test_update_nonexistent_message(self, conn):
        conv = await db.create_conversation(conn)
        result = await db.update_message(conn, conv["id"], "ghost", content="Nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_truncate_after_position(self, conn):
        conv = await db.create_conversation(conn)
        await db.append_message(conn, conv["id"], role="user", content="msg0")
        await db.append_message(conn, conv["id"], role="assistant", content="msg1")
        await db.append_message(conn, conv["id"], role="user", content="msg2")
        await db.append_message(conn, conv["id"], role="assistant", content="msg3")

        # Truncate after position 1 -- should delete msg2 and msg3
        deleted = await db.truncate_messages_after(conn, conv["id"], after_position=1)
        assert deleted == 2

        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert len(fetched["messages"]) == 2
        assert fetched["messages"][0]["content"] == "msg0"
        assert fetched["messages"][1]["content"] == "msg1"

    @pytest.mark.asyncio
    async def test_truncate_preserves_earlier_messages(self, conn):
        conv = await db.create_conversation(conn)
        await db.append_message(conn, conv["id"], role="user", content="keep")
        await db.append_message(conn, conv["id"], role="assistant", content="also keep")

        # Truncate after position 5 -- nothing to delete
        deleted = await db.truncate_messages_after(conn, conv["id"], after_position=5)
        assert deleted == 0

        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert len(fetched["messages"]) == 2

    @pytest.mark.asyncio
    async def test_cascade_delete(self, conn):
        """Deleting a conversation should delete all its messages."""
        conv = await db.create_conversation(conn)
        await db.append_message(conn, conv["id"], role="user", content="Hello")
        await db.append_message(conn, conv["id"], role="assistant", content="Hi")

        await db.delete_conversation(conn, conv["id"])

        # Verify messages are gone (check directly)
        async with conn.execute("SELECT COUNT(*) FROM messages") as cur:
            row = await cur.fetchone()
            assert row[0] == 0

    @pytest.mark.asyncio
    async def test_position_auto_increment(self, conn):
        conv = await db.create_conversation(conn)
        msgs = []
        for i in range(5):
            m = await db.append_message(conn, conv["id"], role="user", content=f"msg{i}")
            msgs.append(m)

        positions = [m["position"] for m in msgs]
        assert positions == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_position_after_truncate_and_reappend(self, conn):
        """After truncating, new messages should continue from the right position."""
        conv = await db.create_conversation(conn)
        await db.append_message(conn, conv["id"], role="user", content="msg0")
        await db.append_message(conn, conv["id"], role="assistant", content="msg1")
        await db.append_message(conn, conv["id"], role="user", content="msg2")

        # Truncate after position 0
        await db.truncate_messages_after(conn, conv["id"], after_position=0)

        # New message should get position 1
        msg = await db.append_message(conn, conv["id"], role="assistant", content="new msg1")
        assert msg is not None
        assert msg["position"] == 1

    @pytest.mark.asyncio
    async def test_append_updates_conversation_timestamp(self, conn):
        conv = await db.create_conversation(conn)
        original_updated = conv["updated_at"]

        await db.append_message(conn, conv["id"], role="user", content="Hello")

        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert fetched["updated_at"] >= original_updated
