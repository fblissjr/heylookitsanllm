# tests/unit/test_conversation_api.py
"""Unit tests for the conversation storage layer (db.py + conversation_api.py).

Tests run against an in-memory SQLite database -- no server required.
"""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from heylook_llm import db
from heylook_llm.conversation_api import conversation_router


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
    async def test_create_with_applied_preset_stamps(self, conn):
        # New-document preset inheritance: a document can START as a preset,
        # which is an explicit apply, so the stamp is written at creation and
        # must round-trip (not just echo in the create response).
        conv = await db.create_conversation(
            conn, title="Inherited", applied_preset_id="preset-123"
        )
        assert conv["applied_preset_id"] == "preset-123"
        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert fetched["applied_preset_id"] == "preset-123"

    @pytest.mark.asyncio
    async def test_create_without_preset_stays_unstamped(self, conn):
        conv = await db.create_conversation(conn, title="Plain")
        assert conv["applied_preset_id"] is None
        fetched = await db.get_conversation(conn, conv["id"])
        assert fetched is not None
        assert fetched["applied_preset_id"] is None

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

    @pytest.mark.asyncio
    async def test_clone_default_title(self, conn):
        conv = await db.create_conversation(
            conn,
            title="Design Talk",
            model_id="qwen-3",
            system_prompt="Be concise",
            params={"temperature": 0.4},
            applied_preset_id="preset-abc",
        )
        m1 = await db.append_message(conn, conv["id"], role="user", content="Hello")
        m2 = await db.append_message(conn, conv["id"], role="assistant", content="Hi!", thinking="Thinking...")

        cloned = await db.clone_conversation(conn, conv["id"])
        assert cloned is not None
        assert cloned["id"] != conv["id"]
        assert cloned["title"] == "Copy of Design Talk"
        assert cloned["model_id"] == "qwen-3"
        assert cloned["system_prompt"] == "Be concise"
        assert cloned["params"] == {"temperature": 0.4}
        assert cloned["applied_preset_id"] == "preset-abc"
        assert len(cloned["messages"]) == 2

        # Check message clone properties
        cm1, cm2 = cloned["messages"]
        assert cm1["id"] != m1["id"]
        assert cm1["role"] == "user"
        assert cm1["content"] == "Hello"
        assert cm1["position"] == 0

        assert cm2["id"] != m2["id"]
        assert cm2["role"] == "assistant"
        assert cm2["content"] == "Hi!"
        assert cm2["thinking"] == "Thinking..."
        assert cm2["position"] == 1

        # Check fetch of cloned conversation
        fetched = await db.get_conversation(conn, cloned["id"])
        assert fetched is not None
        assert fetched["title"] == "Copy of Design Talk"
        assert len(fetched["messages"]) == 2

    @pytest.mark.asyncio
    async def test_clone_custom_title(self, conn):
        conv = await db.create_conversation(conn, title="Original")
        cloned = await db.clone_conversation(conn, conv["id"], title="Branched Chat")
        assert cloned is not None
        assert cloned["title"] == "Branched Chat"

    @pytest.mark.asyncio
    async def test_clone_with_media_blobs(self, conn):
        conv = await db.create_conversation(conn, title="Image Chat")
        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGV5bG9vaw=="},
        }
        await db.append_message(conn, conv["id"], role="user", content=[image_block, {"type": "text", "text": "look"}])

        cloned = await db.clone_conversation(conn, conv["id"])
        assert cloned is not None
        assert len(cloned["messages"]) == 1

        # Check cloned block URL points to cloned conversation
        blocks = cloned["messages"][0]["content_blocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "image"
        media_id = blocks[0]["source"]["media_id"]
        assert blocks[0]["source"]["url"] == f"/v1/conversations/{cloned['id']}/media/{media_id}"

        # Check media blob is readable in cloned conversation
        blob = await db.get_media_blob(conn, cloned["id"], media_id)
        assert blob == ("image/png", b"heylook")

    @pytest.mark.asyncio
    async def test_clone_nonexistent_returns_none(self, conn):
        result = await db.clone_conversation(conn, "ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_clone_independence_after_delete(self, conn):
        conv = await db.create_conversation(conn, title="Parent")
        await db.append_message(conn, conv["id"], role="user", content="msg")
        cloned = await db.clone_conversation(conn, conv["id"])

        # Delete original conversation
        await db.delete_conversation(conn, conv["id"])

        # Cloned conversation should still exist intact
        fetched = await db.get_conversation(conn, cloned["id"])
        assert fetched is not None
        assert len(fetched["messages"]) == 1
        assert fetched["messages"][0]["content"] == "msg"


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

        # Verify messages are gone (check directly -- DuckDB store cascades
        # explicitly, no FK ON DELETE CASCADE)
        count = await conn.run(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        assert count == 0

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
        # Strictly greater: the v3 resume sync skips re-fetching a
        # conversation's body when the list's updated_at has not moved, so
        # "a message write bumps the stamp" is a contract clients rely on
        # (spec section 4). >= would pass with the touch deleted.
        assert fetched["updated_at"] > original_updated


@pytest.mark.unit
class TestConversationParams:
    """Per-conversation sampler settings (params) -- JSON blob, unifies the
    'settings in browser vs server' split by keeping tuning with the conversation
    (next to system_prompt) on the server."""

    @pytest.mark.asyncio
    async def test_create_defaults_to_empty_params(self, conn):
        conv = await db.create_conversation(conn, title="c")
        assert conv["params"] == {}
        assert (await db.get_conversation(conn, conv["id"]))["params"] == {}

    @pytest.mark.asyncio
    async def test_params_round_trip_types(self, conn):
        p = {"temperature": 1.0, "top_p": 0.9, "top_k": 40, "seed": None, "enable_thinking": True}
        conv = await db.create_conversation(conn, title="c", params=p)
        assert conv["params"] == p
        assert (await db.get_conversation(conn, conv["id"]))["params"] == p

    @pytest.mark.asyncio
    async def test_update_params(self, conn):
        conv = await db.create_conversation(conn, title="c", params={"temperature": 1.0})
        updated = await db.update_conversation(conn, conv["id"], params={"temperature": 0.5, "top_k": 20})
        assert updated["params"] == {"temperature": 0.5, "top_k": 20}
        assert (await db.get_conversation(conn, conv["id"]))["params"] == {"temperature": 0.5, "top_k": 20}

    @pytest.mark.asyncio
    async def test_update_params_independent_of_system_prompt(self, conn):
        conv = await db.create_conversation(conn, title="c", system_prompt="be terse",
                                            params={"temperature": 1.0})
        await db.update_conversation(conn, conv["id"], params={"temperature": 0.2})
        got = await db.get_conversation(conn, conv["id"])
        assert got["system_prompt"] == "be terse"     # untouched
        assert got["params"] == {"temperature": 0.2}

    @pytest.mark.asyncio
    async def test_list_omits_params_and_prompt_but_the_body_carries_them(self, conn):
        """The list is a sidebar, not a document.

        It renders a title and orders by recency; it reads neither the sampler
        bag nor the stored system prompt, and both are unbounded. They used to
        ship for every conversation on page load AND on every foreground (the
        resume path re-lists). Same reason list_notebooks omits content.
        """
        conv = await db.create_conversation(
            conn, title="c", params={"temperature": 0.7}, system_prompt="be terse")
        (row,) = await db.list_conversations(conn)
        assert "params" not in row
        assert "system_prompt" not in row
        assert row["title"] == "c"          # what the sidebar actually reads
        assert "updated_at" in row
        # Fetching the conversation is how you get either of them.
        body = await db.get_conversation(conn, conv["id"])
        assert body["params"] == {"temperature": 0.7}
        assert body["system_prompt"] == "be terse"


@pytest.mark.unit
class TestConversationCloneEndpoints:
    @pytest.fixture
    def app(self, conn):
        application = FastAPI()
        application.include_router(conversation_router)
        application.state.db = conn
        return application

    @pytest.mark.asyncio
    async def test_clone_endpoint_default(self, app, conn):
        conv = await db.create_conversation(conn, title="Chat to clone")
        await db.append_message(conn, conv["id"], role="user", content="hello")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(f"/v1/conversations/{conv['id']}/clone")
            assert res.status_code == 201
            data = res.json()
            assert data["title"] == "Copy of Chat to clone"
            assert data["id"] != conv["id"]
            assert len(data["messages"]) == 1
            assert data["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_clone_endpoint_custom_title(self, app, conn):
        conv = await db.create_conversation(conn, title="Original Chat")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(f"/v1/conversations/{conv['id']}/clone", json={"title": "My Cloned Chat"})
            assert res.status_code == 201
            data = res.json()
            assert data["title"] == "My Cloned Chat"

    @pytest.mark.asyncio
    async def test_clone_endpoint_404_nonexistent(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/v1/conversations/nonexistent-id/clone")
            assert res.status_code == 404

