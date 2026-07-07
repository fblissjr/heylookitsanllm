# tests/unit/test_db_blocks.py
"""Content-block storage contract for the DuckDB store (Q5).

Messages persist as CONTENT BLOCK lists (Messages-style) so image
conversations round-trip; the wire stays back-compatible: `content` is the
flattened text of the text blocks, `content_blocks` carries the full list.
String input normalizes to a single text block.
"""

import pytest
import pytest_asyncio

from heylook_llm import db


IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "aGV5bG9vaw=="},
}


@pytest_asyncio.fixture
async def conn():
    connection = await db.get_connection(path=":memory:")
    yield connection
    await connection.close()


@pytest_asyncio.fixture
async def conv(conn):
    return await db.create_conversation(conn, title="blocks")


class TestStringBackCompat:
    @pytest.mark.asyncio
    async def test_string_content_round_trips_as_string(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content="hello")
        assert msg["content"] == "hello"
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_string_content_exposes_single_text_block(self, conn, conv):
        await db.append_message(conn, conv["id"], role="user", content="hello")
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content_blocks"] == [{"type": "text", "text": "hello"}]


class TestBlockStorage:
    @pytest.mark.asyncio
    async def test_image_block_round_trips_exactly(self, conn, conv):
        blocks = [IMAGE_BLOCK, {"type": "text", "text": "what is this?"}]
        msg = await db.append_message(conn, conv["id"], role="user", content=blocks)
        assert msg["content_blocks"] == blocks
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content_blocks"] == blocks

    @pytest.mark.asyncio
    async def test_flattened_content_is_text_blocks_only(self, conn, conv):
        blocks = [IMAGE_BLOCK, {"type": "text", "text": "what is this?"}]
        await db.append_message(conn, conv["id"], role="user", content=blocks)
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content"] == "what is this?"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_flatten_joined(self, conn, conv):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        await db.append_message(conn, conv["id"], role="user", content=blocks)
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content"] == "a\nb"

    @pytest.mark.asyncio
    async def test_update_message_with_blocks(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content="old")
        updated = await db.update_message(
            conn, conv["id"], msg["id"], content=[IMAGE_BLOCK, {"type": "text", "text": "new"}]
        )
        assert updated["content"] == "new"
        assert updated["content_blocks"][0] == IMAGE_BLOCK

    @pytest.mark.asyncio
    async def test_update_message_with_string_normalizes(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        updated = await db.update_message(conn, conv["id"], msg["id"], content="plain again")
        assert updated["content_blocks"] == [{"type": "text", "text": "plain again"}]


class TestStructuralInvariants:
    @pytest.mark.asyncio
    async def test_delete_conversation_deletes_messages(self, conn, conv):
        # DuckDB has no ON DELETE CASCADE -- the store must cascade explicitly.
        await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        assert await db.delete_conversation(conn, conv["id"]) is True
        assert await db.get_conversation(conn, conv["id"]) is None
        counts = await db.clear_all_data(conn)
        assert counts["conversations_deleted"] == 0

    @pytest.mark.asyncio
    async def test_truncate_after_position_with_blocks(self, conn, conv):
        for i in range(4):
            await db.append_message(conn, conv["id"], role="user", content=f"m{i}")
        deleted = await db.truncate_messages_after(conn, conv["id"], 1)
        assert deleted == 2
        got = await db.get_conversation(conn, conv["id"])
        assert [m["content"] for m in got["messages"]] == ["m0", "m1"]
        # positions keep appending after truncation
        msg = await db.append_message(conn, conv["id"], role="assistant", content="m2b")
        assert msg["position"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_appends_serialize(self, conn, conv):
        # The aiosqlite defect class: interleaved handlers bleeding implicit
        # transactions. The store must serialize writes correctly.
        import asyncio
        await asyncio.gather(*[
            db.append_message(conn, conv["id"], role="user", content=f"c{i}")
            for i in range(8)
        ])
        got = await db.get_conversation(conn, conv["id"])
        assert len(got["messages"]) == 8
        assert sorted(m["position"] for m in got["messages"]) == list(range(8))


class TestValidationAndEdgeCases:
    @pytest.mark.asyncio
    async def test_null_text_block_normalizes_to_empty(self, conn, conv):
        # {"type":"text","text":null} must not poison the row: flatten would
        # TypeError on None and make the conversation permanently unreadable.
        msg = await db.append_message(
            conn, conv["id"], role="user", content=[{"type": "text", "text": None}]
        )
        assert msg["content"] == ""
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content"] == ""

    @pytest.mark.asyncio
    async def test_malformed_image_block_rejected_before_persist(self, conn, conv):
        with pytest.raises(ValueError):
            await db.append_message(
                conn, conv["id"], role="user", content=[{"type": "image"}]
            )
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"] == []  # nothing persisted

    @pytest.mark.asyncio
    async def test_non_dict_block_rejected(self, conn, conv):
        with pytest.raises(ValueError):
            await db.append_message(conn, conv["id"], role="user", content=["hi"])

    @pytest.mark.asyncio
    async def test_url_image_source_round_trips(self, conn, conv):
        block = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
        msg = await db.append_message(conn, conv["id"], role="user", content=[block])
        assert msg["content_blocks"] == [block]

    @pytest.mark.asyncio
    async def test_unknown_block_type_passes_through(self, conn, conv):
        block = {"type": "thinking", "thinking": "hmm"}
        msg = await db.append_message(conn, conv["id"], role="user", content=[block])
        assert msg["content_blocks"] == [block]
        assert msg["content"] == ""  # not a text block

    @pytest.mark.asyncio
    async def test_exception_does_not_wedge_connection(self, conn, conv):
        # An op raising mid-transaction must ROLLBACK, not abort the shared
        # connection for every subsequent operation.
        with pytest.raises(ValueError):
            await db.update_message(conn, conv["id"], "ghost")
        msg = await db.append_message(conn, conv["id"], role="user", content="still works")
        assert msg is not None
