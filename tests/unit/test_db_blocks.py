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
    async def test_base64_image_externalizes_to_blob_backed_url(self, conn, conv):
        # Schema v7 contract change: base64 media never persists inline. The
        # stored block carries a url source (serve endpoint path + media_id
        # marker + original media_type), and the bytes live in media_blobs.
        blocks = [IMAGE_BLOCK, {"type": "text", "text": "what is this?"}]
        msg = await db.append_message(conn, conv["id"], role="user", content=blocks)
        src = msg["content_blocks"][0]["source"]
        assert src["type"] == "url"
        assert src["media_type"] == "image/png"
        assert src["url"] == f"/v1/conversations/{conv['id']}/media/{src['media_id']}"
        assert msg["content_blocks"][1] == {"type": "text", "text": "what is this?"}
        # the read returns the same externalized shape, and the bytes survive
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content_blocks"] == msg["content_blocks"]
        blob = await db.get_media_blob(conn, conv["id"], src["media_id"])
        assert blob == ("image/png", b"heylook")  # aGV5bG9vaw== decoded

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
        # updates externalize the same way appends do (schema v7)
        assert updated["content_blocks"][0]["source"]["type"] == "url"
        assert updated["content_blocks"][0]["source"]["media_id"]

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


async def _first_media_id(conn, conv_id):
    got = await db.get_conversation(conn, conv_id)
    for m in got["messages"]:
        for b in m["content_blocks"]:
            if b.get("source", {}).get("media_id"):
                return b["source"]["media_id"]
    return None


class TestMediaByReference:
    """Schema v7: blob lifecycle -- dedup, GC direction, round-trip honesty."""

    @pytest.mark.asyncio
    async def test_same_bytes_dedupe_to_one_blob(self, conn, conv):
        m1 = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        m2 = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        id1 = m1["content_blocks"][0]["source"]["media_id"]
        assert id1 == m2["content_blocks"][0]["source"]["media_id"]  # content-addressed

    @pytest.mark.asyncio
    async def test_deleting_last_reference_collects_the_blob(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        media_id = msg["content_blocks"][0]["source"]["media_id"]
        assert await db.delete_message(conn, conv["id"], msg["id"]) is True
        assert await db.get_media_blob(conn, conv["id"], media_id) is None

    @pytest.mark.asyncio
    async def test_surviving_reference_keeps_the_blob(self, conn, conv):
        m1 = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        media_id = m1["content_blocks"][0]["source"]["media_id"]
        await db.delete_message(conn, conv["id"], m1["id"])
        assert await db.get_media_blob(conn, conv["id"], media_id) is not None

    @pytest.mark.asyncio
    async def test_truncate_collects_orphaned_blobs(self, conn, conv):
        await db.append_message(conn, conv["id"], role="user", content="text first")
        await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        media_id = await _first_media_id(conn, conv["id"])
        await db.truncate_messages_after(conn, conv["id"], 0)
        assert await db.get_media_blob(conn, conv["id"], media_id) is None

    @pytest.mark.asyncio
    async def test_round_tripped_stored_block_keeps_its_media_id(self, conn, conv):
        # A stored row PUT back through a write (the client's retry-save
        # shape) must keep pointing at its blob, not re-externalize or dangle.
        msg = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        stored_block = msg["content_blocks"][0]
        updated = await db.update_message(conn, conv["id"], msg["id"], content=[stored_block])
        assert updated["content_blocks"][0]["source"]["media_id"] == \
            stored_block["source"]["media_id"]

    @pytest.mark.asyncio
    async def test_foreign_media_id_is_stripped(self, conn, conv):
        # A media_id for a blob NOT in this conversation (cross-conversation
        # reference, or dangling after GC) must not persist as a marker --
        # the block degrades to a plain external url.
        other = await db.create_conversation(conn, title="other")
        msg = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        stolen = msg["content_blocks"][0]
        planted = await db.append_message(conn, other["id"], role="user", content=[stolen])
        assert "media_id" not in planted["content_blocks"][0]["source"]

    @pytest.mark.asyncio
    async def test_delete_conversation_deletes_its_blobs(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        media_id = msg["content_blocks"][0]["source"]["media_id"]
        await db.delete_conversation(conn, conv["id"])
        assert await db.get_media_blob(conn, conv["id"], media_id) is None


class TestSingleMessageDelete:
    @pytest.mark.asyncio
    async def test_delete_keeps_neighbors_and_positions(self, conn, conv):
        for i in range(3):
            await db.append_message(conn, conv["id"], role="user", content=f"m{i}")
        got = await db.get_conversation(conn, conv["id"])
        assert await db.delete_message(conn, conv["id"], got["messages"][1]["id"]) is True
        after = await db.get_conversation(conn, conv["id"])
        # neighbors survive with their positions (gaps are fine by design)
        assert [(m["content"], m["position"]) for m in after["messages"]] == \
            [("m0", 0), ("m2", 2)]
        # appends continue past the gap
        msg = await db.append_message(conn, conv["id"], role="user", content="m3")
        assert msg["position"] == 3

    @pytest.mark.asyncio
    async def test_delete_missing_message_returns_false(self, conn, conv):
        assert await db.delete_message(conn, conv["id"], "ghost") is False


class TestModelIdStamp:
    @pytest.mark.asyncio
    async def test_replace_tail_stamps_model_id(self, conn, conv):
        await db.append_message(conn, conv["id"], role="user", content="q")
        row = await db.replace_tail_with_message(
            conn, conv["id"], 0, role="assistant", content="a", model_id="test-model")
        assert row["model_id"] == "test-model"
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][1]["model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_user_rows_carry_no_model_id(self, conn, conv):
        msg = await db.append_message(conn, conv["id"], role="user", content="q")
        assert msg["model_id"] is None

    @pytest.mark.asyncio
    async def test_continuation_keeps_the_original_stamp(self, conn, conv):
        # A merged row was co-written; restamping would misattribute half of
        # it, so replace_tail_with_update leaves model_id alone by design.
        await db.append_message(conn, conv["id"], role="user", content="q")
        row = await db.replace_tail_with_message(
            conn, conv["id"], 0, role="assistant", content="a", model_id="model-one")
        merged = await db.replace_tail_with_update(
            conn, conv["id"], row["position"], row["id"], content="a plus more")
        assert merged["model_id"] == "model-one"
