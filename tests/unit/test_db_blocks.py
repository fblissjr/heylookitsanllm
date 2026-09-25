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
    # A string stores as one text block and reads back as the same string.
    # Rows: the wire field and what it carries, on the write and the read.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("field, expected", [
        pytest.param("content", "hello", id="string_content_round_trips_as_string"),
        pytest.param("content_blocks", [{"type": "text", "text": "hello"}],
                     id="string_content_exposes_single_text_block"),
    ])
    async def test_string_content(self, conn, conv, field, expected):
        msg = await db.append_message(conn, conv["id"], role="user", content="hello")
        assert msg[field] == expected
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0][field] == expected


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

    # `content` is the newline-joined text of the text blocks only, on the
    # write and the read. Rows: (blocks, expected content, blocks stored as
    # given).
    # - null_text: {"type":"text","text":null} must not poison the row: flatten
    #   would TypeError on None and make the conversation permanently
    #   unreadable.
    # - unknown_type: a non-text block passes through untouched and adds no text.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("blocks, expected_content, stored_as_given", [
        pytest.param([IMAGE_BLOCK, {"type": "text", "text": "what is this?"}],
                     "what is this?", False,
                     id="flattened_content_is_text_blocks_only"),
        pytest.param([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                     "a\nb", False, id="multiple_text_blocks_flatten_joined"),
        pytest.param([{"type": "text", "text": None}], "", False,
                     id="null_text_block_normalizes_to_empty"),
        pytest.param([{"type": "thinking", "thinking": "hmm"}], "", True,
                     id="unknown_block_type_passes_through"),
    ])
    async def test_flatten(self, conn, conv, blocks, expected_content, stored_as_given):
        msg = await db.append_message(conn, conv["id"], role="user", content=blocks)
        assert msg["content"] == expected_content
        if stored_as_given:
            assert msg["content_blocks"] == blocks
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"][0]["content"] == expected_content

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
        # the message rows themselves are gone, not just unreachable
        count = await conn.run(
            lambda c: c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        assert count == 0

    # Rows: (after_position, deleted, contents kept, next append's position).
    # A position past the end deletes nothing and keeps every message.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("after, deleted_n, kept, next_position", [
        pytest.param(1, 2, ["m0", "m1"], 2, id="truncate_after_position_with_blocks"),
        pytest.param(5, 0, ["m0", "m1", "m2", "m3"], 4,
                     id="truncate_past_the_end_preserves_earlier_messages"),
    ])
    async def test_truncate_after_position_with_blocks(
        self, conn, conv, after, deleted_n, kept, next_position
    ):
        for i in range(4):
            await db.append_message(conn, conv["id"], role="user", content=f"m{i}")
        deleted = await db.truncate_messages_after(conn, conv["id"], after)
        assert deleted == deleted_n
        got = await db.get_conversation(conn, conv["id"])
        assert [m["content"] for m in got["messages"]] == kept
        # positions keep appending after truncation
        msg = await db.append_message(conn, conv["id"], role="assistant", content="m2b")
        assert msg["position"] == next_position

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
    # A malformed block raises ValueError and persists nothing.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", [
        pytest.param([{"type": "image"}], id="malformed_image_block_rejected_before_persist"),
        pytest.param(["hi"], id="non_dict_block_rejected"),
    ])
    async def test_block_refusal(self, conn, conv, content):
        with pytest.raises(ValueError):
            await db.append_message(conn, conv["id"], role="user", content=content)
        got = await db.get_conversation(conn, conv["id"])
        assert got["messages"] == []  # nothing persisted

    @pytest.mark.asyncio
    async def test_url_image_source_round_trips(self, conn, conv):
        block = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
        msg = await db.append_message(conn, conv["id"], role="user", content=[block])
        assert msg["content_blocks"] == [block]

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


async def _delete_last_reference(conn, cid):
    msg = await db.append_message(conn, cid, role="user", content=[IMAGE_BLOCK])
    assert await db.delete_message(conn, cid, msg["id"]) is True
    return msg["content_blocks"][0]["source"]["media_id"]


async def _delete_one_of_two_references(conn, cid):
    m1 = await db.append_message(conn, cid, role="user", content=[IMAGE_BLOCK])
    await db.append_message(conn, cid, role="user", content=[IMAGE_BLOCK])
    await db.delete_message(conn, cid, m1["id"])
    return m1["content_blocks"][0]["source"]["media_id"]


async def _truncate_the_referencing_message(conn, cid):
    await db.append_message(conn, cid, role="user", content="text first")
    await db.append_message(conn, cid, role="user", content=[IMAGE_BLOCK])
    media_id = await _first_media_id(conn, cid)
    await db.truncate_messages_after(conn, cid, 0)
    return media_id


async def _delete_the_conversation(conn, cid):
    msg = await db.append_message(conn, cid, role="user", content=[IMAGE_BLOCK])
    await db.delete_conversation(conn, cid)
    return msg["content_blocks"][0]["source"]["media_id"]


class TestMediaByReference:
    """Schema v7: blob lifecycle -- dedup, GC direction, round-trip honesty."""

    @pytest.mark.asyncio
    async def test_same_bytes_dedupe_to_one_blob(self, conn, conv):
        m1 = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        m2 = await db.append_message(conn, conv["id"], role="user", content=[IMAGE_BLOCK])
        id1 = m1["content_blocks"][0]["source"]["media_id"]
        assert id1 == m2["content_blocks"][0]["source"]["media_id"]  # content-addressed

    # A media blob exists iff some message in the conversation still
    # references it. Rows: (the removal, whether the blob survives it). Each
    # removal returns the media_id it put at stake.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("remove, survives", [
        pytest.param(_delete_last_reference, False,
                     id="deleting_last_reference_collects_the_blob"),
        pytest.param(_delete_one_of_two_references, True,
                     id="surviving_reference_keeps_the_blob"),
        pytest.param(_truncate_the_referencing_message, False,
                     id="truncate_collects_orphaned_blobs"),
        pytest.param(_delete_the_conversation, False,
                     id="delete_conversation_deletes_its_blobs"),
    ])
    async def test_blob_lifecycle(self, conn, conv, remove, survives):
        media_id = await remove(conn, conv["id"])
        blob = await db.get_media_blob(conn, conv["id"], media_id)
        assert (blob is not None) is survives

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


class TestMessageStats:
    """Per-message generation stats (plan W5): every read carries them, and
    no path that removes a message leaves its stats behind."""

    @pytest.mark.asyncio
    async def test_stats_ride_every_read_and_die_with_their_message(self, conn, conv):
        cid = conv["id"]
        rows = [await db.append_message(conn, cid, role=r, content=r)
                for r in ("user", "assistant", "user", "assistant")]
        for row in rows[1::2]:
            await db.set_message_stats(conn, cid, row["id"], {"output_tokens": 3})

        got = (await db.get_conversation(conn, cid))["messages"]
        assert [m["stats"] for m in got] == [None, {"output_tokens": 3}, None, {"output_tokens": 3}]
        edited = await db.update_message(conn, cid, rows[1]["id"], content="edited")
        assert edited["stats"] == {"output_tokens": 3}

        async def kept():
            def op(c):
                return {r[0] for r in c.execute("SELECT message_id FROM message_stats").fetchall()}
            return await conn.run(op)

        await db.truncate_messages_after(conn, cid, 1)
        assert await kept() == {rows[1]["id"]}
        await db.delete_message(conn, cid, rows[1]["id"])
        assert await kept() == set()
        # a stats write for a message already gone keeps nothing
        await db.set_message_stats(conn, cid, rows[3]["id"], {"output_tokens": 1})
        assert await kept() == set()
        row = await db.append_message(conn, cid, role="assistant", content="a")
        await db.set_message_stats(conn, cid, row["id"], {"output_tokens": 1})
        await db.delete_conversation(conn, cid)
        assert await kept() == set()
