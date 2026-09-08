"""Everything a conversation needs to resume must survive the process dying.

The owner's requirement, stated plainly: DuckDB is the source of truth, and on a
restart the backend and the frontend must read back exactly what they need to
carry on -- conversations, messages, images, thinking, preset linkage, sampler
params, per-row model attribution.

Every other test in this suite uses `:memory:`, which is destroyed with the
connection and therefore cannot see this class of bug at all: a value that is
only ever in the writer's cache, a column that is written but never read back, a
media blob referenced by a row but stored against the wrong key. These use a
FILE and genuinely close the store between the write and the read, so the second
half runs against nothing but bytes on disk.

gguf-first on purpose: llama-server is the provider that matters here, and the
image half is where a store round-trip can look fine and still fail -- the wire
needs BYTES, while the store holds a relative URL nothing outside this process
can fetch.
"""
import base64

import pytest
import pytest_asyncio

from heylook_llm import db


PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001{}".format("0d0a2db4" + "0" * 16)
)).decode()


@pytest_asyncio.fixture
async def restartable(tmp_path):
    """A store that can be closed and reopened, like the server going down."""
    path = tmp_path / "durability.db"

    async def reopen():
        return await db.get_connection(path=path)

    store = await reopen()
    yield store, reopen
    try:
        await store.close()
    except Exception:
        pass


@pytest.mark.unit
class TestConversationSurvivesRestart:
    @pytest.mark.asyncio
    async def test_every_resume_field_reads_back(self, restartable):
        store, reopen = restartable

        preset = await db.create_preset(
            store, name="terse", system_prompt="Be brief.",
            params={"temperature": 0.4, "max_tokens": 128})
        conv = await db.create_conversation(
            store, title="kept", model_id="gguf-model",
            system_prompt="You are helpful.",
            params={"temperature": 0.4, "max_tokens": 128, "enable_thinking": True})
        await db.update_conversation(store, conv["id"], applied_preset_id=preset["id"])
        await db.append_message(store, conv["id"], role="user", content=[
            {"type": "text", "text": "what is this"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        ])
        await db.append_message(
            store, conv["id"], role="assistant",
            content="A picture.", thinking="Looks like a PNG.", model_id="gguf-model")

        # --- the server goes down ---
        await store.close()
        store2 = await reopen()
        try:
            back = await db.get_conversation(store2, conv["id"])
            assert back is not None, "the conversation did not survive the restart"
            assert back["system_prompt"] == "You are helpful."
            assert back["model_id"] == "gguf-model"
            assert back["applied_preset_id"] == preset["id"], \
                "preset linkage lost -- the document forgets which preset it runs"
            assert back["params"] == {
                "temperature": 0.4, "max_tokens": 128, "enable_thinking": True}, \
                f"sampler bag did not round-trip: {back['params']}"

            rows = back["messages"]
            assert [r["role"] for r in rows] == ["user", "assistant"]
            assert rows[1]["thinking"] == "Looks like a PNG.", \
                "thinking lost -- gguf replays it as reasoning_content"
            assert rows[1]["model_id"] == "gguf-model", \
                "per-row model attribution lost"

            img = [b for b in rows[0]["content_blocks"] if b["type"] == "image"]
            assert img, "the image block is gone from the stored row"
            src = img[0]["source"]
            assert src.get("media_id"), \
                "the image lost its blob reference, so nothing can resolve it"

            # The BYTES, which is what the wire needs. A row can round-trip
            # perfectly and still be unusable if the blob went with the process.
            blobs = await db.get_media_blobs(store2, conv["id"], [src["media_id"]])
            assert src["media_id"] in blobs, \
                "the media blob did not survive the restart -- the row references " \
                "a blob that is gone, which the generate path raises on"
            media_type, raw = blobs[src["media_id"]]
            assert media_type == "image/png"
            assert raw == base64.b64decode(PNG), "the stored image bytes differ"

            presets = await db.list_presets(store2)
            assert any(p["id"] == preset["id"] and p["system_prompt"] == "Be brief."
                       for p in presets), "the preset did not survive"
        finally:
            await store2.close()
