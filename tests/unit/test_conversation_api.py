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

# Create / update round-trip title, model_id, system_prompt, params and the
# preset stamp; fields an update does not write are untouched. Each row is
# checked on the call's own result AND on a fresh get.
# Rows: (create kwargs, update kwargs or None, expected fields, fields that
# must equal their created value).
# - preset stamps: new-document preset inheritance -- a document can START as
#   a preset, which is an explicit apply, so the stamp is written at creation
#   and must round-trip (not just echo in the create response).
# - params: per-conversation sampler settings (a JSON blob) keep tuning with
#   the conversation, next to system_prompt, on the server; types survive.
_ROUND_TRIP_ROWS = [
    pytest.param({"title": "Test Chat", "model_id": "llama-3"}, None,
                 {"title": "Test Chat", "model_id": "llama-3", "messages": []}, (),
                 id="create_and_get"),
    pytest.param({"title": "Inherited", "applied_preset_id": "preset-123"}, None,
                 {"applied_preset_id": "preset-123"}, (),
                 id="create_with_applied_preset_stamps"),
    pytest.param({"title": "Plain"}, None, {"applied_preset_id": None}, (),
                 id="create_without_preset_stays_unstamped"),
    pytest.param({"title": "Original"}, {"title": "Renamed", "system_prompt": "Be helpful."},
                 {"title": "Renamed", "system_prompt": "Be helpful."}, ("model_id",),
                 id="update"),
    pytest.param({"title": "Test", "model_id": "llama-3"}, {"model_id": None},
                 {"model_id": None}, ("title",), id="clear_model_id"),
    pytest.param({"title": "c"}, None, {"params": {}}, (),
                 id="create_defaults_to_empty_params"),
    pytest.param({"title": "c", "params": {"temperature": 1.0, "top_p": 0.9, "top_k": 40,
                                           "seed": None, "enable_thinking": True}}, None,
                 {"params": {"temperature": 1.0, "top_p": 0.9, "top_k": 40,
                             "seed": None, "enable_thinking": True}}, (),
                 id="params_round_trip_types"),
    pytest.param({"title": "c", "params": {"temperature": 1.0}},
                 {"params": {"temperature": 0.5, "top_k": 20}},
                 {"params": {"temperature": 0.5, "top_k": 20}}, (),
                 id="update_params"),
    pytest.param({"title": "c", "system_prompt": "be terse", "params": {"temperature": 1.0}},
                 {"params": {"temperature": 0.2}},
                 {"params": {"temperature": 0.2}}, ("system_prompt",),
                 id="update_params_independent_of_system_prompt"),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("create, update, expected, untouched", _ROUND_TRIP_ROWS)
async def test_conversation_round_trip(conn, create, update, expected, untouched):
    conv = await db.create_conversation(conn, **create)
    assert conv["id"]
    result = conv
    if update is not None:
        result = await db.update_conversation(conn, conv["id"], **update)
        assert result is not None
    fetched = await db.get_conversation(conn, conv["id"])
    assert fetched is not None
    for got in (result, fetched):
        for key, value in expected.items():
            assert got[key] == value, key
        for key in untouched:
            assert got[key] == conv[key], f"{key} changed but was not written"


async def _update_missing_conversation(conn):
    return await db.update_conversation(conn, "nonexistent", title="Nope")


async def _delete_missing_conversation(conn):
    return await db.delete_conversation(conn, "ghost")


async def _get_missing_conversation(conn):
    return await db.get_conversation(conn, "nope")


async def _clone_missing_conversation(conn):
    return await db.clone_conversation(conn, "ghost")


async def _append_to_missing_conversation(conn):
    return await db.append_message(conn, "ghost", role="user", content="Hello?")


async def _update_missing_message(conn):
    conv = await db.create_conversation(conn)
    return await db.update_message(conn, conv["id"], "ghost", content="Nope")


# Every store op on a missing conversation or message id answers None/False,
# never raises. The route's 404 for a missing clone id is a row of
# test_clone_route.
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("op, expected", [
    pytest.param(_update_missing_conversation, None, id="update_nonexistent"),
    pytest.param(_delete_missing_conversation, False, id="delete_nonexistent"),
    pytest.param(_get_missing_conversation, None, id="get_nonexistent"),
    pytest.param(_clone_missing_conversation, None, id="clone_nonexistent_returns_none"),
    pytest.param(_append_to_missing_conversation, None,
                 id="append_to_nonexistent_conversation"),
    pytest.param(_update_missing_message, None, id="update_nonexistent_message"),
])
async def test_missing_id(conn, op, expected):
    assert await op(conn) is expected


@pytest.mark.unit
class TestConversationCRUD:
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

    # Clone deep-copies fields and messages under new ids; the title defaults
    # to 'Copy of X' or takes the given one.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("clone_kwargs, expected_title", [
        pytest.param({}, "Copy of Design Talk", id="clone_default_title"),
        pytest.param({"title": "Branched Chat"}, "Branched Chat", id="clone_custom_title"),
    ])
    async def test_clone_copy(self, conn, clone_kwargs, expected_title):
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

        cloned = await db.clone_conversation(conn, conv["id"], **clone_kwargs)
        assert cloned is not None
        assert cloned["id"] != conv["id"]
        assert cloned["title"] == expected_title
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
        assert fetched["title"] == expected_title
        assert len(fetched["messages"]) == 2


# ---------------------------------------------------------------------------
# Message CRUD
# ---------------------------------------------------------------------------

# Append and update round-trip content and thinking independently; positions
# increase from 0. Rows: (append kwargs in order, expected fields on each
# append's result, update kwargs for the last message or None, expected fields
# on the update's result, contents a fresh get must list or None).
_MESSAGE_ROWS = [
    pytest.param(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}],
        [{"position": 0, "role": "user", "content": "Hello"}, {"position": 1}],
        None, None, ["Hello", "Hi there"],
        id="append_and_retrieve",
    ),
    pytest.param(
        [{"role": "assistant", "content": "Answer", "thinking": "Let me think..."}],
        [{"thinking": "Let me think..."}], None, None, None,
        id="append_with_thinking",
    ),
    pytest.param(
        [{"role": "user", "content": "Original"}], [{}],
        {"content": "Edited"}, {"content": "Edited"}, None,
        id="update_content",
    ),
    pytest.param(
        [{"role": "assistant", "content": "Answer", "thinking": "Old thinking"}], [{}],
        {"thinking": "New thinking"}, {"thinking": "New thinking", "content": "Answer"}, None,
        id="update_thinking_only",
    ),
    pytest.param(
        [{"role": "assistant", "content": "Answer", "thinking": "Some thinking"}], [{}],
        {"thinking": None}, {"thinking": None, "content": "Answer"}, None,
        id="clear_thinking",
    ),
    pytest.param(
        [{"role": "user", "content": f"msg{i}"} for i in range(5)],
        [{"position": i} for i in range(5)], None, None, None,
        id="position_auto_increment",
    ),
]


@pytest.mark.unit
class TestMessageCRUD:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "appends, expected_appended, update, expected_updated, fetched_contents",
        _MESSAGE_ROWS,
    )
    async def test_message_round_trip(
        self, conn, appends, expected_appended, update, expected_updated, fetched_contents
    ):
        conv = await db.create_conversation(conn)
        msgs = []
        for kwargs, expected in zip(appends, expected_appended, strict=True):
            m = await db.append_message(conn, conv["id"], **kwargs)
            assert m is not None
            for key, value in expected.items():
                assert m[key] == value, key
            msgs.append(m)
        if update is not None:
            updated = await db.update_message(conn, conv["id"], msgs[-1]["id"], **update)
            assert updated is not None
            for key, value in expected_updated.items():
                assert updated[key] == value, key
        if fetched_contents is not None:
            fetched = await db.get_conversation(conn, conv["id"])
            assert fetched is not None
            assert [m["content"] for m in fetched["messages"]] == fetched_contents

    @pytest.mark.asyncio
    async def test_update_no_fields_raises(self, conn):
        conv = await db.create_conversation(conn)
        msg = await db.append_message(conn, conv["id"], role="user", content="Hello")
        assert msg is not None
        with pytest.raises(ValueError, match="No updatable fields"):
            await db.update_message(conn, conv["id"], msg["id"])

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
    (next to system_prompt) on the server. Round trips are rows of
    test_conversation_round_trip."""

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

    # POST /clone -> 201 with the store's result, custom title honoured, 404
    # for a missing id. Rows: (source title or None for no conversation,
    # message contents to append, request body, status, expected title,
    # expected cloned message contents or None).
    @pytest.mark.asyncio
    @pytest.mark.parametrize("title, contents, body, status, expected_title, expected_contents", [
        pytest.param("Chat to clone", ["hello"], None, 201, "Copy of Chat to clone", ["hello"],
                     id="clone_endpoint_default"),
        pytest.param("Original Chat", [], {"title": "My Cloned Chat"}, 201, "My Cloned Chat",
                     None, id="clone_endpoint_custom_title"),
        pytest.param(None, [], None, 404, None, None, id="clone_endpoint_404_nonexistent"),
    ])
    async def test_clone_route(
        self, app, conn, title, contents, body, status, expected_title, expected_contents
    ):
        conv_id = "nonexistent-id"
        if title is not None:
            conv = await db.create_conversation(conn, title=title)
            conv_id = conv["id"]
            for content in contents:
                await db.append_message(conn, conv_id, role="user", content=content)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            kwargs = {"json": body} if body is not None else {}
            res = await client.post(f"/v1/conversations/{conv_id}/clone", **kwargs)
            assert res.status_code == status
            if status != 201:
                return
            data = res.json()
            assert data["title"] == expected_title
            assert data["id"] != conv_id
            if expected_contents is not None:
                assert [m["content"] for m in data["messages"]] == expected_contents
