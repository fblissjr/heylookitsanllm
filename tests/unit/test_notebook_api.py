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


# One round trip: create (defaults, preset stamp) -> get -> update -> delete.
# Rows: (create kwargs, op, op kwargs, expected fields). For "get" and
# "update" the expected fields hold on the op's result AND a fresh get; for
# "delete" the delete answers True and the notebook is gone.
# - preset stamps: same new-document preset inheritance contract as
#   conversations -- starting-as-a-preset is an explicit apply, stamped at
#   creation.
_NB_ROUND_TRIP_ROWS = [
    pytest.param({"title": "Test", "content": "Hello world"}, "get", None,
                 {"title": "Test", "content": "Hello world"}, id="create_and_get"),
    pytest.param({}, "get", None,
                 {"title": "Untitled", "content": "", "system_prompt": None, "model_id": None},
                 id="create_defaults"),
    pytest.param({"title": "Inherited", "applied_preset_id": "preset-123"}, "get", None,
                 {"applied_preset_id": "preset-123"},
                 id="create_with_applied_preset_stamps"),
    pytest.param({}, "get", None, {"applied_preset_id": None},
                 id="create_without_preset_stays_unstamped"),
    pytest.param({"title": "Original"}, "update", {"title": "Renamed", "content": "New content"},
                 {"title": "Renamed", "content": "New content"}, id="update"),
    pytest.param({"title": "Doomed"}, "delete", None, None, id="delete"),
]


@pytest.mark.unit
class TestNotebookCRUD:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("create, op, op_kwargs, expected", _NB_ROUND_TRIP_ROWS)
    async def test_notebook_round_trip(self, conn, create, op, op_kwargs, expected):
        nb = await db.create_notebook(conn, **create)
        assert nb["id"]
        if op == "delete":
            assert await db.delete_notebook(conn, nb["id"]) is True
            assert await db.get_notebook(conn, nb["id"]) is None
            return
        result = nb
        if op == "update":
            result = await db.update_notebook(conn, nb["id"], **op_kwargs)
            assert result is not None
        fetched = await db.get_notebook(conn, nb["id"])
        assert fetched is not None
        for got in (result, fetched):
            for key, value in expected.items():
                assert got[key] == value, key

    @pytest.mark.asyncio
    async def test_list_ordered_by_updated(self, conn):
        n1 = await db.create_notebook(conn, title="First")
        n2 = await db.create_notebook(conn, title="Second")
        notebooks = await db.list_notebooks(conn)
        assert len(notebooks) == 2
        assert notebooks[0]["id"] == n2["id"]

    # PATCH semantics: keys absent from the update are kept, an explicit None
    # clears. Rows: (create kwargs, update kwargs, expected fields after).
    @pytest.mark.asyncio
    @pytest.mark.parametrize("create, update, expected", [
        pytest.param({"title": "Test", "content": "Keep this", "model_id": "llama"},
                     {"title": "Changed"},
                     {"title": "Changed", "content": "Keep this", "model_id": "llama"},
                     id="update_partial"),
        pytest.param({"model_id": "llama", "system_prompt": "Be helpful"},
                     {"model_id": None},
                     {"model_id": None, "system_prompt": "Be helpful"},
                     id="update_clear_nullable"),
    ])
    async def test_update_patch_semantics(self, conn, create, update, expected):
        nb = await db.create_notebook(conn, **create)
        updated = await db.update_notebook(conn, nb["id"], **update)
        assert updated is not None
        for key, value in expected.items():
            assert updated[key] == value, key

    @pytest.mark.asyncio
    async def test_update_no_fields_raises(self, conn):
        nb = await db.create_notebook(conn)
        with pytest.raises(ValueError, match="No updatable fields"):
            await db.update_notebook(conn, nb["id"])

    # get / update / delete of an unknown id -> None / None / False.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("op, kwargs, expected", [
        pytest.param("update_notebook", {"title": "Nope"}, None, id="update_nonexistent"),
        pytest.param("delete_notebook", {}, False, id="delete_nonexistent"),
        pytest.param("get_notebook", {}, None, id="get_nonexistent"),
    ])
    async def test_missing_id(self, conn, op, kwargs, expected):
        assert await getattr(db, op)(conn, "ghost", **kwargs) is expected


@pytest.mark.unit
class TestNotebookParams:
    """Per-notebook sampler settings -- same JSON-blob shape + shared encode/decode
    as conversations (unified, not a branched copy)."""

    # Rows: (create kwargs, update params or None, expected params on the
    # call's result and on a fresh get).
    @pytest.mark.asyncio
    @pytest.mark.parametrize("create, update_params, expected", [
        pytest.param({"title": "n"}, None, {}, id="create_defaults_to_empty_params"),
        pytest.param({"title": "n", "params": {"temperature": 1.0, "top_k": 40, "seed": None}},
                     None, {"temperature": 1.0, "top_k": 40, "seed": None},
                     id="params_round_trip"),
        pytest.param({"title": "n", "params": {"temperature": 1.0}},
                     {"temperature": 0.3}, {"temperature": 0.3}, id="update_params"),
    ])
    async def test_params_round_trip(self, conn, create, update_params, expected):
        nb = await db.create_notebook(conn, **create)
        result = nb
        if update_params is not None:
            result = await db.update_notebook(conn, nb["id"], params=update_params)
        assert result["params"] == expected
        assert (await db.get_notebook(conn, nb["id"]))["params"] == expected
