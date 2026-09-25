# tests/unit/test_preset_store.py
"""User preset storage contract for the DuckDB store.

Presets are named bundles of system_prompt + sampler params, authored from
the UI (distinct from the bundled TOML sampler registry in presets.py, which
is server-side and request-scoped via ``ChatRequest.preset``). Names are
unique, enforced in code -- the store's single serialized writer makes the
check race-free.

The route tests in test_preset_api.py call these same functions through a
thin pass-through (ValueError -> 400, None -> 404, PresetNameTaken -> 409), so
the partial update, unknown id, no-fields, delete, duplicate name, rename
collision and blank name cases live there. What stays here is what the routes
do not assert.
"""

import pytest
import pytest_asyncio

from heylook_llm import db


@pytest_asyncio.fixture
async def conn():
    connection = await db.get_connection(path=":memory:")
    yield connection
    await connection.close()


class TestPresetCrud:
    @pytest.mark.asyncio
    async def test_create_returns_full_preset(self, conn):
        p = await db.create_preset(
            conn, name="pirate", system_prompt="Talk like a pirate.",
            params={"temperature": 1.2, "top_p": 0.9},
        )
        assert p["name"] == "pirate"
        assert p["system_prompt"] == "Talk like a pirate."
        assert p["params"] == {"temperature": 1.2, "top_p": 0.9}
        assert p["id"] and p["created_at"] and p["updated_at"]

    @pytest.mark.asyncio
    async def test_params_round_trip_types(self, conn):
        params = {"temperature": 0.7, "top_k": 40, "enable_thinking": True, "seed": 0}
        p = await db.create_preset(conn, name="typed", params=params)
        got = await db.list_presets(conn)
        assert got[0]["params"] == params

    @pytest.mark.asyncio
    async def test_defaults_are_empty(self, conn):
        p = await db.create_preset(conn, name="bare")
        assert p["system_prompt"] is None
        assert p["params"] == {}

    @pytest.mark.asyncio
    async def test_list_ordered_by_name(self, conn):
        for name in ("zeta", "alpha", "mid"):
            await db.create_preset(conn, name=name)
        got = await db.list_presets(conn)
        assert [p["name"] for p in got] == ["alpha", "mid", "zeta"]

    @pytest.mark.asyncio
    async def test_update_fields(self, conn):
        p = await db.create_preset(conn, name="v1", system_prompt="old", params={"top_k": 1})
        updated = await db.update_preset(
            conn, p["id"], name="v2", system_prompt="new", params={"top_k": 2}
        )
        assert updated["name"] == "v2"
        assert updated["system_prompt"] == "new"
        assert updated["params"] == {"top_k": 2}
        assert updated["updated_at"] >= p["updated_at"]


class TestPresetNameUniqueness:
    @pytest.mark.asyncio
    async def test_update_keeping_own_name_is_fine(self, conn):
        p = await db.create_preset(conn, name="self")
        updated = await db.update_preset(conn, p["id"], name="self", system_prompt="x")
        assert updated["system_prompt"] == "x"

    @pytest.mark.asyncio
    async def test_rejected_create_does_not_wedge_connection(self, conn):
        await db.create_preset(conn, name="dup")
        with pytest.raises(db.PresetNameTaken):
            await db.create_preset(conn, name="dup")
        assert await db.create_preset(conn, name="fresh") is not None


class TestPresetValidation:
    @pytest.mark.asyncio
    async def test_params_must_be_a_dict(self, conn):
        with pytest.raises(ValueError):
            await db.create_preset(conn, name="bad", params=["not", "a", "dict"])

    @pytest.mark.asyncio
    async def test_unserializable_params_raise_value_error(self, conn):
        # orjson refuses ints beyond 64 bits with TypeError; the boundary must
        # turn that into ValueError so the router 400s instead of 500ing.
        with pytest.raises(ValueError):
            await db.create_preset(conn, name="big", params={"seed": 2**64})


class TestPresetIsolation:
    @pytest.mark.asyncio
    async def test_clear_all_data_keeps_presets(self, conn):
        # Presets are configuration, not conversation data -- the danger-zone
        # clear must not take them.
        await db.create_preset(conn, name="survivor")
        await db.create_conversation(conn, title="doomed")
        await db.clear_all_data(conn)
        assert [p["name"] for p in await db.list_presets(conn)] == ["survivor"]

    @pytest.mark.asyncio
    async def test_presets_survive_schema_recreate(self, tmp_path, monkeypatch):
        # The fresh-start recreate on a _SCHEMA_VERSION mismatch drops the
        # data tables; presets are versionless config and must NOT be in the
        # drop list.
        path = tmp_path / "recreate.duckdb"
        conn = await db.get_connection(path=path)
        await db.create_preset(conn, name="keeper")
        await db.create_conversation(conn, title="doomed")
        await conn.close()

        monkeypatch.setattr(db, "_SCHEMA_VERSION", db._SCHEMA_VERSION + 1)
        conn = await db.get_connection(path=path)
        try:
            assert await db.list_conversations(conn) == []  # recreate happened
            assert [p["name"] for p in await db.list_presets(conn)] == ["keeper"]
        finally:
            await conn.close()
