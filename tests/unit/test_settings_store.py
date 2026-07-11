# tests/unit/test_settings_store.py
"""Settings key->value store contract for the DuckDB App store.

Operational settings (obs level/retention, etc.) are runtime-mutable config
edited via /v1/admin/config, persisted in the App DB alongside presets. The
table is key->JSON value: schema-stable (a new setting is a new row, never a
DDL change), so it survives the drop/recreate schema policy without a carve-out.
"""

import pytest
import pytest_asyncio

from heylook_llm import db


@pytest_asyncio.fixture
async def conn():
    connection = await db.get_connection(path=":memory:")
    yield connection
    await connection.close()


class TestSettingsStore:
    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, conn):
        assert await db.get_setting(conn, "nope") is None

    @pytest.mark.asyncio
    async def test_set_then_get_roundtrips_scalar(self, conn):
        await db.set_setting(conn, "observability_level", "debug")
        assert await db.get_setting(conn, "observability_level") == "debug"

    @pytest.mark.asyncio
    async def test_value_types_roundtrip(self, conn):
        # JSON value column must preserve types, not stringify.
        await db.set_setting(conn, "an_int", 30)
        await db.set_setting(conn, "a_bool", True)
        await db.set_setting(conn, "a_list", ["a", "b"])
        await db.set_setting(conn, "an_obj", {"x": 1})
        assert await db.get_setting(conn, "an_int") == 30
        assert await db.get_setting(conn, "a_bool") is True
        assert await db.get_setting(conn, "a_list") == ["a", "b"]
        assert await db.get_setting(conn, "an_obj") == {"x": 1}

    @pytest.mark.asyncio
    async def test_set_is_upsert(self, conn):
        await db.set_setting(conn, "k", "one")
        await db.set_setting(conn, "k", "two")
        assert await db.get_setting(conn, "k") == "two"
        # upsert must not create a duplicate row
        allv = await db.get_all_settings(conn)
        assert list(allv.keys()).count("k") == 1

    @pytest.mark.asyncio
    async def test_get_all_returns_decoded_map(self, conn):
        await db.set_setting(conn, "observability_level", "standard")
        await db.set_setting(conn, "observability_retention_days", 14)
        allv = await db.get_all_settings(conn)
        assert allv == {"observability_level": "standard", "observability_retention_days": 14}

    @pytest.mark.asyncio
    async def test_delete_returns_existence(self, conn):
        await db.set_setting(conn, "k", "v")
        assert await db.delete_setting(conn, "k") is True
        assert await db.delete_setting(conn, "k") is False
        assert await db.get_setting(conn, "k") is None

    @pytest.mark.asyncio
    async def test_non_json_serializable_rejected(self, conn):
        with pytest.raises(ValueError):
            await db.set_setting(conn, "bad", {1, 2, 3})  # a set is not JSON

    @pytest.mark.asyncio
    async def test_settings_survive_alongside_presets(self, conn):
        # settings and presets coexist in the same store, independent tables
        await db.set_setting(conn, "observability_level", "minimal")
        await db.create_preset(conn, name="p1", params={})
        assert await db.get_setting(conn, "observability_level") == "minimal"
        assert len(await db.list_presets(conn)) == 1
