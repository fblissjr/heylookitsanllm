"""The engine contract (plan W13), checked where clients read it.

Every assertion goes through the routes (/v1/models and /v1/admin/models),
never through describe() alone: a contract that the function honours and no
route carries is the failure this file exists to catch.

Properties, not examples:
- the same `engine` keys on every row, whatever the engine;
- a value with provenance unknown or not_applicable is null;
- a configured value is marked configured;
- the settings cover every configurable field of the row's provider, and
  rows of one provider share one key set, loaded or not;
- the sampler settings carry the cascade's own answer (sampler_defaults);
- no absolute path anywhere in /v1/models (LAN clients read it).
"""
import os

from heylook_llm.config import PROVIDER_CONFIG_CLASSES, configurable_fields
from heylook_llm.providers.contract import EngineDescription

FACT_LEAVES = (("runtime",), ("context", "length"), ("context", "running"),
               ("template", "origin"), ("template", "path"),
               ("template", "sha256"), ("template", "running_sha256"))
FUTURE_SLOTS = ("cache", "thinking", "image", "steering")


def _rows(client):
    listed = client.get("/v1/models").json()["data"]
    admin = client.get("/v1/admin/models").json()["models"]
    return [("v1/models", r) for r in listed] + [("admin", r) for r in admin]


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _check(route, row):
    engine = row["engine"]
    rid = f"{route}:{row['id']}"
    assert set(engine) == set(EngineDescription.model_fields), rid
    for slot in FUTURE_SLOTS:
        assert engine[slot] is None, f"{rid}: {slot} is filled by a later workstream"
    for path in FACT_LEAVES:
        fact = engine
        for key in path:
            fact = fact[key]
        assert set(fact) == {"value", "provenance", "source"}, (rid, path)
        if fact["provenance"] in ("unknown", "not_applicable"):
            assert fact["value"] is None, (rid, path, fact)
    for name, setting in engine["settings"].items():
        assert setting["reason"], (rid, name)
        if setting["configured"] is not None:
            assert setting["provenance"] == "configured", (rid, name, setting)
        if setting["provenance"] in ("unknown", "not_applicable"):
            assert setting["value"] is None, (rid, name, setting)
    provider_cls = PROVIDER_CONFIG_CLASSES[row["provider"]]
    assert configurable_fields(provider_cls) <= set(engine["settings"]), rid
    for key in set(engine["settings"]) & set(row.get("sampler_defaults") or {}):
        assert engine["settings"][key]["value"] == row["sampler_defaults"][key], (rid, key)


def test_every_row_honours_the_contract_loaded_or_not(client):
    before = _rows(client)
    assert {r["provider"] for _, r in before} == set(PROVIDER_CONFIG_CLASSES), (
        "the fixture must cover every engine, or 'same keys on every engine' "
        "is untested")
    for route, row in before:
        _check(route, row)

    # Load one model: the observed half joins, the shape does not change.
    mid = before[0][1]["id"]
    assert client.post(f"/v1/models/{mid}/load").status_code == 200
    after = _rows(client)
    for route, row in after:
        _check(route, row)

    def key_sets(rows):
        sets = {}
        for _, row in rows:
            sets.setdefault(row["provider"], set()).add(frozenset(row["engine"]["settings"]))
        return sets

    for provider, variants in key_sets(before + after).items():
        assert len(variants) == 1, f"{provider} rows disagree on settings keys"


def test_no_absolute_path_on_the_public_list(client, mock_router):
    """With a real absolute path stored on a model (a server_binary override),
    nothing in /v1/models' engine blocks carries it -- neither the stored
    setting nor the binary the spawn would use."""
    gguf = next(m for m in mock_router.app_config.models if m.provider == "gguf")
    stored = "/opt/somewhere/else/llama-server"
    gguf.config.server_binary = stored
    try:
        rows = {r["id"]: r for r in client.get("/v1/models").json()["data"]}
        settings = rows[gguf.id]["engine"]["settings"]
        assert settings["server_binary"]["value"] == "llama-server"
        assert settings["binary"]["value"] == "llama-server"
        for row in rows.values():
            for s in _strings(row["engine"]):
                assert not (os.path.isabs(s) or s.startswith("~")), (row["id"], s)
    finally:
        gguf.config.server_binary = None
