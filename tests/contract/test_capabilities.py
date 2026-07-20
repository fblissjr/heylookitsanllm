# tests/contract/test_capabilities.py
#
# Contract tests for GET /v1/capabilities -- specifically named-sampler
# discovery (added 2026-07-20). Before this, sampler preset names were only
# discoverable via the admin sampler-presets route or the text of a 400.


class TestCapabilitiesSamplers:
    """GET /v1/capabilities advertises the named-sampler registry so scripted
    clients (batch-labeler etc.) can discover names without admin access."""

    def test_samplers_block_present(self, client):
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200
        block = resp.json().get("samplers")
        assert block is not None, "capabilities missing samplers block"
        assert block["request_field"] == "sampler"
        assert block["model_default_field"] == "default_sampler"

    def test_samplers_match_registry(self, client):
        """Names mirror the registry exactly -- the same source of truth the
        cascade resolves against, so this can't drift."""
        from heylook_llm.samplers import get_sampler_registry

        expected = set(get_sampler_registry().list_names())
        resp = client.get("/v1/capabilities")
        block = resp.json()["samplers"]
        returned = {p["name"] for p in block["available"]}
        assert returned == expected
        for entry in block["available"]:
            assert "description" in entry
