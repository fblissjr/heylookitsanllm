# tests/contract/test_capabilities.py
#
# Contract tests for GET /v1/capabilities -- specifically sampler-preset
# discovery (added 2026-07-20). Before this, sampler preset names were only
# discoverable via the admin sampler-presets route or the text of a 400.


class TestCapabilitiesSamplerPresets:
    """GET /v1/capabilities advertises the sampler-preset registry so scripted
    clients (batch-labeler etc.) can discover names without admin access."""

    def test_sampler_presets_block_present(self, client):
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200
        block = resp.json().get("sampler_presets")
        assert block is not None, "capabilities missing sampler_presets block"
        assert block["request_field"] == "preset"
        assert block["model_default_field"] == "default_preset"

    def test_sampler_presets_match_registry(self, client):
        """Names mirror the registry exactly -- the same source of truth the
        cascade resolves against, so this can't drift."""
        from heylook_llm.presets import get_preset_registry

        expected = set(get_preset_registry().list_names())
        resp = client.get("/v1/capabilities")
        block = resp.json()["sampler_presets"]
        returned = {p["name"] for p in block["available"]}
        assert returned == expected
        for preset in block["available"]:
            assert "description" in preset
