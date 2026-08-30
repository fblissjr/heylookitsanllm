# tests/unit/test_startup_banner.py
"""The startup banner's endpoint list is complete (v1.79.45).

It was not. `get_api_endpoints` walked `app.routes` and kept anything with a
`.path`, but a router mounted via `include_router` appears there as an
`_IncludedRouter` carrying neither `.path` nor a `.routes` list to recurse
into. So every endpoint behind a router was silently absent and the banner
printed 12 of 48 -- omitting `/v1/messages` (the wire this project's own
frontend speaks), all of `/v1/conversations`, `/v1/presets` and `/v1/admin`
-- while reading like a complete list.

The property, not a count: whatever the app serves under `/v1` is what the
banner reports. A count would need editing every time a route lands, which is
the maintenance shape this repo treats as a defect with a delay.
"""

import pytest


@pytest.mark.unit
class TestEndpointDiscovery:
    def _app(self):
        from heylook_llm.api import app
        return app

    def test_the_banner_lists_every_v1_path_the_schema_publishes(self):
        """The oracle is the OpenAPI schema -- the surface this repo already
        treats as authoritative -- rather than a hand-written roster."""
        from heylook_llm.server import get_api_endpoints

        app = self._app()
        published = {p for p in app.openapi()["paths"] if p.startswith("/v1/")}
        assert set(get_api_endpoints(app)) == published

    def test_router_mounted_endpoints_are_included(self):
        """The specific failure. These three are each mounted through
        `include_router` and were each invisible; `/v1/messages` is the one
        that matters most, since it is the wire v3 speaks and the one the
        integration guide points external clients at."""
        from heylook_llm.server import get_api_endpoints

        found = set(get_api_endpoints(self._app()))
        for path in ("/v1/messages", "/v1/conversations", "/v1/admin/models"):
            assert path in found, f"{path} missing -- router-mounted routes are invisible again"

    def test_directly_mounted_endpoints_are_still_included(self):
        """The old implementation got these right, so a fix that lost them
        would be trading one blind spot for another."""
        from heylook_llm.server import get_api_endpoints

        found = set(get_api_endpoints(self._app()))
        assert "/v1/chat/completions" in found
        assert "/v1/models" in found

    def test_only_v1_paths_are_reported(self):
        """`/docs`, `/openapi.json` and the `/v3` frontend mount are not API
        endpoints, and the banner's line says "under /v1"."""
        from heylook_llm.server import get_api_endpoints

        assert all(p.startswith("/v1/") for p in get_api_endpoints(self._app()))
