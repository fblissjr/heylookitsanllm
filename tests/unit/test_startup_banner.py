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
        treats as authoritative -- rather than a hand-written roster.

        Set equality also carries three former checks: router-mounted routes
        are listed (the specific failure: `/v1/messages`, the wire v3 speaks,
        `/v1/conversations`, `/v1/admin/models` were each invisible), routes
        mounted on the app itself are still listed (a fix that lost those
        would trade one blind spot for another), and nothing outside /v1
        (`/docs`, `/openapi.json`, the frontend at `/`) is reported. The
        subset assertion keeps the equality from passing on a schema that
        holds only one mount kind."""
        from heylook_llm.server import get_api_endpoints

        app = self._app()
        published = {p for p in app.openapi()["paths"] if p.startswith("/v1/")}
        assert {
            "/v1/messages", "/v1/conversations", "/v1/admin/models",  # via include_router
            "/v1/data/clear",  # mounted on the app itself
            "/v1/models",
        } <= published
        assert set(get_api_endpoints(app)) == published
