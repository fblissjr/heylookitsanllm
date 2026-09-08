# tests/contract/test_openapi_conformance.py
#
# Verify the OpenAPI schema is consistent with actual app routes and
# response shapes. Uses app.openapi() (same as scripts/export_openapi.py).

import pytest


class TestOpenAPISchema:
    """Tests that the OpenAPI schema is complete and consistent."""

    @pytest.fixture(scope="class")
    def schema(self, app):
        """Extract OpenAPI schema from the app."""
        return app.openapi()

    def test_schema_has_info(self, schema):
        """Schema includes title and version."""
        assert "info" in schema
        assert "title" in schema["info"]
        assert "version" in schema["info"]

    def test_schema_has_paths(self, schema):
        """Schema has a non-empty paths section."""
        assert "paths" in schema
        assert len(schema["paths"]) > 0

    def test_core_endpoints_in_schema(self, schema):
        """All core API endpoints appear in the schema."""
        paths = schema["paths"]
        expected_paths = [
            "/v1/models",
            "/v1/messages",
            "/v1/conversations",
            "/v1/system/metrics",
            "/v1/admin/models",
            "/v1/admin/models/scan",
            "/v1/admin/models/samplers",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing endpoint in OpenAPI schema: {path}"

    def test_every_served_v1_route_is_in_the_schema(self, app, schema):
        """A `/v1` route this app SERVES is documented, and vice versa.

        The version of this that walked `app.routes` directly was decorative,
        and measured so: a route added with `include_in_schema=False` -- the
        exact defect it names -- was planted and it stayed green. `app.routes`
        holds 4 Routes, 6 APIRoutes and 17 `_IncludedRouter` entries, and an
        `_IncludedRouter` has no `.path`, so the loop saw ONE `/v1` path
        (`/v1/data/clear`, the only route mounted straight on the app) while
        the schema published 46. The assertion body was reachable for 1 route
        in 46.

        That is the same `_IncludedRouter` blind spot that
        `test_startup_banner.py` commemorates as a shipped bug -- the banner
        printed 12 of 48 -- and the broken walk outlived the fix, one module
        away, inside a contract test.

        `server.get_api_endpoints` is NOT the oracle here, deliberately: it
        reads `app.openapi()`, so comparing it to the schema compares the
        schema to itself. That tautology is why nothing caught the planted
        route. Recursing through `original_router` is what actually reaches
        the served set independently of the schema.
        """
        from fastapi.routing import APIRoute
        from starlette.routing import Route

        def walk(routes):
            for route in routes:
                inner = getattr(route, "original_router", None)
                if inner is not None:
                    yield from walk(inner.routes)
                elif isinstance(route, (APIRoute, Route)):
                    yield route

        # Normalize the path-converter spelling: a route declares
        # `{model_id:path}` where OpenAPI publishes `{model_id}`.
        served = {
            r.path.replace(":path}", "}")
            for r in walk(app.routes) if r.path.startswith("/v1/")
        }
        published = {p for p in schema["paths"] if p.startswith("/v1/")}
        assert served == published, (
            f"served but undocumented: {sorted(served - published)}; "
            f"documented but not served: {sorted(published - served)}"
        )

    def test_openai_chat_routes_are_gone(self, schema):
        """The OpenAI-compatible chat routes were removed in v1.79.66 and must
        not come back by accident: a page that reappears is a second wire to
        keep conformant, which is what the removal was for."""
        for path in ("/v1/chat/completions", "/v1/batch/chat/completions"):
            assert path not in schema["paths"], f"{path} is served again"

    def test_messages_has_post(self, schema):
        """POST /v1/messages is documented as the inference route."""
        endpoint = schema["paths"].get("/v1/messages", {})
        assert "post" in endpoint
        post = endpoint["post"]
        assert "summary" in post
        assert "requestBody" in post

    def test_models_has_get(self, schema):
        """GET /v1/models is documented."""
        endpoint = schema["paths"].get("/v1/models", {})
        assert "get" in endpoint

    def test_schema_has_component_schemas(self, schema):
        """Schema defines component schemas for request/response models."""
        components = schema.get("components", {})
        schemas = components.get("schemas", {})
        assert len(schemas) > 0

        # Key schemas that should exist
        expected_schemas = [
            "MessageCreateRequest",
        ]
        for name in expected_schemas:
            assert name in schemas, f"Missing schema definition: {name}"
        # ChatRequest is the INTERNAL request model providers are driven with;
        # since v1.79.66 no route takes it, so it must not be published.
        assert "ChatRequest" not in schemas

    def test_message_request_schema_has_required_fields(self, schema):
        """MessageCreateRequest schema requires the messages field."""
        schemas = schema["components"]["schemas"]
        req = schemas.get("MessageCreateRequest", {})
        required = req.get("required", [])
        assert "messages" in required
