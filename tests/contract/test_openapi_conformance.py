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
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/system/metrics",
            "/v1/admin/models",
            "/v1/admin/models/scan",
            "/v1/admin/models/profiles",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing endpoint in OpenAPI schema: {path}"

    def test_all_routes_have_schema_entries(self, app, schema):
        """Every registered HTTP route with a /v1/ prefix has a schema entry.

        WebSocket routes are excluded (OpenAPI does not document them).
        """
        from starlette.routing import WebSocketRoute

        paths = set(schema["paths"].keys())
        for route in app.routes:
            if isinstance(route, WebSocketRoute):
                continue
            if hasattr(route, "path") and route.path.startswith("/v1/"):
                # Normalize path params: {model_id:path} -> {model_id}
                normalized = route.path
                if ":path}" in normalized:
                    normalized = normalized.replace(":path}", "}")
                # Only check fixed paths (no path params) -- parameterized
                # routes use OpenAPI's {param} syntax which may differ
                if "{" not in normalized:
                    assert normalized in paths, (
                        f"Route {normalized} not found in OpenAPI schema"
                    )

    def test_chat_completions_has_post(self, schema):
        """POST /v1/chat/completions is documented."""
        endpoint = schema["paths"].get("/v1/chat/completions", {})
        assert "post" in endpoint
        post = endpoint["post"]
        assert "summary" in post
        assert "requestBody" in post

    def test_messages_has_post(self, schema):
        """POST /v1/messages is documented."""
        endpoint = schema["paths"].get("/v1/messages", {})
        assert "post" in endpoint

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
            "ChatRequest",
            "ChatCompletionResponse",
            "MessageCreateRequest",
        ]
        for name in expected_schemas:
            assert name in schemas, f"Missing schema definition: {name}"

    def test_chat_request_schema_has_required_fields(self, schema):
        """ChatRequest schema requires messages field."""
        schemas = schema["components"]["schemas"]
        chat_req = schemas.get("ChatRequest", {})
        required = chat_req.get("required", [])
        assert "messages" in required

    def test_endpoint_count(self, schema):
        """Sanity check: we have a reasonable number of endpoints."""
        paths = schema["paths"]
        endpoint_count = 0
        for path, methods in paths.items():
            for method in methods:
                if method in ("get", "post", "put", "delete", "patch"):
                    endpoint_count += 1
        # We expect at least 10 endpoints (models, chat, messages, admin CRUD, etc.)
        assert endpoint_count >= 10, f"Only {endpoint_count} endpoints found"
