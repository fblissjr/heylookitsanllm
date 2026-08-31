"""Contract: POST /v1/models/{id}/load -- the load an INFERENCE client drives.

Deliberately not in test_admin.py. The route moved off the admin gate in
v1.79.48 because gating it protected nothing: a generate request already
calls the same ``router.get_provider`` and can already evict at
``max_loaded_models=1``, so the admin token only stopped a client from doing
explicitly what it could do implicitly. It is gated like inference now.
"""


import pytest


@pytest.fixture(autouse=True)
def _unload_after(client):
    """Leave the router as this file found it.

    The `client` fixture is SESSION-scoped, so a model loaded here stays
    loaded for every later file -- and `test_admin.py` has a case whose whole
    subject is an UNLOADED model. Harmless in collection order (test_admin
    sorts first), which is exactly why it would have sat here as a trap for
    whoever next ran a subset in a different order.
    """
    yield
    client.post("/v1/admin/models/test-mlx-model/unload")


class TestModelLoadWarm:
    """POST /v1/models/{id}/load?warm=true -- server-side load + warm.

    The server owns readiness semantics: spawn harnesses (scripts/
    dev_server.sh, tests/e2e/lib/server.mjs) call this one endpoint instead
    of inventing poll-the-model-list + hand-rolled warm generations.
    """

    def test_load_without_warm_unchanged(self, client):
        resp = client.post("/v1/models/test-mlx-model/load")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert "warmed" not in data

    def test_load_with_warm_runs_generation(self, client):
        resp = client.post("/v1/models/test-mlx-model/load?warm=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["warmed"] is True
        assert isinstance(data["warm_ms"], int)

    def test_load_warm_unknown_model_400(self, client):
        resp = client.post("/v1/models/nope/load?warm=true")
        assert resp.status_code == 400

    def test_the_old_admin_path_is_gone(self, client):
        """The move is a MOVE, not an alias -- two URLs for one operation is
        the duplication this repo keeps paying for. Pinned so a well-meaning
        back-compat shim has to argue with a test.

        405, not 404: the admin router's catch-all `/{model_id:path}` still
        owns that URL for GET/PATCH/DELETE (as model_id="test-mlx-model/load"),
        so POST is Method Not Allowed. That is the more informative answer and
        it is asserted as what it is rather than loosened to `in (404, 405)`,
        which would pass if the route came back.
        """
        resp = client.post("/v1/admin/models/test-mlx-model/load")
        assert resp.status_code == 405

    def test_listing_still_resolves(self, client):
        """`/v1/models` (exact GET) and `/v1/models/{id}/load` (POST with
        extra segments) share a prefix; the new route must not shadow the
        OpenAI-compatible listing every client discovers ids from."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"], list)


class TestBusyIsBackpressureNotBreakage:
    """MODEL_BUSY must answer 503, not 500.

    Both are "the load did not happen", and a client cannot tell them apart
    from the status alone if they share one -- a 500 on this route means the
    model exists and is genuinely broken, while MODEL_BUSY is transient and
    self-clearing. v1.79.48 added this route without wiring it to
    `busy_response`, so the same condition `/v1/messages` reports as 503 came
    back here as 500 carrying the identical sentence. Measured against a live
    1.79.52 by a consuming client, which classified on status and told its
    user to refresh the model roster.
    """

    def _busy(self, client, monkeypatch):
        router = client.app.state.router_instance

        def boom(model_id):
            raise RuntimeError(
                "MODEL_BUSY: cannot make room -- ['other-model'] is generating. "
                "Stop the generation or wait for it to finish.")

        monkeypatch.setattr(router, "get_provider", boom)
        return client.post("/v1/models/test-mlx-model/load")

    def test_busy_is_503(self, client, monkeypatch):
        assert self._busy(client, monkeypatch).status_code == 503

    def test_busy_carries_the_shared_envelope(self, client, monkeypatch):
        """Same body shape and headers as the inference routes, because it is
        literally the same function -- asserted so a future hand-rolled copy
        here has to disagree with a test."""
        resp = self._busy(client, monkeypatch)
        assert resp.json()["error"]["code"] == "model_overloaded"
        assert resp.headers.get("Retry-After") == "1"

    def test_busy_keeps_the_reason_the_router_gave(self, client, monkeypatch):
        """The eviction-blocked cause names which model is busy and what to do
        about it; collapsing it to a generic queue-full sentence was the defect
        busy_response was created to fix."""
        assert "is generating" in self._busy(client, monkeypatch).json()["error"]["message"]

    def test_a_real_load_failure_is_still_500(self, client, monkeypatch):
        """The 503 must not swallow the case this route's 500 is FOR."""
        router = client.app.state.router_instance

        def boom(model_id):
            raise RuntimeError("safetensors header is corrupt")

        monkeypatch.setattr(router, "get_provider", boom)
        resp = client.post("/v1/models/test-mlx-model/load")
        assert resp.status_code == 500

