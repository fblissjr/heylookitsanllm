"""The monitoring routes answer after the v1.79.67 split of api.py.

They moved from the app module into monitoring_api.py, and the metrics
route kept calling the collector getter by its old private name: the
schema still listed the path, the unit and contract suites stayed green,
and the perf page's RAM value never populated. A route that exists in the
schema is not a route that works; these hit each one.
"""


class TestMonitoringRoutesAnswer:
    def test_system_metrics_answers_with_system_and_models(self, client):
        res = client.get("/v1/system/metrics")
        assert res.status_code == 200, res.text
        body = res.json()
        assert "system" in body and "models" in body

    def test_capabilities_answers_with_the_server_version(self, client):
        res = client.get("/v1/capabilities")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["server_version"] and "messages" in body["endpoints"]
        assert "standard_vision" not in body["endpoints"]  # the OpenAI wire's entry is gone

    def test_performance_profile_answers(self, client):
        assert client.get("/v1/performance/profile/1h").status_code == 200
        assert client.get("/v1/performance/profile/2h").status_code == 400

    def test_cache_list_answers(self, client):
        res = client.get("/v1/cache/list")
        assert res.status_code == 200, res.text
        assert "caches" in res.json()

    def test_each_route_carries_its_tag_once(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/v1/system/metrics", "/v1/capabilities", "/v1/cache/list", "/v1/embeddings",
                     "/v1/hidden_states", "/v1/models"):
            for op in paths[path].values():
                tags = op.get("tags", [])
                assert len(tags) == len(set(tags)) == 1, f"{path}: {tags}"
