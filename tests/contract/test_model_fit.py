# tests/contract/test_model_fit.py
"""POST /v1/admin/models/{id}/fit -- the memory-fit endpoint.

The COMPUTATION is pinned by tests/unit/test_ram_fit.py; these tests pin the
route contract: 404/422 shapes, override semantics (candidate config edits,
null = reset), the provider-derived hard_working_set flag, and that the
response is the structured report the v3 fit meter renders verbatim.

Ceiling reads are monkeypatched -- contract tests run under mocked MLX and
on arbitrary machines, so live vm_stat/device_info numbers would make every
assertion machine-shaped.
"""

import pytest

from heylook_llm import ram_fit


@pytest.fixture
def stable_ceilings(monkeypatch, tmp_path):
    """Deterministic ceilings + a real weights dir wired into the test model."""
    monkeypatch.setattr(ram_fit, "usable_gb", lambda: 100.0)
    monkeypatch.setattr(ram_fit, "metal_ceilings", lambda: {
        "device": "TestGPU", "working_set_gb": 80.0,
        "max_buffer_gb": 60.0, "sysctl_wired_mb": 0,
    })
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"\0" * (1 * ram_fit.GB))
    return weights


class TestFitEndpoint:
    def test_unknown_model_404s(self, client):
        res = client.post("/v1/admin/models/no-such-model/fit", json={})
        assert res.status_code == 404

    def test_missing_model_path_422s_with_field(self, client):
        # The mock service's test model points at /fake/mlx-model.
        res = client.post("/v1/admin/models/test-mlx-model/fit", json={})
        assert res.status_code == 422
        assert res.json()["detail"]["field"] == "model_path"

    def test_fit_report_shape_and_mlx_hard_flag(self, client, stable_ceilings):
        res = client.post(
            "/v1/admin/models/test-mlx-model/fit",
            json={"config_overrides": {"model_path": str(stable_ceilings)},
                  "headroom_gb": 4.0},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["hard_working_set"] is True  # provider mlx, not layout
        assert body["verdict"] == "pass"
        assert body["weights_gb"] == pytest.approx(1.0)
        assert body["headroom_gb"] == 4.0
        assert body["reclaimable_gb"] == 100.0
        assert body["working_set_gb"] == 80.0
        assert body["kv_headroom_gb"] == pytest.approx(79.0)
        assert body["estimated"] is False
        assert body["sysctl_suggest_mb"] is None  # under the ceiling: no hint
        ceilings = [line["ceiling"] for line in body["lines"]]
        assert ceilings == ["reclaimable_ram", "metal_working_set"]
        assert all(line["verdict"] == "pass" for line in body["lines"])

    def test_overrides_are_candidate_state(self, client, stable_ceilings):
        # A sidecar supplied as an override is counted; nulling a key drops
        # it (reset-to-default, the PATCH spelling) instead of crashing
        # sizing. Sidecar lives OUTSIDE the weights dir so it is counted
        # once, via the override, not by the dir rglob.
        sidecar = stable_ceilings.parent / "mmproj.gguf"
        sidecar.write_bytes(b"\0" * ram_fit.GB)
        with_sidecar = client.post(
            "/v1/admin/models/test-mlx-model/fit",
            json={"config_overrides": {
                "model_path": str(stable_ceilings),
                "mmproj_path": str(sidecar),
            }},
        )
        assert with_sidecar.status_code == 200
        assert with_sidecar.json()["weights_gb"] == pytest.approx(2.0)
        assert any("mmproj" in n for n in with_sidecar.json()["sizing_notes"])
        nulled = client.post(
            "/v1/admin/models/test-mlx-model/fit",
            json={"config_overrides": {
                "model_path": str(stable_ceilings),
                "mmproj_path": None,
            }},
        )
        assert nulled.status_code == 200
        assert nulled.json()["weights_gb"] == pytest.approx(1.0)

    def test_fit_is_read_only(self, client, stable_ceilings, mock_service):
        before = mock_service.get_config("test-mlx-model").config.model_dump()
        client.post(
            "/v1/admin/models/test-mlx-model/fit",
            json={"config_overrides": {"model_path": str(stable_ceilings)}},
        )
        after = mock_service.get_config("test-mlx-model").config.model_dump()
        assert before == after, "a fit evaluation must never write config"
