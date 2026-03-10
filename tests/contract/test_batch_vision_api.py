# tests/contract/test_batch_vision_api.py
"""
Contract tests for batch vision labeling API endpoints.
Uses FastAPI TestClient with MockRouter -- no real models needed.
"""

import sys
from unittest.mock import patch

import pytest

from helpers.mlx_mock import create_mlx_module_mocks


# ---------------------------------------------------------------------------
# Add a vision-enabled model to the test config
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def vision_model_config(app):
    """Add a VLM model to the test config so validation passes for some tests."""
    from heylook_llm.config import ModelConfig

    vlm_config = ModelConfig(
        id="test-vlm",
        provider="mlx",
        config={"model_path": "/fake/vlm", "vision": True},
        description="Test VLM",
        tags=["test", "vision"],
        enabled=True,
    )
    # Add to the app config's model list
    router = app.state.router_instance
    existing_ids = {m.id for m in router.app_config.models}
    if "test-vlm" not in existing_ids:
        router.app_config.models.append(vlm_config)
    yield
    # Cleanup: remove the VLM model
    router.app_config.models = [m for m in router.app_config.models if m.id != "test-vlm"]


@pytest.fixture(autouse=True)
def reset_job_manager():
    """Reset the batch pipeline singleton between tests."""
    _mocks = create_mlx_module_mocks()
    with patch.dict(sys.modules, _mocks):
        from heylook_llm.batch_vision_pipeline import _reset_for_testing
        _reset_for_testing()
        yield
        _reset_for_testing()


# ---------------------------------------------------------------------------
# POST /v1/batch/vision/label
# ---------------------------------------------------------------------------

class TestStartBatchVisionLabel:
    def test_missing_fields_422(self, client):
        resp = client.post("/v1/batch/vision/label", json={})
        assert resp.status_code == 422

    def test_nonexistent_model_404(self, client):
        resp = client.post("/v1/batch/vision/label", json={
            "image_dir": "/tmp",
            "model": "nonexistent-model",
            "system_prompt": "label this",
            "output_db": "/tmp/test.db",
        })
        assert resp.status_code == 404

    def test_non_vision_model_400(self, client):
        resp = client.post("/v1/batch/vision/label", json={
            "image_dir": "/tmp",
            "model": "test-mlx-model",  # text-only model
            "system_prompt": "label this",
            "output_db": "/tmp/test.db",
        })
        assert resp.status_code == 400
        assert "not a vision model" in resp.json()["detail"]

    def test_nonexistent_directory_400(self, client):
        resp = client.post("/v1/batch/vision/label", json={
            "image_dir": "/nonexistent/directory/path",
            "model": "test-vlm",
            "system_prompt": "label this",
            "output_db": "/tmp/test.db",
        })
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /v1/batch/vision/label/{job_id}
# ---------------------------------------------------------------------------

class TestGetBatchVisionStatus:
    def test_unknown_job_404(self, client):
        resp = client.get("/v1/batch/vision/label/nonexistent-job-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/batch/vision/label/{job_id}/cancel
# ---------------------------------------------------------------------------

class TestCancelBatchVisionJob:
    def test_unknown_job_404(self, client):
        resp = client.post("/v1/batch/vision/label/nonexistent-job-id/cancel")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/batch/vision/label (list)
# ---------------------------------------------------------------------------

class TestListBatchVisionJobs:
    def test_empty_list(self, client):
        resp = client.get("/v1/batch/vision/label")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
