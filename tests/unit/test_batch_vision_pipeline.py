# tests/unit/test_batch_vision_pipeline.py
"""
Unit tests for batch_vision_pipeline.py -- pure functions and classes
that don't need MLX or real model loading.
"""

import os
import sys
import time
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helpers.mlx_mock import create_mlx_module_mocks

# Patch MLX modules before importing the pipeline
_mlx_mocks = create_mlx_module_mocks()
with patch.dict(sys.modules, _mlx_mocks):
    from heylook_llm.batch_vision_pipeline import (
        _extract_json,
        _file_hash,
        _scan_images,
        IMAGE_EXTENSIONS,
        ResultDatabase,
        BatchVisionConfig,
        BatchVisionLabelJob,
        BatchVisionJobManager,
        JobStatus,
        _reset_for_testing,
        get_job_manager,
    )


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_valid_json_object(self):
        assert _extract_json('{"key": "value"}') == '{"key":"value"}'

    def test_valid_json_array(self):
        result = _extract_json('[1, 2, 3]')
        assert result == '[1,2,3]'

    def test_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key":"value"}'

    def test_markdown_fences_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key":"value"}'

    def test_surrounding_text(self):
        text = 'Here is the result: {"key": "value"} as you can see'
        assert _extract_json(text) == '{"key":"value"}'

    def test_invalid_json(self):
        assert _extract_json('not json at all') is None

    def test_empty_string(self):
        assert _extract_json('') is None

    def test_whitespace_only(self):
        assert _extract_json('   \n  ') is None

    def test_nested_json(self):
        text = '{"outer": {"inner": true}}'
        result = _extract_json(text)
        assert '"outer"' in result
        assert '"inner"' in result


# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"fake image data")
        h1 = _file_hash(str(f))
        h2 = _file_hash(str(f))
        assert h1 == h2

    def test_different_content_differs(self, tmp_path):
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert _file_hash(str(f1)) != _file_hash(str(f2))

    def test_nonexistent_file_raises(self):
        with pytest.raises(OSError):
            _file_hash("/nonexistent/file.jpg")

    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"data")
        h = _file_hash(str(f))
        assert isinstance(h, str)
        # blake2b digest_size=16 -> 32 hex chars
        assert len(h) == 32
        int(h, 16)  # should not raise


# ---------------------------------------------------------------------------
# _scan_images
# ---------------------------------------------------------------------------

class TestScanImages:
    def test_finds_supported_extensions(self, tmp_path):
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            (tmp_path / f"img{ext}").write_bytes(b"fake")
        files = _scan_images(str(tmp_path), recursive=False, extensions=IMAGE_EXTENSIONS)
        assert len(files) == 4

    def test_ignores_non_image(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not an image")
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        files = _scan_images(str(tmp_path), recursive=False, extensions=IMAGE_EXTENSIONS)
        assert len(files) == 1
        assert files[0].name == "photo.jpg"

    def test_recursive(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"fake")
        (subdir / "nested.png").write_bytes(b"fake")
        files = _scan_images(str(tmp_path), recursive=True, extensions=IMAGE_EXTENSIONS)
        assert len(files) == 2

    def test_non_recursive(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "top.jpg").write_bytes(b"fake")
        (subdir / "nested.png").write_bytes(b"fake")
        files = _scan_images(str(tmp_path), recursive=False, extensions=IMAGE_EXTENSIONS)
        assert len(files) == 1

    def test_empty_dir(self, tmp_path):
        files = _scan_images(str(tmp_path), recursive=True, extensions=IMAGE_EXTENSIONS)
        assert files == []

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            _scan_images("/nonexistent/dir", recursive=True, extensions=IMAGE_EXTENSIONS)

    def test_sorted_output(self, tmp_path):
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"fake")
        files = _scan_images(str(tmp_path), recursive=False, extensions=IMAGE_EXTENSIONS)
        names = [f.name for f in files]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]


# ---------------------------------------------------------------------------
# ResultDatabase
# ---------------------------------------------------------------------------

class TestResultDatabase:
    def test_schema_creation(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        # Check tables exist
        cur = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
        assert {'files', 'labels', 'corrections', 'jobs'}.issubset(tables)
        db.close()

    def test_store_and_check_file(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        db.store_file("hash123", "/path/to/img.jpg", "img.jpg", 1024, "2024-01-01T00:00:00")
        cur = db._conn.execute("SELECT file_hash, file_path FROM files")
        row = cur.fetchone()
        assert row == ("hash123", "/path/to/img.jpg")
        db.close()

    def test_is_processed_before_label(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        db.store_file("hash123", "/path/img.jpg", "img.jpg", 1024, "2024-01-01T00:00:00")
        assert db.is_processed("hash123", "model-a") is False
        db.close()

    def test_is_processed_after_label(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        db.store_file("hash123", "/path/img.jpg", "img.jpg", 1024, "2024-01-01T00:00:00")
        db.store_label("hash123", "model-a", '{"label":"cat"}', "raw", 10, 500)
        assert db.is_processed("hash123", "model-a") is True
        db.close()

    def test_replace_semantics(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        db.store_file("hash123", "/path/img.jpg", "img.jpg", 1024, "2024-01-01T00:00:00")
        db.store_label("hash123", "model-a", '{"label":"cat"}', "raw1", 10, 500)
        db.store_label("hash123", "model-a", '{"label":"dog"}', "raw2", 12, 600)
        cur = db._conn.execute("SELECT label_json FROM labels WHERE file_hash='hash123'")
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == '{"label":"dog"}'
        db.close()

    def test_store_and_update_job(self, tmp_path):
        db = ResultDatabase(str(tmp_path / "test.db"))
        config = BatchVisionConfig(
            image_dir="/images", model_id="test-model",
            system_prompt="label this", output_db=str(tmp_path / "test.db")
        )
        db.store_job("job-1", config)
        db.update_job("job-1", "completed", 100, 5, 105)
        cur = db._conn.execute("SELECT status, processed, failed FROM jobs WHERE job_id='job-1'")
        row = cur.fetchone()
        assert row == ("completed", 100, 5)
        db.close()


# ---------------------------------------------------------------------------
# BatchVisionLabelJob
# ---------------------------------------------------------------------------

class TestBatchVisionLabelJob:
    def _make_job(self, tmp_path, **overrides):
        defaults = {
            'image_dir': str(tmp_path),
            'model_id': 'test-vlm',
            'system_prompt': 'Label this image',
            'output_db': str(tmp_path / 'results.db'),
        }
        defaults.update(overrides)
        config = BatchVisionConfig(**defaults)
        router = MagicMock()
        return BatchVisionLabelJob(config, router)

    def test_job_id_format(self, tmp_path):
        job = self._make_job(tmp_path)
        assert job.job_id.startswith("batch-vision-")
        assert len(job.job_id) == len("batch-vision-") + 12

    def test_initial_progress(self, tmp_path):
        job = self._make_job(tmp_path)
        progress = job.get_progress()
        assert progress.status == JobStatus.PENDING
        assert progress.total_images == 0
        assert progress.completed == 0

    def test_cancel_sets_event(self, tmp_path):
        job = self._make_job(tmp_path)
        assert not job._cancel.is_set()
        job.cancel()
        assert job._cancel.is_set()

    def test_empty_dir_completes(self, tmp_path):
        """Empty image dir should complete immediately without errors."""
        job = self._make_job(tmp_path)
        job.run()
        assert job.status == JobStatus.COMPLETED
        assert job.total_images == 0

    def test_unpins_on_error(self, tmp_path):
        """If get_provider raises, model should still be attempted to unpin."""
        job = self._make_job(tmp_path)
        # Put an image so it tries to get a provider
        (tmp_path / "test.jpg").write_bytes(b"fake")
        job.router.get_provider.side_effect = ValueError("model not found")
        job.run()
        assert job.status == JobStatus.FAILED
        assert "model not found" in job.error


# ---------------------------------------------------------------------------
# BatchVisionJobManager
# ---------------------------------------------------------------------------

class TestBatchVisionJobManager:
    def test_submit_returns_id(self, tmp_path):
        manager = BatchVisionJobManager()
        config = BatchVisionConfig(
            image_dir=str(tmp_path), model_id='test-vlm',
            system_prompt='test', output_db=str(tmp_path / 'results.db'),
        )
        router = MagicMock()
        job = BatchVisionLabelJob(config, router)
        job_id = manager.submit(job)
        assert job_id == job.job_id
        # Wait for the background thread to finish (empty dir -> fast)
        time.sleep(0.1)

    def test_unknown_job_returns_none(self):
        manager = BatchVisionJobManager()
        assert manager.get_progress("nonexistent-id") is None

    def test_cancel_unknown_returns_false(self):
        manager = BatchVisionJobManager()
        assert manager.cancel("nonexistent-id") is False

    def test_list_jobs_empty(self):
        manager = BatchVisionJobManager()
        assert manager.list_jobs() == []

    def test_list_jobs_after_submit(self, tmp_path):
        manager = BatchVisionJobManager()
        config = BatchVisionConfig(
            image_dir=str(tmp_path), model_id='test-vlm',
            system_prompt='test', output_db=str(tmp_path / 'results.db'),
        )
        router = MagicMock()
        job = BatchVisionLabelJob(config, router)
        manager.submit(job)
        jobs = manager.list_jobs()
        assert len(jobs) == 1
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Module singleton reset
# ---------------------------------------------------------------------------

class TestModuleSingleton:
    def test_reset_creates_new_manager(self):
        mgr1 = get_job_manager()
        _reset_for_testing()
        mgr2 = get_job_manager()
        assert mgr1 is not mgr2
        _reset_for_testing()  # cleanup
