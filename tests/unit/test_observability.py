# tests/unit/test_observability.py
"""Observability spine core: record_event -- level-gated, best-effort JSONL.

One ingestion path appends a line to the right stream (metrics vs events),
gated by the configured verbosity level, and NEVER raises (observability must
not break inference). Every line carries ts + iso + type + source.
"""

import os

import orjson
import pytest

from heylook_llm import observability as obs


@pytest.fixture
def logs(tmp_path):
    """Point the spine at a temp dir; reset to a known level per test."""
    obs.configure(level="debug", log_dir=tmp_path)
    yield tmp_path


def _read(path):
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


class TestRecord:
    def test_writes_line_with_ts_iso_type_source(self, logs):
        obs.record_event("request_complete", tier="metrics", min_level="minimal",
                          model="m", generation_tps=42.0)
        (rec,) = _read(logs / "metrics.jsonl")
        assert rec["type"] == "request_complete"
        assert rec["source"] == "backend"
        assert isinstance(rec["ts"], float)
        assert rec["iso"].endswith(tuple("0123456789")) or "T" in rec["iso"]
        assert rec["model"] == "m"
        assert rec["generation_tps"] == 42.0

    def test_tier_routes_to_stream(self, logs):
        obs.record_event("request_complete", tier="metrics", min_level="minimal", n=1)
        obs.record_event("request_error", tier="events", min_level="minimal", n=2)
        assert len(_read(logs / "metrics.jsonl")) == 1
        assert len(_read(logs / "events.jsonl")) == 1

    def test_source_override(self, logs):
        obs.record_event("client_error", tier="events", min_level="standard",
                         source="frontend-v3", msg="boom")
        (rec,) = _read(logs / "events.jsonl")
        assert rec["source"] == "frontend-v3"


class TestLevelGating:
    def test_off_writes_nothing(self, tmp_path):
        obs.configure(level="off", log_dir=tmp_path)
        obs.record_event("x", tier="metrics", min_level="minimal", a=1)
        assert not (tmp_path / "metrics.jsonl").exists()

    def test_minimal_gates_out_higher_levels(self, tmp_path):
        obs.configure(level="minimal", log_dir=tmp_path)
        obs.record_event("counted", tier="metrics", min_level="minimal", a=1)   # kept
        obs.record_event("detailed", tier="events", min_level="standard", a=2)  # dropped
        obs.record_event("chunk", tier="events", min_level="debug", a=3)        # dropped
        assert len(_read(tmp_path / "metrics.jsonl")) == 1
        assert not (tmp_path / "events.jsonl").exists()

    def test_debug_keeps_everything(self, tmp_path):
        obs.configure(level="debug", log_dir=tmp_path)
        for lvl in ("minimal", "standard", "debug"):
            obs.record_event("e", tier="events", min_level=lvl, lvl=lvl)
        assert len(_read(tmp_path / "events.jsonl")) == 3


class TestRotation:
    def test_size_cap_rolls_active_to_archive(self, tmp_path):
        obs.configure(level="debug", log_dir=tmp_path)
        obs.record_event("e", tier="events", min_level="minimal", a=1)
        active = tmp_path / "events.jsonl"
        assert active.exists()
        # tiny cap forces a roll; active file renamed to events.<ts>.jsonl
        obs.rotate_streams(retention_days=30, max_bytes=1, now=1000.0)
        archives = list(tmp_path.glob("events.*.jsonl"))
        assert len(archives) == 1
        assert not active.exists()
        # writing again creates a fresh active file
        obs.record_event("e2", tier="events", min_level="minimal", a=2)
        assert active.exists()

    def test_age_cap_deletes_old_archives(self, tmp_path):
        obs.configure(level="debug", log_dir=tmp_path)
        obs.record_event("e", tier="events", min_level="minimal", a=1)
        obs.rotate_streams(retention_days=30, max_bytes=1, now=1000.0)  # roll active -> archive
        (archive,) = list(tmp_path.glob("events.*.jsonl"))
        # age is measured by the archive's mtime; backdate it 40 days before ref
        ref_now = 100 * 86400.0
        os.utime(archive, (ref_now - 40 * 86400, ref_now - 40 * 86400))
        obs.rotate_streams(retention_days=30, max_bytes=10**9, now=ref_now)  # 40d > 30d -> gone
        assert len(list(tmp_path.glob("events.*.jsonl"))) == 0

    def test_retention_zero_keeps_archives(self, tmp_path):
        obs.configure(level="debug", log_dir=tmp_path)
        obs.record_event("e", tier="events", min_level="minimal", a=1)
        obs.rotate_streams(retention_days=0, max_bytes=1, now=0.0)
        obs.rotate_streams(retention_days=0, max_bytes=10**9, now=10**6)
        assert len(list(tmp_path.glob("events.*.jsonl"))) == 1

    def test_maybe_rotate_throttles(self, tmp_path):
        obs.configure(level="debug", log_dir=tmp_path, retention_days=30)
        assert obs.maybe_rotate(now=100000.0) is True    # first sweep runs
        assert obs.maybe_rotate(now=100001.0) is False   # within the hour -> skip
        assert obs.maybe_rotate(now=100000.0 + 4000) is True  # >1h later -> runs


class TestNeverRaises:
    def test_bad_log_dir_does_not_raise(self, tmp_path):
        # point at a path that can't be created (a file where a dir is needed)
        clash = tmp_path / "clash"
        clash.write_text("i am a file")
        obs.configure(level="debug", log_dir=clash / "sub")
        # must swallow, not raise
        obs.record_event("x", tier="metrics", min_level="minimal", a=1)

    def test_unknown_tier_does_not_raise(self, logs):
        obs.record_event("x", tier="nonsense", min_level="minimal", a=1)
