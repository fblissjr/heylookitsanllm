"""Tests for watch folders + discovered-models surface (C3 of Slice 1.5).

Watch folders are a passive discovery mechanism: the server periodically
scans configured folders and the HF cache, caches the list of "discovered
but not yet imported" models, and surfaces the list via a read-only admin
endpoint. There is NO auto-import -- the frontend Models page (C5) shows an
Add button that hits the existing /v1/admin/models/import endpoint.

Tests use a fake scanner (monkeypatched onto MemoryManager) to avoid real
filesystem I/O and to inject deterministic discovered lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


@dataclass
class _FakeScannedModel:
    id: str
    path: str
    provider: str = "mlx"
    size_gb: float = 1.0
    vision: bool = False
    quantization: str | None = None
    already_configured: bool = False
    tags: list[str] | None = None
    description: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TestScanConfig:
    def test_scan_section_is_optional(self):
        """models.toml without [scan] still loads; AppConfig.scan is None."""
        from heylook_llm.config import AppConfig

        cfg = AppConfig(models=[])
        assert cfg.scan is None

    def test_scan_section_parses(self):
        from heylook_llm.config import AppConfig, ScanConfig

        cfg = AppConfig(
            models=[],
            scan=ScanConfig(
                folders=["/models", "~/cache/mlx"],
                watch_hf_cache=True,
                scan_interval_seconds=600,
            ),
        )
        assert cfg.scan is not None
        assert cfg.scan.folders == ["/models", "~/cache/mlx"]
        assert cfg.scan.watch_hf_cache is True
        assert cfg.scan.scan_interval_seconds == 600

    def test_scan_defaults(self):
        """ScanConfig with only folders uses sensible defaults."""
        from heylook_llm.config import ScanConfig

        cfg = ScanConfig()
        assert cfg.folders == []
        assert cfg.watch_hf_cache is False
        assert cfg.scan_interval_seconds == 900  # 15min


class TestDiscoveredModelsRescan:
    """``MemoryManager._maybe_rescan_models`` drives periodic discovery from
    ``tick()``. The scan is I/O so it's offloaded to a worker thread; tests
    exercise the interval gate + cache semantics synchronously.
    """

    def _mgr(self, monkeypatch, tmp_path, *, scan_interval=900, initial_ts=None):
        from heylook_llm.config import AppConfig, ScanConfig
        from heylook_llm.memory import MemoryManager

        router = MagicMock()
        app_config = AppConfig(
            models=[],
            scan=ScanConfig(
                folders=["/fake"],
                watch_hf_cache=False,
                scan_interval_seconds=scan_interval,
            ),
        )
        mgr = MemoryManager(router=router, app_config=app_config, log_dir=tmp_path)
        if initial_ts is not None:
            mgr._last_scan_ts = initial_ts
        return mgr

    def test_no_scan_config_is_noop(self, monkeypatch, tmp_path):
        """Without a [scan] section, tick doesn't touch the discovery cache."""
        from heylook_llm.config import AppConfig
        from heylook_llm.memory import MemoryManager

        router = MagicMock()
        app_config = AppConfig(models=[])
        mgr = MemoryManager(router=router, app_config=app_config, log_dir=tmp_path)

        called = []

        def fake_scan(paths, scan_hf):
            called.append((paths, scan_hf))
            return []

        monkeypatch.setattr(mgr, "_scan_paths_sync", fake_scan)

        mgr._maybe_rescan_models(now_ts=10_000.0)

        assert called == []

    def test_first_tick_triggers_scan(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, initial_ts=0.0)

        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: [_FakeScannedModel(id="m1", path="/p/m1")],
        )

        mgr._maybe_rescan_models(now_ts=1_000.0)

        assert "m1" in mgr._discovered
        assert mgr._last_scan_ts == 1_000.0

    def test_rescan_within_interval_is_skipped(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, scan_interval=900, initial_ts=1_000.0)
        calls = []
        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: calls.append(1) or [],
        )

        # 300s after last scan, 900s interval -- must not re-scan.
        mgr._maybe_rescan_models(now_ts=1_300.0)

        assert calls == []

    def test_rescan_after_interval_triggers_scan(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, scan_interval=900, initial_ts=1_000.0)
        calls = []
        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: (calls.append(1) or [
                _FakeScannedModel(id="m2", path="/p/m2")
            ]),
        )

        mgr._maybe_rescan_models(now_ts=2_000.0)

        assert len(calls) == 1
        assert "m2" in mgr._discovered

    def test_already_configured_models_excluded(self, monkeypatch, tmp_path):
        """The scanner returns `already_configured=True` for models in
        models.toml; the discovery cache must exclude them so the UI doesn't
        surface duplicates."""
        mgr = self._mgr(monkeypatch, tmp_path, initial_ts=0.0)
        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: [
                _FakeScannedModel(id="new", path="/p/new", already_configured=False),
                _FakeScannedModel(id="old", path="/p/old", already_configured=True),
            ],
        )

        mgr._maybe_rescan_models(now_ts=1_000.0)

        assert "new" in mgr._discovered
        assert "old" not in mgr._discovered

    def test_scan_interval_zero_disables(self, monkeypatch, tmp_path):
        """scan_interval_seconds=0 disables periodic rescan; initial scan still
        happens on first tick so the endpoint has something to return."""
        mgr = self._mgr(monkeypatch, tmp_path, scan_interval=0, initial_ts=0.0)
        calls = []
        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: calls.append(1) or [],
        )

        mgr._maybe_rescan_models(now_ts=1.0)
        mgr._maybe_rescan_models(now_ts=1_000_000.0)

        assert calls == []

    def test_tick_also_drives_rescan(self, monkeypatch, tmp_path):
        """MemoryManager.tick() must call _maybe_rescan_models, not just
        unload_idle_models. Guards against regressions where the discovery
        path silently stops firing."""
        mgr = self._mgr(monkeypatch, tmp_path, initial_ts=0.0)
        calls = []
        monkeypatch.setattr(
            mgr, "_maybe_rescan_models",
            lambda: calls.append(1) or None,
        )

        mgr.tick()

        assert calls == [1]

    def test_discovered_snapshot_shape(self, monkeypatch, tmp_path):
        """``discovered_snapshot()`` is the endpoint contract."""
        mgr = self._mgr(monkeypatch, tmp_path, initial_ts=0.0)
        monkeypatch.setattr(
            mgr, "_scan_paths_sync",
            lambda paths, scan_hf: [
                _FakeScannedModel(
                    id="qwen3-4b",
                    path="/models/qwen3-4b",
                    provider="mlx",
                    size_gb=3.2,
                    vision=False,
                    quantization="4bit",
                ),
            ],
        )

        mgr._maybe_rescan_models(now_ts=1_000.0)
        snap = mgr.discovered_snapshot()

        assert snap["last_scan_ts"] == 1_000.0
        assert len(snap["discovered"]) == 1
        entry = snap["discovered"][0]
        assert entry["id"] == "qwen3-4b"
        assert entry["size_gb"] == 3.2
        assert entry["quantization"] == "4bit"
        # Path must be home-normalized for content-invariant discipline.
        assert "path" in entry
