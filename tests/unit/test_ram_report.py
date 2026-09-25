# tests/unit/test_ram_report.py
"""Memory pre-flight arithmetic (heylook_llm.ram_fit; formerly inline in
scripts/ram_report.py, which now renders this module -- v1.60.0).

This arithmetic is a GATE twice over: dev_server.sh refuses to start a model
when it says no, and the admin fit endpoint disables v3's Load button on a
FAIL. A wrong answer here is not a cosmetic reporting bug, it is a refused
load -- which is exactly what happened before the reclaimable-RAM fix, on a
model that then ran with zero swapins and zero swapouts.

Claims (what breaks if a test is deleted):
- vm_stat parsing: the counters silently go missing, reclaimable_gb returns
  None, and the gate quietly falls back to the conservative figure that
  under-reports by ~35 GiB on this hardware.
- reclaimable arithmetic: the gate drifts back toward free+inactive and
  starts refusing loads that fit.
- fallback: the computation stops working off macOS instead of degrading.

(The verdict/asymmetry layer on top is pinned by test_ram_fit.py.)
"""
from __future__ import annotations

import pytest

from heylook_llm import ram_fit as ram_report

GB = 1024 ** 3

# Real `vm_stat` output shape, trimmed. Values are the ones measured on the
# 192 GiB M2 Ultra right after a 127 GiB model unloaded -- the state in which
# the old free+inactive figure wrongly refused a 138 GiB load.
_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  131072.
Pages active:                               5033164.
Pages inactive:                             6721536.
Pages speculative:                           184963.
Pages throttled:                                  0.
Pages wired down:                            294912.
Pages purgeable:                               6553.
File-backed pages:                         10118758.
Anonymous pages:                            1757184.
Pages occupied by compressor:                255567.
Swapins:                                          0.
Swapouts:                                         0.
"""


@pytest.mark.unit
class TestVmStatParsing:
    def test_counters_are_converted_to_bytes_with_the_reported_page_size(self, monkeypatch):
        # The page size is 16 KiB on Apple Silicon, not the 4 KiB default.
        # Assuming 4 KiB understates every figure by 4x.
        monkeypatch.setattr(ram_report, "_vm_stat_pages",
                            ram_report._vm_stat_pages)  # keep the real function
        import subprocess

        class _Result:
            stdout = _VM_STAT
            returncode = 0

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        stats = ram_report._vm_stat_pages()
        assert stats["anonymous pages"] == 1757184 * 16384
        assert stats["pages wired down"] == 294912 * 16384
        assert stats["file-backed pages"] == 10118758 * 16384

    def test_unreadable_vm_stat_yields_empty(self, monkeypatch):
        import subprocess

        def _boom(*a, **k):
            raise OSError("no vm_stat here")

        monkeypatch.setattr(subprocess, "run", _boom)
        assert ram_report._vm_stat_pages() == {}


def _warm_machine(monkeypatch):
    """The regression fixture: 192 GiB total, 27 GiB anonymous, 5 GiB wired,
    125 GiB free+inactive -- a cache-warm machine after a big unload."""
    monkeypatch.setattr(ram_report, "_vm_stat_pages", lambda: {
        "anonymous pages": 27 * GB, "pages wired down": 5 * GB,
    })
    monkeypatch.setattr(
        ram_report.__dict__.setdefault("psutil", __import__("psutil")),
        "virtual_memory", lambda: type("VM", (), {"total": 192 * GB, "available": 125 * GB})(),
    )


@pytest.mark.unit
class TestReclaimable:
    @pytest.mark.parametrize("check", [
        # reclaimable == total - anonymous - wired, nothing else subtracted.
        pytest.param(lambda reclaimable, available: reclaimable == pytest.approx(160.0),
                     id="reclaimable_excludes_only_anonymous_and_wired"),
        # THE regression. File-backed pages macOS parked in the active queue
        # are clean and evictable, but free+inactive cannot see them -- so the
        # conservative figure refused a load that then ran without paging.
        pytest.param(lambda reclaimable, available: reclaimable > available + 30,
                     id="reclaimable_beats_free_plus_inactive_on_a_cache_warm_machine"),
    ])
    def test_reclaimable_on_the_regression_fixture(self, monkeypatch, check):
        _warm_machine(monkeypatch)
        reclaimable, available = ram_report.reclaimable_gb(), ram_report.available_gb()
        assert check(reclaimable, available), (reclaimable, available)

    def test_usable_falls_back_when_vm_stat_is_unavailable(self, monkeypatch):
        # Off macOS there are no such counters; the script must still answer.
        monkeypatch.setattr(ram_report, "_vm_stat_pages", lambda: {})
        monkeypatch.setattr(ram_report, "available_gb", lambda: 42.0)
        assert ram_report.reclaimable_gb() is None
        assert ram_report.usable_gb() == 42.0

    def test_partial_counters_are_not_half_trusted(self, monkeypatch):
        # Anonymous without wired (or vice versa) would silently overstate
        # headroom; refuse to compute rather than guess the missing term.
        monkeypatch.setattr(ram_report, "_vm_stat_pages",
                            lambda: {"anonymous pages": 27 * GB})
        assert ram_report.reclaimable_gb() is None


# Shard-set sizing lives in test_ram_fit.py::TestSizing::test_shard_sizing.
