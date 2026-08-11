# tests/unit/test_ram_fit.py
"""ram_fit: sizing + ceilings + the structured fit verdict.

The load-bearing behavior is the ENGINE ASYMMETRY: over the Metal working
set is FAIL for MLX (mx.set_wired_limit refuses above the recommendation)
but WARN for llama.cpp (a debug-build warning upstream; the model loads and
degrades into paging). The fit endpoint and ram_report.py both render this
module, so these tests pin the one computation.

Ceilings are monkeypatched -- the real ones read vm_stat/sysctl/MLX and
would make every assertion machine-shaped.
"""

import pytest

from heylook_llm import ram_fit
from heylook_llm.ram_fit import evaluate_fit, fit_for_config, is_mlx_config, size_config_gb


def _patch_ceilings(monkeypatch, *, usable: float = 100.0,
                    working_set: float | None = 80.0,
                    max_buffer: float = 60.0, sysctl: int = 0):
    monkeypatch.setattr(ram_fit, "usable_gb", lambda: usable)
    metal = None if working_set is None else {
        "device": "TestGPU",
        "working_set_gb": working_set,
        "max_buffer_gb": max_buffer,
        "sysctl_wired_mb": sysctl,
    }
    monkeypatch.setattr(ram_fit, "metal_ceilings", lambda: metal)


@pytest.mark.unit
class TestEngineAsymmetry:
    def test_over_working_set_fails_for_mlx(self, monkeypatch):
        _patch_ceilings(monkeypatch, usable=200.0, working_set=80.0)
        report = evaluate_fit(90.0, 8.0, hard_working_set=True)
        assert report.verdict == "fail"
        assert not report.fits
        ws = next(l for l in report.lines if l.ceiling == "metal_working_set")
        assert ws.verdict == "fail"

    def test_over_working_set_warns_for_gguf(self, monkeypatch):
        # Same numbers, other engine: llama.cpp loads past the recommendation
        # and degrades into paging -- calling it FAIL would be wrong.
        _patch_ceilings(monkeypatch, usable=200.0, working_set=80.0)
        report = evaluate_fit(90.0, 8.0, hard_working_set=False)
        assert report.verdict == "warn"
        assert report.fits
        ws = next(l for l in report.lines if l.ceiling == "metal_working_set")
        assert ws.verdict == "warn"

    def test_reclaimable_fail_is_a_refusal_on_both_engines(self, monkeypatch):
        _patch_ceilings(monkeypatch, usable=50.0, working_set=200.0)
        for hard in (True, False):
            report = evaluate_fit(90.0, 8.0, hard_working_set=hard)
            assert report.verdict == "fail", f"hard_working_set={hard}"


@pytest.mark.unit
class TestReportFields:
    def test_pass_report_numbers(self, monkeypatch):
        _patch_ceilings(monkeypatch, usable=100.0, working_set=80.0)
        report = evaluate_fit(60.0, 8.0, hard_working_set=True)
        assert report.verdict == "pass"
        assert report.weights_gb == 60.0
        assert report.reclaimable_gb == 100.0
        assert report.working_set_gb == 80.0
        assert report.kv_headroom_gb == pytest.approx(20.0)
        assert report.estimated is False

    def test_sysctl_hint_only_at_os_default(self, monkeypatch):
        # Over the working set with sysctl=0 -> actionable hint. With the
        # sysctl already raised, the ceiling is a deliberate choice: no hint.
        _patch_ceilings(monkeypatch, usable=200.0, working_set=80.0, sysctl=0)
        hinted = evaluate_fit(90.0, 8.0, hard_working_set=False)
        assert hinted.sysctl_suggest_mb == int((98.0 + 8) * 1024)
        _patch_ceilings(monkeypatch, usable=200.0, working_set=80.0, sysctl=178176)
        raised = evaluate_fit(90.0, 8.0, hard_working_set=False)
        assert raised.sysctl_suggest_mb is None

    def test_max_buffer_warn_is_weights_only(self, monkeypatch):
        # The per-allocation cap compares WEIGHTS (one allocation's worth),
        # not weights+headroom, and never fails the fit by itself.
        _patch_ceilings(monkeypatch, usable=200.0, working_set=200.0, max_buffer=50.0)
        report = evaluate_fit(60.0, 8.0, hard_working_set=True)
        buf = next(l for l in report.lines if l.ceiling == "metal_max_buffer")
        assert buf.verdict == "warn"
        assert buf.need_gb == 60.0
        assert report.verdict == "warn"

    def test_no_metal_reports_ram_only(self, monkeypatch):
        _patch_ceilings(monkeypatch, usable=100.0, working_set=None)
        report = evaluate_fit(60.0, 8.0, hard_working_set=True)
        assert [l.ceiling for l in report.lines] == ["reclaimable_ram"]
        assert report.working_set_gb is None
        assert report.kv_headroom_gb is None
        assert report.verdict == "pass"


@pytest.mark.unit
class TestSizing:
    def _gguf(self, path, size):
        path.write_bytes(b"\0" * size)
        return path

    def test_shard_set_counts_the_whole_set(self, tmp_path):
        # The named shard can be a tiny index file; sizing it alone is wrong
        # by orders of magnitude (the DeepSeek trap ram_report exists for).
        self._gguf(tmp_path / "m-00001-of-00003.gguf", 10)
        self._gguf(tmp_path / "m-00002-of-00003.gguf", 1000)
        self._gguf(tmp_path / "m-00003-of-00003.gguf", 1000)
        size_gb, notes = size_config_gb({"model_path": str(tmp_path / "m-00001-of-00003.gguf")})
        assert size_gb == pytest.approx(2010 / ram_fit.GB)
        assert any("3-shard set" in n for n in notes)

    def test_dir_sizing_and_sidecars(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"\0" * 500)
        (d / "config.json").write_bytes(b"\0" * 99)  # not weights, not counted
        mmproj = self._gguf(tmp_path / "mmproj.gguf", 200)
        size_gb, notes = size_config_gb({"model_path": str(d), "mmproj_path": str(mmproj)})
        assert size_gb == pytest.approx(700 / ram_fit.GB)
        assert any("mmproj" in n for n in notes)

    def test_missing_path_notes_it(self, tmp_path):
        size_gb, notes = size_config_gb({"model_path": str(tmp_path / "nope")})
        assert size_gb == 0.0
        assert notes == ["model_path does not exist"]

    def test_is_mlx_config_is_layout_based(self):
        assert is_mlx_config({"model_path": "/models/some-dir"})
        assert not is_mlx_config({"model_path": "/models/model.gguf"})

    def test_fit_for_config_derives_hard_flag_from_layout(self, tmp_path, monkeypatch):
        _patch_ceilings(monkeypatch, usable=1000.0, working_set=1000.0)
        gguf = self._gguf(tmp_path / "m.gguf", 10)
        report = fit_for_config({"model_path": str(gguf)})
        assert report.hard_working_set is False
        d = tmp_path / "mlxdir"
        d.mkdir()
        report = fit_for_config({"model_path": str(d)})
        assert report.hard_working_set is True

    def test_fit_for_config_honors_explicit_hard_flag(self, tmp_path, monkeypatch):
        # The server passes the provider-derived truth; layout must not win.
        _patch_ceilings(monkeypatch, usable=1000.0, working_set=1000.0)
        gguf = self._gguf(tmp_path / "m.gguf", 10)
        report = fit_for_config({"model_path": str(gguf)}, hard_working_set=True)
        assert report.hard_working_set is True
