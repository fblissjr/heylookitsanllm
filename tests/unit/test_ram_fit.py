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
    # The suggestion caps itself against total RAM; a unit test must not
    # read the host's.
    monkeypatch.setattr(ram_fit, "total_ram_gb", lambda: None)


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

    def test_the_suggested_limit_is_enough_and_only_offered_at_os_default(self, monkeypatch):
        # Property: raising the sysctl to what the report suggests gives the
        # same model a working set it passes with headroom to spare, on both
        # engines (hard = MLX). Over the working set with sysctl=0 -> a hint;
        # applied -> pass, not thin, and no second hint. With the sysctl
        # already raised, the ceiling is a deliberate choice: no hint.
        for hard in (True, False):
            for weights, headroom, working_set in ((90.0, 8.0, 80.0), (145.0, 8.0, 161.0),
                                                   (30.0, 40.0, 60.0)):
                _patch_ceilings(monkeypatch, usable=500.0, working_set=working_set,
                                max_buffer=500.0, sysctl=0)
                hinted = evaluate_fit(weights, headroom, hard_working_set=hard)
                assert hinted.sysctl_suggest_mb is not None, (weights, working_set)
                applied_gb = hinted.sysctl_suggest_mb / 1024
                _patch_ceilings(monkeypatch, usable=500.0, working_set=applied_gb,
                                max_buffer=500.0, sysctl=hinted.sysctl_suggest_mb)
                after = evaluate_fit(weights, headroom, hard_working_set=hard)
                assert after.verdict == "pass", (hard, weights, applied_gb)
                assert after.headroom_thin is False, (hard, weights, applied_gb)
                assert after.sysctl_suggest_mb is None
            _patch_ceilings(monkeypatch, usable=200.0, working_set=80.0, sysctl=178176)
            assert evaluate_fit(90.0, 8.0, hard_working_set=hard).sysctl_suggest_mb is None

    def test_thin_headroom_is_flagged_and_hinted(self, monkeypatch):
        # Under the working set, but with less than THIN_HEADROOM_GB left for
        # KV + compute: the report says so (the provider inherits llama's
        # micro-batch off this) and, at the OS-default sysctl, suggests the
        # raise. The 2026-09-07 case: 145 GiB of weights under a 161 GiB set.
        _patch_ceilings(monkeypatch, usable=200.0, working_set=161.0, max_buffer=200.0, sysctl=0)
        thin = evaluate_fit(145.0, 8.0, hard_working_set=False)
        assert thin.verdict == "pass" and thin.headroom_thin is True
        assert thin.sysctl_suggest_mb is not None
        roomy = evaluate_fit(100.0, 8.0, hard_working_set=False)
        assert roomy.headroom_thin is False and roomy.sysctl_suggest_mb is None
        _patch_ceilings(monkeypatch, usable=200.0, working_set=161.0, max_buffer=200.0, sysctl=178176)
        assert evaluate_fit(145.0, 8.0, hard_working_set=False).sysctl_suggest_mb is None

    def test_suggestion_never_crowds_the_os(self, monkeypatch):
        # Property: whatever the machine and the model, the suggested wired
        # limit leaves at least OS_RESERVE_GB (the limit the module sets) of
        # total RAM to everything that is not the model.
        offered = 0
        for total in (64.0, 96.0, 128.0, 192.0):
            for weights in (total * 0.5, total * 0.8, total * 0.95):
                _patch_ceilings(monkeypatch, usable=total, working_set=total * 0.75,
                                sysctl=0)
                monkeypatch.setattr(ram_fit, "total_ram_gb", lambda t=total: t)
                report = evaluate_fit(weights, 8.0, hard_working_set=False)
                if report.sysctl_suggest_mb is None:
                    continue  # roomy: nothing suggested
                offered += 1
                assert report.sysctl_suggest_mb / 1024 <= total - ram_fit.OS_RESERVE_GB, (
                    total, weights, report.sysctl_suggest_mb)
        assert offered >= 8

    def test_max_buffer_compares_ONE_allocation_not_the_sum(self, monkeypatch):
        # The cap limits a single buffer, so the summed weight is the wrong
        # thing to compare against it. The old check used the sum and therefore
        # told a SHARDED model it "needs a sharded/split layout" -- advice it
        # had already taken. Never fails the fit by itself either way.
        _patch_ceilings(monkeypatch, usable=200.0, working_set=200.0, max_buffer=50.0)

        # sum over the cap, no single file near it -> NO warning
        sharded = evaluate_fit(60.0, 8.0, hard_working_set=True, largest_alloc_gb=20.0)
        assert not [l for l in sharded.lines if l.ceiling == "metal_max_buffer"]
        assert sharded.verdict == "pass"

        # one file over the cap -> warning, reporting THAT file, not the sum
        monolith = evaluate_fit(60.0, 8.0, hard_working_set=True, largest_alloc_gb=55.0)
        buf = next(l for l in monolith.lines if l.ceiling == "metal_max_buffer")
        assert buf.verdict == "warn" and buf.need_gb == 55.0
        assert monolith.verdict == "warn"

        # unsizeable -> no verdict at all, rather than a wrong one
        unknown = evaluate_fit(60.0, 8.0, hard_working_set=True)
        assert not [l for l in unknown.lines if l.ceiling == "metal_max_buffer"]

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

    # One property, four rows (two came from test_ram_report.py): a shard set
    # is sized as the WHOLE set, its largest shard as ONE allocation, and a
    # standalone file as itself. Every row asserts both public measures.
    @pytest.mark.parametrize("sizes, named, set_bytes, largest_bytes", [
        # The Metal cap limits ONE buffer: the largest shard, not the set.
        pytest.param({"m-00001-of-00003.gguf": 30 << 20, "m-00002-of-00003.gguf": 40 << 20,
                      "m-00003-of-00003.gguf": 20 << 20},
                     "m-00001-of-00003.gguf", 90 << 20, 40 << 20,
                     id="largest_alloc_reads_the_biggest_shard_not_the_set"),
        # The named shard can be a tiny index file; sizing it alone is wrong
        # by orders of magnitude (the DeepSeek trap ram_report exists for).
        pytest.param({"m-00001-of-00003.gguf": 10, "m-00002-of-00003.gguf": 1000,
                      "m-00003-of-00003.gguf": 1000},
                     "m-00001-of-00003.gguf", 2010, 1000,
                     id="shard_set_counts_the_whole_set"),
        # The other way this gate lies: sizing the named first shard called a
        # 127 GiB model "5 MB".
        pytest.param({"m-00001-of-00003.gguf": 100, "m-00002-of-00003.gguf": 90_000,
                      "m-00003-of-00003.gguf": 80_000},
                     "m-00001-of-00003.gguf", 170_100, 90_000,
                     id="a_shard_is_sized_as_its_whole_set"),
        pytest.param({"solo.gguf": 1234}, "solo.gguf", 1234, 1234,
                     id="a_standalone_file_is_sized_as_itself"),
    ])
    def test_shard_sizing(self, tmp_path, sizes, named, set_bytes, largest_bytes):
        from heylook_llm.ram_fit import largest_alloc_gb
        for name, size in sizes.items():
            self._gguf(tmp_path / name, size)
        cfg = {"model_path": str(tmp_path / named)}
        size_gb, notes = size_config_gb(cfg)
        assert size_gb == pytest.approx(set_bytes / ram_fit.GB)
        assert largest_alloc_gb(cfg) == pytest.approx(largest_bytes / ram_fit.GB)
        is_set = len(sizes) > 1
        assert any(f"{len(sizes)}-shard set" in n for n in notes) is is_set

    def test_dir_sizing_and_sidecars(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"\0" * 500)
        (d / "config.json").write_bytes(b"\0" * 99)  # not weights, not counted
        mmproj = self._gguf(tmp_path / "mmproj.gguf", 200)
        size_gb, notes = size_config_gb({"model_path": str(d), "mmproj_path": str(mmproj)})
        assert size_gb == pytest.approx(700 / ram_fit.GB)
        assert any("mmproj" in n for n in notes)

    def test_rpc_tensor_split_scales_primary_gguf_weights_only(self, tmp_path):
        from heylook_llm.ram_fit import local_weight_fraction

        primary = self._gguf(tmp_path / "target.gguf", 1000)
        drafter = self._gguf(tmp_path / "draft.gguf", 100)
        cfg = {
            "model_path": str(primary),
            "draft_model_path": str(drafter),
            "rpc": "worker.local:50052",
            "tensor_split": "20,80",
        }
        frac, note = local_weight_fraction(cfg)
        assert frac == pytest.approx(0.8)
        assert note and "80% local" in note

        size_gb, notes = size_config_gb(cfg)
        # 80% of 1000-byte target + 100% of 100-byte drafter = 900 bytes local
        assert size_gb == pytest.approx(900 / ram_fit.GB)
        assert any("80% local" in n for n in notes)

        # Mismatched count or MLX dir falls back to 1.0
        assert local_weight_fraction({**cfg, "tensor_split": "80"})[0] == 1.0
        assert local_weight_fraction({"model_path": str(tmp_path), "rpc": "1.2.3.4:50052", "tensor_split": "20,80"})[0] == 1.0

        # MLX pipeline_ratios scales rank-0 primary weights
        mlx_dir = tmp_path / "mlx_model"
        mlx_dir.mkdir()
        (mlx_dir / "model.safetensors").write_bytes(b"\0" * 1000)
        mlx_cfg = {"model_path": str(mlx_dir), "pipeline_ratios": [0.8, 0.2]}
        mlx_frac, mlx_note = local_weight_fraction(mlx_cfg)
        assert mlx_frac == pytest.approx(0.8)
        assert mlx_note and "80% rank-0 local" in mlx_note
        mlx_size, mlx_notes = size_config_gb(mlx_cfg)
        assert mlx_size == pytest.approx(800 / ram_fit.GB)
        assert any("80% rank-0 local" in n for n in mlx_notes)

    def test_missing_path_notes_it(self, tmp_path):
        size_gb, notes = size_config_gb({"model_path": str(tmp_path / "nope")})
        assert size_gb == 0.0
        assert notes == ["model_path does not exist"]

    def test_is_mlx_config_is_layout_based(self):
        assert is_mlx_config({"model_path": "/models/some-dir"})
        assert not is_mlx_config({"model_path": "/models/model.gguf"})

    @pytest.mark.parametrize("cases", [
        # (layout, explicit hard flag, expected report.hard_working_set)
        pytest.param([("gguf", None, False), ("mlx_dir", None, True)],
                     id="fit_for_config_derives_hard_flag_from_layout"),
        # The server passes the provider-derived truth; layout must not win.
        pytest.param([("gguf", True, True)],
                     id="fit_for_config_honors_explicit_hard_flag"),
    ])
    def test_fit_for_config_hard_flag(self, tmp_path, monkeypatch, cases):
        _patch_ceilings(monkeypatch, usable=1000.0, working_set=1000.0)
        paths = {"gguf": self._gguf(tmp_path / "m.gguf", 10), "mlx_dir": tmp_path / "mlxdir"}
        paths["mlx_dir"].mkdir()
        for layout, explicit, expected in cases:
            kwargs = {} if explicit is None else {"hard_working_set": explicit}
            report = fit_for_config({"model_path": str(paths[layout])}, **kwargs)
            assert report.hard_working_set is expected, (layout, explicit)

    def test_can_fallback_to_single_node_and_recommend_split_ratios(self, tmp_path, monkeypatch):
        from heylook_llm.ram_fit import can_fallback_to_single_node, recommend_split_ratios

        mlx_dir = tmp_path / "mlx_model"
        mlx_dir.mkdir()
        # 100 GiB model (represented by mocked size_config_gb via file size or monkeypatch)
        (mlx_dir / "model.safetensors").write_bytes(b"\0" * 1000)
        cfg = {
            "model_path": str(mlx_dir),
            "pipeline_ratios": [0.8, 0.2],
            "pipeline_peers": ["peer0.local", "peer1.local"],
        }

        # When local host working set is tiny, full model exceeds working set -> cannot fallback
        _patch_ceilings(monkeypatch, usable=500.0, working_set=80.0)
        orig_size = ram_fit.size_config_gb
        monkeypatch.setattr(
            ram_fit,
            "size_config_gb",
            lambda c: (80.0, ["pipeline_ratios=[0.8, 0.2]"])
            if c.get("pipeline_ratios")
            else (100.0, []),
        )
        can_solo, solo_report = can_fallback_to_single_node(cfg, hard_working_set=True)
        assert can_solo is False
        assert solo_report.verdict == "fail"

        # When local host has 180 GiB working set, 100 GiB full model fits solo -> can fallback!
        _patch_ceilings(monkeypatch, usable=500.0, working_set=180.0)
        can_solo_ok, solo_report_ok = can_fallback_to_single_node(cfg, hard_working_set=True)
        assert can_solo_ok is True
        assert solo_report_ok.fits is True
        monkeypatch.setattr(ram_fit, "size_config_gb", orig_size)

        # recommend_split_ratios computes local/worker split from ceilings
        r_local, r_worker = recommend_split_ratios(180.0, local_working_set_gb=180.0, worker_working_set_gb=48.0)
        assert 0.7 <= r_local <= 0.9
        assert pytest.approx(r_local + r_worker, abs=0.02) == 1.0

    def test_provider_single_node_fallback_vs_fail_fast_when_peer_offline(self, tmp_path, monkeypatch):
        import socket
        from heylook_llm.providers.llama_server_provider import LlamaServerProvider

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_unused:
            s_unused.bind(("127.0.0.1", 0))
            unused_port = s_unused.getsockname()[1]

        gguf_file = self._gguf(tmp_path / "model.gguf", 1024)
        base_cfg = {
            "model_path": str(gguf_file),
            "rpc": f"127.0.0.1:{unused_port}",
            "tensor_split": "20,80",
        }

        # Case 1: Model fits solo on current host -> auto-fallback to single-node (strips rpc + tensor_split)
        _patch_ceilings(monkeypatch, usable=180.0, working_set=180.0)
        prov_ok = LlamaServerProvider("test-gguf", dict(base_cfg), verbose=False)
        prov_ok._preflight_rpc_servers()
        assert "rpc" not in prov_ok.config
        assert "tensor_split" not in prov_ok.config

        # Case 2: Model exceeds current host solo RAM -> fail fast with RuntimeError naming peer + size
        _patch_ceilings(monkeypatch, usable=10.0, working_set=10.0)
        monkeypatch.setattr(
            ram_fit,
            "can_fallback_to_single_node",
            lambda _cfg, **kw: (False, evaluate_fit(190.0, 8.0, hard_working_set=False)),
        )
        prov_big = LlamaServerProvider("big-gguf", dict(base_cfg), verbose=False)
        with pytest.raises(RuntimeError, match="exceed this host's"):
            prov_big._preflight_rpc_servers()

