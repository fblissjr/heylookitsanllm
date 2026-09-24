# tests/unit/test_engine_classifier.py
#
# The shared engine taxonomy (tests/helpers/engines.py) that tests/smoke and
# tests/eval both consume. It is test code, but it is the thing that decides
# whether a live run gets to claim coverage, so it gets checked like product
# code: the two harnesses only report what it says.
#
# Model-free and server-free -- `classify` takes an injectable fetcher.

from helpers.engines import ARMS, classify, format_coverage


def _server(models, admin_models=None, admin_status=200):
    """A fake (path) -> (status, body) fetcher over two canned endpoints."""
    def fetch(path):
        if path == "/v1/models":
            return 200, {"data": models}
        if path == "/v1/admin/models":
            return admin_status, ({"models": admin_models or []}
                                  if admin_status == 200 else None)
        raise AssertionError(f"unexpected path {path}")
    return fetch


def _row(mid, runtime, **extra):
    return {"id": mid, "engine": {"runtime": {"value": runtime}}, **extra}


def _cov(*rows_caps):
    """rows_caps: (admin row, capabilities) pairs."""
    return classify("x", get_json=_server(
        models=[{"id": r["id"], "capabilities": caps} for r, caps in rows_caps],
        admin_models=[r for r, _ in rows_caps]))


class TestArms:
    def test_runtime_and_vision_decide_the_arm(self):
        cov = _cov((_row("t", "mlx-vlm"), ["chat"]),
                   (_row("v", "mlx-vlm"), ["chat", "vision"]),
                   (_row("g", "llama.cpp"), ["chat", "vision"]))
        assert cov.by_engine == {"t": "mlx-text", "v": "mlx-vision", "g": "gguf"}
        assert not cov.unclassified
        assert cov.engines_of(["t", "g"]) == ["mlx-text", "gguf"]

    def test_no_runtime_is_a_named_hole_never_a_guess(self):
        # A token-guarded admin endpoint, or a server older than the engine
        # contract: the model is reported, not classified by inference.
        cov = classify("x", get_json=_server(
            models=[{"id": "m", "capabilities": ["vision"]}], admin_status=401))
        assert cov.by_engine == {}
        assert "m" in cov.unclassified

    def test_residency_is_carried_through(self):
        # The only cheapness signal a live harness has when choosing an arm's
        # model; losing it makes the smoke run cost a cold load it did not need.
        cov = _cov((_row("a", "mlx-vlm", loaded=True), []), (_row("b", "mlx-vlm"), []))
        assert cov.resident == {"a"}


class TestCoverageReport:
    def test_every_arm_is_reported_and_an_empty_one_is_never_green(self):
        cov = _cov((_row("t", "mlx-vlm"), []), (_row("v", "mlx-vlm"), ["vision"]))
        text = format_coverage(cov, spanned=["mlx-text"])
        for arm in ARMS:
            assert arm in text
        assert "mlx-text   covered" in text
        assert "mlx-vision UNCOVERED -- 1 model(s) served, none run" in text
        assert "gguf       UNCOVERED -- no model served for this arm" in text
