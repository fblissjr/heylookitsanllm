# tests/unit/test_engine_classifier.py
#
# The shared engine taxonomy (tests/helpers/engines.py) that tests/smoke and
# tests/eval both consume. It is test code, but it is the thing that decides
# whether a live run gets to claim coverage, so it gets checked like product
# code: the two harnesses only report what it says.
#
# Model-free and server-free -- `classify` takes an injectable fetcher.

import pytest

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


class TestEngineIdentity:
    """`effective_loader` is the engine; provider and capabilities are not."""

    def test_server_answer_wins_over_the_vision_capability(self):
        # The conflation the whole module exists to prevent: a dual-capable VLM
        # pinned to `loader = "mlx-lm"` still REPORTS vision. Splitting on the
        # capability puts it in the mlx-vlm arm, and the run goes green having
        # decoded with mlx-lm twice.
        cov = classify("x", get_json=_server(
            models=[{"id": "pinned-vlm", "capabilities": ["vision"]}],
            admin_models=[{"id": "pinned-vlm", "provider": "mlx",
                           "effective_loader": "mlx-lm"}],
        ))
        assert cov.by_engine == {"pinned-vlm": "mlx-lm"}
        assert not cov.unconfirmable

    def test_gguf_is_one_engine_named_by_provider(self):
        cov = classify("x", get_json=_server(
            models=[{"id": "g", "capabilities": ["vision"]}],
            admin_models=[{"id": "g", "provider": "gguf",
                           "effective_loader": None}],
        ))
        assert cov.by_engine == {"g": "gguf"}

    def test_embeddings_are_excluded_not_unclassified(self):
        # Two different answers that must not print the same: "we chose not to
        # cover this" versus "we could not tell what this is".
        cov = classify("x", get_json=_server(
            models=[{"id": "e"}],
            admin_models=[{"id": "e", "provider": "mlx_embedding"}],
        ))
        assert "e" in cov.excluded
        assert "e" not in cov.unclassified
        assert "e" not in cov.by_engine

    def test_residency_is_carried_through(self):
        # The only cheapness signal a live harness has when choosing an arm's
        # model; losing it makes the smoke run cost a cold load it did not need.
        cov = classify("x", get_json=_server(
            models=[{"id": "a"}, {"id": "b"}],
            admin_models=[{"id": "a", "provider": "mlx", "effective_loader": "mlx-lm",
                           "loaded": True},
                          {"id": "b", "provider": "mlx", "effective_loader": "mlx-lm"}],
        ))
        assert cov.resident == {"a"}


class TestDegradedServers:
    """A classifier that lies when the server is old or locked down is worse
    than one that says it does not know."""

    def test_missing_effective_loader_infers_but_records_unconfirmable(self):
        cov = classify("x", get_json=_server(
            models=[{"id": "v", "capabilities": ["vision"]},
                    {"id": "t", "capabilities": []}],
            admin_models=[{"id": "v", "provider": "mlx"},
                          {"id": "t", "provider": "mlx"}],
        ))
        assert cov.by_engine == {"v": "mlx-vlm", "t": "mlx-lm"}
        assert set(cov.unconfirmable) == {"v", "t"}

    def test_admin_401_degrades_to_the_inference_it_replaced(self):
        # Both harnesses now depend on an endpoint behind `require_admin_token`,
        # so the token-gated server is a REAL state, not a hypothetical. It
        # degrades rather than collapsing because `/v1/models` carries
        # `provider` (api.list_models sets it from the config): gguf is still
        # named outright, and mlx falls back to the capability inference and is
        # reported as not-confirmed. A run against such a server says what it
        # could not confirm instead of reporting a coverage hole for every
        # model it serves.
        cov = classify("x", get_json=_server(
            models=[{"id": "v", "provider": "mlx", "capabilities": ["vision"]},
                    {"id": "t", "provider": "mlx", "capabilities": []},
                    {"id": "g", "provider": "gguf"}],
            admin_status=401,
        ))
        assert cov.by_engine == {"v": "mlx-vlm", "t": "mlx-lm", "g": "gguf"}
        assert set(cov.unconfirmable) == {"v", "t"}   # gguf needs no loader
        assert not cov.unclassified

    def test_a_model_with_no_provider_anywhere_is_unclassified_not_guessed(self):
        # `/v1/models` omits `provider` when the router serves an id with no
        # resolvable config. Nothing to classify FROM -- and guessing "mlx"
        # would let a coverage hole print as a covered arm.
        cov = classify("x", get_json=_server(
            models=[{"id": "a", "capabilities": ["vision"]}],
            admin_status=401,
        ))
        assert cov.by_engine == {}
        assert "a" in cov.unclassified

    def test_models_endpoint_failure_raises(self):
        def fetch(_path):
            return 500, None
        with pytest.raises(RuntimeError):
            classify("x", get_json=fetch)


class TestCoverageReport:
    """The sentence the plan exists to make printable."""

    def test_every_served_model_classifies_or_is_named(self):
        # Phase 1's own check: an unclassifiable model is a coverage hole with
        # no name, so every id must land in exactly one bucket.
        models = [{"id": "t"}, {"id": "v", "capabilities": ["vision"]},
                  {"id": "g"}, {"id": "e"}, {"id": "weird"}]
        cov = classify("x", get_json=_server(
            models=models,
            admin_models=[
                {"id": "t", "provider": "mlx", "effective_loader": "mlx-lm"},
                {"id": "v", "provider": "mlx", "effective_loader": "mlx-vlm"},
                {"id": "g", "provider": "gguf"},
                {"id": "e", "provider": "mlx_embedding"},
                {"id": "weird", "provider": "quantum-goat"},
            ],
        ))
        placed = set(cov.by_engine) | set(cov.excluded) | set(cov.unclassified)
        assert placed == {m["id"] for m in models}
        assert cov.unclassified.keys() == {"weird"}

    def test_absent_engine_reads_as_uncovered_never_as_green(self):
        cov = classify("x", get_json=_server(
            models=[{"id": "t"}],
            admin_models=[{"id": "t", "provider": "mlx", "effective_loader": "mlx-lm"}],
        ))
        assert cov.arms_present() == ["mlx-lm"]
        assert set(cov.arms_absent()) == {"mlx-vlm", "gguf"}
        text = format_coverage(cov)
        assert "mlx-lm   covered" in text
        for absent in ("mlx-vlm", "gguf"):
            line = next(l for l in text.splitlines() if l.strip().startswith(absent))
            assert "UNCOVERED" in line

    def test_a_served_but_unrun_engine_is_still_uncovered(self):
        # The failure this plan is aimed at: the engine EXISTS on the server and
        # the run simply did not touch it. Reporting that as covered because a
        # model was available is the quiet green.
        cov = classify("x", get_json=_server(
            models=[{"id": "t"}, {"id": "v", "capabilities": ["vision"]}],
            admin_models=[
                {"id": "t", "provider": "mlx", "effective_loader": "mlx-lm"},
                {"id": "v", "provider": "mlx", "effective_loader": "mlx-vlm"},
            ],
        ))
        text = format_coverage(cov, spanned=["mlx-lm"])
        assert "mlx-vlm  UNCOVERED -- 1 model(s) served, none run" in text

    def test_engines_of_reports_what_a_model_list_spans(self):
        cov = classify("x", get_json=_server(
            models=[{"id": "t"}, {"id": "v", "capabilities": ["vision"]},
                    {"id": "g"}],
            admin_models=[
                {"id": "t", "provider": "mlx", "effective_loader": "mlx-lm"},
                {"id": "v", "provider": "mlx", "effective_loader": "mlx-vlm"},
                {"id": "g", "provider": "gguf"},
            ],
        ))
        assert cov.engines_of(["t", "g"]) == ["mlx-lm", "gguf"]
        assert cov.engines_of(["t"]) == ["mlx-lm"]
        # An unknown id spans nothing rather than raising -- the eval harness
        # warns about a bad --models entry and keeps going.
        assert cov.engines_of(["nope"]) == []

    def test_arms_order_is_stable(self):
        assert ARMS == ("mlx-lm", "mlx-vlm", "gguf")
