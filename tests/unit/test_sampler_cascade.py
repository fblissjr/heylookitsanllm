"""The sampler cascade: what a request actually runs with.

There is no registry any more. The five bundled TOMLs, the `SamplerRegistry`
that loaded them, `ChatRequest.sampler` and models.toml `default_sampler` were
removed in v2.0.30 -- three of the five had no consumer, `thinking` was a
no-op behind a hardcoded fallback, and the frontend never touched any of it.

What is left is four layers, and these tests pin their ORDER, because both
providers resolve through this one function and a drift between them is the
failure the shared cascade exists to prevent:

1.  Floor -- two fallback values, a max_tokens stop, and four off-switches
1b. Vendor -- the model's OWN published settings (generation_config.json on
    MLX, the GGUF header's general.sampling.* on gguf). The per-model answer,
    and the one that should normally win over anything global.
2.  Model sampler fields from models.toml
3.  Request explicit fields -- always win
"""

from __future__ import annotations

from unittest import mock

import pytest
from heylook_llm.samplers import GLOBAL_SAMPLER_FLOOR

class TestVendorSampling:
    """load_vendor_sampling: source of the per-model vendor cascade layer.

    Claim: a model dir's generation_config.json yields exactly the sampling
    keys (temperature/top_p/top_k), and a missing or broken file yields {}
    rather than blocking a load.
    """

    def test_reads_sampling_keys_only(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            '{"temperature": 1.0, "top_k": 64, "top_p": 0.95,'
            ' "do_sample": true, "eos_token_id": [1, 106, 50]}'
        )
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {
            "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        }

    @pytest.mark.parametrize("body", [None, "{nope"], ids=["missing-file", "malformed-json"])
    def test_missing_or_malformed_file_is_empty(self, tmp_path, body):
        if body is not None:
            (tmp_path / "generation_config.json").write_text(body)
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {}

    def test_non_numeric_values_dropped(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            '{"temperature": "high", "top_k": true, "top_p": 0.9}'
        )
        from heylook_llm.samplers import load_vendor_sampling

        assert load_vendor_sampling(str(tmp_path)) == {"top_p": 0.9}


class TestResolveEffectiveSampling:
    """resolve_effective_sampling: the ONE cascade shared by both providers.

    Claim: MLX and gguf resolve request sampling through the same function
    with the same semantics; the two hand-mirrored implementations (and
    their duplicated thinking fallbacks) are gone. Deleting any test here
    lets the providers' cascades drift apart again.
    """

    @staticmethod
    def _req(**kw):
        from heylook_llm.config import ChatRequest

        body = {"messages": [{"role": "user", "content": "hi"}]}
        body.update(kw)
        return ChatRequest.model_validate(body)

    # floor < vendor < model config < request, per key
    @pytest.mark.parametrize(
        "request_fields, config, vendor, expected",
        [
            ({}, {}, None,
             {"temperature": GLOBAL_SAMPLER_FLOOR["temperature"],
              "max_tokens": GLOBAL_SAMPLER_FLOOR["max_tokens"]}),
            ({}, {}, {"temperature": 1.0, "top_k": 64}, {"temperature": 1.0, "top_k": 64}),
            ({"enable_thinking": True, "presence_penalty": 0.2, "temperature": 0.9},
             {"temperature": 0.3}, {"temperature": 1.0},
             {"temperature": 0.9, "presence_penalty": 0.2}),
        ],
        ids=["floor-only", "vendor-overlay-beats-floor", "explicit-request-fields-win"],
    )
    def test_layer_order(self, request_fields, config, vendor, expected):
        from heylook_llm.samplers import resolve_effective_sampling

        merged = resolve_effective_sampling(self._req(**request_fields), config, vendor=vendor)
        for key, value in expected.items():
            assert merged[key] == value, key

    def test_thinking_resolves_the_switch_and_changes_nothing_else(self):
        """Thinking flips the switch and NO sampler value with it (v2.0.32).

        Until then it dragged `presence_penalty 1.5` along on every
        thinking-capable model on both engines -- a value derived from one
        family's July 2026 docs, made automatic on the strength of a single
        observed gemma loop, and never measured. Both engines still have to
        read the same resolved bool, which is why the switch is materialized
        here; what is gone is the sampler change riding on it.
        """
        from heylook_llm.samplers import resolve_effective_sampling, GLOBAL_SAMPLER_FLOOR

        on = resolve_effective_sampling(self._req(enable_thinking=True), {})
        assert on["enable_thinking"] is True
        assert on["presence_penalty"] == GLOBAL_SAMPLER_FLOOR["presence_penalty"], \
            "thinking still applies a repetition penalty nobody asked for"

        off = resolve_effective_sampling(self._req(enable_thinking=False), {})
        # The switch is the ONLY thing that differs between the two.
        assert {k: v for k, v in on.items() if k != "enable_thinking"} == \
               {k: v for k, v in off.items() if k != "enable_thinking"}

    def test_repetition_control_is_still_reachable_per_model(self):
        """Removing the automatic overlay must not remove the ability.

        A model that genuinely loops is fixed on THAT model rather than by a
        global default, and both engines can carry the field -- gguf gained it
        in the same change, having had no per-model lever before.
        """
        from heylook_llm.config import PROVIDER_CONFIG_CLASSES
        from heylook_llm.samplers import resolve_effective_sampling

        for provider in ("mlx", "gguf"):
            assert "presence_penalty" in PROVIDER_CONFIG_CLASSES[provider].model_fields, \
                f"{provider} cannot set presence_penalty per model"

        merged = resolve_effective_sampling(
            self._req(enable_thinking=True), {"presence_penalty": 1.5})
        assert merged["presence_penalty"] == 1.5


class TestThinkingDefault:
    """The thinking switch resolves request > models.toml > CAPABILITY
    (v1.79.62). From v1.50.0 unset meant off on every model; a thinking
    model that silently did not think unless someone found the config flag
    was the standing complaint, and the reason for off-by-default (no way
    to send an explicit off) is gone now that the UI can."""

    @pytest.mark.parametrize(
        "cases",
        [
            # (request enable_thinking, model config, capable, resolved)
            [(None, {"enable_thinking": True}, False, True),     # model config sets it
             (False, {"enable_thinking": True}, False, False),   # request False beats config
             (None, {}, True, True)],                            # silent: capability
            [(None, {}, True, True), (None, {}, False, False)],
            [(None, {"enable_thinking": False}, True, False),
             (None, {"enable_thinking": True}, False, True)],
            [(False, {"enable_thinking": True}, True, False),
             (True, {"enable_thinking": False}, False, True)],
        ],
        ids=["documented-order", "unset-everywhere-follows-capability",
             "config-pins-either-way-over-capability", "request-explicit-wins"],
    )
    def test_switch_resolution(self, cases):
        from heylook_llm.config import ChatRequest
        from heylook_llm.samplers import resolve_effective_sampling

        for request_value, config, capable, resolved in cases:
            req = ChatRequest.model_validate({
                "messages": [{"role": "user", "content": "hi"}],
                "enable_thinking": request_value})
            got = resolve_effective_sampling(req, config, thinking_capable=capable)
            assert got["enable_thinking"] is resolved, (request_value, config, capable)


@pytest.mark.unit
class TestSamplerDefaultsReporting:
    """`sampler_defaults` is what the settings panel prints in a blank field,
    so it has to be the cascade's ANSWER, not a second implementation of it.
    Every check here compares against `resolve_effective_sampling` itself or
    against a layer the cascade owns."""

    # The strongest available oracle: run the real cascade for an empty
    # request and demand agreement on every reported key, over (cfg,
    # capability, vendor) combos. Each case also pins the reported thinking
    # value to thinking_default's own answer (one bag since v2.0.33, so the
    # two cannot disagree by construction), and a row's pins name the value a
    # layer must produce: temperature/top_p/top_k ARE the vendor keys, and
    # dropping the vendor argument would report the global floor for every
    # model whose generation_config.json overrides it -- silently, and
    # precisely on the models where the number is worth showing. The panel
    # writes these keys back as request params, so a key no request accepts
    # would render a control that cannot do anything.
    @pytest.mark.parametrize(
        "cases",
        [
            [({"temperature": 0.42}, True, {"top_k": 64, "top_p": 0.9}, {})],
            [({}, False, {"top_k": 64}, {"top_k": 64}),
             ({}, False, None, {"top_k": GLOBAL_SAMPLER_FLOOR["top_k"]})],
            [({"temperature": 0.6}, False, {"temperature": 0.7}, {"temperature": 0.6})],
            [({}, True, None, {}), ({}, False, None, {}),
             ({"enable_thinking": False}, True, None, {})],
            [({}, True, None, {"enable_thinking": True}),
             ({"enable_thinking": False}, True, None, {"enable_thinking": False})],
            [({}, True, None, {})],
        ],
        ids=[
            "it-is-the-cascade", "vendor-layer-reaches-the-report",
            "models-toml-beats-vendor", "thinking-value-is-thinking-defaults-answer",
            "thinking-default-reports-the-same-answer", "keys-are-request-sampler-fields",
        ],
    )
    def test_it_is_the_cascade_not_a_re_derivation(self, cases):
        from heylook_llm.samplers import (REQUEST_SAMPLER_FIELDS, _NoRequest,
                                          resolve_effective_sampling, sampler_defaults,
                                          thinking_default)

        assert GLOBAL_SAMPLER_FLOOR["top_k"] != 64, "pick a value the floor does not already have"
        for cfg, cap, vendor, pins in cases:
            got = sampler_defaults(cfg, thinking_capable=cap, vendor=vendor)
            want = resolve_effective_sampling(_NoRequest(), cfg, vendor, thinking_capable=cap)
            assert got == {k: want[k] for k in REQUEST_SAMPLER_FIELDS if k in want}
            assert set(got) <= set(REQUEST_SAMPLER_FIELDS)
            assert got["enable_thinking"] is thinking_default(cfg, thinking_capable=cap)
            for key, value in pins.items():
                assert got[key] == value, (cfg, cap, vendor, key)

    def test_vendor_layer_reaches_the_report_on_every_engine(self):
        """Each engine's vendor layer must reach the reported defaults.

        This is the check the drift needed and did not have: `sampler_defaults`
        takes `vendor` from `capabilities._vendor_sampling_pairs`, which names
        the engines by hand, and gguf gained a vendor layer in its provider one
        commit after that gate was written -- so every gguf row advertised the
        global floor while generation used the GGUF header's values. Silent,
        because a plausible number is indistinguishable from the right one.

        Sentinels rather than real files: the claim is about the WIRING, and a
        fixture model would only prove it for whichever engine the fixture used.
        """
        from heylook_llm import capabilities, gguf_metadata, samplers
        from heylook_llm.config import ModelConfig

        cases = {
            "mlx": ("load_vendor_sampling", "/fake/mlx-model", 11),
            "gguf": ("_gguf_vendor", "/fake/model.gguf", 22),
        }
        for provider, (_, path, sentinel) in cases.items():
            capabilities._vendor_sampling_pairs.cache_clear()
            # Patched at the SOURCE readers (v2.0.74), which each engine's
            # describer calls through the module: any caller is caught.
            with mock.patch.object(samplers, "load_vendor_sampling",
                                   return_value={"top_k": sentinel}), \
                 mock.patch.object(gguf_metadata, "vendor_sampling",
                                   return_value={"top_k": sentinel}):
                mc = ModelConfig(id=f"t-{provider}", provider=provider,
                                 config={"model_path": path})
                got = capabilities.derived_model_facts(mc).sampler_defaults
            assert got["top_k"] == sentinel, (
                f"{provider}: the reported default ignored that engine's vendor "
                f"layer, so its rows will advertise the global floor while "
                f"generation uses the vendor values"
            )
        capabilities._vendor_sampling_pairs.cache_clear()
