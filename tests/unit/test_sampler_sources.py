# tests/unit/test_sampler_sources.py
"""`sampler_sources`: which layer each reported sampler default came from.

The panel prints it beside the value ("0.6 · vendor"), so it has to name the
layer the cascade actually used: the model's own stored value beats its
vendor values, which beat heylook's floor. Driven through the same
derivation both model rows read, on a gguf entry (a stored per-model sampler
field there is new in v2.0.150).
"""
from unittest import mock

import pytest


@pytest.mark.unit
def test_each_default_names_the_layer_it_came_from():
    from heylook_llm import capabilities, gguf_metadata
    from heylook_llm.config import ModelConfig

    capabilities._vendor_sampling_pairs.cache_clear()
    with mock.patch.object(gguf_metadata, "vendor_sampling",
                           return_value={"top_k": 20, "temperature": 0.6}):
        mc = ModelConfig(id="t-sources", provider="gguf",
                         config={"model_path": "/fake/model.gguf", "temperature": 0.3})
        facts = capabilities.derived_model_facts(mc)
    capabilities._vendor_sampling_pairs.cache_clear()

    assert (facts.sampler_defaults["temperature"], facts.sampler_sources["temperature"]) == (0.3, "model")
    assert (facts.sampler_defaults["top_k"], facts.sampler_sources["top_k"]) == (20, "vendor")
    assert facts.sampler_sources["min_p"] == "default"
    assert "enable_thinking" not in facts.sampler_sources
