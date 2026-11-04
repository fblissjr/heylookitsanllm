"""Embedding semantics and lifecycle checks, without loading model weights."""
import base64
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from heylook_llm.embedding_models import embedding_info, gguf_embedding_profile
from heylook_llm.schema.embeddings import EmbeddingRequest
from heylook_llm.providers.base import BaseProvider, InvalidGenerationRequest
from heylook_llm.providers.common.embedding import finish, resolved_items
from heylook_llm.providers.common.generation_gate import GenerationGate, GenerationCancelled
from heylook_llm.providers.abort import AbortEvent


@pytest.fixture
def info(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "embedding_gemma2", "vision_config": {}, "audio_config": {},
    }))
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps({"prompts": {"query": "query: "}}))
    return embedding_info("mlx", {"model_path": str(tmp_path)})


def test_multimodal_item_boundaries_and_prompt_applied_once(info):
    body = EmbeddingRequest(model="embed", input=[{"prompt_name": "query", "content": [
        {"type": "text", "text": "water"},
        *[{"type": kind, "source": {"type": "base64", "media_type": mime,
                                     "data": base64.b64encode(kind.encode()).decode()}}
          for kind, mime in (("image", "image/png"), ("audio", "audio/wav"), ("video", "video/mp4"))],
    ]}, "second"])
    with resolved_items(body, info, None) as items:
        assert len(items) == 2
        assert items[0][0] == "query: "
        assert [p["type"] for p in items[0][1]] == ["text", "image", "audio", "video"]
        paths = [Path(p["path"]) for p in items[0][1][1:]]
        assert [p.read_bytes() for p in paths] == [b"image", b"audio", b"video"]
    assert all(not p.exists() for p in paths)


@pytest.mark.parametrize("value", [[], [{"content": []}], [{"content": [{"type": "bogus"}]}]])
def test_invalid_input_rejected(value):
    with pytest.raises(ValidationError):
        EmbeddingRequest(model="embed", input=value)


def test_truncation_renormalizes_and_preserves_order(info):
    body = EmbeddingRequest(model="embed", input=["a", "b"], dimensions=128)
    result = finish(body, info, [[1.0] * 768, [2.0, *([0.0] * 767)]], 5)
    assert [v.index for v in result.data] == [0, 1]
    assert len(result.data[0].embedding) == 128
    assert math.isclose(sum(v*v for v in result.data[0].embedding), 1.0)
    assert result.data[1].embedding == [1.0] + [0.0] * 127


@pytest.mark.parametrize("vector", [[0.0] * 768, [float("nan")] * 768, [1.0]])
def test_bad_vectors_never_escape(info, vector):
    with pytest.raises(RuntimeError):
        finish(EmbeddingRequest(model="embed", input="x"), info, [vector], None)


def test_profile_cannot_split_noncausal_input():
    assert gguf_embedding_profile({}) == {"ctx_size": 8192, "n_batch": 8192, "n_ubatch": 8192}
    with pytest.raises(ValueError, match="ctx_size"):
        gguf_embedding_profile({"ctx_size": 4096, "n_ubatch": 512})


class Provider(BaseProvider):
    provider_name = "gguf"

    def load_model(self):
        pass

    def create_chat_completion(self, request, abort_event=None):
        raise NotImplementedError


def test_failure_and_queued_cancel_release_activity_and_gate(info):
    provider = Provider("embed", {}, False)
    provider._embedding = info
    provider._gen_gate = GenerationGate()
    body = EmbeddingRequest(model="embed", input="x")
    with patch("heylook_llm.providers.gguf_embeddings.embed", side_effect=ValueError("failure")):
        with pytest.raises(ValueError, match="failure"):
            provider.embed(body)
    assert provider.active_generations == 0
    assert not provider._gen_gate.busy
    abort = AbortEvent()
    abort.set()
    with pytest.raises(GenerationCancelled):
        provider.embed(body, abort)
    assert provider.active_generations == 0
    assert not provider._gen_gate.busy


def test_gguf_wire_preserves_mixed_content_and_batch_indices(info):
    from heylook_llm.providers.gguf_embeddings import embed
    import io
    body = EmbeddingRequest(model="embed", input=[{"content": [
        {"type": "text", "text": "scene"},
        {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "eA=="}},
        {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "eQ=="}},
    ]}])
    provider = Provider("embed", {}, False)
    provider._base_url = "http://localhost:1234"
    response = {"data": [{"embedding": [1.0]*768}], "usage": {"prompt_tokens": 4}}
    with resolved_items(body, info, None) as items:
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            vectors, tokens = embed(provider, items, None)
    wire = json.loads(send.call_args.args[0].data)
    parts = wire["input"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "input_video", "input_audio"]
    assert parts[1]["input_video"]["data"] == "data:video/mp4;base64,eA=="
    assert tokens == 4 and len(vectors) == 1


def test_mlx_text_batches_keep_order_lengths_and_the_failing_index():
    # Fake processor/model: the "vector" of a row is its unpadded token sum, so a
    # padding leak, a row mix-up or a wrong chunk slice changes the value.
    mx = pytest.importorskip("mlx.core")
    from types import SimpleNamespace
    from heylook_llm.providers import mlx_embeddings
    forwards = []
    class Processor:
        tokenizer = SimpleNamespace(padding_side="right")
        def apply_chat_template(self, conversations, **kw):
            single = isinstance(conversations[0], dict)
            rows = [[len(m["content"][0]["text"]) for m in c if m["role"] == "user"] for c in
                    ([conversations] if single else conversations)]
            width = max(r[0] for r in rows)
            ids = [[1] * r[0] + [7] * (width - r[0]) for r in rows]       # pad id 7: a leak shows
            mask = [[1] * r[0] + [0] * (width - r[0]) for r in rows]
            return {"input_ids": mx.array(ids), "attention_mask": mx.array(mask)}
    class Model:
        def __call__(self, input_ids, attention_mask, **kw):
            forwards.append(tuple(input_ids.shape))
            return SimpleNamespace(text_embeds=mx.stack([mx.sum(input_ids * attention_mask, axis=-1)] * 2, axis=-1))
    provider = SimpleNamespace(processor=Processor(), model=Model(), context_length=12)
    text = lambda n: ("", [{"type": "text", "text": "x" * n}])
    media = ("", [{"type": "text", "text": "xx"}, {"type": "image", "path": "/f.png"}])
    vectors, tokens = mlx_embeddings.embed(provider, [text(5), media, text(3), text(6), text(2)], None)
    assert [v[0] for v in vectors] == [5, 2, 3, 6, 2] and tokens == 18
    assert forwards[:2] == [(2, 5), (2, 6)]   # budget 12: [5,3] then [6,2], each cut to its longest row
    with pytest.raises(mlx_embeddings.InvalidGenerationRequest, match=r"input\[1\]"):
        mlx_embeddings.embed(provider, [text(2), text(13)], None)


def test_visual_padding_trims_to_the_longest_row_and_leaves_full_grids():
    mx = pytest.importorskip("mlx.core")
    from heylook_llm.providers.mlx_embeddings import trim_visual_padding
    pos = mx.array([[[0, 0], [1, 0], [-1, -1], [-1, -1]], [[0, 0], [1, 0], [2, 0], [-1, -1]]])
    out = trim_visual_padding({"pixel_values": mx.arange(8).reshape(2, 4, 1), "image_position_ids": pos})
    assert out["pixel_values"].shape == (2, 3, 1) and out["image_position_ids"][0, 2].tolist() == [-1, -1]
    full = {"pixel_values_videos": mx.zeros((1, 2, 1)), "video_position_ids": mx.zeros((1, 2, 2), dtype=mx.int32)}
    assert trim_visual_padding(full)["pixel_values_videos"].shape == (1, 2, 1)


def test_mlx_load_materializes_private_buffers_across_workers():
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    import mlx.nn as nn
    from heylook_llm.providers.mlx_embeddings import materialize_state
    from heylook_llm.streaming_utils import _executor_pool
    stream = mx.new_thread_local_stream(mx.default_device())
    first, second = _executor_pool.acquire(), _executor_pool.acquire()
    try:
        def build():
            with mx.stream(stream):
                model = nn.Module()
                model._private = mx.arange(4) + 1
                assert '_private' not in model.parameters()
                materialize_state(model)
                return model
        model = first.submit(build).result()
        def consume():
            with mx.stream(stream):
                return (model._private * 2).tolist()
        assert second.submit(consume).result() == [2,4,6,8]
    finally:
        _executor_pool.release(first)
        _executor_pool.release(second)
