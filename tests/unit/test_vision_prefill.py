# tests/unit/test_vision_prefill.py
"""`_prefill_language_model`: the vision path's prefill, driven with a
recording fake.

What these CAN pin is the shape of the calls -- which tokens go to the language
model, in what pieces, with which kwargs, and what the caller is told on the
way. What they CANNOT pin is whether a real model then generates the right
thing: cache and position state are invisible to a fake. That half is
`scripts/vlm_parity_probe.py`, which compares against mlx-vlm's own loop on a
real model.
"""
import pytest

mx = pytest.importorskip("mlx.core", reason="the prefill slices real mx arrays")

from heylook_llm.providers import mlx_provider as mp  # noqa: E402
from heylook_llm.providers.abort import AbortEvent  # noqa: E402
from heylook_llm.providers.base import GenerationFailed  # noqa: E402

HIDDEN = 4


class _Embeds:
    def __init__(self, inputs_embeds, **extra):
        self.inputs_embeds = inputs_embeds
        self._extra = extra

    def to_dict(self):
        return {"inputs_embeds": self.inputs_embeds, **self._extra}


class _Cache:
    state = mx.zeros((1,))


class _LanguageModel:
    def __init__(self, chunkable, abort_after=None, abort_event=None):
        self.calls = []
        self._chunkable = chunkable
        self._abort_after = abort_after
        self._abort_event = abort_event

    def chunked_prefill_policy(self, **_):
        return self._chunkable

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._abort_after is not None and len(self.calls) == self._abort_after:
            self._abort_event.set()


class _Model:
    def __init__(self, n, *, chunkable=True, embed_extra=None, **lm):
        self.language_model = _LanguageModel(chunkable, **lm)
        self._n = n
        self._embed_extra = embed_extra or {}
        self.embed_calls = []

    def get_input_embeddings(self, input_ids, pixel_values, **kwargs):
        self.embed_calls.append(kwargs)
        return _Embeds(mx.zeros((1, self._n, HIDDEN)), **self._embed_extra)


def _prefill(model, n, *, step, extras=None, abort_event=None):
    ids = mx.arange(n).reshape(1, n)
    return mp._prefill_language_model(
        model, model.language_model, ids, mx.zeros((1, 3, 2, 2)),
        mx.ones((1, n), dtype=mx.int32), extras or {}, [_Cache()],
        step_size=step, abort_event=abort_event)


@pytest.mark.parametrize("n,step", [(2, 8), (9, 4), (9, 8), (17, 4), (17, 16), (33, 1)])
def test_every_prompt_token_but_the_last_is_prefilled_exactly_once(n, step):
    model = _Model(n)
    event = AbortEvent()
    assert _prefill(model, n, step=step, abort_event=event) is True

    seen = []
    for call in model.language_model.calls:
        chunk = call["inputs"][0].tolist()
        assert 1 <= len(chunk) <= step
        assert call["inputs_embeds"].shape[1] == len(chunk) == call["n_to_process"]
        seen += chunk
    assert seen == list(range(n - 1)), "the last prompt token belongs to mlx-lm"
    assert event.prefill_progress() == (n - 1, n)


def test_no_mask_and_no_pixels_reach_the_language_model():
    """A caller-supplied mask REPLACES the model's own causal / sliding-window /
    bidirectional masks on the families that honour it, and an int32 ones mask
    is "no mask" -- non-causal attention. Both belong to the embedding step."""
    model = _Model(9)
    _prefill(model, 9, step=4, extras={"image_grid_thw": mx.array([[1, 2, 2]])})

    assert "mask" in model.embed_calls[0]
    for call in model.language_model.calls:
        assert "mask" not in call and "pixel_values" not in call
        assert "image_grid_thw" in call, "extras travel on, as they do upstream"


def test_progress_only_moves_forward():
    seen = []

    class _Recorder(AbortEvent):
        def set_prefill_progress(self, processed, total):
            seen.append((processed, total))

    _prefill(_Model(17), 17, step=4, abort_event=_Recorder())
    assert seen[0] == (0, 17) and seen[-1] == (16, 17)
    assert seen == sorted(seen) and {t for _, t in seen} == {17}


def test_an_abort_between_chunks_stops_the_prefill():
    event = AbortEvent()
    model = _Model(17, abort_after=2, abort_event=event)
    assert _prefill(model, 17, step=4, abort_event=event) is False
    assert len(model.language_model.calls) == 2


def test_a_family_that_refuses_chunking_gets_one_call_with_per_token_kwargs_cut():
    """gemma-4 with images: its vision blocks attend bidirectionally and its
    overlay silently no-ops when `mm_token_type_ids` and the mask differ in
    length -- so a single N-1 call must carry N-1 of them. Chunked mode is
    left alone: upstream passes them whole there and each family aligns them
    by cache offset."""
    n = 9
    per_token = {"mm_token_type_ids": mx.zeros((1, n), dtype=mx.int32),
                 "position_ids": mx.zeros((3, 1, n), dtype=mx.int32)}

    single = _Model(n, chunkable=False, embed_extra=dict(per_token))
    _prefill(single, n, step=4)
    (call,) = single.language_model.calls
    assert call["n_to_process"] == n - 1
    assert call["mm_token_type_ids"].shape == (1, n - 1)
    assert call["position_ids"].shape == (3, 1, n - 1)

    chunked = _Model(n, chunkable=True, embed_extra=dict(per_token))
    _prefill(chunked, n, step=4)
    assert all(c["mm_token_type_ids"].shape == (1, n) for c in chunked.language_model.calls)


def test_an_unknown_per_token_input_is_refused_rather_than_passed_unsliced():
    n = 9
    model = _Model(n, chunkable=False,
                   embed_extra={"some_new_per_token_thing": mx.zeros((1, n, 2))})
    with pytest.raises(GenerationFailed, match="some_new_per_token_thing"):
        _prefill(model, n, step=4)

    # ...while something that merely HAS a dimension of N is left alone:
    # cached image features are (tokens, hidden), and a hidden size can equal
    # a prompt length by coincidence.
    lookalike = _Model(n, chunkable=False,
                       embed_extra={"cached_image_features": mx.zeros((5, n))})
    _prefill(lookalike, n, step=4)
    assert lookalike.language_model.calls[0]["cached_image_features"].shape == (5, n)


class TestTheStrategyHandsMlxLmTheLastPromptToken:
    """`VLMVisionStrategy.generate` end to end with everything model-shaped
    faked: what it must hand `run_generation`, and what it must refuse before
    spending a forward pass."""

    IMAGE_TOKEN = 99

    def _drive(self, monkeypatch, ids, *, continuing=False, context_length=None):
        from contextlib import nullcontext
        from types import SimpleNamespace

        from heylook_llm.config import ChatRequest
        from heylook_llm.providers.base import GenerationChunk

        handed = {}

        def fake_run_generation(**kwargs):
            handed.update(kwargs)
            yield GenerationChunk(text="x", token=1, prompt_tokens=1, prompt_tps=1.0)

        monkeypatch.setattr(mp, "run_generation", fake_run_generation)
        monkeypatch.setattr(mp, "build_sampler", lambda *a, **k: ("sampler", ["proc"]))
        monkeypatch.setattr(mp, "vlm_prepare_inputs", lambda *a, **k: {
            "input_ids": mx.array([ids]), "pixel_values": mx.zeros((1, 3, 2, 2))})
        monkeypatch.setattr(mp, "make_prompt_cache", lambda m: [_Cache()])
        monkeypatch.setattr(mp, "wrap_language_model", lambda m: m.language_model)
        monkeypatch.setattr(mp, "wired_limit", lambda *a, **k: nullcontext())
        prefilled = {}
        monkeypatch.setattr(mp, "_prefill_language_model",
                            lambda *a, **k: prefilled.update(k) or True)

        strategy = mp.VLMVisionStrategy(model_id="m", context_length=context_length)
        monkeypatch.setattr(strategy, "_prepare_vlm_inputs_parallel",
                            lambda *a, **k: (["img"], "prompt", True, ["url"]))
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="fake", image_token_index=self.IMAGE_TOKEN),
            language_model=object())
        messages = [{"role": "user", "content": "hi"}]
        if continuing:
            messages.append({"role": "assistant", "content": "The goat is"})
        request = ChatRequest.model_validate({"model": "m", "messages": messages})
        chunks = list(strategy.generate(
            request, {"max_tokens": 8}, model, SimpleNamespace(tokenizer=object())))
        return handed, chunks

    def test_the_last_prompt_token_is_the_whole_prompt(self, monkeypatch):
        handed, chunks = self._drive(monkeypatch, [5, 6, 7, 8], continuing=True)
        assert handed["prompt_tokens"] == [8]
        assert handed["prefill_progress_offset"] == 3
        # Never passed before v2.0.55 -- a `pre_filled_cache` exemption stood in
        # for it, and removing that exemption without this would have cost
        # every vision continuation its seam space, silently.
        assert handed["continuing"] is True
        assert handed["processors"] == ["proc"], "the first token gets them too now"
        # mlx-lm sees a one-token prompt; the client must see the real one.
        assert [c.prompt_tokens for c in chunks] == [4]

    def test_a_prompt_ending_on_a_media_placeholder_is_refused(self, monkeypatch):
        with pytest.raises(GenerationFailed, match="placeholder"):
            self._drive(monkeypatch, [5, 6, self.IMAGE_TOKEN])

    def test_an_over_length_prompt_is_refused_before_any_compute(self, monkeypatch):
        from heylook_llm.providers.base import InvalidGenerationRequest
        with pytest.raises(InvalidGenerationRequest, match="context"):
            self._drive(monkeypatch, [5, 6, 7, 8], context_length=3)
