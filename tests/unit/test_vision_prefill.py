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


@pytest.mark.parametrize("prompt_len", [1, 3, 40])
def test_a_penalty_sees_the_reply_and_never_the_prompt(prompt_len):
    """The SAME reply must reach a processor the same way whatever mlx-lm
    happened to prefill: one prompt token (the vision path), an uncached
    suffix (a prompt-cache hit) or the whole prompt (a cold text request).
    Before v2.0.60 those were three different penalty histories."""
    from heylook_llm.providers.common.generation_core import generated_only

    seen = []
    (scoped,) = generated_only([lambda tokens, logits: seen.append(tokens.tolist()) or logits])

    prompt, reply = list(range(100, 100 + prompt_len)), [7, 8, 9]
    for n in range(len(reply) + 1):          # call n samples reply token n
        scoped(mx.array(prompt + reply[:n]), mx.zeros((1, 16)))
    assert seen == [[], [7], [7, 8], [7, 8, 9]]
    assert generated_only([]) == [] and generated_only(None) is None
