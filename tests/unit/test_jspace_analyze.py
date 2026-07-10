"""Unit tests for analyze.py helpers (content-block messages, stop tokens,
greedy first-token) with fakes — no model/download needed — plus the full
analyze() pipeline on a tiny random-weight gpt2 (heatmap per-cell top-k)."""
import mlx.core as mx

from heylook_llm.jspace import JSpaceLens
from heylook_llm.jspace.analyze import (
    _eos_ids, _message_text, analyze, format_prompt, greedy_generate)


def test_message_text_str_and_blocks():
    assert _message_text({"content": "hi"}) == "hi"
    assert _message_text({"content": [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {}},
        {"type": "text", "text": "b"}]}) == "ab"
    assert _message_text({"content": None}) == ""
    assert _message_text({}) == ""


class _FakeTok:
    bos_token_id = 2
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        body = [10, 11]
        return [self.bos_token_id] + body if add_special_tokens else body

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TPL " + " ".join(m["content"] for m in messages)

    def convert_tokens_to_ids(self, t):
        return -1


def test_format_prompt_content_blocks_no_crash_and_bos():
    # Regression (#7): list content must not crash the raw-completion join.
    tok = _FakeTok()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    ids = format_prompt(model=None, processor=tok, is_vlm=False, messages=msgs, chat=False)
    assert ids[0] == 2                      # BOS prepended
    ids_chat = format_prompt(model=None, processor=tok, is_vlm=False, messages=msgs, chat=True)
    assert ids_chat[0] == 2                 # chat path also fine with block content


def test_eos_ids_uses_plural():
    # Regression (#8): eos_token_ids (plural) must be honored.
    class Plural:
        eos_token_ids = [1, 106]

        def convert_tokens_to_ids(self, t):
            return 106 if t == "<end_of_turn>" else -1

    assert _eos_ids(Plural()) == {1, 106}


class _FixedArgmaxAdapter:
    """logits whose last-position argmax is always `token`."""
    def __init__(self, token, vocab=5):
        self.token, self.vocab = token, vocab

    def logits(self, ids):
        L = ids.shape[1]
        row = [0.0] * self.vocab
        row[self.token] = 10.0
        return mx.array([[row] * L])


def test_greedy_generate_returns_first_token():
    first, gen, logps = greedy_generate(_FixedArgmaxAdapter(3), [1, 2], max_tokens=2, eos_ids={1})
    assert first == 3 and gen == [3, 3] and len(logps) == 2


def test_greedy_generate_eos_first_token_reported():
    # Regression (#9): first_token is the model's real prediction even when it's EOS.
    first, gen, logps = greedy_generate(_FixedArgmaxAdapter(1), [1, 2], max_tokens=4, eos_ids={1})
    assert first == 1 and gen == [] and logps == []


# ---------------------------------------------------------------------------
# full analyze() pipeline on a tiny random-weight gpt2 (no downloads)
# ---------------------------------------------------------------------------

class _TinyTok:
    """Deterministic in-vocab fake tokenizer for the tiny gpt2 (vocab 50)."""
    bos_token_id = None
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [1 + (i % 40) for i in range(len(text.split()) + 2)]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{i}" for i in ids)

    def convert_tokens_to_ids(self, t):
        return -1


class _TinyProvider:
    is_vlm = False
    model_id = "tiny"

    def __init__(self, model):
        self.model = model
        self.processor = _TinyTok()


def _tiny_analyze(**kwargs):
    from mlx_lm.models.gpt2 import Model, ModelArgs
    mx.random.seed(0)
    args = ModelArgs(model_type="gpt2", n_ctx=64, n_embd=32, n_head=4, n_layer=3,
                     n_positions=64, layer_norm_epsilon=1e-5, vocab_size=50)
    model = Model(args)
    mx.eval(model.parameters())
    eye = mx.eye(32, dtype=mx.float32)
    lens = JSpaceLens(jacobians={0: eye, 1: eye, 2: eye},
                      source_layers=[0, 1, 2], d_model=32)
    return analyze(_TinyProvider(model), lens,
                   [{"role": "user", "content": "one two three four"}],
                   max_answer_tokens=2, top_k=4, **kwargs)


def test_analyze_heatmap_top_k_per_cell():
    out = _tiny_analyze(heatmap=True, heatmap_top_k=3)
    assert out["heatmap"], "expected a heatmap grid"
    for row in out["heatmap"]:
        for cell in row["cells"]:
            tk = cell["top_k"]
            assert len(tk) == 3
            logits = [c["logit"] for c in tk]
            assert logits == sorted(logits, reverse=True)
            assert tk[0]["token"] == cell["token"]   # top-1 agrees with the cell


def test_analyze_heatmap_default_has_no_top_k():
    # Back-compat: without heatmap_top_k, cells stay {token, entropy} only.
    out = _tiny_analyze(heatmap=True)
    for row in out["heatmap"]:
        for cell in row["cells"]:
            assert "top_k" not in cell
