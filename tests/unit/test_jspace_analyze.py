"""Unit tests for analyze.py helpers (content-block messages, stop tokens,
greedy first-token) with fakes — no model/download needed."""
import mlx.core as mx

from heylook_llm.jspace.analyze import (
    _eos_ids, _message_text, format_prompt, greedy_generate)


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
